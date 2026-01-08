from functools import partial

import torch
import torch.nn as nn
import copy

from ...utils.spconv_utils import replace_feature, spconv
from ...utils import common_utils

from spconv.pytorch import functional as Fsp
from ..domain_general_models.pointnet2_encoder import pointnet2_perceptual_backbone

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
# from mobile_sam import sam_model_registry as mobile_sam_registry
# from mobile_sam import SamPredictor as MobileSamPredictor

from groundingdino.util.inference import load_model, load_image, predict, annotate
import groundingdino.datasets.transforms as T
from groundingdino.util import box_ops
import torchvision.transforms as transforms
from PIL import Image
import cv2
from .GDINO_utils2 import gdino_processing, remove_overlap_boxes, map_class_name_to_id
from .GDINO_utils2 import score_down_logit, rename_str_in_phrases

from pcdet.utils.box2d_utils import pairwise_iou

import numpy as np
# from .gen_3D_box import generate_box_from_points
from .gen_3D_box_kitti_faster import generate_box_from_points

def post_act_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,
                   conv_type='subm', norm_fn=None):

    if conv_type == 'subm':
        conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
    elif conv_type == 'spconv':
        conv = spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                   bias=False, indice_key=indice_key)
    elif conv_type == 'inverseconv':
        conv = spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False)
    else:
        raise NotImplementedError

    m = spconv.SparseSequential(
        conv,
        norm_fn(out_channels),
        nn.ReLU(),
    )

    return m


class SparseBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, norm_fn=None, downsample=None, indice_key=None):
        super(SparseBasicBlock, self).__init__()

        assert norm_fn is not None
        bias = norm_fn is not None
        self.conv1 = spconv.SubMConv3d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()
        self.conv2 = spconv.SubMConv3d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn2 = norm_fn(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = replace_feature(out, self.bn1(out.features))
        out = replace_feature(out, self.relu(out.features))

        out = self.conv2(out)
        out = replace_feature(out, self.bn2(out.features))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = replace_feature(out, out.features + identity.features)
        out = replace_feature(out, self.relu(out.features))

        return out


class VoxelBackBone8x_ImgGDinoSam_2_faster2(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, voxel_size, point_cloud_range, **kwargs):
        super().__init__()
        
        self.use_fp16_sam = True
        self.use_fp16_gdino = True
        
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(64, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )
        self.num_point_features = 128
        self.backbone_channels = {
            'x_conv1': 16,
            'x_conv2': 32,
            'x_conv3': 64,
            'x_conv4': 64
        }

        self.inv_idx =  torch.Tensor([2, 1, 0]).long().cuda()
        self.voxel_size = torch.Tensor(voxel_size).cuda()
        self.point_cloud_range = torch.Tensor(point_cloud_range).cuda()

        self.DINO_model = load_model(self.model_cfg.DINO_CONFIG, self.model_cfg.DINO_PRETRAIN).cuda() 
        self.DINO_model.eval()

        self.img_transformer = TorchImageTransform(
            short_edge=800, 
            max_size=1333, 
            device='cuda'
        )
        

        sam = sam_model_registry[self.model_cfg.MODEL_TYPE](checkpoint=self.model_cfg.SAM_PRETRAIN).cuda()
        
        sam.eval()
        
        # use_fp16:
        if self.use_fp16_sam:
            sam = sam.half()
        
        for param in sam.parameters():
            param.requires_grad = False
        self.predictor = SamPredictor(sam)
        sam.image_encoder.image_size = 512

        
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.channel_expander = nn.Sequential(
            nn.Conv1d(self.model_cfg.MAX_NUM_MASK, self.model_cfg.OUTPUT_CHANNEL, 1),
            norm_fn(self.model_cfg.OUTPUT_CHANNEL),
            nn.ReLU(),
        )
      
        self.BN_RELU = nn.Sequential(
            norm_fn(self.model_cfg.OUTPUT_CHANNEL+1),
            nn.ReLU(),
        )

        self.multi_layer_channel_expanders = nn.ModuleList()
        self.multi_layer_BN_RELUs = nn.ModuleList()
        for layer_id in self.model_cfg.MULTI_SRC_CHANNEL:
            output_channel = self.model_cfg.MULTI_SRC_CHANNEL[layer_id]

            single_channel_expander = nn.Sequential(
                nn.Conv1d(self.model_cfg.MAX_NUM_MASK, output_channel, 1),
                norm_fn(output_channel),
                nn.ReLU(),
            )
            single_BN_RELU = nn.Sequential(
                norm_fn(output_channel+1),
                nn.ReLU(),
            )
            self.multi_layer_channel_expanders.append(single_channel_expander)
            self.multi_layer_BN_RELUs.append(single_BN_RELU)
    
    def gen_img_seg_feat_2D(self, image_cur, boxes=None, labels=None, score_rois=None):
        '''
        Args:

        Return:
            image_seg_feat: torch.tensor (H, W, CLS_NUM)
        '''
        H, W = image_cur.shape[0], image_cur.shape[1]
        if boxes == None and labels == None and score_rois == None:
            use_gdino_box = True
        else:
            use_gdino_box = False

        if use_gdino_box:
            img_dino = self.img_transformer(image_cur)

            unified_config = self.model_cfg.GDINO_PROCESS['unified']
            
            with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
                boxes_dino, logits_dino, phrases_dino = gdino_processing(
                    model=self.DINO_model,
                    image=img_dino,
                    txt_prompt=unified_config['TEXT_PROMPT'],
                    box_thresh=unified_config['BOX_TRESHOLD'],
                    txt_thresh=unified_config['TEXT_TRESHOLD'],
                    valid_labels=unified_config["DT_labels"],
                    denial_labels=unified_config.get('repel_labels', None),
                )
        
            boxes_dino_xyxy = box_ops.box_cxcywh_to_xyxy(boxes_dino) * torch.Tensor([W, H, W, H])
            boxes = self.predictor.transform.apply_boxes_torch(boxes_dino_xyxy, image_cur.shape[:2])
            

            if unified_config.get('scores_down', False):
                phrases_dino, logits_dino = score_down_logit(
                    phrases_dino, 
                    logits_dino, 
                    unified_config['scores_down']
                )

            if unified_config.get('replace_labels', False):
                phrases_dino = rename_str_in_phrases(
                    phrases_dino, 
                    unified_config['replace_labels']
                )

            iou_thresh = unified_config.get('nms_iou_thresh', 0.75)
            boxes, logits_dino, phrases_dino = remove_overlap_boxes(
                boxes, logits_dino, phrases_dino, iou_thresh=iou_thresh
            )

            boxes, labels, score_rois = map_class_name_to_id(
                boxes, logits_dino, phrases_dino, 
                unified_config["DT_labels"]
            )

            if len(boxes) > 0:
                boxes, labels, score_rois = self.filter_boxes_by_score(boxes, labels, score_rois)
   
            
            if len(boxes) == 0:
                boxes, labels, score_rois = torch.zeros((0,4)), [], []
            else:
                boxes = torch.vstack(boxes)
        
        else:
            boxes = self.predictor.transform.apply_boxes_torch(boxes, image_cur.shape[:2])
            labels = labels.tolist()
            score_rois = score_rois

        
        if boxes.shape[0] > 0:
            
            self.predictor.set_image(image_cur)
            with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
                masks, score_masks, _ = self.predictor.predict_torch(
                    point_coords = None,
                    point_labels = None,
                    boxes = boxes.to(device=self.predictor.device),
                    multimask_output = False,
                )

        num_cls = self.model_cfg.PROMPT_DETECTOR.CLS_NUM
        if boxes.shape[0] > 0:
            image_seg_feat = torch.zeros((H, W, num_cls), device=masks.device, dtype=torch.float32)
            
            for i in range(boxes.shape[0]):
                label_idx = int(labels[i] - 1)
                if 0 <= label_idx < num_cls:
                    score_feat = masks[i][0].float() * score_masks[i]
                    image_seg_feat[:, :, label_idx] = torch.maximum(
                        image_seg_feat[:, :, label_idx],
                        score_feat
                    )
            
            save_dict = {
                'masks': [masks[i][0] for i in range(boxes.shape[0])],
                'points': [(boxes[i][0:2]+boxes[i][2:4])/2 for i in range(boxes.shape[0])],
                'labels': labels,
                'score_rois': score_rois,
                'score_seg': [score_masks[i] for i in range(boxes.shape[0])],
            }
        else:
            image_seg_feat = torch.zeros((H, W, num_cls), device='cuda', dtype=torch.float32)
            save_dict = {
                'masks': [],
                'points': [],
                'labels': [],
                'score_rois': [],
                'score_seg': [],
            }
        
        return image_seg_feat, save_dict

    def filter_boxes_by_score(self, boxes, labels, score_rois):
        """根据类别阈值过滤boxes"""
        if len(boxes) == 0 or not hasattr(self.model_cfg, 'final_score_cut'):
            return boxes, labels, score_rois
        
        score_thresholds = self.model_cfg.final_score_cut
        filtered_boxes, filtered_labels, filtered_scores = [], [], []
        
        for box, label, score in zip(boxes, labels, score_rois):
            threshold = score_thresholds.get(str(int(label)), 0.1)
            if score >= threshold:
                filtered_boxes.append(box)
                filtered_labels.append(label)
                filtered_scores.append(score)
        
        return filtered_boxes, filtered_labels, filtered_scores
    
    def construct_multimodal_features(self, x, x_rgb, batch_dict, voxel_stride_XYZ, fuse_sum=False):
        """
            Construct the multimodal features with both lidar sparse features and image features.
            Args:
                x: [N, C] lidar sparse features
                x_rgb: [b, c, h, w] image features
                batch_dict: input and output information during forward
                fuse_sum: bool, manner for fusion, True - sum, False - concat

            Return:
                image_with_voxelfeatures: [N, C] fused multimodal features
        """
        batch_index = x.indices[:, 0]
        spatial_indices = x.indices[:, 1:] * voxel_stride_XYZ[self.inv_idx]
        voxels_3d = spatial_indices * self.voxel_size[self.inv_idx] + self.point_cloud_range[:3][self.inv_idx]
        calibs = batch_dict['calib']
        batch_size = batch_dict['batch_size']
        h, w = batch_dict['images'].shape[2:]

        assert x_rgb.shape[2:] == batch_dict['images'].shape[2:]

        image_with_voxelfeatures = []
        voxels_2d_int_list = []
        filter_idx_list = []
        
        for b in range(batch_size):
            
            x_rgb_batch = x_rgb[b]

            calib = calibs[b]
            voxels_3d_batch = voxels_3d[batch_index==b]
            voxel_features_sparse = x.features[batch_index==b]

            # Reverse the point cloud transformations to the original coords.
            if 'noise_scale' in batch_dict:
                voxels_3d_batch[:, :3] /= batch_dict['noise_scale'][b]
            if 'noise_rot' in batch_dict:
                voxels_3d_batch = common_utils.rotate_points_along_z(voxels_3d_batch[:, self.inv_idx].unsqueeze(0), -batch_dict['noise_rot'][b].unsqueeze(0))[0, :, self.inv_idx]
            if 'flip_x' in batch_dict:
                voxels_3d_batch[:, 1] *= -1 if batch_dict['flip_x'][b] else 1
            if 'flip_y' in batch_dict:
                voxels_3d_batch[:, 2] *= -1 if batch_dict['flip_y'][b] else 1
            if 'shift_coor' in batch_dict:
                voxels_3d_batch[:, :3] -= batch_dict['shift_coor'][b, self.inv_idx]
                
            # print('h, w', h, w)
            voxels_2d, depth_2d = calib.lidar_to_img(voxels_3d_batch[:, self.inv_idx].cpu().numpy())

            voxels_2d_int = torch.Tensor(voxels_2d).to(x_rgb_batch.device).long()

            filter_idx = (0<=voxels_2d_int[:, 1]) * (voxels_2d_int[:, 1] < h) * (0<=voxels_2d_int[:, 0]) * (voxels_2d_int[:, 0] < w)            

            filter_idx_list.append(filter_idx)
            voxels_2d_int = voxels_2d_int[filter_idx]
            voxels_2d_int_list.append(voxels_2d_int)

            image_features_batch = torch.zeros((voxel_features_sparse.shape[0], x_rgb_batch.shape[0]), device=x_rgb_batch.device)
            image_features_batch[filter_idx] = x_rgb_batch[:, voxels_2d_int[:, 1], voxels_2d_int[:, 0]].permute(1, 0).contiguous()
            
            if fuse_sum:
                voxels_3d_batch_org = voxels_3d[batch_index==b]
                distance_3d_batch = torch.sqrt(voxels_3d_batch_org[:,0]**2 + voxels_3d_batch_org[:,1]**2 + voxels_3d_batch_org[:,2]**2)
                distance_std = torch.sqrt(self.point_cloud_range[0]**2 + self.point_cloud_range[1]**2 + self.point_cloud_range[2]**2)
                distance_3d_batch = (distance_3d_batch / distance_std).unsqueeze(dim=-1)
                image_with_voxelfeature = voxel_features_sparse + torch.cat((image_features_batch, distance_3d_batch),dim=-1) 
            else:
                image_with_voxelfeature = torch.cat([image_features_batch, voxel_features_sparse], dim=1)
            
            image_with_voxelfeatures.append(image_with_voxelfeature)
        
        image_with_voxelfeatures = torch.cat(image_with_voxelfeatures)
        return image_with_voxelfeatures

    def image_pc_feat_fusion(self, batch_dict, use_2d_gt):
        '''
        Args:
            batch_dict:
                'imges': tensor RBG(B, 3, H, W)

        '''
        
        from pcdet.utils import box_utils
        image_batch = batch_dict['images'].permute(0,2,3,1) # BHW3
        image_batch *= 255.0
        h, w = batch_dict['images'].shape[2:]

        feat_pools = []
        image_mask_dicts = []
        for i in range(image_batch.shape[0]):
            image_cur = image_batch[i, :, :, :].cpu().numpy().astype(np.uint8)
 
            if self.training or use_2d_gt:
                # print('Use image gt infos.')
                labels = batch_dict['gt_boxes'][i,:, -1]
                valid_mask = labels > 0
                labels = labels[valid_mask]
                scores = torch.ones_like(labels)
                boxes = batch_dict['gt_boxes2d'][i,valid_mask,:]
                image_seg_feat, image_mask_dict = self.gen_img_seg_feat_2D(image_cur, boxes, labels, scores)
            else:   
                image_seg_feat, image_mask_dict = self.gen_img_seg_feat_2D(image_cur)


            feat_pools.append(image_seg_feat.unsqueeze(dim=0).to(device=batch_dict['images'].device))
            image_mask_dicts.append(image_mask_dict)

        
        x_rgb = torch.cat(feat_pools, dim=0) # BHWC
        x_rgb = x_rgb.permute(0,3,1,2) # BCHW
        x_rgb = x_rgb.view(x_rgb.shape[0], x_rgb.shape[1], -1)

        
        x_img = self.channel_expander(x_rgb) 
        x_img = x_img.view(x_img.shape[0], x_img.shape[1], h ,w) # BSHW
         
        voxel_stride_XYZ = torch.tensor([8, 8, 16], dtype=torch.int, device=x_img.device)

        fused_feat_batch = self.construct_multimodal_features(
            x = batch_dict['encoded_spconv_tensor'], 
            x_rgb = x_img, 
            batch_dict=batch_dict, 
            voxel_stride_XYZ = voxel_stride_XYZ, 
            fuse_sum=True,
        )

        batch_dict['encoded_spconv_tensor'] = batch_dict['encoded_spconv_tensor'].replace_feature(self.BN_RELU(fused_feat_batch))

        for k, layer_id in enumerate(self.model_cfg.MULTI_SRC_CHANNEL):
            x_img = self.multi_layer_channel_expanders[k](x_rgb) 
            x_img = x_img.view(x_img.shape[0], x_img.shape[1], h ,w) # BSHW
            fused_feat_batch = self.construct_multimodal_features(
                x = batch_dict['multi_scale_3d_features'][layer_id], 
                x_rgb = x_img, 
                batch_dict=batch_dict, 
                voxel_stride_XYZ = torch.tensor([batch_dict['multi_scale_3d_strides'][layer_id]]*3, dtype=int, device=x_img.device), 
                fuse_sum=True,
            )
            batch_dict['multi_scale_3d_features'][layer_id] = \
                batch_dict['multi_scale_3d_features'][layer_id].replace_feature( \
                    self.multi_layer_BN_RELUs[k](fused_feat_batch)
                )
     
        return batch_dict, image_mask_dicts

    def forward(self, batch_dict, output_add_roi=True, use_2d_gt=False):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        
        batch_dict['input_sp_tensor'] = input_sp_tensor     
        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)

        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })
        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        })

        batch_dict, image_mask_dicts = self.image_pc_feat_fusion(batch_dict, use_2d_gt)
        
        if output_add_roi:
            obj_points_dicts = self.get_points_by_image_mask(image_mask_dicts, batch_dict['points'], batch_dict)

            added_boxes_dicts = generate_box_from_points(obj_points_dicts, 
                scale_info = batch_dict['noise_scale'] if 'noise_scale' in batch_dict else None, 
                box_fit_config=self.model_cfg.BOX_FIT_CONFIG
            )
            batch_dict['add_roi_boxes'] = added_boxes_dicts
        return batch_dict
    
    def get_points_by_image_mask(self, image_mask_dicts, points, batch_dict):
        '''
        Args:
            image_mask_dicts: lists of dict (batch_size)
                'masks': list (num_obj)
                    mask: numpy.array (H, W)
                'labels': list (num_obj)
                    label: int or float
            
            points: tensor (batch_idx, x, y, z)
        
        Return:
            obj_points_dicts: list of dicts (batch_size)
                'points_objs': list (num_obj)
                    points: numpy.array (N,3)
                'label_obj': list (num_obj)
                    labels: int or float
        '''

        obj_points_dicts = []
        h, w = batch_dict['images'].shape[2:]

        for b in range(len(image_mask_dicts)):
            point_cur = points[points[:,0]==b, 1:4] #xyz
            point_org = point_cur.detach().clone()
            calib = batch_dict['calib'][b]
            # Reverse the transformations on boxes
            if 'noise_scale' in batch_dict:
                point_cur /= batch_dict['noise_scale'][b]
            if 'noise_rot' in batch_dict:
                point_cur = common_utils.rotate_points_along_z(point_cur.unsqueeze(0), -batch_dict['noise_rot'][b].unsqueeze(0))[0, :, :]
            if 'flip_x' in batch_dict:
                point_cur[:, 1] *= -1 if batch_dict['flip_x'][b] else 1
            if 'flip_y' in batch_dict:
                point_cur[:, 0] *= -1 if batch_dict['flip_y'][b] else 1
            if 'shift_coor' in batch_dict:
                point_cur[:, :3] -= batch_dict['shift_coor'][b, :]

            point_2D, _ = calib.lidar_to_img(point_cur.cpu().numpy())
            point_2D_int = point_2D.astype(int)

            filter_idx = (0<=point_2D_int[:, 1]) * (point_2D_int[:, 1] < h) * (0<=point_2D_int[:, 0]) * (point_2D_int[:, 0] < w)
            point_2D_int = point_2D_int[filter_idx]
            point_org = point_org[filter_idx]

            obj_points_dict = {}
            obj_points_list = []
            obj_labels_list = []
            num_obj = image_mask_dicts[b]['masks'].__len__()
            for i_obj in range(num_obj):
                mask = image_mask_dicts[b]['masks'][i_obj]
                label = image_mask_dicts[b]['labels'][i_obj]
                mask_feat = mask[point_2D_int[:, 1], point_2D_int[:, 0]]
                point_obj = point_org[mask_feat==True, :]
                obj_points_list.append(point_obj.cpu().numpy())
                obj_labels_list.append(label)
         
            obj_points_dict = {
                'points_objs': obj_points_list,
                'label_objs': obj_labels_list,
            }

        obj_points_dicts.append(obj_points_dict)

        return obj_points_dicts

def removed_overlap_low_priority_obj(boxes, masks, labels, score_rois, score_masks, txt_to_cls_dict, objects=['person'], overlap_thresh=0.7):

    cls_tars = [txt_to_cls_dict[obj_] for obj_ in objects]

    del_idx=[]
    for cls_tar in cls_tars:
        for i, label in enumerate(labels):
            if label == cls_tar:
                mask_obj = masks[i, 0,:,:].reshape(1,-1) # (HxW)
                mask_all = masks[:, 0,:,:].reshape(masks.shape[0],-1) # (N, Hxw)
                overlaps = (mask_obj & mask_all).sum(dim=1).float()/mask_obj.sum().float()
                overlaps[i] = 0
                overlapped_flag = overlaps > overlap_thresh
                if overlapped_flag.sum() > 0:
                    indices = torch.nonzero(overlapped_flag, as_tuple=True)[0].tolist()
                    for label_overlapped in [labels[i] for i in indices]:
                        if label_overlapped not in cls_tars:
                            del_idx.append(i)
                            break

    if len(del_idx) > 0:
        # print('delete people:', del_idx, [labels[i] for i in del_idx])
        save_flag = torch.ones(masks.size(0), dtype=bool, device=masks.device)
        save_flag[del_idx] = False
        indices_save = torch.nonzero(save_flag, as_tuple=True)[0].tolist()
    
        return boxes[save_flag], masks[save_flag], \
        [labels[i] for i in indices_save],\
        [score_rois[i] for i in indices_save],\
        [score_masks[i] for i in indices_save]
    
    return boxes, masks, labels, score_rois, score_masks

def show_mask(mask, image, random_color=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.8])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.cpu().reshape(h, w, 1) * color.reshape(1, 1, -1)
    
    annotated_frame_pil = Image.fromarray(image).convert("RGBA")
    mask_image_pil = Image.fromarray((mask_image.cpu().numpy() * 255).astype(np.uint8)).convert("RGBA")

    return np.array(Image.alpha_composite(annotated_frame_pil, mask_image_pil))


import torch
import torch.nn.functional as F

class TorchImageTransform:
    def __init__(self, short_edge=800, max_size=1333, device='cuda'):
        self.short_edge = short_edge
        self.max_size = max_size
        self.device = device
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    
    def __call__(self, image_np):
        # numpy → tensor
        img = torch.from_numpy(image_np).to(self.device).permute(2, 0, 1).float()
        img = img.div(255.0).unsqueeze(0)  # (1, 3, H, W), [0-1]
        
        # resize
        h, w = img.shape[2:]
        if max(h, w) == h:
            new_h, new_w = self.short_edge, int(self.short_edge * w / h)
        else:
            new_h, new_w = int(self.short_edge * h / w), self.short_edge
        
        if max(new_h, new_w) > self.max_size:
            scale = self.max_size / max(new_h, new_w)
            new_h, new_w = int(new_h * scale), int(new_w * scale)
        
        if (new_h, new_w) != (h, w):
            img = F.interpolate(img, size=(new_h, new_w), mode='bilinear', align_corners=False)
        
        # normalize
        img = (img - self.mean) / self.std
        return img.squeeze(0)