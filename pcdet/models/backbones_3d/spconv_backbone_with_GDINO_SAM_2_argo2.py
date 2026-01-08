from functools import partial

import torch
import torch.nn as nn
import copy

from ...utils.spconv_utils import replace_feature, spconv
from ...utils import common_utils

from spconv.pytorch import functional as Fsp
from ..domain_general_models.pointnet2_encoder import pointnet2_perceptual_backbone

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

from groundingdino.util.inference import load_model, load_image, predict, annotate
import groundingdino.datasets.transforms as T
from groundingdino.util import box_ops
import torchvision.transforms as transforms
from PIL import Image
import cv2
from .GDINO_utils import gdino_processing, remove_overlap_boxes, map_class_name_to_id
from .GDINO_utils import score_down_logit, rename_str_in_phrases, score_threshold
from pcdet.utils import box_utils

from pcdet.utils.box2d_utils import pairwise_iou

import numpy as np

from .gen_3D_box_argo import generate_box_from_points
from .nusc_utils import project_3d_boxes_to_2d_torch, project_lidar_to_image_torch
from .nusc_utils import remove_points_in_boxes_efficient, shrink_mask_pure_torch
import pickle, os

CAMS_ = ['ring_front_right', 'ring_side_right', 'ring_rear_right', 'ring_rear_left', 'ring_side_left', 'ring_front_left', 'ring_front_center']
IMG_SIZE_CAMS_ = [(1550, 2048), (1550, 2048), (1550, 2048), (1550, 2048), (1550, 2048), (1550, 2048), (2048, 1550)] 
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


class VoxelBackBone8x_ImgGDinoSam_2_argo2(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, voxel_size, point_cloud_range, **kwargs):
        super().__init__()
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

        self.DINO_model = None 
        self.test_mask_read = self.model_cfg.get('TEST_MASK_READER', None)
        # load_model(self.model_cfg.DINO_CONFIG, self.model_cfg.DINO_PRETRAIN).cuda()


        self.img_transformer = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        if self.test_mask_read is None or self.training:
            sam = sam_model_registry[self.model_cfg.MODEL_TYPE](checkpoint=self.model_cfg.SAM_PRETRAIN).cuda()
            for param in sam.parameters():
                param.requires_grad = False
            self.predictor = SamPredictor(sam)
        
        # masks = self.segmentor.generate(image) # one image with RGB [0~255]
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

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
        # np.save('image.npy', image_cur)
        if use_gdino_box:
            # GDINO
            pil_image = Image.fromarray(image_cur)
            img_dino, _ = self.img_transformer(pil_image, None)
            # print(img_dino[:,0,0], image_cur[0,0,:])

            # common classes
            first_config = self.model_cfg.GDINO_PROCESS['first']
            boxes_dino_first, logits_dino_first, phrases_dino_first = gdino_processing(
                model=self.DINO_model,
                image=img_dino,
                txt_prompt=first_config['TEXT_PROMPT'],
                box_thresh=first_config['BOX_TRESHOLD'],
                txt_thresh=first_config['TEXT_TRESHOLD'],
                valid_labels=first_config["DT_labels"],
                denial_labels=first_config.get('repel_labels', None),
            )
            boxes_dino_first, logits_dino_first, phrases_dino_first = score_threshold(boxes_dino_first, logits_dino_first, phrases_dino_first, first_config['score_thresh_dict'])
            
            boxes_dino_xyxy_first = box_ops.box_cxcywh_to_xyxy(boxes_dino_first) * torch.Tensor([W, H, W, H])
            boxes_dino_trans_first = self.predictor.transform.apply_boxes_torch(boxes_dino_xyxy_first, image_cur.shape[:2])
            boxes_first, logits_first, phrases_first = boxes_dino_trans_first, logits_dino_first, phrases_dino_first

            # novel classes
            second_config = self.model_cfg.GDINO_PROCESS['second']
            boxes_dino_second, logits_dino_second, phrases_dino_second = gdino_processing(
                model=self.DINO_model,
                image=img_dino,
                txt_prompt=second_config['TEXT_PROMPT'],
                box_thresh=second_config['BOX_TRESHOLD'],
                txt_thresh=second_config['TEXT_TRESHOLD'],
                valid_labels=second_config["DT_labels"],
                denial_labels=second_config.get('repel_labels', None),
            )
            boxes_dino_second, logits_dino_second, phrases_dino_second = score_threshold(boxes_dino_second, logits_dino_second, phrases_dino_second, second_config['score_thresh_dict'])
            
            boxes_dino_xyxy_second = box_ops.box_cxcywh_to_xyxy(boxes_dino_second) * torch.Tensor([W, H, W, H])
            boxes_dino_trans_second = self.predictor.transform.apply_boxes_torch(boxes_dino_xyxy_second, image_cur.shape[:2])
            boxes_second, logits_second, phrases_second = boxes_dino_trans_second, logits_dino_second, phrases_dino_second
            # print(boxes_dino.shape, logits_dino, phrases_dino)

            # remove overlapped boxes (first)
            boxes_first, logits_first, phrases_first = remove_overlap_boxes(
                boxes_first, logits_first, phrases_first, iou_thresh=first_config['self_iou_thresh'])
            if first_config.get('scores_down', False):
                phrases_first, logits_first = score_down_logit(phrases_first, logits_first, first_config['scores_down'])
            if first_config.get('replace_labels', False):
                phrases_first = rename_str_in_phrases(phrases_first, first_config['replace_labels'])
            
            # remove overlapped boxes (second)
            boxes_second, logits_second, phrases_second = remove_overlap_boxes(
                boxes_second, logits_second, phrases_second, iou_thresh=second_config['self_iou_thresh'])
            if second_config.get('scores_down', False):
                phrases_second, logits_second = score_down_logit(phrases_second, logits_second, second_config['scores_down'])
            if second_config.get('replace_labels', False):
                phrases_second = rename_str_in_phrases(phrases_second, second_config['replace_labels'])

            # remove overlapped boxes between first and second
            boxes_first, logits_first, phrases_first, boxes_second, logits_second, phrases_second = remove_overlap_boxes(
                boxes_first, logits_first, phrases_first, boxes_second, logits_second, phrases_second
            )
            

            boxes_first, labels_first, score_rois_first = map_class_name_to_id(
                boxes_first, logits_first, phrases_first, first_config["DT_labels"])
            boxes_second, labels_second, score_rois_second = map_class_name_to_id(
                boxes_second, logits_second, phrases_second, second_config["DT_labels"])

            if len(boxes_first+boxes_second) !=0:
                boxes = torch.vstack(boxes_first+boxes_second)
                labels = labels_first + labels_second
                score_rois = score_rois_first + score_rois_second
            else:
                boxes, labels, score_rois = torch.zeros((0,4)), [], []
        
        else:
            original_boxes = copy.deepcopy(boxes)
            boxes = self.predictor.transform.apply_boxes_torch(boxes, image_cur.shape[:2])
            labels = labels.tolist()
            score_rois = score_rois

        # SAM
        if boxes.shape[0] > 0:
            with torch.no_grad(): 
                self.predictor.set_image(image_cur)
                masks, score_masks, _ = self.predictor.predict_torch(
                    point_coords = None,
                    point_labels = None,
                    boxes = boxes.cuda(),
                    multimask_output = False,
                )


        num_cls = self.model_cfg.PROMPT_DETECTOR.CLS_NUM
        image_seg_feat = torch.zeros((H, W, num_cls, max(boxes.shape[0], 1)), dtype=boxes.dtype) # H,W,CLS_NUM,ROI_NUM
        
        save_masks = []
        save_points = []
        save_labels = []
        save_score_rois = []
        save_score_seg = []
        save_dict = {}
        for i in range(boxes.shape[0]):
            score_feat = masks[i][0].float() * score_masks[i] # * score_rois[i].item()
            image_seg_feat[:,:,int(labels[i]-1), i] = score_feat

            save_masks.append(masks[i][0])
            save_points.append((boxes[i][0:2]+boxes[i][2:4])/2)
            save_labels.append(labels[i])
            save_score_rois.append(score_rois[i])
            save_score_seg.append(score_masks[i])

        save_dict = {'masks': save_masks, 'points': save_points, 'labels': save_labels, 'score_rois': save_score_rois, 'score_seg': save_score_seg}

        image_seg_feat = torch.max(image_seg_feat, dim=-1)[0].cuda()
        return image_seg_feat, save_dict


    def image_pc_feat_fusion(self, batch_dict, use_2d_gt):
        '''
        Args:
            batch_dict:
                'imges': tensor RBG(B, 3, H, W)

        '''
        if self.test_mask_read is None or self.training:
            image_batch = batch_dict['camera_imgs'].permute(0,1,3,4,2) # B x Cams x HW3
            image_batch_fc = batch_dict['camera_imgs_fc'].permute(0,1,3,4,2)
            image_batch *= 255.0
            image_batch_fc *= 255.0
            h, w = batch_dict['camera_imgs'].shape[3:]
            h_fc, w_fc = batch_dict['camera_imgs_fc'].shape[3:]

        feat_pools = []
        image_mask_dicts = []
        for i in range(batch_dict['batch_size']):
            image_mask_dicts_cur_batch = []
            feat_pools_cur_batch = []
            if (self.test_mask_read is not None) and (not self.training) and (not use_2d_gt):
                # print(batch_dict.keys())
                token = batch_dict['frame_id'][i]
                
                assert os.path.exists(self.test_mask_read['mask_dir']), 'mask_dir not exist!'
                for cam_ in CAMS_:
                    cam_pkl_file = os.path.join(self.test_mask_read['mask_dir'], token, '%s.pkl' % cam_)
                    if os.path.exists(cam_pkl_file):
                        cam_mask_info = pickle.load(open(cam_pkl_file, 'rb')) 
                        image_mask_dict = {}

                        image_mask_dict['masks'] = []
                        image_mask_dict['labels'] = []

                        for mask, label in zip(cam_mask_info['masks'], cam_mask_info['labels']):
                            if label in self.test_mask_read['label_to_id']:
                                image_mask_dict['masks'].append((mask !=0).int())
                                image_mask_dict['labels'].append(self.test_mask_read['label_to_id'][label])

                        image_mask_dicts_cur_batch.append(image_mask_dict)
                    else:
                        image_mask_dicts_cur_batch.append(None)

            else:
                for i_cam in range(image_batch.shape[1]+1):
                    if i_cam != image_batch.shape[1]:
                        # i_cam = 4
                        image_cur = image_batch[i, i_cam, :, :, :].cpu().numpy().astype(np.uint8)
                    else:
                        image_cur = image_batch_fc[i, 0, :, :, :].cpu().numpy().astype(np.uint8)

                    if self.training or use_2d_gt:
                        # preprocess the gt boxes
                        gt_boxes_3d = batch_dict['gt_boxes'][i, :, :7].clone().detach()
                        labels = batch_dict['gt_boxes'][i,:, -1].clone().detach()
                        gt_boxes_3d = gt_boxes_3d[batch_dict['gt_boxes_added'][i] == 0]
                        labels = labels[batch_dict['gt_boxes_added'][i] == 0]
                        # Reverse the transformations on boxes
                        if 'noise_scale' in batch_dict:
                            gt_boxes_3d[:, :6] /= batch_dict['noise_scale'][i]
                        if 'noise_rot' in batch_dict:
                            gt_boxes_3d[:, :3] = common_utils.rotate_points_along_z(gt_boxes_3d[:, :3].unsqueeze(0), -batch_dict['noise_rot'][i].unsqueeze(0))[0, :, :]
                            gt_boxes_3d[:, 6] -= batch_dict['noise_rot'][i]

                        if 'flip_x' in batch_dict:
                            gt_boxes_3d[:, 1] *= -1 if batch_dict['flip_x'][i] else 1
                            gt_boxes_3d[:, 6] *= -1 if batch_dict['flip_x'][i] else 1

                        if 'flip_y' in batch_dict:
                            gt_boxes_3d[:, 0] *= -1 if batch_dict['flip_y'][i] else 1
                            gt_boxes_3d[:, 6]  = -(gt_boxes_3d[:, 6] + np.pi) if batch_dict['flip_y'][i] else gt_boxes_3d[:, 6]
                        if 'shift_coor' in batch_dict:
                            gt_boxes_3d[:, :3] -= batch_dict['shift_coor'][i, :]

                        boxes, val_flags = project_3d_boxes_to_2d_torch(
                            gt_boxes_3d[:, :7],
                            batch_dict['lidar2image'][i, i_cam] if i_cam != image_batch.shape[1] else batch_dict['lidar2image_fc'][i, 0],
                            (h, w) if i_cam != image_batch.shape[1] else (h_fc, w_fc),
                        )
                        boxes = boxes[val_flags]
                        labels = labels[val_flags]
                        valid_mask = labels > 0
                        labels = labels[valid_mask]
                        scores = torch.ones_like(labels)
                        boxes = boxes[valid_mask]
                        image_seg_feat, image_mask_dict = self.gen_img_seg_feat_2D(image_cur, boxes, labels, scores)
                    else:  
                        raise NotImplementedError('Not testing yet!')        
                        image_seg_feat, image_mask_dict = self.gen_img_seg_feat_2D(image_cur)

    

                    feat_pools_cur_batch.append(image_seg_feat.unsqueeze(dim=0).to(device=batch_dict['camera_imgs'].device))
                    image_mask_dicts_cur_batch.append(image_mask_dict)
            
            feat_pools.append(feat_pools_cur_batch)
            image_mask_dicts.append(image_mask_dicts_cur_batch)
     
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

        if self.DINO_model == None and self.test_mask_read is None and not self.training:
            self.DINO_model = load_model(self.model_cfg.DINO_CONFIG, self.model_cfg.DINO_PRETRAIN).cuda()

        if output_add_roi:
            batch_dict, image_mask_dicts = self.image_pc_feat_fusion(batch_dict, use_2d_gt)

            import time
            init_time = time.time()
            obj_points_dicts = self.get_points_by_image_mask(image_mask_dicts, batch_dict['points'], batch_dict)
            print('get points time:', time.time() - init_time)

            init_time = time.time()
            added_boxes_dicts = generate_box_from_points(obj_points_dicts, 
                scale_info = batch_dict['noise_scale'] if 'noise_scale' in batch_dict else None, 
                box_fit_config=self.model_cfg.BOX_FIT_CONFIG
            )
            print('get bboxes time:', time.time() - init_time)
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
        # h, w = batch_dict['camera_imgs'].shape[3:]

        for b in range(batch_dict['batch_size']):
            if batch_dict.get('gt_boxes_added', None) is not None:
                points_with_aug_boxes = points[points[:,0]==b, 1:4].detach().clone()
                gt_boxes_3d_aug = batch_dict['gt_boxes'][b, :, :7].clone().detach()
                gt_boxes_3d_aug = gt_boxes_3d_aug[batch_dict['gt_boxes_added'][b] == 1]
                point_cur, _ =  remove_points_in_boxes_efficient(points_with_aug_boxes, gt_boxes_3d_aug) #xyz
            else:
                point_cur = points[points[:,0]==b, 1:4].detach().clone()
            
            point_org = point_cur.detach().clone()

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

            obj_points_dict = {}
            obj_points_list = []
            obj_labels_list = []
            
            for c in range(CAMS_.__len__()):
                if image_mask_dicts[b][c] is None:
                    continue
                h, w = IMG_SIZE_CAMS_[c]
                lidar2image = batch_dict['lidar2image'][b, c] if c != CAMS_.__len__()-1 else batch_dict['lidar2image_fc'][b, 0]
       
                point_2D, _ = project_lidar_to_image_torch(
                    point_cur[:, :3], lidar2image, (h, w)
                    )
                point_2D_int = point_2D.cpu().numpy().astype(int)

                filter_idx = (0<=point_2D_int[:, 1]) * (point_2D_int[:, 1] < h) * (0<=point_2D_int[:, 0]) * (point_2D_int[:, 0] < w)
                point_2D_int = point_2D_int[filter_idx]
                point_org_cur_cam = point_org[filter_idx]

                num_obj = image_mask_dicts[b][c]['masks'].__len__()
                for i_obj in range(num_obj):
                    mask = image_mask_dicts[b][c]['masks'][i_obj]

                    label = image_mask_dicts[b][c]['labels'][i_obj]
                    mask_feat = mask[point_2D_int[:, 1], point_2D_int[:, 0]]
                    point_obj = point_org_cur_cam[mask_feat==True, :]


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


