from .detector3d_template import Detector3DTemplate
from .detector3d_template_ada import ActiveDetector3DTemplate
from .detector3d_template_multi_db import Detector3DTemplate_M_DB
from .detector3d_template_multi_db_3 import Detector3DTemplate_M_DB_3
from .detector3d_template_dg_2_src_d import Detector3DTemplate_DG_2_Source_Domain
from pcdet.utils import common_utils
from pcdet.ops.iou3d_nms import iou3d_nms_utils
from pcdet.config import cfg
from spconv.pytorch import functional as Fsp
from ...utils.spconv_utils import replace_feature, spconv
import torch

class VoxelRCNN(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset, class_names):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset, class_names=class_names)
        self.module_list = self.build_networks()

    def forward(self, batch_dict):
        # print(batch_dict.keys())
        # print(batch_dict['frame_id'])
        # # print(batch_dict['frame_id'], batch_dict['metadata'])
        # import pickle
        # data_save ={
        #     'frame_id': batch_dict['frame_id'],
        #     'gt_boxes': batch_dict['gt_boxes'].cpu().numpy(),
        #     'points': batch_dict['points'].cpu().numpy(),
        #     # 'camera_imgs': batch_dict['camera_imgs'].cpu().numpy(),
        #     # 'camera_imgs_fc': batch_dict['camera_imgs_fc'].cpu().numpy(),
        #     # 'lidar2image': batch_dict['lidar2image'].cpu().numpy(),
        #     # 'lidar2image_fc': batch_dict['lidar2image_fc'].cpu().numpy(),
        # }
        # with open('argo_batch_dict.pkl', 'wb') as f:
        #     pickle.dump(data_save, f)
        # raise NotImplementedError('stop')

        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}
        loss = 0
        
        loss_rpn, tb_dict = self.dense_head.get_loss()
        loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)

        loss = loss + loss_rpn + loss_rcnn

        if hasattr(self.backbone_3d, 'get_loss'):
            loss_backbone3d, tb_dict = self.backbone_3d.get_loss(tb_dict)
            loss += loss_backbone3d
            
        return loss, tb_dict, disp_dict

class VoxelRCNN_adv(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    def forward(self, batch_dict):
        for i, cur_module in enumerate(self.module_list):
            if i < 5:
                batch_dict = cur_module(batch_dict)

        if self.training:
            loss  = self.get_adversarial_loss(batch_dict)

            ret_dict = {
                'loss': loss
            }
            return ret_dict, {}, {}

        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts



    def get_adversarial_loss(self, data_dict):

        # data_dict['rois'] # B, N, 7+C
        # data_dict['roi_scores'] # B, N
        # data_dict['roi_labels'] # B, N
        # print(data_dict['roi_labels'])

        batch_size = data_dict['batch_size']

        assert batch_size == data_dict['rois'].shape[0] 

        prp_box = data_dict['rois'] 
        prp_scr = data_dict['roi_scores']
        prp_labels = data_dict['roi_labels'] 
        # print(data_dict['rois'].shape, prp_scr.shape, prp_labels.shape)

        gt_boxes = data_dict['gt_boxes'] # (batch_size, n_obj, 8)
        # print(gt_boxes)
             
        loss = 0
        for i_sample in range(batch_size):
            box_cur_sample = prp_box[i_sample, :, :]
            scr_cur_sample = prp_scr[i_sample, :]
            lab_cur_sample = prp_labels[i_sample, :]

            gt_boxes_cur = gt_boxes[i_sample, :, :7]
            gt_label_cur = gt_boxes[i_sample, :, 7].view(-1)

            ## select relevant prps and objects of interest in prediction
            relevant_index = (scr_cur_sample > 0.1) & (lab_cur_sample != 0)
            # print(relevant_index.sum())
            scr_cur_sample = scr_cur_sample[relevant_index]
            box_cur_sample = box_cur_sample[relevant_index, :]
            lab_cur_sample = lab_cur_sample[relevant_index]

            ## select Car in GT
            relevant_index = (gt_label_cur != 0)
            # print(relevant_index.sum())
            gt_boxes_cur = gt_boxes_cur[relevant_index, :]
            gt_label_cur = gt_label_cur[relevant_index]

            # iou
            iou = iou3d_nms_utils.boxes_iou3d_gpu(box_cur_sample, gt_boxes_cur)
            score_loss = - torch.log(1 - scr_cur_sample).view(1, -1)
            loss_cur_sample = torch.sum(torch.mm(score_loss, iou))

            loss += loss_cur_sample / iou.shape[0]
            # print(loss)
        return loss / batch_size

class VoxelRCNN_AE(Detector3DTemplate_DG_2_Source_Domain):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    def forward(self, batch_dict):
        
        
        # print(batch_dict['input_sp_tensor'].indices[33:43])
        # print(batch_dict['upsam_sp_tensor'].indices[33:43])
        # print(batch_dict['multi_scale_3d_features']['x_conv4'].indice_dict)

        if self.training:
            
            # reconstruction preparations
            batch_dict_org = {
                'points': batch_dict['points_org'],
                'batch_size': batch_dict['batch_size'],
            }
            for idx_module, cur_module in enumerate(self.module_list):
                if idx_module in [0, 1]:
                    batch_dict_org = cur_module(batch_dict_org)
            # output => ['input_sp_tensor'], ['multi_scale_3d_features']['x_conv4']

            # normal detection loss
            for idx_module, cur_module in enumerate(self.module_list):
                if idx_module == 2:
                    continue
                batch_dict = cur_module(batch_dict)
            loss_det, tb_dict, disp_dict = self.get_training_loss()
        
            # reconstruction loss
            x_conv4_remain = batch_dict['multi_scale_3d_features']['x_conv4'] 
            x_conv4_all = batch_dict_org['multi_scale_3d_features']['x_conv4']
            assert x_conv4_all.spatial_shape == x_conv4_remain.spatial_shape
            scalar_coord_remain = x_conv4_remain.indices[:, 0] * x_conv4_all.spatial_shape[0] * x_conv4_all.spatial_shape[1] * x_conv4_all.spatial_shape[2] + \
                                    x_conv4_remain.indices[:, 1] * x_conv4_all.spatial_shape[1] * x_conv4_all.spatial_shape[2] + \
                                    x_conv4_remain.indices[:, 2] * x_conv4_all.spatial_shape[2] + \
                                    x_conv4_remain.indices[:, 3]
            scalar_coord_all = x_conv4_all.indices[:, 0] * x_conv4_all.spatial_shape[0] * x_conv4_all.spatial_shape[1] * x_conv4_all.spatial_shape[2] + \
                                    x_conv4_all.indices[:, 1] * x_conv4_all.spatial_shape[1] * x_conv4_all.spatial_shape[2] + \
                                    x_conv4_all.indices[:, 2] * x_conv4_all.spatial_shape[2] + \
                                    x_conv4_all.indices[:, 3]
            mask_tran = (scalar_coord_all.view(-1,1) == scalar_coord_remain.view(1,-1)).float()
            
            # print('@@@',mask_tran.sum())
            features_tran = mask_tran @ x_conv4_remain.features.clone()
            # print('remain:', x_conv4_remain.indices.shape)
            # print('all:', x_conv4_all.indices.shape) 
            # print('ratio:', mask_tran.sum()/x_conv4_all.indices.shape[0])

            x_conv4_remain_ups = spconv.SparseConvTensor(
                features=features_tran,
                indices=x_conv4_all.indices,
                spatial_shape=x_conv4_all.spatial_shape,
                batch_size=x_conv4_all.batch_size,
                grid=x_conv4_all.grid,
                voxel_num=x_conv4_all.grid,
                indice_dict = x_conv4_all.indice_dict,
            )
            x_sp_ups = self.decoder_3d(x_conv4_remain_ups).features
            batch_fun = torch.nn.BatchNorm1d(3, affine=False).cuda()
            x_sp_gts = batch_fun(batch_dict_org['input_sp_tensor'].features)
            loss_recon = torch.mean((x_sp_ups-x_sp_gts)**2)
            
            # print('after upsampling')
            # print('remain:', x_sp_ups.indices.shape)
            # print('remain:', x_sp_ups.features[44:49])
            # print('all:', batch_dict_org['input_sp_tensor'].indices.shape) 
            # print('all:', x_sp_gts[44:49])  
            
            # raise TypeError('stop')

            ret_dict = {
                'loss': {
                    'loss_det': loss_det,
                    'loss_recon': loss_recon,
                }             
            }
            return ret_dict, tb_dict, disp_dict

        else:
            for idx_module, cur_module in enumerate(self.module_list):
                if idx_module == 2:
                    continue
                batch_dict = cur_module(batch_dict)

            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}
        loss = 0
        
        loss_rpn, tb_dict = self.dense_head.get_loss()
        loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)

        loss = loss + loss_rpn + loss_rcnn

        if hasattr(self.backbone_3d, 'get_loss'):
            loss_backbone3d, tb_dict = self.backbone_3d.get_loss(tb_dict)
            loss += loss_backbone3d
            
        return loss, tb_dict, disp_dict
    
class VoxelRCNN_AE_TestTimeTraining(Detector3DTemplate_DG_2_Source_Domain):
    def __init__(self, model_cfg, num_class, dataset, class_names):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset, class_names=class_names)
        self.module_list = self.build_networks()

    def trans_conv4_feat_with_indice(self, batch_dict, batch_dict_org):

        x_conv4_remain = batch_dict['multi_scale_3d_features']['x_conv4'] 
        x_conv4_all = batch_dict_org['multi_scale_3d_features']['x_conv4']
        assert x_conv4_all.spatial_shape == x_conv4_remain.spatial_shape
        scalar_coord_remain = x_conv4_remain.indices[:, 0] * x_conv4_all.spatial_shape[0] * x_conv4_all.spatial_shape[1] * x_conv4_all.spatial_shape[2] + \
                                x_conv4_remain.indices[:, 1] * x_conv4_all.spatial_shape[1] * x_conv4_all.spatial_shape[2] + \
                                x_conv4_remain.indices[:, 2] * x_conv4_all.spatial_shape[2] + \
                                x_conv4_remain.indices[:, 3]
        scalar_coord_all = x_conv4_all.indices[:, 0] * x_conv4_all.spatial_shape[0] * x_conv4_all.spatial_shape[1] * x_conv4_all.spatial_shape[2] + \
                                x_conv4_all.indices[:, 1] * x_conv4_all.spatial_shape[1] * x_conv4_all.spatial_shape[2] + \
                                x_conv4_all.indices[:, 2] * x_conv4_all.spatial_shape[2] + \
                                x_conv4_all.indices[:, 3]
        mask_tran = (scalar_coord_all.view(-1,1) == scalar_coord_remain.view(1,-1)).float()      
        features_tran = mask_tran @ x_conv4_remain.features
        x_conv4_remain_ups = spconv.SparseConvTensor(
            features=features_tran,
            indices=x_conv4_all.indices,
            spatial_shape=x_conv4_all.spatial_shape,
            batch_size=x_conv4_all.batch_size,
            grid=x_conv4_all.grid,
            voxel_num=x_conv4_all.grid,
            indice_dict = x_conv4_all.indice_dict,
        )
        return x_conv4_remain_ups

    def forward(self, batch_dict):
        
        
        # print(batch_dict['input_sp_tensor'].indices[33:43])
        # print(batch_dict['upsam_sp_tensor'].indices[33:43])
        # print(batch_dict['multi_scale_3d_features']['x_conv4'].indice_dict)

        if self.training:
            
            # reconstruction preparations
            batch_dict_org = {
                'points': batch_dict['points_org'],
                'batch_size': batch_dict['batch_size'],
            }
            for idx_module, cur_module in enumerate(self.module_list):
                if idx_module in [0, 1]:
                    batch_dict_org = cur_module(batch_dict_org)
            # output => ['input_sp_tensor'], ['multi_scale_3d_features']['x_conv4']

            # normal detection
            for idx_module, cur_module in enumerate(self.module_list):
                if idx_module in [0, 1]:
                    batch_dict = cur_module(batch_dict)
            # loss_det, tb_dict, disp_dict = self.get_training_loss()
        
            # reconstruction loss
            x_conv4_for_ups = self.trans_conv4_feat_with_indice(batch_dict, batch_dict_org)
            x_sp_ups = self.decoder_3d(x_conv4_for_ups)
            loss_recon = self.decoder_3d.get_loss(x_sp_ups, batch_dict_org['input_sp_tensor'])

            ret_dict = {
                'loss': {}             
            }
            ret_dict['loss'].update(loss_recon)
            return ret_dict, {}, {}

        else:
            for idx_module, cur_module in enumerate(self.module_list):
                if idx_module == 2:
                    continue
                batch_dict = cur_module(batch_dict)

            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts


class VoxelRCNN_AE_2ChanDet(Detector3DTemplate_DG_2_Source_Domain):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    def forward(self, batch_dict):
        
        
        # print(batch_dict['input_sp_tensor'].indices[33:43])
        # print(batch_dict['upsam_sp_tensor'].indices[33:43])
        # print(batch_dict['multi_scale_3d_features']['x_conv4'].indice_dict)

        if self.training:
           
            # data_org detection
            batch_dict_org = {}
            batch_dict_org.update(batch_dict)
            batch_dict_org.update({'points': batch_dict['points_org']})
            for idx_module, cur_module in enumerate(self.module_list):
                if idx_module == 2:
                    continue
                batch_dict_org = cur_module(batch_dict_org)
            loss_det_org, tb_dict_org, disp_dict_org = self.get_training_loss()   
            # output => ['input_sp_tensor'], ['multi_scale_3d_features']['x_conv4']

            # normal detection loss
            for idx_module, cur_module in enumerate(self.module_list):
                if idx_module == 2:
                    continue
                batch_dict = cur_module(batch_dict)
            loss_det, tb_dict, disp_dict = self.get_training_loss()
        
            # reconstruction loss
            x_conv4_remain = batch_dict['multi_scale_3d_features']['x_conv4'] 
            x_conv4_all = batch_dict_org['multi_scale_3d_features']['x_conv4']
            assert x_conv4_all.spatial_shape == x_conv4_remain.spatial_shape
            scalar_coord_remain = x_conv4_remain.indices[:, 0] * x_conv4_all.spatial_shape[0] * x_conv4_all.spatial_shape[1] * x_conv4_all.spatial_shape[2] + \
                                    x_conv4_remain.indices[:, 1] * x_conv4_all.spatial_shape[1] * x_conv4_all.spatial_shape[2] + \
                                    x_conv4_remain.indices[:, 2] * x_conv4_all.spatial_shape[2] + \
                                    x_conv4_remain.indices[:, 3]
            scalar_coord_all = x_conv4_all.indices[:, 0] * x_conv4_all.spatial_shape[0] * x_conv4_all.spatial_shape[1] * x_conv4_all.spatial_shape[2] + \
                                    x_conv4_all.indices[:, 1] * x_conv4_all.spatial_shape[1] * x_conv4_all.spatial_shape[2] + \
                                    x_conv4_all.indices[:, 2] * x_conv4_all.spatial_shape[2] + \
                                    x_conv4_all.indices[:, 3]
            mask_tran = (scalar_coord_all.view(-1,1) == scalar_coord_remain.view(1,-1)).float()
            
            # print('@@@',mask_tran.sum())
            features_tran = mask_tran @ x_conv4_remain.features
            # print('remain:', x_conv4_remain.indices.shape)
            # print('all:', x_conv4_all.indices.shape) 
            # print('ratio:', mask_tran.sum()/x_conv4_all.indices.shape[0])

            x_conv4_remain_ups = spconv.SparseConvTensor(
                features=features_tran,
                indices=x_conv4_all.indices,
                spatial_shape=x_conv4_all.spatial_shape,
                batch_size=x_conv4_all.batch_size,
                grid=x_conv4_all.grid,
                voxel_num=x_conv4_all.grid,
                indice_dict = x_conv4_all.indice_dict,
            )
            x_sp_ups = self.decoder_3d(x_conv4_remain_ups).features

            batch_fun = torch.nn.BatchNorm1d(3, affine=False).cuda()
            x_sp_gts = batch_fun(batch_dict_org['input_sp_tensor'].features)

            loss_recon = torch.mean((x_sp_ups-x_sp_gts)**2)


            ret_dict = {
                'loss': {
                    'loss_det_org': loss_det_org,
                    'loss_det': loss_det,
                    'loss_recon': loss_recon,
                }             
            }
            return ret_dict, tb_dict, disp_dict

        else:
            for idx_module, cur_module in enumerate(self.module_list):
                if idx_module == 2:
                    continue
                batch_dict = cur_module(batch_dict)

            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}
        loss = 0
        
        loss_rpn, tb_dict = self.dense_head.get_loss()
        loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)

        loss = loss + loss_rpn + loss_rcnn

        if hasattr(self.backbone_3d, 'get_loss'):
            loss_backbone3d, tb_dict = self.backbone_3d.get_loss(tb_dict)
            loss += loss_backbone3d
            
        return loss, tb_dict, disp_dict


class VoxelRCNN_AE_ChanOrg(Detector3DTemplate_DG_2_Source_Domain):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()
    
    def trans_conv4_feat_with_indice(self, batch_dict, batch_dict_org):

        x_conv4_remain = batch_dict['multi_scale_3d_features']['x_conv4'] 
        x_conv4_all = batch_dict_org['multi_scale_3d_features']['x_conv4']
        assert x_conv4_all.spatial_shape == x_conv4_remain.spatial_shape
        scalar_coord_remain = x_conv4_remain.indices[:, 0] * x_conv4_all.spatial_shape[0] * x_conv4_all.spatial_shape[1] * x_conv4_all.spatial_shape[2] + \
                                x_conv4_remain.indices[:, 1] * x_conv4_all.spatial_shape[1] * x_conv4_all.spatial_shape[2] + \
                                x_conv4_remain.indices[:, 2] * x_conv4_all.spatial_shape[2] + \
                                x_conv4_remain.indices[:, 3]
        scalar_coord_all = x_conv4_all.indices[:, 0] * x_conv4_all.spatial_shape[0] * x_conv4_all.spatial_shape[1] * x_conv4_all.spatial_shape[2] + \
                                x_conv4_all.indices[:, 1] * x_conv4_all.spatial_shape[1] * x_conv4_all.spatial_shape[2] + \
                                x_conv4_all.indices[:, 2] * x_conv4_all.spatial_shape[2] + \
                                x_conv4_all.indices[:, 3]
        mask_tran = (scalar_coord_all.view(-1,1) == scalar_coord_remain.view(1,-1)).float()      
        features_tran = mask_tran @ x_conv4_remain.features
        x_conv4_remain_ups = spconv.SparseConvTensor(
            features=features_tran,
            indices=x_conv4_all.indices,
            spatial_shape=x_conv4_all.spatial_shape,
            batch_size=x_conv4_all.batch_size,
            grid=x_conv4_all.grid,
            voxel_num=x_conv4_all.grid,
            indice_dict = x_conv4_all.indice_dict,
        )
        return x_conv4_remain_ups

    def forward(self, batch_dict):

        if self.training:
           
            # data_org detection loss
            batch_dict_org = {}
            batch_dict_org.update(batch_dict)
            batch_dict_org.update({'points': batch_dict['points_org']})
            for idx_module, cur_module in enumerate(self.module_list):
                if idx_module == 2:
                    continue
                batch_dict_org = cur_module(batch_dict_org)
            loss_det_org, tb_dict, disp_dict = self.get_training_loss()   
            # output => ['input_sp_tensor'], ['multi_scale_3d_features']['x_conv4']

            # normal detection 
            for idx_module, cur_module in enumerate(self.module_list):
                if idx_module >= 2:
                    continue
                batch_dict = cur_module(batch_dict)
        
            # reconstruction loss
            x_conv4_for_ups = self.trans_conv4_feat_with_indice(batch_dict, batch_dict_org)
            x_sp_ups = self.decoder_3d(x_conv4_for_ups)
            loss_recon = self.decoder_3d.get_loss(x_sp_ups, batch_dict_org['input_sp_tensor'])

            ret_dict = {
                'loss': {
                    'loss_det_org': loss_det_org,
                }             
            }
            ret_dict['loss'].update(loss_recon)
            return ret_dict, tb_dict, disp_dict

        else:
            for idx_module, cur_module in enumerate(self.module_list):
                if idx_module == 2:
                    continue
                batch_dict = cur_module(batch_dict)

            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}
        loss = 0
        
        loss_rpn, tb_dict = self.dense_head.get_loss()
        loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)

        loss = loss + loss_rpn + loss_rcnn

        if hasattr(self.backbone_3d, 'get_loss'):
            loss_backbone3d, tb_dict = self.backbone_3d.get_loss(tb_dict)
            loss += loss_backbone3d
            
        return loss, tb_dict, disp_dict
        
class VoxelRCNN_DG_2_Source_Domain(Detector3DTemplate_DG_2_Source_Domain):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()
    
    def merge_DIR_and_DSR(self, batch_dict):    
        # print('DIR ID', id(batch_dict['encoded_spconv_tensor_DIR']) )
        # print('DSR ID', id(batch_dict['encoded_spconv_tensor_DSR']) )
        # print('**')
        batch_dict.update({
            'encoded_spconv_tensor': Fsp.sparse_add(batch_dict['encoded_spconv_tensor_DIR'], batch_dict['encoded_spconv_tensor_DSR']),
            'encoded_spconv_tensor_stride': batch_dict['encoded_spconv_tensor_stride_DIR']
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': Fsp.sparse_add(batch_dict['multi_scale_3d_features_DIR']['x_conv1'], batch_dict['multi_scale_3d_features_DSR']['x_conv1']),
                'x_conv2': Fsp.sparse_add(batch_dict['multi_scale_3d_features_DIR']['x_conv2'], batch_dict['multi_scale_3d_features_DSR']['x_conv2']),
                'x_conv3': Fsp.sparse_add(batch_dict['multi_scale_3d_features_DIR']['x_conv3'], batch_dict['multi_scale_3d_features_DSR']['x_conv3']),
                'x_conv4': Fsp.sparse_add(batch_dict['multi_scale_3d_features_DIR']['x_conv4'], batch_dict['multi_scale_3d_features_DSR']['x_conv4']),
            }
        })
        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': batch_dict['multi_scale_3d_strides_DIR']['x_conv1'],
                'x_conv2': batch_dict['multi_scale_3d_strides_DIR']['x_conv2'],
                'x_conv3': batch_dict['multi_scale_3d_strides_DIR']['x_conv3'],
                'x_conv4': batch_dict['multi_scale_3d_strides_DIR']['x_conv4'],
            }
        })
        return batch_dict

    def transfer_DSR(self, batch_dict, batch_dict_target):
        batch_dict.update({
            'encoded_spconv_tensor_DSR': batch_dict_target['encoded_spconv_tensor_DSR'],
            'encoded_spconv_tensor_stride_DSR': batch_dict_target['encoded_spconv_tensor_stride_DSR'],
            'multi_scale_3d_features_DSR': batch_dict_target['multi_scale_3d_features_DSR'],
            'multi_scale_3d_strides_DSR': batch_dict_target['multi_scale_3d_strides_DSR'],
            'DSR_domain': batch_dict_target['DSR_domain'],
        })
        return batch_dict

    def forward(self, batch_dict_1, batch_dict_2=None):
  
        all_datasets = ['waymo', 'kitti', 'nuscenes']

        if self.training: # when training
            assert batch_dict_1['dataset_domain'] in all_datasets and batch_dict_2['dataset_domain'] in all_datasets
            # first 4 moduels: ['vfe', 'backbone_3d_src_1', 'backbone_3d_src_2', 'backbone_3d']
            
            for idx_module, cur_module in enumerate(self.module_list):
                if idx_module > 3:
                    break
                if (idx_module in [1, 2]) and (cur_module.domain_name != batch_dict_1['dataset_domain']):
                    continue
                batch_dict_1 = cur_module(batch_dict_1)
            
            for idx_module, cur_module in enumerate(self.module_list):
                if idx_module > 3:
                    break
                if (idx_module in [1, 2]) and (cur_module.domain_name != batch_dict_2['dataset_domain']):
                    continue
                batch_dict_2 = cur_module(batch_dict_2)

            # print(batch_dict_1['DSR_domain'])
            # print(batch_dict_2['DSR_domain'])
            
            batch_dict_1 = self.merge_DIR_and_DSR(batch_dict_1)
            for idx_module, cur_module in enumerate(self.module_list):
                if idx_module <= 3:
                    continue
                batch_dict_1 = cur_module(batch_dict_1)
            loss1, tb_dict, disp_dict = self.get_training_loss()
            
            batch_dict_2 = self.merge_DIR_and_DSR(batch_dict_2)
            for idx_module, cur_module in enumerate(self.module_list):
                if idx_module <= 3:
                    continue
                batch_dict_2 = cur_module(batch_dict_2)
            loss2, _, _ = self.get_training_loss()

            DSR_1 = self.transfer_DSR({}, batch_dict_1)
            DSR_2 = self.transfer_DSR({}, batch_dict_2)

            # DIR1 + DSR2   
            batch_dict_1 = self.transfer_DSR(batch_dict_1, DSR_2)
            batch_dict_1 = self.merge_DIR_and_DSR(batch_dict_1)
            for idx_module, cur_module in enumerate(self.module_list):
                if idx_module <= 3:
                    continue
                batch_dict_1 = cur_module(batch_dict_1)
            loss3, _, _ = self.get_training_loss()
            
            batch_dict_2 = self.transfer_DSR(batch_dict_2, DSR_1)
            batch_dict_2 = self.merge_DIR_and_DSR(batch_dict_2)
            for idx_module, cur_module in enumerate(self.module_list):
                if idx_module <= 3:
                    continue
                batch_dict_2 = cur_module(batch_dict_2)
            loss4, _, _ = self.get_training_loss()
            
        
        else: # when testing
            # first 4 moduels: ['vfe', 'backbone_3d_src_1', 'backbone_3d_src_2', 'backbone_3d']
            for idx_module, cur_module in enumerate(self.module_list):

                # if (idx_module in [1, 2]):
                if (idx_module in [1, 2]) and batch_dict_1['dataset_domain'] != cur_module.domain_name:
                    continue

                batch_dict_1 = cur_module(batch_dict_1)

                if idx_module==3:
                    # replace with DIR
                    batch_dict_1.update({
                        'encoded_spconv_tensor': batch_dict_1['encoded_spconv_tensor_DIR'],
                        'encoded_spconv_tensor_stride': batch_dict_1['encoded_spconv_tensor_stride_DIR'],
                        'multi_scale_3d_features': batch_dict_1['multi_scale_3d_features_DIR'],
                        'multi_scale_3d_strides': batch_dict_1['multi_scale_3d_strides_DIR'],
                    })

        if self.training:
            # loss, tb_dict, disp_dict = self.get_training_loss()
            ret_dict = {
                'loss': { 
                    'loss1': loss1,
                    'loss2': loss2,
                    'loss3': loss3,
                    'loss4': loss4, 
                },
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict_1)
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}
        loss = 0
        
        loss_rpn, tb_dict = self.dense_head.get_loss()
        loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)

        loss = loss + loss_rpn + loss_rcnn

        if hasattr(self.backbone_3d, 'get_loss'):
            loss_backbone3d, tb_dict = self.backbone_3d.get_loss(tb_dict)
            loss += loss_backbone3d
            
        return loss, tb_dict, disp_dict

class VoxelRCNN_DG_2_Source_Domain_Single_DSR_Encoder(Detector3DTemplate_DG_2_Source_Domain):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()
    
    def merge_DIR_and_DSR(self, batch_dict):    
        # print('DIR ID', id(batch_dict['encoded_spconv_tensor_DIR']) )
        # print('DSR ID', id(batch_dict['encoded_spconv_tensor_DSR']) )
        # print('**')
        batch_dict.update({
            'encoded_spconv_tensor': Fsp.sparse_add(batch_dict['encoded_spconv_tensor_DIR'], batch_dict['encoded_spconv_tensor_DSR']),
            'encoded_spconv_tensor_stride': batch_dict['encoded_spconv_tensor_stride_DIR']
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': Fsp.sparse_add(batch_dict['multi_scale_3d_features_DIR']['x_conv1'], batch_dict['multi_scale_3d_features_DSR']['x_conv1']),
                'x_conv2': Fsp.sparse_add(batch_dict['multi_scale_3d_features_DIR']['x_conv2'], batch_dict['multi_scale_3d_features_DSR']['x_conv2']),
                'x_conv3': Fsp.sparse_add(batch_dict['multi_scale_3d_features_DIR']['x_conv3'], batch_dict['multi_scale_3d_features_DSR']['x_conv3']),
                'x_conv4': Fsp.sparse_add(batch_dict['multi_scale_3d_features_DIR']['x_conv4'], batch_dict['multi_scale_3d_features_DSR']['x_conv4']),
            }
        })
        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': batch_dict['multi_scale_3d_strides_DIR']['x_conv1'],
                'x_conv2': batch_dict['multi_scale_3d_strides_DIR']['x_conv2'],
                'x_conv3': batch_dict['multi_scale_3d_strides_DIR']['x_conv3'],
                'x_conv4': batch_dict['multi_scale_3d_strides_DIR']['x_conv4'],
            }
        })
        return batch_dict

    def transfer_DSR(self, batch_dict, batch_dict_target):
        batch_dict.update({
            'encoded_spconv_tensor_DSR': batch_dict_target['encoded_spconv_tensor_DSR'],
            'encoded_spconv_tensor_stride_DSR': batch_dict_target['encoded_spconv_tensor_stride_DSR'],
            'multi_scale_3d_features_DSR': batch_dict_target['multi_scale_3d_features_DSR'],
            'multi_scale_3d_strides_DSR': batch_dict_target['multi_scale_3d_strides_DSR'],
            'DSR_domain': batch_dict_target['DSR_domain'],
        })
        return batch_dict

    def forward(self, batch_dict_1, batch_dict_2=None):
  
        all_datasets = ['waymo', 'kitti', 'nuscenes']

        '''
        first 4 modules ['vfe', 'backbone_3d_src_1', 'backbone_3d', 'discriminator_domain']
        '''

        if self.training: # when training
            assert batch_dict_1['dataset_domain'] in all_datasets and batch_dict_2['dataset_domain'] in all_datasets
            
            for idx_module, cur_module in enumerate(self.module_list):
                if idx_module >= 3:
                    break
                batch_dict_1 = cur_module(batch_dict_1)
            
            for idx_module, cur_module in enumerate(self.module_list):
                if idx_module >= 3:
                    break
                batch_dict_2 = cur_module(batch_dict_2)

            # print(batch_dict_1['DSR_domain'])
            # print(batch_dict_2['DSR_domain'])
            
            batch_dict_1 = self.merge_DIR_and_DSR(batch_dict_1)
            for idx_module, cur_module in enumerate(self.module_list):
                if idx_module <= 3:
                    continue
                batch_dict_1 = cur_module(batch_dict_1)
            loss1, tb_dict, disp_dict = self.get_training_loss()
            
            batch_dict_2 = self.merge_DIR_and_DSR(batch_dict_2)
            for idx_module, cur_module in enumerate(self.module_list):
                if idx_module <= 3:
                    continue
                batch_dict_2 = cur_module(batch_dict_2)
            loss2, _, _ = self.get_training_loss()

            DSR_1 = self.transfer_DSR({}, batch_dict_1)
            DSR_2 = self.transfer_DSR({}, batch_dict_2)

            # DIR1 + DSR2   
            batch_dict_1 = self.transfer_DSR(batch_dict_1, DSR_2)
            batch_dict_1 = self.merge_DIR_and_DSR(batch_dict_1)
            for idx_module, cur_module in enumerate(self.module_list):
                if idx_module <= 3:
                    continue
                batch_dict_1 = cur_module(batch_dict_1)
            loss3, _, _ = self.get_training_loss()

            # DIR2 + DSR1
            batch_dict_2 = self.transfer_DSR(batch_dict_2, DSR_1)
            batch_dict_2 = self.merge_DIR_and_DSR(batch_dict_2)
            for idx_module, cur_module in enumerate(self.module_list):
                if idx_module <= 3:
                    continue
                batch_dict_2 = cur_module(batch_dict_2)
            loss4, _, _ = self.get_training_loss()
            
            DSR_1['gt_boxes'] = batch_dict_1['gt_boxes']
            DSR_2['gt_boxes'] = batch_dict_2['gt_boxes']
            DSR_1['batch_size'] = batch_dict_1['batch_size']
            DSR_2['batch_size'] = batch_dict_2['batch_size']

            result_dict = self.discriminator_domain(DSR_1, DSR_2)
            discr_losses = self.discriminator_domain.get_discriminator_loss(result_dict)
            
        
        else: # when testing
            # first 4 moduels: ['vfe', 'backbone_3d_src_1', 'backbone_3d', 'discriminator_domain']
            for idx_module, cur_module in enumerate(self.module_list):

                if idx_module==3:
                    continue

                batch_dict_1 = cur_module(batch_dict_1)

                if idx_module==2:
                    # replace with DIR
                    batch_dict_1.update({
                        'encoded_spconv_tensor': batch_dict_1['encoded_spconv_tensor_DIR'],
                        'encoded_spconv_tensor_stride': batch_dict_1['encoded_spconv_tensor_stride_DIR'],
                        'multi_scale_3d_features': batch_dict_1['multi_scale_3d_features_DIR'],
                        'multi_scale_3d_strides': batch_dict_1['multi_scale_3d_strides_DIR'],
                    })

        if self.training:
            # loss, tb_dict, disp_dict = self.get_training_loss()
            ret_dict = {
                'loss': { 
                    'loss1': loss1,
                    'loss2': loss2,
                    'loss3': loss3,
                    'loss4': loss4, 
                },
            }
            ret_dict['loss'].update(discr_losses)
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict_1)
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}
        loss = 0
        
        loss_rpn, tb_dict = self.dense_head.get_loss()
        loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)

        loss = loss + loss_rpn + loss_rcnn

        if hasattr(self.backbone_3d, 'get_loss'):
            loss_backbone3d, tb_dict = self.backbone_3d.get_loss(tb_dict)
            loss += loss_backbone3d
            
        return loss, tb_dict, disp_dict

####
class VoxelRCNN_DG_2_Source_Domain_Single_DSR_Encoder_Orthognal_DIR_DSR(Detector3DTemplate_DG_2_Source_Domain):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()
    
    def merge_DIR_and_DSR(self, batch_dict):    
        # print('DIR ID', id(batch_dict['encoded_spconv_tensor_DIR']) )
        # print('DSR ID', id(batch_dict['encoded_spconv_tensor_DSR']) )
        # print('DSR:', batch_dict['encoded_spconv_tensor_DSR'].features.requires_grad )
        # print('**')
        batch_dict.update({
            'encoded_spconv_tensor': Fsp.sparse_add(batch_dict['encoded_spconv_tensor_DIR'], batch_dict['encoded_spconv_tensor_DSR']),
            'encoded_spconv_tensor_stride': batch_dict['encoded_spconv_tensor_stride_DIR']
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': Fsp.sparse_add(batch_dict['multi_scale_3d_features_DIR']['x_conv1'], batch_dict['multi_scale_3d_features_DSR']['x_conv1']),
                'x_conv2': Fsp.sparse_add(batch_dict['multi_scale_3d_features_DIR']['x_conv2'], batch_dict['multi_scale_3d_features_DSR']['x_conv2']),
                'x_conv3': Fsp.sparse_add(batch_dict['multi_scale_3d_features_DIR']['x_conv3'], batch_dict['multi_scale_3d_features_DSR']['x_conv3']),
                'x_conv4': Fsp.sparse_add(batch_dict['multi_scale_3d_features_DIR']['x_conv4'], batch_dict['multi_scale_3d_features_DSR']['x_conv4']),
            }
        })
        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': batch_dict['multi_scale_3d_strides_DIR']['x_conv1'],
                'x_conv2': batch_dict['multi_scale_3d_strides_DIR']['x_conv2'],
                'x_conv3': batch_dict['multi_scale_3d_strides_DIR']['x_conv3'],
                'x_conv4': batch_dict['multi_scale_3d_strides_DIR']['x_conv4'],
            }
        })
        return batch_dict

    def transfer_DSR(self, batch_dict, batch_dict_target):
        batch_dict.update({
            'encoded_spconv_tensor_DSR': batch_dict_target['encoded_spconv_tensor_DSR'],
            'encoded_spconv_tensor_stride_DSR': batch_dict_target['encoded_spconv_tensor_stride_DSR'],
            'multi_scale_3d_features_DSR': batch_dict_target['multi_scale_3d_features_DSR'],
            'multi_scale_3d_strides_DSR': batch_dict_target['multi_scale_3d_strides_DSR'],
            'DSR_domain': batch_dict_target['DSR_domain'],
        })
        return batch_dict

    def forward(self, batch_dict_1, batch_dict_2=None):
  
        all_datasets = ['waymo', 'kitti', 'nuscenes']

        '''
        first 5 modules ['vfe', 'backbone_3d_src_1', 'backbone_3d', 'discriminator_domain', 'separator_dir_dsr']
        '''

        if self.training: # when training
            assert batch_dict_1['dataset_domain'] in all_datasets and batch_dict_2['dataset_domain'] in all_datasets
            
            for idx_module, cur_module in enumerate(self.module_list):
                if idx_module >= 3:
                    break
                batch_dict_1 = cur_module(batch_dict_1)
            
            for idx_module, cur_module in enumerate(self.module_list):
                if idx_module >= 3:
                    break
                batch_dict_2 = cur_module(batch_dict_2)

            # print(batch_dict_1['DSR_domain'])
            # print(batch_dict_2['DSR_domain'])

            if self.training:
                orth_loss_1_dict = self.separator_dir_dsr(batch_dict_1)
                orth_loss_2_dict = self.separator_dir_dsr(batch_dict_2)
                orth_loss = 0
                for key in list(orth_loss_1_dict.keys()):
                   orth_loss += orth_loss_1_dict[key] + orth_loss_2_dict[key]
            
            batch_dict_1 = self.merge_DIR_and_DSR(batch_dict_1)
            for idx_module, cur_module in enumerate(self.module_list):
                if idx_module <= 4:
                    continue
                batch_dict_1 = cur_module(batch_dict_1)
            loss1, tb_dict, disp_dict = self.get_training_loss()
            
            batch_dict_2 = self.merge_DIR_and_DSR(batch_dict_2)
            for idx_module, cur_module in enumerate(self.module_list):
                if idx_module <= 4:
                    continue
                batch_dict_2 = cur_module(batch_dict_2)
            loss2, _, _ = self.get_training_loss()

            DSR_1 = self.transfer_DSR({}, batch_dict_1)
            DSR_2 = self.transfer_DSR({}, batch_dict_2)

            # DIR1 + DSR2   
            batch_dict_1 = self.transfer_DSR(batch_dict_1, DSR_2)
            batch_dict_1 = self.merge_DIR_and_DSR(batch_dict_1)
            for idx_module, cur_module in enumerate(self.module_list):
                if idx_module <= 4:
                    continue
                batch_dict_1 = cur_module(batch_dict_1)
            loss3, _, _ = self.get_training_loss()
            
            batch_dict_2 = self.transfer_DSR(batch_dict_2, DSR_1)
            batch_dict_2 = self.merge_DIR_and_DSR(batch_dict_2)
            for idx_module, cur_module in enumerate(self.module_list):
                if idx_module <= 4:
                    continue
                batch_dict_2 = cur_module(batch_dict_2)
            loss4, _, _ = self.get_training_loss()
            
            DSR_1['gt_boxes'] = batch_dict_1['gt_boxes']
            DSR_2['gt_boxes'] = batch_dict_2['gt_boxes']
            DSR_1['batch_size'] = batch_dict_1['batch_size']
            DSR_2['batch_size'] = batch_dict_2['batch_size']

            result_dict = self.discriminator_domain(DSR_1, DSR_2)
            discr_losses = self.discriminator_domain.get_discriminator_loss(result_dict)
            
        
        else: # when testing
            # first 5 modules ['vfe', 'backbone_3d_src_1', 'backbone_3d', 'discriminator_domain', 'separator_dir_dsr']
            for idx_module, cur_module in enumerate(self.module_list):

                if (idx_module in [3, 4]):
                    continue

                batch_dict_1 = cur_module(batch_dict_1)

                if idx_module==2:
                    # # replace final features with DIR
                    batch_dict_1.update({
                        'encoded_spconv_tensor': batch_dict_1['encoded_spconv_tensor_DIR'],
                        'encoded_spconv_tensor_stride': batch_dict_1['encoded_spconv_tensor_stride_DIR'],
                        'multi_scale_3d_features': batch_dict_1['multi_scale_3d_features_DIR'],
                        'multi_scale_3d_strides': batch_dict_1['multi_scale_3d_strides_DIR'],
                    })
                    # # merge DIR and DSR
                    # batch_dict_1 = self.merge_DIR_and_DSR(batch_dict_1)


        if self.training:
            # loss, tb_dict, disp_dict = self.get_training_loss()
            ret_dict = {
                'loss': { 
                    'loss1': loss1,
                    'loss2': loss2,
                    'loss3': loss3,
                    'loss4': loss4, 
                },
            }
            ret_dict['loss'].update(discr_losses)
            ret_dict['loss'].update({'orth_loss': orth_loss})
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict_1)
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}
        loss = 0
        
        loss_rpn, tb_dict = self.dense_head.get_loss()
        loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)

        loss = loss + loss_rpn + loss_rcnn

        if hasattr(self.backbone_3d, 'get_loss'):
            loss_backbone3d, tb_dict = self.backbone_3d.get_loss(tb_dict)
            loss += loss_backbone3d
            
        return loss, tb_dict, disp_dict

class VoxelRCNN_DG_2_Source_Domain_Single_DSR_Encoder_Orthognal_DIR_DSR_ConLoss_DSR(Detector3DTemplate_DG_2_Source_Domain):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()
    
    def merge_DIR_and_DSR(self, batch_dict):    
        batch_dict.update({
            'encoded_spconv_tensor': Fsp.sparse_add(batch_dict['encoded_spconv_tensor_DIR'], batch_dict['encoded_spconv_tensor_DSR']),
            'encoded_spconv_tensor_stride': batch_dict['encoded_spconv_tensor_stride_DIR']
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': Fsp.sparse_add(batch_dict['multi_scale_3d_features_DIR']['x_conv1'], batch_dict['multi_scale_3d_features_DSR']['x_conv1']),
                'x_conv2': Fsp.sparse_add(batch_dict['multi_scale_3d_features_DIR']['x_conv2'], batch_dict['multi_scale_3d_features_DSR']['x_conv2']),
                'x_conv3': Fsp.sparse_add(batch_dict['multi_scale_3d_features_DIR']['x_conv3'], batch_dict['multi_scale_3d_features_DSR']['x_conv3']),
                'x_conv4': Fsp.sparse_add(batch_dict['multi_scale_3d_features_DIR']['x_conv4'], batch_dict['multi_scale_3d_features_DSR']['x_conv4']),
            }
        })
        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': batch_dict['multi_scale_3d_strides_DIR']['x_conv1'],
                'x_conv2': batch_dict['multi_scale_3d_strides_DIR']['x_conv2'],
                'x_conv3': batch_dict['multi_scale_3d_strides_DIR']['x_conv3'],
                'x_conv4': batch_dict['multi_scale_3d_strides_DIR']['x_conv4'],
            }
        })
        return batch_dict

    def transfer_DSR(self, batch_dict, batch_dict_target):
        batch_dict.update({
            'encoded_spconv_tensor_DSR': batch_dict_target['encoded_spconv_tensor_DSR'],
            'encoded_spconv_tensor_stride_DSR': batch_dict_target['encoded_spconv_tensor_stride_DSR'],
            'multi_scale_3d_features_DSR': batch_dict_target['multi_scale_3d_features_DSR'],
            'multi_scale_3d_strides_DSR': batch_dict_target['multi_scale_3d_strides_DSR'],
            'DSR_domain': batch_dict_target['DSR_domain'],
        })
        return batch_dict

    def forward(self, batch_dict_1, batch_dict_2=None):
  
        all_datasets = ['waymo', 'kitti', 'nuscenes']

        '''
        first 5 modules ['vfe', 'backbone_3d_src_1', 'backbone_3d', 'discriminator_domain', 'separator_dir_dsr']
        '''

        if self.training: # when training
            assert batch_dict_1['dataset_domain'] in all_datasets and batch_dict_2['dataset_domain'] in all_datasets
            
            for idx_module, cur_module in enumerate(self.module_list):
                if idx_module >= 3:
                    break
                batch_dict_1 = cur_module(batch_dict_1)
            
            for idx_module, cur_module in enumerate(self.module_list):
                if idx_module >= 3:
                    break
                batch_dict_2 = cur_module(batch_dict_2)

            if self.training:
                orth_loss_1_dict = self.separator_dir_dsr(batch_dict_1)
                orth_loss_2_dict = self.separator_dir_dsr(batch_dict_2)
                orth_loss = 0
                for key in list(orth_loss_1_dict.keys()):
                   orth_loss += orth_loss_1_dict[key] + orth_loss_2_dict[key]
            
            # detection loss 1
            batch_dict_1 = self.merge_DIR_and_DSR(batch_dict_1)
            for idx_module, cur_module in enumerate(self.module_list):
                if idx_module <= 4:
                    continue
                batch_dict_1 = cur_module(batch_dict_1)
            loss1, tb_dict, disp_dict = self.get_training_loss()
            
            # detection loss 2
            batch_dict_2 = self.merge_DIR_and_DSR(batch_dict_2)
            for idx_module, cur_module in enumerate(self.module_list):
                if idx_module <= 4:
                    continue
                batch_dict_2 = cur_module(batch_dict_2)
            loss2, _, _ = self.get_training_loss()

            DSR_1 = self.transfer_DSR({}, batch_dict_1)
            DSR_2 = self.transfer_DSR({}, batch_dict_2)

            # detection loss 3: DIR1 + DSR2   
            batch_dict_1 = self.transfer_DSR(batch_dict_1, DSR_2)
            batch_dict_1 = self.merge_DIR_and_DSR(batch_dict_1)
            for idx_module, cur_module in enumerate(self.module_list):
                if idx_module <= 4:
                    continue
                batch_dict_1 = cur_module(batch_dict_1)
            loss3, _, _ = self.get_training_loss()

            # detection loss 4: DIR2 + DSR1
            batch_dict_2 = self.transfer_DSR(batch_dict_2, DSR_1)
            batch_dict_2 = self.merge_DIR_and_DSR(batch_dict_2)
            for idx_module, cur_module in enumerate(self.module_list):
                if idx_module <= 4:
                    continue
                batch_dict_2 = cur_module(batch_dict_2)
            loss4, _, _ = self.get_training_loss()
            
            DSR_1['gt_boxes'] = batch_dict_1['gt_boxes']
            DSR_2['gt_boxes'] = batch_dict_2['gt_boxes']
            DSR_1['batch_size'] = batch_dict_1['batch_size']
            DSR_2['batch_size'] = batch_dict_2['batch_size']

            discr_losses = self.discriminator_domain(DSR_1, DSR_2)
            discr_loss = 0
            for key in list(discr_losses.keys()):
                discr_loss += discr_losses[key]

        
        else: # when testing
            # first 5 modules ['vfe', 'backbone_3d_src_1', 'backbone_3d', 'discriminator_domain', 'separator_dir_dsr']
            for idx_module, cur_module in enumerate(self.module_list):

                if (idx_module in [3, 4]):
                    continue

                batch_dict_1 = cur_module(batch_dict_1)

                if idx_module==2:
                    # replace final features with DIR
                    batch_dict_1.update({
                        'encoded_spconv_tensor': batch_dict_1['encoded_spconv_tensor_DIR'],
                        'encoded_spconv_tensor_stride': batch_dict_1['encoded_spconv_tensor_stride_DIR'],
                        'multi_scale_3d_features': batch_dict_1['multi_scale_3d_features_DIR'],
                        'multi_scale_3d_strides': batch_dict_1['multi_scale_3d_strides_DIR'],
                    })

                    # # merge DIR and DSR
                    # batch_dict_1 = self.merge_DIR_and_DSR(batch_dict_1)

        if self.training:
            # loss, tb_dict, disp_dict = self.get_training_loss()
            ret_dict = {
                'loss': { 
                    'loss1': loss1,
                    'loss2': loss2,
                    'loss3': loss3,
                    'loss4': loss4, 
                },
            }
            # ret_dict['loss'].update({'discr_loss': discr_loss})
            ret_dict['loss'].update(discr_losses)
            ret_dict['loss'].update({'orth_loss': orth_loss})
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict_1)
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}
        loss = 0
        
        loss_rpn, tb_dict = self.dense_head.get_loss()
        loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)

        loss = loss + loss_rpn + loss_rcnn

        if hasattr(self.backbone_3d, 'get_loss'):
            loss_backbone3d, tb_dict = self.backbone_3d.get_loss(tb_dict)
            loss += loss_backbone3d
            
        return loss, tb_dict, disp_dict

class VoxelRCNN_DG_2_Source_Domain_FineTune(Detector3DTemplate_DG_2_Source_Domain):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    def detach_DIR(self, batch_dict):

        batch_dict['encoded_spconv_tensor_DIR'] = replace_feature(
            batch_dict['encoded_spconv_tensor_DIR'], 
            batch_dict['encoded_spconv_tensor_DIR'].features.detach()
            )

        for conv_layer in ['x_conv1', 'x_conv2', 'x_conv3', 'x_conv4']:
            batch_dict['multi_scale_3d_features_DIR'][conv_layer] = replace_feature(
                batch_dict['multi_scale_3d_features_DIR'][conv_layer], 
                batch_dict['multi_scale_3d_features_DIR'][conv_layer].features.detach()
                )

        return batch_dict
    
    def forward(self, batch_dict):
        all_datasets = ['waymo', 'kitti', 'nuscenes']
        skipped_encoder = all_datasets[:]

        if self.training: # when training

            # first 2 moduels: ['vfe', 'backbone_3d']
            for idx_module, cur_module in enumerate(self.module_list):

                batch_dict = cur_module(batch_dict)
                
                if idx_module==1:
                    batch_dict = self.detach_DIR(batch_dict)
                    # replace with DIR
                    batch_dict.update({
                        'encoded_spconv_tensor': batch_dict['encoded_spconv_tensor_DIR'],
                        'encoded_spconv_tensor_stride': batch_dict['encoded_spconv_tensor_stride_DIR'],
                        'multi_scale_3d_features': batch_dict['multi_scale_3d_features_DIR'],
                        'multi_scale_3d_strides': batch_dict['multi_scale_3d_strides_DIR'],
                    })

            # print('DIR_domain:', batch_dict['dataset_domain'], '\tDSR_domain:', print(batch_dict['DSR_domain']))

        else: # when testing
            for idx_module, cur_module in enumerate(self.module_list):

                batch_dict = cur_module(batch_dict)

                if idx_module==1:
                    # replace with DIR
                    # replace with DIR
                    batch_dict.update({
                        'encoded_spconv_tensor': batch_dict['encoded_spconv_tensor_DIR'],
                        'encoded_spconv_tensor_stride': batch_dict['encoded_spconv_tensor_stride_DIR'],
                        'multi_scale_3d_features': batch_dict['multi_scale_3d_features_DIR'],
                        'multi_scale_3d_strides': batch_dict['multi_scale_3d_strides_DIR'],
                    })

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss,
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}
        loss = 0
        
        loss_rpn, tb_dict = self.dense_head.get_loss()
        loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)

        loss = loss + loss_rpn + loss_rcnn

        if hasattr(self.backbone_3d, 'get_loss'):
            loss_backbone3d, tb_dict = self.backbone_3d.get_loss(tb_dict)
            loss += loss_backbone3d
            
        return loss, tb_dict, disp_dict

class VoxelRCNN_DG_2_Source_Domain_DSRSubGen_DomClass_OrthognalDIRDSR(Detector3DTemplate_DG_2_Source_Domain):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()
    
    # def merge_DIR_and_DSR(self, batch_dict):    
    #     # print('DIR ID', id(batch_dict['encoded_spconv_tensor_DIR']) )
    #     # print('DSR ID', id(batch_dict['encoded_spconv_tensor_DSR']) )
    #     # print('DSR:', batch_dict['encoded_spconv_tensor_DSR'].features.requires_grad )
    #     # print('**')
    #     batch_dict.update({
    #         'encoded_spconv_tensor': Fsp.sparse_add(batch_dict['encoded_spconv_tensor_DIR'], batch_dict['encoded_spconv_tensor_DSR']),
    #         'encoded_spconv_tensor_stride': batch_dict['encoded_spconv_tensor_stride_DIR']
    #     })
    #     batch_dict.update({
    #         'multi_scale_3d_features': {
    #             'x_conv1': Fsp.sparse_add(batch_dict['multi_scale_3d_features_DIR']['x_conv1'], batch_dict['multi_scale_3d_features_DSR']['x_conv1']),
    #             'x_conv2': Fsp.sparse_add(batch_dict['multi_scale_3d_features_DIR']['x_conv2'], batch_dict['multi_scale_3d_features_DSR']['x_conv2']),
    #             'x_conv3': Fsp.sparse_add(batch_dict['multi_scale_3d_features_DIR']['x_conv3'], batch_dict['multi_scale_3d_features_DSR']['x_conv3']),
    #             'x_conv4': Fsp.sparse_add(batch_dict['multi_scale_3d_features_DIR']['x_conv4'], batch_dict['multi_scale_3d_features_DSR']['x_conv4']),
    #         }
    #     })
    #     batch_dict.update({
    #         'multi_scale_3d_strides': {
    #             'x_conv1': batch_dict['multi_scale_3d_strides_DIR']['x_conv1'],
    #             'x_conv2': batch_dict['multi_scale_3d_strides_DIR']['x_conv2'],
    #             'x_conv3': batch_dict['multi_scale_3d_strides_DIR']['x_conv3'],
    #             'x_conv4': batch_dict['multi_scale_3d_strides_DIR']['x_conv4'],
    #         }
    #     })
    #     return batch_dict

    def transfer_DSR(self, batch_dict, batch_dict_target):
        batch_dict.update({
            'encoded_spconv_tensor_DSR': batch_dict_target['encoded_spconv_tensor_DSR'],
            'encoded_spconv_tensor_stride_DSR': batch_dict_target['encoded_spconv_tensor_stride_DSR'],
            'multi_scale_3d_features_DSR': batch_dict_target['multi_scale_3d_features_DSR'],
            'multi_scale_3d_strides_DSR': batch_dict_target['multi_scale_3d_strides_DSR'],
            'DSR_domain': batch_dict_target['DSR_domain'],
        })
        return batch_dict

    def forward(self, batch_dict_1, batch_dict_2=None):
  
        all_datasets = ['waymo', 'kitti', 'nuscenes']

        '''
        first 5 modules ['vfe', 'backbone_3d', 'backbone_3d_dir_dsr_gen', 'discriminator_domain', 'separator_dir_dsr']
        '''

        if self.training: # when training
            assert batch_dict_1['dataset_domain'] in all_datasets and batch_dict_2['dataset_domain'] in all_datasets
            
            for idx_module, cur_module in enumerate(self.module_list):
                if idx_module >= 3:
                    break
                batch_dict_1 = cur_module(batch_dict_1)
            
            for idx_module, cur_module in enumerate(self.module_list):
                if idx_module >= 3:
                    break
                batch_dict_2 = cur_module(batch_dict_2)

            # loss for orthogonal 
            orth_loss_1_dict = self.separator_dir_dsr(batch_dict_1)
            orth_loss_2_dict = self.separator_dir_dsr(batch_dict_2)
            orth_loss = 0
            for key in list(orth_loss_1_dict.keys()):
                orth_loss += orth_loss_1_dict[key] + orth_loss_2_dict[key]
            
            # batch_dict_1 = self.merge_DIR_and_DSR(batch_dict_1)
            batch_dict_1.update({
                'encoded_spconv_tensor': batch_dict_1['encoded_spconv_tensor_DIR'],
                'encoded_spconv_tensor_stride': batch_dict_1['encoded_spconv_tensor_stride_DIR'],
                'multi_scale_3d_features': batch_dict_1['multi_scale_3d_features_DIR'],
                'multi_scale_3d_strides': batch_dict_1['multi_scale_3d_strides_DIR'],
            })
            for idx_module, cur_module in enumerate(self.module_list):
                if idx_module <= 4:
                    continue
                batch_dict_1 = cur_module(batch_dict_1)
            loss1, tb_dict, disp_dict = self.get_training_loss()  

            # batch_dict_2 = self.merge_DIR_and_DSR(batch_dict_2)
            batch_dict_2.update({
                'encoded_spconv_tensor': batch_dict_2['encoded_spconv_tensor_DIR'],
                'encoded_spconv_tensor_stride': batch_dict_2['encoded_spconv_tensor_stride_DIR'],
                'multi_scale_3d_features': batch_dict_2['multi_scale_3d_features_DIR'],
                'multi_scale_3d_strides': batch_dict_2['multi_scale_3d_strides_DIR'],
            })
            for idx_module, cur_module in enumerate(self.module_list):
                if idx_module <= 4:
                    continue
                batch_dict_2 = cur_module(batch_dict_2)
            loss2, _, _ = self.get_training_loss()

            DSR_1 = self.transfer_DSR({}, batch_dict_1)
            DSR_2 = self.transfer_DSR({}, batch_dict_2)
           
            DSR_1['gt_boxes'] = batch_dict_1['gt_boxes']
            DSR_2['gt_boxes'] = batch_dict_2['gt_boxes']
            DSR_1['batch_size'] = batch_dict_1['batch_size']
            DSR_2['batch_size'] = batch_dict_2['batch_size']

            result_dict = self.discriminator_domain(DSR_1, DSR_2)
            discr_losses = self.discriminator_domain.get_discriminator_loss(result_dict)
            
        
        else: # when testing
            # first 5 modules ['vfe', 'backbone_3d', 'backbone_3d_dir_dsr_gen', 'discriminator_domain', 'separator_dir_dsr']
            for idx_module, cur_module in enumerate(self.module_list):

                if (idx_module in [3, 4]):
                    continue

                batch_dict_1 = cur_module(batch_dict_1)

                if idx_module==2:
                    # # replace final features with DIR
                    batch_dict_1.update({
                        'encoded_spconv_tensor': batch_dict_1['encoded_spconv_tensor_DIR'],
                        'encoded_spconv_tensor_stride': batch_dict_1['encoded_spconv_tensor_stride_DIR'],
                        'multi_scale_3d_features': batch_dict_1['multi_scale_3d_features_DIR'],
                        'multi_scale_3d_strides': batch_dict_1['multi_scale_3d_strides_DIR'],
                    })
                    # # merge DIR and DSR
                    # batch_dict_1 = self.merge_DIR_and_DSR(batch_dict_1)


        if self.training:
            # loss, tb_dict, disp_dict = self.get_training_loss()
            ret_dict = {
                'loss': { 
                    'loss1': loss1,
                    'loss2': loss2,
                },
            }
            ret_dict['loss'].update(discr_losses)
            ret_dict['loss'].update({'orth_loss': orth_loss})
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict_1)
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}
        loss = 0
        
        loss_rpn, tb_dict = self.dense_head.get_loss()
        loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)

        loss = loss + loss_rpn + loss_rcnn

        if hasattr(self.backbone_3d, 'get_loss'):
            loss_backbone3d, tb_dict = self.backbone_3d.get_loss(tb_dict)
            loss += loss_backbone3d
            
        return loss, tb_dict, disp_dict

class VoxelRCNN_DG_2_Source_Domain_DSRSubGen_DSRDomClass_DIRAdvDomClass_OrthognalDIRDSR(Detector3DTemplate_DG_2_Source_Domain):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    def transfer_DSR(self, batch_dict, batch_dict_target):
        batch_dict.update({
            'encoded_spconv_tensor_DSR': batch_dict_target['encoded_spconv_tensor_DSR'],
            'encoded_spconv_tensor_stride_DSR': batch_dict_target['encoded_spconv_tensor_stride_DSR'],
            'multi_scale_3d_features_DSR': batch_dict_target['multi_scale_3d_features_DSR'],
            'multi_scale_3d_strides_DSR': batch_dict_target['multi_scale_3d_strides_DSR'],
            'DSR_domain': batch_dict_target['DSR_domain'],
        })
        return batch_dict

    def forward(self, batch_dict_1, batch_dict_2=None):
  
        all_datasets = ['waymo', 'kitti', 'nuscenes']

        '''
        first 6 modules ['vfe', 'backbone_3d', 'backbone_3d_dir_dsr_gen', 'discriminator_domain','discriminator_domain_2', 'separator_dir_dsr']
        '''

        if self.training: # when training
            assert batch_dict_1['dataset_domain'] in all_datasets and batch_dict_2['dataset_domain'] in all_datasets
            
            for idx_module, cur_module in enumerate(self.module_list):
                if idx_module >= 3:
                    break
                batch_dict_1 = cur_module(batch_dict_1)
            
            for idx_module, cur_module in enumerate(self.module_list):
                if idx_module >= 3:
                    break
                batch_dict_2 = cur_module(batch_dict_2)

            # loss for orthogonal 
            orth_loss_1_dict = self.separator_dir_dsr(batch_dict_1)
            orth_loss_2_dict = self.separator_dir_dsr(batch_dict_2)
            orth_loss = 0
            for key in list(orth_loss_1_dict.keys()):
                orth_loss += orth_loss_1_dict[key] + orth_loss_2_dict[key]

            # loss for discrimative DSR
            DSR_1 = self.transfer_DSR({}, batch_dict_1)
            DSR_2 = self.transfer_DSR({}, batch_dict_2)
           
            DSR_1['gt_boxes'] = batch_dict_1['gt_boxes']
            DSR_2['gt_boxes'] = batch_dict_2['gt_boxes']
            DSR_1['batch_size'] = batch_dict_1['batch_size']
            DSR_2['batch_size'] = batch_dict_2['batch_size']

            result_dict = self.discriminator_domain(DSR_1, DSR_2)
            discr_losses = self.discriminator_domain.get_discriminator_loss(result_dict)

            discr_loss = 0
            for key in list(discr_losses.keys()):
                discr_loss += discr_losses[key]


            # loss for undiscrimative DIR
            result_dict_dir = self.discriminator_domain_2(batch_dict_1, batch_dict_2)
            undiscr_losses = self.discriminator_domain_2.get_discriminator_loss(result_dict_dir)

            undiscr_loss = 0
            for key in list(undiscr_losses.keys()):
                undiscr_loss += undiscr_losses[key]
            
            # detection loss of batch_1
            batch_dict_1.update({
                'encoded_spconv_tensor': batch_dict_1['encoded_spconv_tensor_DIR'],
                'encoded_spconv_tensor_stride': batch_dict_1['encoded_spconv_tensor_stride_DIR'],
                'multi_scale_3d_features': batch_dict_1['multi_scale_3d_features_DIR'],
                'multi_scale_3d_strides': batch_dict_1['multi_scale_3d_strides_DIR'],
            })
            for idx_module, cur_module in enumerate(self.module_list):
                if idx_module <= 5:
                    continue
                batch_dict_1 = cur_module(batch_dict_1)
            loss1, tb_dict, disp_dict = self.get_training_loss()  

            # detection loss of batch_2
            batch_dict_2.update({
                'encoded_spconv_tensor': batch_dict_2['encoded_spconv_tensor_DIR'],
                'encoded_spconv_tensor_stride': batch_dict_2['encoded_spconv_tensor_stride_DIR'],
                'multi_scale_3d_features': batch_dict_2['multi_scale_3d_features_DIR'],
                'multi_scale_3d_strides': batch_dict_2['multi_scale_3d_strides_DIR'],
            })
            for idx_module, cur_module in enumerate(self.module_list):
                if idx_module <= 5:
                    continue
                batch_dict_2 = cur_module(batch_dict_2)
            loss2, _, _ = self.get_training_loss()  
        
        else: # when testing
            '''
            first 6 modules ['vfe', 'backbone_3d', 'backbone_3d_dir_dsr_gen', 'discriminator_domain','discriminator_domain_2', 'separator_dir_dsr']
            '''
            for idx_module, cur_module in enumerate(self.module_list):

                if (idx_module in [3, 4, 5]):
                    continue

                batch_dict_1 = cur_module(batch_dict_1)

                if idx_module==2:
                    # # replace final features with DIR
                    batch_dict_1.update({
                        'encoded_spconv_tensor': batch_dict_1['encoded_spconv_tensor_DIR'],
                        'encoded_spconv_tensor_stride': batch_dict_1['encoded_spconv_tensor_stride_DIR'],
                        'multi_scale_3d_features': batch_dict_1['multi_scale_3d_features_DIR'],
                        'multi_scale_3d_strides': batch_dict_1['multi_scale_3d_strides_DIR'],
                    })

        if self.training:
            # loss, tb_dict, disp_dict = self.get_training_loss()
            ret_dict = {
                'loss': { 
                    'loss1': loss1,
                    'loss2': loss2,
                },
            }
            ret_dict['loss'].update({'discr_loss': discr_loss})
            ret_dict['loss'].update({'undiscr_loss': undiscr_loss})
            ret_dict['loss'].update({'orth_loss': orth_loss})
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict_1)
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}
        loss = 0
        
        loss_rpn, tb_dict = self.dense_head.get_loss()
        loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)

        loss = loss + loss_rpn + loss_rcnn

        if hasattr(self.backbone_3d, 'get_loss'):
            loss_backbone3d, tb_dict = self.backbone_3d.get_loss(tb_dict)
            loss += loss_backbone3d
            
        return loss, tb_dict, disp_dict


class VoxelRCNN_DG_2_Source_Domain_Discriminator(Detector3DTemplate_DG_2_Source_Domain):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()
    
    def merge_DIR_and_DSR(self, batch_dict):    
        # print('DIR ID', id(batch_dict['encoded_spconv_tensor_DIR']) )
        # print('DSR ID', id(batch_dict['encoded_spconv_tensor_DSR']) )
        # print('**')
        batch_dict.update({
            'encoded_spconv_tensor': Fsp.sparse_add(batch_dict['encoded_spconv_tensor_DIR'], batch_dict['encoded_spconv_tensor_DSR']),
            'encoded_spconv_tensor_stride': batch_dict['encoded_spconv_tensor_stride_DIR']
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': Fsp.sparse_add(batch_dict['multi_scale_3d_features_DIR']['x_conv1'], batch_dict['multi_scale_3d_features_DSR']['x_conv1']),
                'x_conv2': Fsp.sparse_add(batch_dict['multi_scale_3d_features_DIR']['x_conv2'], batch_dict['multi_scale_3d_features_DSR']['x_conv2']),
                'x_conv3': Fsp.sparse_add(batch_dict['multi_scale_3d_features_DIR']['x_conv3'], batch_dict['multi_scale_3d_features_DSR']['x_conv3']),
                'x_conv4': Fsp.sparse_add(batch_dict['multi_scale_3d_features_DIR']['x_conv4'], batch_dict['multi_scale_3d_features_DSR']['x_conv4']),
            }
        })
        batch_dict.update({
            'multi_scale_3d_strides': batch_dict['multi_scale_3d_strides_DIR'],
            'DSR_domain': batch_dict['DSR_domain'],
        })
        return batch_dict

    # def transfer_DSR(self, batch_dict, batch_dict_target):
    #     batch_dict.update({
    #         'encoded_spconv_tensor_DSR': batch_dict_target['encoded_spconv_tensor_DSR'],
    #         'encoded_spconv_tensor_stride_DSR': batch_dict_target['encoded_spconv_tensor_stride_DSR'],
    #         'multi_scale_3d_features_DSR': batch_dict_target['multi_scale_3d_features_DSR'],
    #         'multi_scale_3d_strides_DSR': batch_dict_target['multi_scale_3d_strides_DSR'],
    #         'DSR_domain': batch_dict_target['DSR_domain'],
    #     })
    #     return batch_dict
    
    def transfer_DSR(self, batch_dict, batch_dict_target):
        batch_dict.update({
            'encoded_spconv_tensor_DSR': batch_dict_target['encoded_spconv_tensor_DSR'],
            'encoded_spconv_tensor_stride_DSR': batch_dict_target['encoded_spconv_tensor_stride_DSR'],
            'multi_scale_3d_features_DSR': batch_dict_target['multi_scale_3d_features_DSR'],
            'multi_scale_3d_strides_DSR': batch_dict_target['multi_scale_3d_strides_DSR'],
            'DSR_domain': batch_dict_target['DSR_domain'],
        })
        return batch_dict
    
    def transfer_DIR(self, batch_dict, batch_dict_target):
        batch_dict.update({
            'encoded_spconv_tensor_DIR': batch_dict_target['encoded_spconv_tensor_DIR'],
            'encoded_spconv_tensor_stride_DIR': batch_dict_target['encoded_spconv_tensor_stride_DIR'],
            'multi_scale_3d_features_DIR': batch_dict_target['multi_scale_3d_features_DIR'],
            'multi_scale_3d_strides_DIR': batch_dict_target['multi_scale_3d_strides_DIR'],
        })
        return batch_dict

    def forward(self, batch_dict_1, batch_dict_2=None):
  
        all_datasets = ['waymo', 'kitti', 'nuscenes']

        
        assert batch_dict_1['dataset_domain'] in all_datasets and batch_dict_2['dataset_domain'] in all_datasets
        # first 4 moduels: ['vfe', 'backbone_3d_src_1', 'backbone_3d_src_2', 'backbone_3d']
        
        for idx_module, cur_module in enumerate(self.module_list):
            if idx_module > 3:
                break
            if (idx_module in [1, 2]) and (cur_module.domain_name != batch_dict_1['dataset_domain']):
                continue
            batch_dict_1 = cur_module(batch_dict_1)
        
        for idx_module, cur_module in enumerate(self.module_list):
            if idx_module > 3:
                break
            if (idx_module in [1, 2]) and (cur_module.domain_name != batch_dict_2['dataset_domain']):
                continue
            batch_dict_2 = cur_module(batch_dict_2)

        # print(batch_dict_1['DSR_domain'])
        # print(batch_dict_2['DSR_domain'])

        DIR_1 = self.transfer_DIR({}, batch_dict_1)
        DIR_2 = self.transfer_DIR({}, batch_dict_2)

        DSR_1 = self.transfer_DSR({}, batch_dict_1)
        DSR_2 = self.transfer_DSR({}, batch_dict_2)

        print(batch_dict_1['multi_scale_3d_features_DSR']['x_conv2'].dense().grad_fn)
        print(batch_dict_1['multi_scale_3d_features_DSR']['x_conv2'].dense().is_leaf)

        # import pickle
        # import os
        # save_root_dir = '/dataset/shuangzhi/TT/feature_saved_for_TT'

        # # batch_size == 1
        # frame_id_1 = batch_dict_1['frame_id'][0]
        # # print(batch_dict_1['frame_id'])
        # DSR_domain_1 =  DSR_1['DSR_domain']
        # save_dir_1 = save_root_dir + '/' + DSR_domain_1
        # if not os.path.exists(save_dir_1):
        #     os.makedirs(save_dir_1)
        # with open((save_dir_1 + '/%s.pkl' % frame_id_1), 'wb') as f:
        #     dump_data = DSR_1
        #     pickle.dump(dump_data, f) 
        # with open((save_root_dir + '/%s_sample_list.txt' % DSR_domain_1), 'a') as f:
        #     f.writelines(frame_id_1+'\n')
        
        # frame_id_2 = batch_dict_2['frame_id'][0]
        # DSR_domain_2 =  DSR_2['DSR_domain']
        # save_dir_2 = save_root_dir + '/' + DSR_domain_2
        # if not os.path.exists(save_dir_2):
        #     os.makedirs(save_dir_2)
        # with open((save_dir_2 + '/%s.pkl' % frame_id_2), 'wb') as f:
        #     dump_data = DSR_2
        #     pickle.dump(dump_data, f) 
        # with open((save_root_dir + '/%s_sample_list.txt' % DSR_domain_2), 'a') as f:
        #     f.writelines(frame_id_2+'\n')
        
        # if len(os.listdir(save_dir_2)) >=100:
        #     raise TypeError('Finished the collection. Stop!')
        
        
        DSR_1_DIR_1 = {}
        DSR_1_DIR_1.update(DSR_1)
        DSR_1_DIR_1.update(DIR_1)
        self.merge_DIR_and_DSR(DSR_1_DIR_1)

        DSR_2_DIR_2 = {}
        DSR_2_DIR_2.update(DSR_2)
        DSR_2_DIR_2.update(DIR_2)
        self.merge_DIR_and_DSR(DSR_2_DIR_2)

        DSR_1_DIR_2 = {}
        DSR_1_DIR_2.update(DSR_1)
        DSR_1_DIR_2.update(DIR_2)
        self.merge_DIR_and_DSR(DSR_1_DIR_2)

        DSR_2_DIR_1 = {}
        DSR_2_DIR_1.update(DSR_2)
        DSR_2_DIR_1.update(DIR_1)
        self.merge_DIR_and_DSR(DSR_2_DIR_1)

        result_dict = self.discriminator_domain([DSR_1_DIR_1, DSR_1_DIR_2], [DSR_2_DIR_2, DSR_2_DIR_1], batch_size=batch_dict_1['batch_size'])

        if self.training:
            loss = self.discriminator_domain.get_discriminator_loss(result_dict)
            ret_dict = {
                'loss': loss,
            }
            return ret_dict, {}, {}
        else:
            pred_acc_dicts = {}
            for src_name in list(result_dict['domain_pred'].keys()):
                pred_prop = torch.nn.Softmax(dim=1)(result_dict['domain_pred'][src_name])
                # print(pred_prop)
                pred_label = torch.argmax(pred_prop, dim=1)
                gt_label = result_dict['domain_gt_label'][src_name]
                # print(gt_label)
                pred_acc_dicts[src_name] = float((pred_label==gt_label).sum().item()) / pred_label.shape[0]
                # print(pred_acc_dicts[src_name])

            return pred_acc_dicts

    # def get_training_loss(self):
    #     disp_dict = {}
    #     loss = 0
        
    #     loss_rpn, tb_dict = self.dense_head.get_loss()
    #     loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)

    #     loss = loss + loss_rpn + loss_rcnn

    #     if hasattr(self.backbone_3d, 'get_loss'):
    #         loss_backbone3d, tb_dict = self.backbone_3d.get_loss(tb_dict)
    #         loss += loss_backbone3d
            
    #     return loss, tb_dict, disp_dict


class VoxelRCNN_DG_2_Source_Domain_TestTime(Detector3DTemplate_DG_2_Source_Domain):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()
    
    def sp_tensor_to_device(self, sp_tensor, device):
        new_spt = spconv.SparseConvTensor(
            sp_tensor.features.to(device), 
            sp_tensor.indices.to(device), 
            sp_tensor.spatial_shape,
            sp_tensor.batch_size)
        return new_spt

    
    def merge_DIR_and_DSR(self, batch_dict):    
        # print('DIR ID', id(batch_dict['encoded_spconv_tensor_DIR']) )
        # print('DSR ID', id(batch_dict['encoded_spconv_tensor_DSR']) )
        # print('**')
        # print('DIR device', batch_dict['encoded_spconv_tensor_DIR'].indices.device )
        # print('DSR device', batch_dict['encoded_spconv_tensor_DSR'].features.device )
        # print('**')

        if batch_dict['encoded_spconv_tensor_DIR'].features.device != batch_dict['encoded_spconv_tensor_DSR'].features.device:
            device_target = batch_dict['encoded_spconv_tensor_DIR'].features.device
            batch_dict['encoded_spconv_tensor_DSR'] = self.sp_tensor_to_device(batch_dict['encoded_spconv_tensor_DSR'], device_target)
            for conv_layer in ['x_conv1', 'x_conv2', 'x_conv3', 'x_conv4']:
                batch_dict['multi_scale_3d_features_DSR'][conv_layer] = self.sp_tensor_to_device(
                    batch_dict['multi_scale_3d_features_DSR'][conv_layer],
                    device_target
                )

        batch_dict.update({
            'encoded_spconv_tensor': Fsp.sparse_add(batch_dict['encoded_spconv_tensor_DIR'], batch_dict['encoded_spconv_tensor_DSR']),
            'encoded_spconv_tensor_stride': batch_dict['encoded_spconv_tensor_stride_DIR']
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': Fsp.sparse_add(batch_dict['multi_scale_3d_features_DIR']['x_conv1'], batch_dict['multi_scale_3d_features_DSR']['x_conv1']),
                'x_conv2': Fsp.sparse_add(batch_dict['multi_scale_3d_features_DIR']['x_conv2'], batch_dict['multi_scale_3d_features_DSR']['x_conv2']),
                'x_conv3': Fsp.sparse_add(batch_dict['multi_scale_3d_features_DIR']['x_conv3'], batch_dict['multi_scale_3d_features_DSR']['x_conv3']),
                'x_conv4': Fsp.sparse_add(batch_dict['multi_scale_3d_features_DIR']['x_conv4'], batch_dict['multi_scale_3d_features_DSR']['x_conv4']),
            }
        })
        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': batch_dict['multi_scale_3d_strides_DIR']['x_conv1'],
                'x_conv2': batch_dict['multi_scale_3d_strides_DIR']['x_conv2'],
                'x_conv3': batch_dict['multi_scale_3d_strides_DIR']['x_conv3'],
                'x_conv4': batch_dict['multi_scale_3d_strides_DIR']['x_conv4'],
            }
        })
        return batch_dict

    def transfer_DSR(self, batch_dict, batch_dict_target):
        batch_dict.update({
            'encoded_spconv_tensor_DSR': batch_dict_target['encoded_spconv_tensor_DSR'],
            'encoded_spconv_tensor_stride_DSR': batch_dict_target['encoded_spconv_tensor_stride_DSR'],
            'multi_scale_3d_features_DSR': batch_dict_target['multi_scale_3d_features_DSR'],
            'multi_scale_3d_strides_DSR': batch_dict_target['multi_scale_3d_strides_DSR'],
            'DSR_domain': batch_dict_target['DSR_domain'],
        })
        return batch_dict

    def transfer_DIR(self, batch_dict, batch_dict_target):
        batch_dict.update({
            'encoded_spconv_tensor_DIR': batch_dict_target['encoded_spconv_tensor_DIR'],
            'encoded_spconv_tensor_stride_DIR': batch_dict_target['encoded_spconv_tensor_stride_DIR'],
            'multi_scale_3d_features_DIR': batch_dict_target['multi_scale_3d_features_DIR'],
            'multi_scale_3d_strides_DIR': batch_dict_target['multi_scale_3d_strides_DIR'],
        })
        return batch_dict

    def detach_DSR(self, batch_dict):

        batch_dict['encoded_spconv_tensor_DSR'] = replace_feature(
            batch_dict['encoded_spconv_tensor_DSR'], 
            batch_dict['encoded_spconv_tensor_DSR'].features.detach()
            )

        for conv_layer in ['x_conv1', 'x_conv2', 'x_conv3', 'x_conv4']:
            batch_dict['multi_scale_3d_features_DSR'][conv_layer] = replace_feature(
                batch_dict['multi_scale_3d_features_DSR'][conv_layer], 
                batch_dict['multi_scale_3d_features_DSR'][conv_layer].features.detach()
                )

        return batch_dict


    def forward(self, batch_dict_target, batch_dict_1=None, batch_dict_2=None, DSR_TT=None):
  
        all_datasets = ['waymo', 'kitti', 'nuscenes']

        if self.training: # when training
            assert batch_dict_1['dataset_domain'] in all_datasets and batch_dict_2['dataset_domain'] in all_datasets 
            # first 4 moduels: ['vfe', 'backbone_3d_src_1', 'backbone_3d_src_2', 'backbone_3d' ], 5th ['discriminator_domain']
            
            for idx_module, cur_module in enumerate(self.module_list):
                if idx_module > 3:
                    break
                if (idx_module in [1, 2]) and (cur_module.domain_name != batch_dict_1['dataset_domain']):
                    continue
                batch_dict_1 = cur_module(batch_dict_1)
            
            for idx_module, cur_module in enumerate(self.module_list):
                if idx_module > 3:
                    break
                if (idx_module in [1, 2]) and (cur_module.domain_name != batch_dict_2['dataset_domain']):
                    continue
                batch_dict_2 = cur_module(batch_dict_2)
            
            for idx_module, cur_module in enumerate(self.module_list):
                if idx_module in [0, 3]:
                    batch_dict_target = cur_module(batch_dict_target)
            

            # detach DSR for batch_dict_1 and batch_dict_2
            batch_dict_1 = self.detach_DSR(batch_dict_1)
            batch_dict_2 = self.detach_DSR(batch_dict_2)

            # print(batch_dict_1['DSR_domain'])
            # print(batch_dict_2['DSR_domain'])
            
            batch_dict_1 = self.merge_DIR_and_DSR(batch_dict_1)
            for idx_module, cur_module in enumerate(self.module_list):
                if idx_module <= 4:
                    continue
                batch_dict_1 = cur_module(batch_dict_1)
            loss1, tb_dict, disp_dict = self.get_training_det_loss()
            
            batch_dict_2 = self.merge_DIR_and_DSR(batch_dict_2)
            for idx_module, cur_module in enumerate(self.module_list):
                if idx_module <= 4:
                    continue
                batch_dict_2 = cur_module(batch_dict_2)
            loss2, _, _ = self.get_training_det_loss()

            DSR_1 = self.transfer_DSR({}, batch_dict_1)
            DSR_2 = self.transfer_DSR({}, batch_dict_2)

            # DIR1 + DSR2   
            batch_dict_1 = self.transfer_DSR(batch_dict_1, DSR_2)
            batch_dict_1 = self.merge_DIR_and_DSR(batch_dict_1)
            for idx_module, cur_module in enumerate(self.module_list):
                if idx_module <= 4:
                    continue
                batch_dict_1 = cur_module(batch_dict_1)
            loss3, _, _ = self.get_training_det_loss()
            
            # DIR2 + DSR1
            batch_dict_2 = self.transfer_DSR(batch_dict_2, DSR_1)
            batch_dict_2 = self.merge_DIR_and_DSR(batch_dict_2)
            for idx_module, cur_module in enumerate(self.module_list):
                if idx_module <= 4:
                    continue
                batch_dict_2 = cur_module(batch_dict_2)
            loss4, _, _ = self.get_training_det_loss()

            DIR_target = self.transfer_DIR({}, batch_dict_target)

            # discriminator
            DIR_target_DSR_1 = {}
            DIR_target_DSR_1.update(DIR_target)
            DIR_target_DSR_1.update(DSR_1)
            DIR_target_DSR_1 = self.merge_DIR_and_DSR(DIR_target_DSR_1) 

            DIR_target_DSR_2 = {}
            DIR_target_DSR_2.update(DIR_target)
            DIR_target_DSR_2.update(DSR_2)
            DIR_target_DSR_2 = self.merge_DIR_and_DSR(DIR_target_DSR_2)  

            result_dict = self.discriminator_domain([DIR_target_DSR_1], [DIR_target_DSR_2], batch_size=batch_dict_1['batch_size'])
            losses_discrinimator = self.discriminator_domain.get_discriminator_loss(result_dict)

            # loss, tb_dict, disp_dict = self.get_training_loss()
            ret_dict = {
                'loss': { 
                    'loss1': loss1,
                    'loss2': loss2,
                    'loss3': loss3,
                    'loss4': loss4, 
                },
            }
            ret_dict['loss'].update(losses_discrinimator)

            return ret_dict, tb_dict, disp_dict
           
        
        else: # when testing
            # first 4 moduels: ['vfe', 'backbone_3d_src_1', 'backbone_3d_src_2', 'backbone_3d']
            for idx_module, cur_module in enumerate(self.module_list):

                # if (idx_module in [1, 2]):
                if (idx_module in [1, 2]) and batch_dict_target['dataset_domain'] != cur_module.domain_name:
                    continue
                
                if idx_module == 4:
                    continue

                batch_dict_target = cur_module(batch_dict_target)

                if idx_module==3:
                    # final-feature-filled method 1: replace with DIR
                    batch_dict_target.update({
                        'encoded_spconv_tensor': batch_dict_target['encoded_spconv_tensor_DIR'],
                        'encoded_spconv_tensor_stride': batch_dict_target['encoded_spconv_tensor_stride_DIR'],
                        'multi_scale_3d_features': batch_dict_target['multi_scale_3d_features_DIR'],
                        'multi_scale_3d_strides': batch_dict_target['multi_scale_3d_strides_DIR'],
                    })

                    # # final-feature-filled method 1: replace with DIR + storged DSR of 2 src domains
                    # batch_dict_target = self.transfer_DSR(batch_dict_target, DSR_TT)
                    # batch_dict_target = self.merge_DIR_and_DSR(batch_dict_target)

            
            pred_dicts, recall_dicts = self.post_processing(batch_dict_target)
            return pred_dicts, recall_dicts
    # end of forward

    def get_training_det_loss(self):
        disp_dict = {}
        loss = 0
        
        loss_rpn, tb_dict = self.dense_head.get_loss()
        loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)

        loss = loss + loss_rpn + loss_rcnn

        if hasattr(self.backbone_3d, 'get_loss'):
            loss_backbone3d, tb_dict = self.backbone_3d.get_loss(tb_dict)
            loss += loss_backbone3d
            
        return loss, tb_dict, disp_dict

class VoxelRCNN_M_DB(Detector3DTemplate_M_DB):
    def __init__(self, model_cfg, num_class, num_class_s2, dataset, dataset_s2, source_one_name):
        super().__init__(model_cfg=model_cfg, num_class=num_class, num_class_s2=num_class_s2, dataset=dataset,
                         dataset_s2=dataset_s2, source_one_name=source_one_name)
        self.module_list = self.build_networks()
        self.source_one_name = source_one_name

    def forward(self, batch_dict):

        # Split the Concat dataset batch into batch_1 and batch_2
        split_tag_s1, split_tag_s2 = common_utils.split_batch_dict(self.source_one_name, batch_dict)

        batch_s1 = {}
        batch_s2 = {}
   
        len_of_module = len(self.module_list)
        for k, cur_module in enumerate(self.module_list):
            if k < len_of_module-4:
                batch_dict = cur_module(batch_dict)
            
            if k == len_of_module-4 or k == len_of_module-3:
                if len(split_tag_s1) == batch_dict['batch_size']:
                    batch_dict = cur_module(batch_dict)
                elif len(split_tag_s2) == batch_dict['batch_size']:
                    continue
                else:
                    if k == len_of_module-4:
                        batch_s1, batch_s2 = common_utils.split_two_batch_dict_gpu(split_tag_s1, split_tag_s2, batch_dict)
                    batch_s1 = cur_module(batch_s1)

            if k == len_of_module-2 or k == len_of_module-1:
                if len(split_tag_s2) == batch_dict['batch_size']:
                    batch_dict = cur_module(batch_dict)
                elif len(split_tag_s1) == batch_dict['batch_size']:
                    continue
                else:
                    batch_s2 = cur_module(batch_s2)
            
        if self.training:
            split_tag_s1, split_tag_s2 = common_utils.split_batch_dict(self.source_one_name, batch_dict)
            if len(split_tag_s1) == batch_dict['batch_size']:
                loss, tb_dict, disp_dict = self.get_training_loss_s1()
            
                ret_dict = {
                    'loss': loss
                }
                return ret_dict, tb_dict, disp_dict
            elif len(split_tag_s2) == batch_dict['batch_size']:
                loss, tb_dict, disp_dict = self.get_training_loss_s2()
            
                ret_dict = {
                    'loss': loss
                }
                return ret_dict, tb_dict, disp_dict
            else:
                loss_1, tb_dict_1, disp_dict_1 = self.get_training_loss_s1()
                loss_2, tb_dict_2, disp_dict_2 = self.get_training_loss_s2()
                ret_dict = {
                    'loss': loss_1 + loss_2
                }
                return ret_dict, tb_dict_1, disp_dict_1
              
        else:
            # NOTE: When peform the inference, only one dataset can be accessed.
            if 'batch_box_preds' in batch_dict.keys():
                pred_dicts, recall_dicts = self.post_processing(batch_dict)
                return pred_dicts, recall_dicts
            elif 'batch_box_preds' in batch_s1.keys():
                pred_dicts_s1, recall_dicts_s1 = self.post_processing(batch_s1)
                pred_dicts_s2, recall_dicts_s2 = self.post_processing(batch_s2)
                return pred_dicts_s1, recall_dicts_s1, pred_dicts_s2, recall_dicts_s2

    def get_training_loss_s1(self):
        disp_dict = {}
        loss = 0
        
        loss_rpn, tb_dict = self.dense_head_s1.get_loss()
        loss_rcnn, tb_dict = self.roi_head_s1.get_loss(tb_dict)

        loss = loss + loss_rpn + loss_rcnn
        return loss, tb_dict, disp_dict
    
    def get_training_loss_s2(self):
        disp_dict = {}
        loss = 0
        
        loss_rpn, tb_dict = self.dense_head_s2.get_loss()
        loss_rcnn, tb_dict = self.roi_head_s2.get_loss(tb_dict)

        loss = loss + loss_rpn + loss_rcnn
        return loss, tb_dict, disp_dict

class VoxelRCNN_M_DB_3(Detector3DTemplate_M_DB_3):
    def __init__(self, model_cfg, num_class, num_class_s2, num_class_s3, dataset, dataset_s2, dataset_s3, source_one_name, source_1):
        super().__init__(model_cfg=model_cfg, num_class=num_class, num_class_s2=num_class_s2, num_class_s3=num_class_s3,
                        dataset=dataset, dataset_s2=dataset_s2, dataset_s3=dataset_s3, source_one_name=source_one_name, source_1=source_1)
        self.module_list = self.build_networks()
        self.source_one_name = source_one_name
        self.source_1 = source_1

    def forward(self, batch_dict):
        batch_s1 = {}
        batch_s2 = {}
        batch_s3 = {}

        if self.training:
            len_of_module = len(self.module_list)
            for k, cur_module in enumerate(self.module_list):
                if k < len_of_module-6:
                    batch_dict = cur_module(batch_dict)
                
                if k == len_of_module-6 or k == len_of_module-5:
                    # Split the Concat dataset batch into batch_1, batch_2, and batch_3
                    if k == len_of_module-6:
                        split_tag_s1, split_tag_s2_pre = common_utils.split_batch_dict('waymo', batch_dict)
                        batch_s1, batch_s2_pre = common_utils.split_two_batch_dict_gpu(split_tag_s1, split_tag_s2_pre, batch_dict)
                        split_tag_s2, split_tag_s3 = common_utils.split_batch_dict(self.source_one_name, batch_s2_pre)
                        batch_s2, batch_s3 = common_utils.split_two_batch_dict_gpu(split_tag_s2, split_tag_s3, batch_s2_pre)
                    batch_s1 = cur_module(batch_s1)

                if k == len_of_module-4 or k == len_of_module-3:              
                    batch_s2 = cur_module(batch_s2)

                if k == len_of_module-2 or k == len_of_module-1:
                    batch_s3 = cur_module(batch_s3)
        else:
            len_of_module = len(self.module_list)
            for k, cur_module in enumerate(self.module_list):
                if k < len_of_module-6:
                    batch_dict = cur_module(batch_dict)
                
                if k == len_of_module-6 or k == len_of_module-5:
                    if self.source_1 == 1:
                        batch_dict = cur_module(batch_dict)
                    else:
                        continue
                if k == len_of_module-4 or k == len_of_module-3:
                    if self.source_1 == 2:         
                        batch_dict = cur_module(batch_dict)
                    else:
                        continue

                if k == len_of_module-2 or k == len_of_module-1:
                    if self.source_1 == 3:  
                        batch_dict = cur_module(batch_dict)
                    else:
                        continue

        if self.training:
            loss_1, tb_dict_1, disp_dict_1 = self.get_training_loss_s1()
            loss_2, tb_dict_2, disp_dict_2 = self.get_training_loss_s2()
            loss_3, tb_dict_3, disp_dict_3 = self.get_training_loss_s3()
            ret_dict = {
                'loss': loss_1 + loss_2 + loss_3
            }
            return ret_dict, tb_dict_1, disp_dict_1
              
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss_s1(self):
        disp_dict = {}
        loss = 0
        
        loss_rpn, tb_dict = self.dense_head_s1.get_loss()
        loss_rcnn, tb_dict = self.roi_head_s1.get_loss(tb_dict)

        loss = loss + loss_rpn + loss_rcnn
        return loss, tb_dict, disp_dict
    
    def get_training_loss_s2(self):
        disp_dict = {}
        loss = 0
        
        loss_rpn, tb_dict = self.dense_head_s2.get_loss()
        loss_rcnn, tb_dict = self.roi_head_s2.get_loss(tb_dict)

        loss = loss + loss_rpn + loss_rcnn
        return loss, tb_dict, disp_dict

    def get_training_loss_s3(self):
        disp_dict = {}
        loss = 0
        
        loss_rpn, tb_dict = self.dense_head_s3.get_loss()
        loss_rcnn, tb_dict = self.roi_head_s3.get_loss(tb_dict)

        loss = loss + loss_rpn + loss_rcnn
        return loss, tb_dict, disp_dict

class ActiveDualVoxelRCNN(ActiveDetector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    def forward(self, batch_dict, **forward_args):
        batch_dict['mode'] = forward_args.get('mode', None) if forward_args is not None else None
        
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training and forward_args.get('mode', None) == 'train_discriminator':
            loss = self.discriminator.get_discriminator_loss(batch_dict, source=forward_args['source'])
            return loss
        
        if self.training and forward_args.get('mode', None) == 'train_detector':
            loss, tb_dict, disp_dict = self.get_detector_loss()
        
        elif not self.training and forward_args.get('mode', None) == 'active_evaluate':
            batch_dict = self.post_processing(batch_dict)
            sample_score = self.get_evaluate_score(batch_dict, forward_args['domain'])
            return sample_score
        elif not self.training and forward_args.get('mode', None) == None:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

        ret_dict = {
            'loss': loss
        }
        return ret_dict, tb_dict, disp_dict

    def get_detector_loss(self):
        disp_dict = {}
        loss_rpn, tb_dict = self.dense_head.get_loss()
        loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)

        loss = loss_rpn + loss_rcnn
        return loss, tb_dict, disp_dict

    def get_evaluate_score(self, batch_dict, domain):
        batch_dict = self.discriminator.domainness_evaluate(batch_dict)
        batch_size = batch_dict['batch_size']
        frame_id = [str(id) for id in batch_dict['frame_id']]
        domainness_evaluate = batch_dict['domainness_evaluate'].cpu()
        reweight_roi = batch_dict['reweight_roi']
        sample_score = []

        for i in range(batch_size):
            for i in range(batch_size):
                frame_score = {
                    'frame_id': frame_id[i],
                    'domainness_evaluate': domainness_evaluate[i].cpu(),
                    'roi_feature': reweight_roi[i],
                    'total_score': domainness_evaluate[i].cpu()
                }
                sample_score.append(frame_score)
            return sample_score


class VoxelRCNN_TQS(ActiveDetector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    def forward(self, batch_dict, **forward_args):
        batch_dict['mode'] = forward_args.get('mode', None) if forward_args is not None else None
        
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training and forward_args.get('mode', None) == 'train_discriminator':
            loss = self.discriminator.get_discriminator_loss(batch_dict, source=forward_args['source'])
            return loss
        
        if self.training and forward_args.get('mode', None) == 'train_detector':
            loss, tb_dict, disp_dict = self.get_detector_loss()
        elif self.training and forward_args.get('mode', 'train_mul_cls'):
            loss, tb_dict, disp_dict = self.get_mul_cls_loss()
        
        elif not self.training and forward_args.get('mode', None) == 'active_evaluate':
            batch_dict = self.post_processing(batch_dict)
            sample_score = self.get_evaluate_score(batch_dict, forward_args['domain'])
            return sample_score
        elif not self.training and forward_args.get('mode', None) == None:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

        ret_dict = {
            'loss': loss
        }
        return ret_dict, tb_dict, disp_dict

    def get_mul_cls_loss(self, mode='train_mul_cls'):
        disp_dict = {}
        loss, loss_mul, tb_dict = self.roi_head.get_mul_cls_loss()
        return loss_mul, tb_dict, disp_dict
    
    def get_detector_loss(self):
        disp_dict = {}
        loss_rpn, tb_dict = self.dense_head.get_loss()
        loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)

        loss = loss_rpn + loss_rcnn
        return loss, tb_dict, disp_dict

    def get_evaluate_score(self, batch_dict, domain):
        batch_dict = self.discriminator.domainness_evaluate(batch_dict)
        batch_dict = self.roi_head.committee_evaluate(batch_dict)
        batch_dict = self.roi_head.uncertainty_evaluate(batch_dict)
        batch_size = batch_dict['batch_size']
        frame_id = [str(id) for id in batch_dict['frame_id']]
        domainness_evaluate = batch_dict['domainness_evaluate'].cpu()
        reweight_roi = batch_dict['reweight_roi']
        committee_evaluate = batch_dict['committee_score'].cpu()
        uncertainty_evaluate = batch_dict['uncertainty'].cpu()
        roi_score = batch_dict['cls_preds']
        sample_score = []

        for i in range(batch_size):

            frame_score = {
                'frame_id': frame_id[i],
                'committee_evaluate': committee_evaluate[i],
                'uncertainty_evaluate': uncertainty_evaluate[i],
                'domainness_evaluate': domainness_evaluate[i],
                'roi_feature': reweight_roi[i],
                'roi_score': roi_score[i],
                'total_score': committee_evaluate[i] + uncertainty_evaluate[i] + domainness_evaluate[i]
            }
            sample_score.append(frame_score)
        return sample_score

class VoxelRCNN_CLUE(ActiveDetector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    def forward(self, batch_dict, **forward_args):
        batch_dict['mode'] = forward_args.get('mode', None) if forward_args is not None else None
        
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training and forward_args.get('mode', None) == 'train_discriminator':
            loss = self.discriminator.get_discriminator_loss(batch_dict, source=forward_args['source'])
            return loss
        
        if self.training and forward_args.get('mode', None) == 'train_detector':
            loss, tb_dict, disp_dict = self.get_detector_loss()
        elif self.training and forward_args.get('mode', 'train_mul_cls'):
            loss, tb_dict, disp_dict = self.get_mul_cls_loss()
        
        elif not self.training and forward_args.get('mode', None) == 'active_evaluate':
            batch_dict = self.post_processing(batch_dict)
            sample_score = self.get_evaluate_score(batch_dict, forward_args['domain'])
            return sample_score
        elif not self.training and forward_args.get('mode', None) == None:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

        ret_dict = {
            'loss': loss
        }
        return ret_dict, tb_dict, disp_dict

    def get_mul_cls_loss(self, mode='train_mul_cls'):
        disp_dict = {}
        loss, mul_loss, tb_dict = self.roi_head.get_mul_cls_loss()
        return mul_loss, tb_dict, disp_dict
    
    def get_detector_loss(self):
        disp_dict = {}
        loss_rpn, tb_dict = self.dense_head.get_loss()
        loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)

        loss = loss_rpn + loss_rcnn
        return loss, tb_dict, disp_dict

    def get_evaluate_score(self, batch_dict, domain):
        # batch_dict = self.discriminator.domainness_evaluate(batch_dict)
        batch_size = batch_dict['batch_size']
        frame_id = [str(id) for id in batch_dict['frame_id']]
        # domainness_evaluate = batch_dict['domainness_evaluate'].cpu()
        reweight_roi = batch_dict['reweight_roi']
        roi_score = batch_dict['cls_preds']
        sample_score = []

        for i in range(batch_size):
            for i in range(batch_size):
                frame_score = {
                    'frame_id': frame_id[i],
                    # 'domainness_evaluate': domainness_evaluate[i].cpu(),
                    'roi_feature': reweight_roi[i],
                    'roi_score': roi_score[i]
                    # 'total_score': domainness_evaluate[i].cpu()
                }
                sample_score.append(frame_score)
            return sample_score
