import torch
import torch.nn as nn
import torch.nn.functional as F
# from ....ops.pointnet2.pointnet2_stack import voxel_pool_modules as voxelpool_stack_modules
from ....utils import common_utils, box_utils
import pickle
import copy
import math


class Siamese_Searcher(nn.Module):
    def __init__(self, backbone_channels, model_cfg, point_cloud_range, voxel_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.pool_cfg = model_cfg.ROI_GRID_POOL
        LAYER_cfg = self.pool_cfg.POOL_LAYERS
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size
        self.roi_ext = self.model_cfg.ROI_EXTEND_FACTOR

        self.init_weights()

    def init_weights(self):
        init_func = nn.init.xavier_normal_
        # for module_list in [self.shared_fc_layer, self.cls_fc_layers, self.reg_fc_layers]:
        #     for m in module_list.modules():
        #         if isinstance(m, nn.Linear):
        #             init_func(m.weight)
        #             if m.bias is not None:
        #                 nn.init.constant_(m.bias, 0)
                    
        # nn.init.normal_(self.cls_pred_layer.weight, 0, 0.01)
        # nn.init.constant_(self.cls_pred_layer.bias, 0)
        # nn.init.normal_(self.reg_pred_layer.weight, mean=0, std=0.001)
        # nn.init.constant_(self.reg_pred_layer.bias, 0)

    # def _init_weights(self):
    #     init_func = nn.init.xavier_normal_
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
    #             init_func(m.weight)
    #             if m.bias is not None:
    #                 nn.init.constant_(m.bias, 0)
    #     nn.init.normal_(self.reg_layers[-1].weight, mean=0, std=0.001)
    
    def roi_grid_pool(self, batch_dict, rois, roi_grid_pool_layers, roi_ext=None):
        """
        Args:
            batch_dict:
                batch_size:
                rois: (B, num_rois, 7 + C)
                point_coords: (num_points, 4)  [bs_idx, x, y, z]
                point_features: (num_points, C)
                point_cls_scores: (N1 + N2 + N3 + ..., 1)
                point_part_offset: (N1 + N2 + N3 + ..., 3)
        Returns:

        """
        batch_size = batch_dict['batch_size']
        with_vf_transform = batch_dict.get('with_voxel_feature_transform', False)
        
        grid_size = self.pool_cfg.GRID_SIZE if roi_ext is None else int(self.pool_cfg.GRID_SIZE * roi_ext)
        if roi_ext: # XY only
            rois[:, :, 3:5] *= roi_ext
        
        roi_grid_xyz, _ = self.get_global_grid_points_of_roi(
            rois, grid_size = grid_size
        )  # (BxN, 6x6x6, 3)
        # roi_grid_xyz: (B, Nx6x6x6, 3)
        roi_grid_xyz = roi_grid_xyz.view(batch_size, -1, 3) 
        roi_grid_xyz_output = roi_grid_xyz.clone() 

        # compute the voxel coordinates of grid points
        roi_grid_coords_x = (roi_grid_xyz[:, :, 0:1] - self.point_cloud_range[0]) // self.voxel_size[0]
        roi_grid_coords_y = (roi_grid_xyz[:, :, 1:2] - self.point_cloud_range[1]) // self.voxel_size[1]
        roi_grid_coords_z = (roi_grid_xyz[:, :, 2:3] - self.point_cloud_range[2]) // self.voxel_size[2]
        # roi_grid_coords: (B, Nx6x6x6, 3)
        roi_grid_coords = torch.cat([roi_grid_coords_x, roi_grid_coords_y, roi_grid_coords_z], dim=-1)

        batch_idx = rois.new_zeros(batch_size, roi_grid_coords.shape[1], 1)
        for bs_idx in range(batch_size):
            batch_idx[bs_idx, :, 0] = bs_idx
        # roi_grid_coords: (B, Nx6x6x6, 4)
        # roi_grid_coords = torch.cat([batch_idx, roi_grid_coords], dim=-1)
        # roi_grid_coords = roi_grid_coords.int()
        roi_grid_batch_cnt = rois.new_zeros(batch_size).int().fill_(roi_grid_coords.shape[1])

        pooled_features_list = []
        for k, src_name in enumerate(self.pool_cfg.FEATURES_SOURCE):
            pool_layer = roi_grid_pool_layers[self.pool_cfg.POOL_LAYERS.index(src_name)]
            cur_stride = batch_dict['multi_scale_3d_strides'][src_name]
            cur_sp_tensors = batch_dict['multi_scale_3d_features'][src_name]

            if with_vf_transform:
                cur_sp_tensors = batch_dict['multi_scale_3d_features_post'][src_name]
            else:
                cur_sp_tensors = batch_dict['multi_scale_3d_features'][src_name]

            # compute voxel center xyz and batch_cnt
            cur_coords = cur_sp_tensors.indices
            cur_voxel_xyz = common_utils.get_voxel_centers(
                cur_coords[:, 1:4],
                downsample_times=cur_stride,
                voxel_size=self.voxel_size,
                point_cloud_range=self.point_cloud_range
            )
            cur_voxel_xyz_batch_cnt = cur_voxel_xyz.new_zeros(batch_size).int()
            for bs_idx in range(batch_size):
                cur_voxel_xyz_batch_cnt[bs_idx] = (cur_coords[:, 0] == bs_idx).sum()
            # get voxel2point tensor
            v2p_ind_tensor = common_utils.generate_voxel2pinds(cur_sp_tensors)
            # compute the grid coordinates in this scale, in [batch_idx, x y z] order
            cur_roi_grid_coords = roi_grid_coords // cur_stride
            cur_roi_grid_coords = torch.cat([batch_idx, cur_roi_grid_coords], dim=-1)
            cur_roi_grid_coords = cur_roi_grid_coords.int()
            # voxel neighbor aggregation
            pooled_features = pool_layer(
                xyz=cur_voxel_xyz.contiguous(),
                xyz_batch_cnt=cur_voxel_xyz_batch_cnt,
                new_xyz=roi_grid_xyz.contiguous().view(-1, 3),
                new_xyz_batch_cnt=roi_grid_batch_cnt,
                new_coords=cur_roi_grid_coords.contiguous().view(-1, 4),
                features=cur_sp_tensors.features.contiguous(),
                voxel2point_indices=v2p_ind_tensor
            )

            pooled_features = pooled_features.view(
                batch_size, -1, grid_size, grid_size, self.pool_cfg.GRID_SIZE,
                pooled_features.shape[-1]
            ) 
            pooled_features_list.append(pooled_features)
        
        ms_pooled_features = torch.cat(pooled_features_list, dim=-1)  # (B, N, 6, 6, 6, C)

        roi_grid_xyz_output = roi_grid_xyz_output.view(batch_size, ms_pooled_features.shape[1], 
            grid_size, grid_size, self.pool_cfg.GRID_SIZE, 
            3,
        )
        return ms_pooled_features, roi_grid_xyz_output


    def get_global_grid_points_of_roi(self, rois, grid_size):
        rois = rois.view(-1, rois.shape[-1])
        batch_size_rcnn = rois.shape[0]

        local_roi_grid_points = self.get_dense_grid_points(rois, batch_size_rcnn, grid_size)  # (B, 6x6x6, 3)
        global_roi_grid_points = common_utils.rotate_points_along_z(
            local_roi_grid_points.clone(), rois[:, 6]
        ).squeeze(dim=1)
        global_center = rois[:, 0:3].clone()
        global_roi_grid_points += global_center.unsqueeze(dim=1)
        return global_roi_grid_points, local_roi_grid_points

    # @staticmethod
    def get_dense_grid_points(self, rois, batch_size_rcnn, grid_size):
        faked_features = rois.new_ones((grid_size, grid_size, self.pool_cfg.GRID_SIZE))
        dense_idx = faked_features.nonzero()  # (N, 3) [x_idx, y_idx, z_idx]
        dense_idx = dense_idx.repeat(batch_size_rcnn, 1, 1).float()  # (B, 6x6x6, 3)

        local_roi_size = rois.view(batch_size_rcnn, -1)[:, 3:6]
        grid_size_xyz = torch.tensor([grid_size, grid_size, self.pool_cfg.GRID_SIZE], dtype=rois.dtype, device=rois.device)
        roi_grid_points = (dense_idx + 0.5) / grid_size_xyz * local_roi_size.unsqueeze(dim=1) \
                          - (local_roi_size.unsqueeze(dim=1) / 2)  # (B, 6x6x6, 3)
        return roi_grid_points
    
    def forward(self, batch_dict, rois, roi_labels, roi_grid_pool_layers, box_proto_dict, atten_cls):
        '''
        Args:
            batch_dict:
            rois: (B,M,7)
            roi_labels: (B, M)
            box_proto_dict: 
                cls_ids:
                    boxes_proto: (K_shots, 7)
                    feats_proto: (K_shots, grid_size, grid_size, grid_size, C)
        Return:
            rois_new
        '''

        feats_rois, grid_coors_rois = self.roi_grid_pool(batch_dict, rois, roi_grid_pool_layers, self.roi_ext)
        grid_size_roi = int(self.pool_cfg.GRID_SIZE * self.roi_ext)

        roi_atten_mask = torch.zeros_like(roi_labels)

        rois_new = rois.clone()
        for i_batch in range(rois.shape[0]):
            for i_obj in range(rois.shape[1]):
                roi_cur = rois[i_batch, i_obj, :]
                label_cur = roi_labels[i_batch, i_obj].cpu().item()

                if label_cur not in atten_cls:
                    continue

                roi_atten_mask[i_batch, i_obj] = 1           
                
                feat_cur_roi = feats_rois[i_batch, i_obj, :, :, :, :].unsqueeze(0) # 1,6,6,6,C
                grid_coor_roi = grid_coors_rois[i_batch, i_obj, :, :, :, :]
                # print(roi_cur[:3]-grid_coor_roi.view(self.pool_cfg.GRID_SIZE**3, 3)[171,:])

                boxes_proto = box_proto_dict[label_cur]['boxes_proto']
                feats_proto = box_proto_dict[label_cur]['feats_proto'] # K_shots,6,6,6,C
                
                assert self.pool_cfg.GRID_SIZE % 2 == 1

                padding = int((self.pool_cfg.GRID_SIZE-1)/2)
                match_map = F.conv3d(feat_cur_roi.permute(0,4,1,2,3), feats_proto.permute(0,4,1,2,3),\
                    padding=padding)

                BN3D = torch.nn.BatchNorm3d(feats_proto.shape[0], affine=False).cuda()
                match_map = BN3D(match_map)
                # print(match_map.shape)
                match_map = match_map.squeeze(dim=0)
                # print(match_map[0,:,:,3])
                match_map = match_map.view(match_map.shape[0], -1)
                idx_max = torch.argmax(match_map)
                idx_proto_selected = idx_max // match_map.shape[1]
                idx_loc_selected = idx_max % match_map.shape[1]

                rois_new[i_batch, i_obj, :3] = grid_coor_roi.view(grid_size_roi**2*self.pool_cfg.GRID_SIZE, 3)[idx_loc_selected,:]
                # rois_new[i_batch, i_obj, 3:6] = boxes_proto[idx_proto_selected, 3:6]
                rois_new[i_batch, i_obj, 6] = rois[i_batch, i_obj, 6]

        return rois_new






