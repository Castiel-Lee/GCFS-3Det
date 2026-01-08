import torch
import torch.nn as nn
import torch.nn.functional as F
# from ....ops.pointnet2.pointnet2_stack import voxel_pool_modules as voxelpool_stack_modules
from ....utils import common_utils, box_utils
import pickle
import copy



class Siamese_Searcher(nn.Module):
    def __init__(self, backbone_channels, model_cfg, point_cloud_range, voxel_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.pool_cfg = model_cfg.ROI_GRID_POOL
        LAYER_cfg = self.pool_cfg.POOL_LAYERS
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size

        # c_out = 0
        # self.roi_grid_pool_layers = nn.ModuleList()
        # for src_name in self.pool_cfg.FEATURES_SOURCE:
        #     mlps = LAYER_cfg[src_name].MLPS
        #     for k in range(len(mlps)):
        #         mlps[k] = [backbone_channels[src_name]] + mlps[k]
        #     pool_layer = voxelpool_stack_modules.NeighborVoxelSAModuleMSG(
        #         query_ranges=LAYER_cfg[src_name].QUERY_RANGES,
        #         nsamples=LAYER_cfg[src_name].NSAMPLE,
        #         radii=LAYER_cfg[src_name].POOL_RADIUS,
        #         mlps=mlps,
        #         pool_method=LAYER_cfg[src_name].POOL_METHOD,
        #     )
            
        #     self.roi_grid_pool_layers.append(pool_layer)

        #     c_out += sum([x[-1] for x in mlps])

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
    
    def roi_grid_pool(self, batch_dict, box, roi_grid_pool_layers, xy_dim_extend=None):
        """
        Args:
            Note! batch_size of batch_dict == 1
            batch_dict:
                batch_size:
                box: (B, num_box, 7 + C)
                point_coords: (num_points, 4)  [bs_idx, x, y, z]
                point_features: (num_points, C)
                point_cls_scores: (N1 + N2 + N3 + ..., 1)
                point_part_offset: (N1 + N2 + N3 + ..., 3)
        Returns:

        """
        batch_size = batch_dict['batch_size']
        assert batch_size == 1
        with_vf_transform = batch_dict.get('with_voxel_feature_transform', False)
        if xy_dim_extend is not None:
            box = box.clone()
            box[3:5] *= xy_dim_extend
        
        grid_size =  box[3:6] / torch.tensor(self.voxel_size, dtype=box.dtype, device=box.device) / self.pool_cfg.STRIDE
        grid_size = torch.round(grid_size)
        roi_grid_xyz, _ = self.get_global_grid_points_of_roi(
            box, grid_size=grid_size
        )  # (XxYxZ, 3)
 
        # compute the voxel coordinates of grid points
        roi_grid_coords_x = (roi_grid_xyz[:, 0:1] - self.point_cloud_range[0]) // self.voxel_size[0]
        roi_grid_coords_y = (roi_grid_xyz[:, 1:2] - self.point_cloud_range[1]) // self.voxel_size[1]
        roi_grid_coords_z = (roi_grid_xyz[:, 2:3] - self.point_cloud_range[2]) // self.voxel_size[2]
        # roi_grid_coords: (XYZ, 3)
        roi_grid_coords = torch.cat([roi_grid_coords_x, roi_grid_coords_y, roi_grid_coords_z], dim=-1)

        batch_idx = box.new_zeros(roi_grid_coords.shape[0], 1) # (XYZ, 1)

        # roi_grid_coords: (B, Nx6x6x6, 4)
        # roi_grid_coords = torch.cat([batch_idx, roi_grid_coords], dim=-1)
        # roi_grid_coords = roi_grid_coords.int()
        roi_grid_batch_cnt = box.new_zeros(1).int().fill_(roi_grid_coords.shape[0])

        pooled_features_list = []
        for k, src_name in enumerate(self.pool_cfg.FEATURES_SOURCE):
            pool_layer = roi_grid_pool_layers[k]
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
            cur_voxel_xyz_batch_cnt = cur_voxel_xyz.new_zeros(1).int()
            for bs_idx in range(1):
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
                -1, grid_size[0].int().item(), grid_size[1].int().item(), grid_size[2].int().item(),
                pooled_features.shape[-1]
            )  # (1,X,Y,Z,C)
            pooled_features_list.append(pooled_features)
        
        ms_pooled_features = torch.cat(pooled_features_list, dim=-1)
        
        return ms_pooled_features, roi_grid_xyz


    def get_global_grid_points_of_roi(self, box, grid_size):
        local_roi_grid_points = self.get_dense_grid_points(box, grid_size)  # (XYZ, 3)
        global_roi_grid_points = common_utils.rotate_points_along_z(
            local_roi_grid_points.clone().unsqueeze(dim=0), box[6].unsqueeze(dim=0)
        ).squeeze(dim=0)
        global_center = box[0:3].clone()
        global_roi_grid_points += global_center
        return global_roi_grid_points, local_roi_grid_points

    @staticmethod
    def get_dense_grid_points(box, grid_size):
        faked_features = box.new_ones((grid_size[0].int().item(), grid_size[1].int().item(), grid_size[2].int().item()))
        dense_idx = faked_features.nonzero().float()  # (N, 3) [x_idx, y_idx, z_idx]

        local_roi_size = box[3:6]
        roi_grid_points = (dense_idx + 0.5) / grid_size * local_roi_size \
                          - (local_roi_size / 2)  # (XYZ, 3)
        return roi_grid_points
    
    def forward(self, batch_dict, proto_box, feat_proto, rois, roi_grid_pool_layers):
        '''
        Args:
            batch_dict:
            proto_box: (7)
            rois: (M,7)
        Return:
            rois_new
        '''
        rois_new = torch.zeros_like(rois)
        assert proto_box.shape[0] == rois.shape[1]
        for i in range(rois.shape[0]):
            roi_cur = rois[i, :]
            feat_roi_cur, roi_grid_coor = self.roi_grid_pool(batch_dict, roi_cur, roi_grid_pool_layers, \
                xy_dim_extend=self.model_cfg.ROI_EXTEND_FACTOR)

            padding = torch.ceil((torch.tensor(feat_proto.shape[1:4]).float()-1)/2).long().tolist()
            match_map = F.conv3d(feat_roi_cur.permute(0,4,1,2,3), feat_proto.permute(0,4,1,2,3),\
                padding=tuple(padding))
            BN3D = torch.nn.BatchNorm3d(1).cuda()
            match_map = BN3D(match_map)
            match_map = match_map.squeeze(dim=0).squeeze(dim=0)
            # print(match_map.shape, feat_roi_cur.shape, feat_proto.shape)
            idx_max = torch.argmax(match_map)
            idx_max = min([idx_max.item(), roi_grid_coor.shape[0]-1])
            box_selected_loc = roi_grid_coor[idx_max]
            box_new = copy.deepcopy(proto_box)
            box_new[:3] = box_selected_loc   
            rois_new[i,:] = box_new

        return rois_new






