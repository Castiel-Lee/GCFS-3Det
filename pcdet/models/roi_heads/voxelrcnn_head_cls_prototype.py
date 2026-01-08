import torch
import torch.nn as nn
from ...ops.pointnet2.pointnet2_stack import voxel_pool_modules as voxelpool_stack_modules
from ...utils import common_utils, box_utils
from .roi_head_template import RoIHeadTemplate
from .utils.cls_prototype_attention import CLS_Attention_Layer
from .utils.box_siamese_search import Siamese_Searcher


class VoxelRCNNHead_CLS_PROTOTYPE(RoIHeadTemplate):
    def __init__(self, backbone_channels, model_cfg, point_cloud_range, voxel_size, class_names, num_class=1, **kwargs):
        super().__init__(num_class=num_class, model_cfg=model_cfg)
        self.model_cfg = model_cfg
        self.pool_cfg = model_cfg.ROI_GRID_POOL
        LAYER_cfg = self.pool_cfg.POOL_LAYERS
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size
        self.class_names = class_names

        if model_cfg.get('SIAMESE_REF_LAYER', False):
            self.siamese_search_layer = Siamese_Searcher(backbone_channels=backbone_channels,
                model_cfg = model_cfg.SIAMESE_REF_LAYER, 
                point_cloud_range = point_cloud_range, 
                voxel_size=voxel_size)

        c_out = 0
        self.roi_grid_pool_layers = nn.ModuleList()
        for src_name in self.pool_cfg.FEATURES_SOURCE:
            mlps = LAYER_cfg[src_name].MLPS
            for k in range(len(mlps)):
                mlps[k] = [backbone_channels[src_name]] + mlps[k]
            pool_layer = voxelpool_stack_modules.NeighborVoxelSAModuleMSG(
                query_ranges=LAYER_cfg[src_name].QUERY_RANGES,
                nsamples=LAYER_cfg[src_name].NSAMPLE,
                radii=LAYER_cfg[src_name].POOL_RADIUS,
                mlps=mlps,
                pool_method=LAYER_cfg[src_name].POOL_METHOD,
            )
            
            self.roi_grid_pool_layers.append(pool_layer)

            c_out += sum([x[-1] for x in mlps])

      
        if self.model_cfg.get('ROI_GRID_POOL_LINEAR', None):

            self.roi_grid_pool_linear_functions = []
            for src_name in self.model_cfg.ROI_GRID_POOL_LINEAR.FEATURES_SOURCE:
                pool_function = voxelpool_stack_modules.NeighborVoxelLinearPoolModuleMSG(
                    query_ranges=self.model_cfg.ROI_GRID_POOL_LINEAR.POOL_LAYERS[src_name].QUERY_RANGES,
                    nsamples=self.model_cfg.ROI_GRID_POOL_LINEAR.POOL_LAYERS[src_name].NSAMPLE,
                    radii=self.model_cfg.ROI_GRID_POOL_LINEAR.POOL_LAYERS[src_name].POOL_RADIUS,
                    pool_method=self.model_cfg.ROI_GRID_POOL_LINEAR.POOL_LAYERS[src_name].POOL_METHOD,
                )
                self.roi_grid_pool_linear_functions.append(pool_function)
        
        
        GRID_SIZE = self.model_cfg.ROI_GRID_POOL.GRID_SIZE
        pre_channel = GRID_SIZE * GRID_SIZE * GRID_SIZE * c_out


        shared_fc_list = []
        for k in range(0, self.model_cfg.SHARED_FC.__len__()):
            shared_fc_list.extend([
                nn.Linear(pre_channel, self.model_cfg.SHARED_FC[k], bias=False),
                nn.BatchNorm1d(self.model_cfg.SHARED_FC[k]),
                nn.ReLU(inplace=True)
            ])
            pre_channel = self.model_cfg.SHARED_FC[k]

            if k != self.model_cfg.SHARED_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                shared_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))
        self.shared_fc_layer = nn.Sequential(*shared_fc_list)

        if self.model_cfg.get('CLS_PROTOTYPES_ATTEN', None):
            self.cls_prototypes_reweight_layer = CLS_Attention_Layer(dim_model=pre_channel, 
                nhead=self.model_cfg.CLS_PROTOTYPES_ATTEN.N_head, 
                dropout=self.model_cfg.CLS_PROTOTYPES_ATTEN.Dropout,
                two_linear=self.model_cfg.CLS_PROTOTYPES_ATTEN.get('USE_TWO_LINEAR', False),
            )
            self.no_skip_atten = self.model_cfg.CLS_PROTOTYPES_ATTEN.get('NO_USE_SKIP', False)

        if self.model_cfg.get('LEARNABLE_CLS_PROTOTYPES_ATTEN', None):
            self.cls_embedding = torch.nn.Embedding(
                num_embeddings=self.model_cfg.LEARNABLE_CLS_PROTOTYPES_ATTEN.N_cls,
                embedding_dim=pre_channel,
            ).cuda()
            self.cls_embedding_reweight_layer = CLS_Attention_Layer(dim_model=pre_channel, 
                nhead=self.model_cfg.LEARNABLE_CLS_PROTOTYPES_ATTEN.N_head, 
                dropout=self.model_cfg.LEARNABLE_CLS_PROTOTYPES_ATTEN.Dropout,
            )

        cls_fc_list = []
        for k in range(0, self.model_cfg.CLS_FC.__len__()):
            cls_fc_list.extend([
                nn.Linear(pre_channel, self.model_cfg.CLS_FC[k], bias=False),
                nn.BatchNorm1d(self.model_cfg.CLS_FC[k]),
                nn.ReLU()
            ])
            pre_channel = self.model_cfg.CLS_FC[k]

            if k != self.model_cfg.CLS_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                cls_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))
        self.cls_fc_layers = nn.Sequential(*cls_fc_list)
        self.cls_pred_layer = nn.Linear(pre_channel, self.num_class, bias=True)

        reg_fc_list = []
        for k in range(0, self.model_cfg.REG_FC.__len__()):
            reg_fc_list.extend([
                nn.Linear(pre_channel, self.model_cfg.REG_FC[k], bias=False),
                nn.BatchNorm1d(self.model_cfg.REG_FC[k]),
                nn.ReLU()
            ])
            pre_channel = self.model_cfg.REG_FC[k]

            if k != self.model_cfg.REG_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                reg_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))
        self.reg_fc_layers = nn.Sequential(*reg_fc_list)
        self.reg_pred_layer = nn.Linear(pre_channel, self.box_coder.code_size * self.num_class, bias=True)

        self.init_weights()

    def init_weights(self):
        init_func = nn.init.xavier_normal_
        for module_list in [self.shared_fc_layer, self.cls_fc_layers, self.reg_fc_layers]:
            for m in module_list.modules():
                if isinstance(m, nn.Linear):
                    init_func(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                    
        nn.init.normal_(self.cls_pred_layer.weight, 0, 0.01)
        nn.init.constant_(self.cls_pred_layer.bias, 0)
        nn.init.normal_(self.reg_pred_layer.weight, mean=0, std=0.001)
        nn.init.constant_(self.reg_pred_layer.bias, 0)

    # def _init_weights(self):
    #     init_func = nn.init.xavier_normal_
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
    #             init_func(m.weight)
    #             if m.bias is not None:
    #                 nn.init.constant_(m.bias, 0)
    #     nn.init.normal_(self.reg_layers[-1].weight, mean=0, std=0.001)
    
    def roi_grid_pool(self, batch_dict, use_gt_as_rois=False):
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
        rois = batch_dict['rois'] if not use_gt_as_rois else batch_dict['gt_boxes']
        batch_size = batch_dict['batch_size']
        with_vf_transform = batch_dict.get('with_voxel_feature_transform', False)
        
        roi_grid_xyz, _ = self.get_global_grid_points_of_roi(
            rois, grid_size=self.pool_cfg.GRID_SIZE
        )  # (BxN, 6x6x6, 3)
        # roi_grid_xyz: (B, Nx6x6x6, 3)
        roi_grid_xyz = roi_grid_xyz.view(batch_size, -1, 3)  

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
            pool_layer = self.roi_grid_pool_layers[k]
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
                -1, self.pool_cfg.GRID_SIZE ** 3,
                pooled_features.shape[-1]
            )  # (BxN, 6x6x6, C)
            pooled_features_list.append(pooled_features)
        
        ms_pooled_features = torch.cat(pooled_features_list, dim=-1)
        
        return ms_pooled_features

    def gt_grid_pool_output(self, batch_dict, sampling_ratio=1, output_index=2, dsr_used=False, dsr_dsr_used=False):
        """
        Args:
            batch_dict:
                batch_size:
                gt_boxes: (B, max_gt, 7 + 1) [x, y, z, dx, dy, dz, heading, gt_label]
                output_index:  2:x_conv4, 0:x_conv2
        Returns:

        """
        import random

        rois = batch_dict['gt_boxes']

        def filter_boxes_by_label(boxes, ratio=1):
            """
            Args:
                boxes: (B, max_gt, 7 + 1) [x, y, z, dx, dy, dz, heading, gt_label] # sample get_labels within [1,3] 
            Returns:
                pooled_features_dict:
                    roi_pooled_features: (BxN, 6x6x6, C)
                    roi_label: (BxN, 1)

            """

            batch_size = boxes.shape[0]
            N_gt = boxes.shape[1]
            N_sample = N_gt
            for i_b in range(batch_size):
                N_sample = min(N_sample, torch.logical_and(boxes[i_b, :, 7]>0, boxes[i_b, :, 7]<4).sum().item())
            if ratio != 1:
                N_sample = int(N_sample * ratio)
            
            new_boxes = boxes.new_zeros(batch_size, N_sample, boxes.shape[-1])
            for i_b in range(batch_size):
                boxes_cur_bat = boxes[i_b, torch.logical_and(boxes[i_b, :, 7]>0, boxes[i_b, :, 7]<4), :] # (N, 8)
                index = torch.LongTensor(random.sample(range(boxes_cur_bat.shape[0]), N_sample)).to(boxes_cur_bat.device)
                boxes_cur_bat_sampled = torch.index_select(boxes_cur_bat, 0, index)

                new_boxes[i_b, :, :] = boxes_cur_bat_sampled
            
            return new_boxes, N_sample

        rois, N_box_per_image = filter_boxes_by_label(rois, sampling_ratio)
        batch_size = batch_dict['batch_size']
        with_vf_transform = batch_dict.get('with_voxel_feature_transform', False)
        pooled_features_dict = {}
        if N_box_per_image>0:
            roi_grid_xyz, _ = self.get_global_grid_points_of_roi(
                rois, grid_size=self.model_cfg.ROI_GRID_POOL_LINEAR.GRID_SIZE
            )  # (BxN, 6x6x6, 3)
            # roi_grid_xyz: (B, Nx6x6x6, 3)
            roi_grid_xyz = roi_grid_xyz.view(batch_size, -1, 3)  

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

            # for k, src_name in enumerate(self.pool_cfg.FEATURES_SOURCE):
            
            src_index = output_index # output [2:x_conv4, 0:x_conv2]
            src_name = self.model_cfg.ROI_GRID_POOL_LINEAR.FEATURES_SOURCE[src_index]
            pool_function = self.roi_grid_pool_linear_functions[src_index]
            cur_stride = batch_dict['multi_scale_3d_strides'][src_name]
            cur_sp_tensors = batch_dict['multi_scale_3d_features'][src_name]

            if dsr_used:
                cur_stride = batch_dict['multi_scale_3d_strides_DSR'][src_name]
                cur_sp_tensors = batch_dict['multi_scale_3d_features_DSR'][src_name]

            # if with_vf_transform:
            #     cur_sp_tensors = batch_dict['multi_scale_3d_features_post'][src_name]
            # else:
            #     cur_sp_tensors = batch_dict['multi_scale_3d_features'][src_name]

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
            pooled_features = pool_function(
                xyz=cur_voxel_xyz.contiguous(),
                xyz_batch_cnt=cur_voxel_xyz_batch_cnt,
                new_xyz=roi_grid_xyz.contiguous().view(-1, 3),
                new_xyz_batch_cnt=roi_grid_batch_cnt,
                new_coords=cur_roi_grid_coords.contiguous().view(-1, 4),
                features=cur_sp_tensors.features.contiguous(),
                voxel2point_indices=v2p_ind_tensor
            )

            pooled_features = pooled_features.view(
                -1, self.model_cfg.ROI_GRID_POOL_LINEAR.GRID_SIZE ** 3,
                pooled_features.shape[-1]
            )  # (BxN, 6x6x6, C)

            pooled_features_dict.update({
                'roi_pooled_features': pooled_features, # (BxN, 6x6x6, C)
                'roi_label': rois[:, :, -1].view(-1,1), # (BxN, 1)
                'boxes_gt_roi': rois,
            })

            assert batch_size == 1
            corners_lidar_roi = box_utils.boxes_to_corners_3d(rois[0, :, :7])  
            N_obj_roi = pooled_features_dict['roi_label'].shape[0]
            pts_num_obj_roi = torch.zeros_like(pooled_features_dict['roi_label'])
            for i_obj in range(N_obj_roi):
                flag = box_utils.in_hull(batch_dict['points'][:, 1:].cpu(), corners_lidar_roi[i_obj].cpu())
                pts_num_obj_roi[i_obj] = (flag.sum())/(rois[0, i_obj, 3].item() * rois[0, i_obj, 4].item() * rois[0, i_obj, 5].item())
            # print(pts_num_obj_roi)
            pooled_features_dict.update({
                'pts_num_obj_roi': pts_num_obj_roi, 
            })
        
        return pooled_features_dict


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

    @staticmethod
    def get_dense_grid_points(rois, batch_size_rcnn, grid_size):
        faked_features = rois.new_ones((grid_size, grid_size, grid_size))
        dense_idx = faked_features.nonzero()  # (N, 3) [x_idx, y_idx, z_idx]
        dense_idx = dense_idx.repeat(batch_size_rcnn, 1, 1).float()  # (B, 6x6x6, 3)

        local_roi_size = rois.view(batch_size_rcnn, -1)[:, 3:6]
        roi_grid_points = (dense_idx + 0.5) / grid_size * local_roi_size.unsqueeze(dim=1) \
                          - (local_roi_size.unsqueeze(dim=1) / 2)  # (B, 6x6x6, 3)
        return roi_grid_points

    def forward(self, batch_dict):
        """
        :param input_data: input dict
        :return:
        """

        targets_dict = self.proposal_layer(
            batch_dict, nms_config=self.model_cfg.NMS_CONFIG['TRAIN' if self.training else 'TEST']
        )

        if batch_dict.get('add_roi_boxes', False):
            batch_size = batch_dict['rois'].shape[0]
            assert batch_size == 1
            max_num_add_box = max([frame['boxes'].shape[0] for frame in batch_dict['add_roi_boxes']])
            max_num_add_box = max(1, max_num_add_box)  # at least one faked rois to avoid error
            rois_exist = batch_dict['rois']
            rois_add = rois_exist.new_zeros((batch_size, max_num_add_box, rois_exist.shape[-1]))
            roi_labels_add = rois_exist.new_zeros((batch_size, max_num_add_box)).long()
            roi_scores_add = rois_exist.new_zeros((batch_size, max_num_add_box))
            for b in range(batch_size):
                num_obj = batch_dict['add_roi_boxes'][b]['boxes'].shape[0]
                rois_add[b, :num_obj, :] = batch_dict['add_roi_boxes'][b]['boxes']
                roi_labels_add[b, :num_obj] = batch_dict['add_roi_boxes'][b]['labels']
                roi_scores_add[b, :num_obj] = 0.8
            batch_dict['rois'] = torch.cat([batch_dict['rois'], rois_add], dim=1)
            batch_dict['roi_labels'] = torch.cat([batch_dict['roi_labels'], roi_labels_add], dim=1)
            batch_dict['roi_scores'] = torch.cat([batch_dict['roi_scores'], roi_scores_add], dim=1)
            
        if self.training:
            targets_dict = self.assign_targets(batch_dict)
            batch_dict['rois'] = targets_dict['rois']
            batch_dict['roi_labels'] = targets_dict['roi_labels']
            
        pooled_features = self.roi_grid_pool(batch_dict)  # (BxN, 6x6x6, C)

        
        # Box Refinement
        pooled_features = pooled_features.view(pooled_features.size(0), -1)
        shared_features = self.shared_fc_layer(pooled_features)
        
        # cls prototype
        if self.model_cfg.get('CLS_PROTOTYPES_ATTEN', None):
            atten_cls = self.model_cfg.CLS_PROTOTYPES_ATTEN.get('Attension_cls', False)
            if atten_cls:
                atten_cls_ids = [self.class_names.index(name)+1 for name in atten_cls]
                proto_atten = torch.cat([batch_dict['cls_prototypes'][id_] for id_ in batch_dict['cls_prototypes'] if id_ in atten_cls_ids ], dim=0)
                if self.model_cfg.get('LEARNABLE_CLS_PROTOTYPES_ATTEN', None):
                    learnable_proto_atten = self.cls_embedding.weight #.clone()
                    assert learnable_proto_atten.shape[0] ==  proto_atten.shape[0]
                    proto_atten = self.cls_embedding_reweight_layer(proto_atten.unsqueeze(0), 
                        learnable_proto_atten.unsqueeze(0)
                    ).squeeze(0)
                # print(proto_atten.shape)
                roi_attend = torch.zeros(shared_features.shape[0], dtype=bool).cuda()
                for id_ in atten_cls_ids:
                    # print(batch_dict['roi_labels'].view(-1), id_)
                    roi_attend |= (batch_dict['roi_labels'].view(-1) == id_)
                # print('roi_attend num', roi_attend.sum())
                if roi_attend.sum()>1:
                    # print(shared_features[:3,:3])
                    temp_feat = shared_features.clone()
                    temp_feat[roi_attend] = self.cls_prototypes_reweight_layer(
                        temp_feat[roi_attend].unsqueeze(0), 
                        proto_atten.unsqueeze(0),
                        no_skip = self.no_skip_atten
                    ).squeeze(0)
                    shared_features= temp_feat.clone()
            else:
                # for id_ in batch_dict['cls_prototypes']:
                #     print(id_, batch_dict['cls_prototypes'][id_].shape)
                proto_atten = torch.cat([batch_dict['cls_prototypes'][id_] for id_ in batch_dict['cls_prototypes']], dim=0)
                if self.model_cfg.get('LEARNABLE_CLS_PROTOTYPES_ATTEN', None):
                    learnable_proto_atten = self.cls_embedding.weight #.clone()
                    # assert learnable_proto_atten.shape[0] ==  proto_atten.shape[0]
                    proto_atten = self.cls_embedding_reweight_layer(proto_atten.unsqueeze(0), 
                        learnable_proto_atten.unsqueeze(0)
                    ).squeeze(0)
                shared_features = self.cls_prototypes_reweight_layer(shared_features.unsqueeze(0), proto_atten.unsqueeze(0), no_skip = self.no_skip_atten).squeeze(0)
        # Learnable cls embedding
        elif self.model_cfg.get('LEARNABLE_CLS_PROTOTYPES_ATTEN', None):
            atten_cls = self.model_cfg.LEARNABLE_CLS_PROTOTYPES_ATTEN.get('Attension_cls', False)
            if atten_cls:
                atten_cls_ids = [self.class_names.index(name)+1 for name in atten_cls]
                proto_atten = self.cls_embedding.weight#.clone()
                roi_attend = torch.zeros(shared_features.shape[0], dtype=bool).cuda()
                for id_ in atten_cls_ids:
                    roi_attend |= (batch_dict['roi_labels'].view(-1) == id_)
                if roi_attend.sum()>1:
                    # print(roi_attend.sum())
                    temp_feat = shared_features.clone()
                    temp_feat[roi_attend] = self.cls_embedding_reweight_layer(
                        temp_feat[roi_attend].unsqueeze(0), 
                        proto_atten.unsqueeze(0)
                    ).squeeze(0)
                    shared_features= temp_feat.clone()
            else:
                proto_atten = self.cls_embedding.weight#.clone()
                shared_features = self.cls_embedding_reweight_layer(shared_features.unsqueeze(0), proto_atten.unsqueeze(0)).squeeze(0)
                
        # print(shared_features)
        rcnn_cls = self.cls_pred_layer(self.cls_fc_layers(shared_features))
        rcnn_reg = self.reg_pred_layer(self.reg_fc_layers(shared_features))
        
        
        # grid_size = self.model_cfg.ROI_GRID_POOL.GRID_SIZE
        # batch_size_rcnn = pooled_features.shape[0]
        # pooled_features = pooled_features.permute(0, 2, 1).\
        #     contiguous().view(batch_size_rcnn, -1, grid_size, grid_size, grid_size)  # (BxN, C, 6, 6, 6)

        # shared_features = self.shared_fc_layer(pooled_features.view(batch_size_rcnn, -1, 1))
        # rcnn_cls = self.cls_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, 1 or 2)
        # rcnn_reg = self.reg_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, C)
        if not self.training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=batch_dict['batch_size'], rois=batch_dict['rois'], cls_preds=rcnn_cls, box_preds=rcnn_reg
            )
            batch_dict['batch_cls_preds'] = batch_cls_preds
            batch_dict['batch_box_preds'] = batch_box_preds
            batch_dict['cls_preds_normalized'] = False
        else:
            targets_dict['rcnn_cls'] = rcnn_cls
            targets_dict['rcnn_reg'] = rcnn_reg

            self.forward_ret_dict = targets_dict

        return batch_dict


