from .detector3d_template import Detector3DTemplate
from .detector3d_template_multi_db import Detector3DTemplate_M_DB
from pcdet.utils import common_utils
import torch, random
import numpy as np

class PVRCNNPlusPlus_COMMON_NwayNovel_CLS_PROTOTYPE_2DATA_LOADED(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset, class_names):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset, class_names=class_names)
        self.module_list = self.build_networks()
        self.dataset_support_cfg = model_cfg.DATA_CONFIG_SUPPORT
        assert class_names == self.dataset_support_cfg.CLASS_NAMES
        self.K_shot = self.dataset_support_cfg.get('K_SHOTS', None)
        self.common_classes = self.dataset_support_cfg.COMMON_CLS

        self.cls_prototypes_for_test=None
        self.cls_feat_protos_for_test=None

        self.CL_domain_loss = model_cfg.PRE_SETTING.CL_FOR_DOMAIN_ALI
        self.use_support_anchor = model_cfg.PRE_SETTING.USE_SUPPORT_AS_ANCHOR
        self.cls_embedding_in_CL = model_cfg.PRE_SETTING.get('CLS_EMB_IN_CONTRAST', False)
        self.use_cls_embe_in_CL_only = model_cfg.PRE_SETTING.get('CLS_EMB_IN_CL_ONLY', False)
        self.use_all_sup_train = model_cfg.PRE_SETTING.get('ALL_SUP_DATA', False)
        self.use_all_cls_for_pro = model_cfg.PRE_SETTING.get('ALL_CLS_FOR_PRO', False)
        self.all_obj_as_pro = model_cfg.PRE_SETTING.get('ALL_OBJ_AS_PRO', False)
        self.use_siamese_refine = model_cfg.PRE_SETTING.get('USE_SIAMESE_REF', False)
        self.no_grad_cls_pro = model_cfg.PRE_SETTING.get('CLS_PRO_NO_GRAD', False)

        self.init_cls_prototype_before_training = model_cfg.PRE_SETTING.get('CLS_PRO_INIT_BEFORE_TRAIN', False)
        self.init_cls_prototype_flag = False

    
    def init_cls_prototypes(self):
        self.cls_prototypes_for_test=None
        self.cls_feat_protos_for_test=None
    
    def set_cls_prototypes_by_mean_features(self):
        cls_features, _ = self.get_cls_prototype_whole_dataset(no_grad=False, use_2d_gt=True)
        cls_features = torch.cat([cls_features[key] for key in cls_features], dim=0)
        self.roi_head.cls_embedding.weight.data.copy_(cls_features)
        print('Class prototypes are initialized with mean features of few-shot data')


    
    def add_additional_boxes_to_rois(self, batch_dict):     
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
            return batch_dict
        else:
            # print('no_roi_boxes')
            return batch_dict

    def forward(self, batch_dict, batch_dict_2=None):

        if self.training:

            if (not self.init_cls_prototype_flag) and (self.init_cls_prototype_before_training):
                self.set_cls_prototypes_by_mean_features()
                self.init_cls_prototype_flag = True

            assert batch_dict_2 is not None
            Nway_cons_cls = batch_dict['novel_classes'] if batch_dict.get('novel_classes', False) else None
            batch_dict['cls_prototypes'], support_features_dict = \
                self.get_cls_prototype(data_batch=batch_dict_2, Kshots_cons=True, Nway_cons_cls=Nway_cons_cls, no_grad=self.no_grad_cls_pro, use_2d_gt=True)

            batch_dict = self.vfe(batch_dict)
            if self.unetscn:
                batch_dict = self.unetscn(batch_dict)
            batch_dict = self.backbone_3d(batch_dict)
            if self.CL_domain_loss:
                if not self.use_cls_embe_in_CL_only:
                    raise NotImplementedError('Not implemented.')
                    # _, query_features_dict = self.get_cls_prototype(data_batch=batch_dict, Kshots_cons=False, use_2d_gt=True, output_add_roi=True)
                else:
                    query_features_dict = {}

            batch_dict = self.map_to_bev_module(batch_dict)
            batch_dict = self.backbone_2d(batch_dict)
            batch_dict = self.dense_head(batch_dict)

            batch_dict = self.roi_head.proposal_layer(
                batch_dict, nms_config=self.roi_head.model_cfg.NMS_CONFIG['TRAIN']
            )
            batch_dict = self.add_additional_boxes_to_rois(batch_dict)

            # roi assign targets for training only
            targets_dict = self.roi_head.assign_targets(batch_dict)
            batch_dict['rois'] = targets_dict['rois']
            batch_dict['roi_labels'] = targets_dict['roi_labels']
            batch_dict['roi_targets_dict'] = targets_dict
            num_rois_per_scene = targets_dict['rois'].shape[1]
            if 'roi_valid_num' in batch_dict:
                batch_dict['roi_valid_num'] = [num_rois_per_scene for _ in range(batch_dict['batch_size'])]

            batch_dict = self.pfe(batch_dict)
            batch_dict = self.point_head(batch_dict)
            batch_dict = self.roi_head(batch_dict)
            
            loss, tb_dict, disp_dict = self.get_training_loss()

            # ret_dict = {
            #     'loss': loss
            # }
            if self.CL_domain_loss: 
                if self.cls_embedding_in_CL:
                    cls_embedding_feats = self.roi_head.cls_embedding.weight #.clone()
                    common_cls_ids=[ self.class_names.index(cls_)+1 for cls_ in self.common_classes]  
                    if Nway_cons_cls is not None:
                        novel_labels = [self.class_names.index(cls_)+1 for cls_ in Nway_cons_cls]
                    else:
                        novel_labels = [self.class_names.index(item)+1 for item in self.dataset_support_cfg.NOVEL_CLS]
                        novel_labels.sort()
                    # print(novel_labels)
                    # print(Nway_cons_cls, novel_labels)
                    if not self.use_all_cls_for_pro:  
                        raise NotImplementedError('Not implemented.')                      
                        # assert cls_embedding_feats.shape[0] == len(novel_labels)           
                        # cls_embedding_labels = novel_labels
                    else:
                        assert cls_embedding_feats.shape[0] == len(common_cls_ids) + len(novel_labels), '%d, %d, %d' %(cls_embedding_feats.shape[0], len(common_cls_ids), len(novel_labels))
                        cls_embedding_labels = common_cls_ids + novel_labels

                    for i, label in enumerate(cls_embedding_labels):
                        # print(i, label)
                        cls_feat_embed = cls_embedding_feats[i].view(1, -1)
                        if not self.use_cls_embe_in_CL_only:
                            raise NotImplementedError('Not implemented.')
                            # query_features_dict[label] = torch.cat((query_features_dict[label], cls_feat_embed), dim=0)  \
                            #     if label in query_features_dict else cls_feat_embed
                        else:
                            query_features_dict[label] = cls_feat_embed
                    # if self.use_cls_embe_in_CL_only:
                    #     for key in support_features_dict:
                    #         support_features_dict[key] = support_features_dict[key].detach()
                CL_loss = self.get_CL_domian_loss(support_features_dict, query_features_dict, self.use_support_anchor)
            else:
                CL_loss = torch.zeros(1, dtype=loss.dtype).cuda().mean()

            ret_dict = {
                'loss': {
                    'det_loss': loss,
                    'CL_loss': CL_loss,
                }
            }
            return ret_dict, tb_dict, disp_dict
        else:
            if self.cls_prototypes_for_test == None:
                # use gt-2d info of few-shot dataset for generate prototypes
                self.cls_prototypes_for_test, obj_feats = self.get_cls_prototype_whole_dataset(use_2d_gt=True)
                if self.all_obj_as_pro:
                    raise NotImplementedError('Not implemented.')
                    # self.cls_prototypes_for_test = obj_feats
                print('Acquire class prototypes!')
            batch_dict['cls_prototypes'] = self.cls_prototypes_for_test

            batch_dict = self.vfe(batch_dict)
            if self.unetscn:
                batch_dict = self.unetscn(batch_dict)
            batch_dict = self.backbone_3d(batch_dict)
            batch_dict = self.map_to_bev_module(batch_dict)
            batch_dict = self.backbone_2d(batch_dict)
            batch_dict = self.dense_head(batch_dict)
            batch_dict = self.roi_head.proposal_layer(
                batch_dict, nms_config=self.roi_head.model_cfg.NMS_CONFIG['TEST']
            )
            batch_dict = self.add_additional_boxes_to_rois(batch_dict)

            batch_dict = self.pfe(batch_dict)
            batch_dict = self.point_head(batch_dict)
            batch_dict = self.roi_head(batch_dict)

            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}
        loss_rpn, tb_dict = self.dense_head.get_loss()
        if self.point_head is not None:
            loss_point, tb_dict = self.point_head.get_loss(tb_dict)
        else:
            loss_point = 0
        loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)

        loss = loss_rpn + loss_point + loss_rcnn
        return loss, tb_dict, disp_dict

    def get_CL_domian_loss(self, support_features_dict, query_features_dict, use_support_anchor):

        support_features, support_labels, query_features, query_labels = [], [], [], []

        for cls_id in support_features_dict:
            N_temp = support_features_dict[cls_id].shape[0]
            support_features.append(support_features_dict[cls_id])
            support_labels += [cls_id]*N_temp
        support_features = torch.cat(support_features, dim=0)
        support_labels = torch.tensor(support_labels, dtype=float)

        for cls_id in query_features_dict:
            N_temp = query_features_dict[cls_id].shape[0]
            query_features.append(query_features_dict[cls_id])
            query_labels += [cls_id]*N_temp
        query_features = torch.cat(query_features, dim=0)
        query_labels = torch.tensor(query_labels, dtype=float)

        return self.sup_con_loss(support_features, support_labels, query_features, query_labels, use_support_anchor)

    def process_batch_dict_with_gt(self, batch_dict, use_2d_gt):
        batch_dict = self.vfe(batch_dict)
        if self.unetscn:
            batch_dict = self.unetscn(batch_dict)
        batch_dict = self.backbone_3d(batch_dict, output_add_roi=False, use_2d_gt=use_2d_gt)
        batch_dict = self.map_to_bev_module(batch_dict)
        batch_dict = self.backbone_2d(batch_dict)

        batch_dict['rois'] = batch_dict['gt_boxes'][:,:,:7]
        batch_dict['roi_labels'] = batch_dict['gt_boxes'][:,:,7]
        # batch_dict['roi_scores'] = batch_dict['roi_labels'].
        batch_dict['has_class_labels'] = True

        batch_dict = self.pfe(batch_dict)
        batch_dict = self.point_head(batch_dict)
        return batch_dict

    def get_cls_prototype(self, data_batch, Kshots_cons, Nway_cons_cls=None, no_grad=False, use_2d_gt=False, output_add_roi=True): 
        assert self.training == True  
        support_class_names = self.dataset_support_cfg.CLASS_NAMES
        cls_feats = {(support_class_names.index(cls_)+1):[] for cls_ in support_class_names}
        cls_nums = {(support_class_names.index(cls_)+1):0 for cls_ in support_class_names}
        cls_protos = {}

        if no_grad:
            with torch.no_grad():
                data_batch = self.process_batch_dict_with_gt(batch_dict = data_batch, use_2d_gt=use_2d_gt)

                features_gt_box = self.roi_head.roi_grid_pool(data_batch, use_gt_as_rois=True) # (B x N_gt, 6x6x6, C)
                grid_size = self.roi_head.model_cfg.ROI_GRID_POOL.GRID_SIZE
                batch_size_rcnn = features_gt_box.shape[0]
                features_gt_box = features_gt_box.permute(0, 2, 1).\
                    contiguous().view(batch_size_rcnn, -1, grid_size, grid_size, grid_size)  # (BxN, C, 6, 6, 6)
                features_gt_box = self.roi_head.shared_fc_layer(features_gt_box.view(batch_size_rcnn, -1, 1)).squeeze(dim=2)

        else:
            data_batch = self.process_batch_dict_with_gt(batch_dict = data_batch, use_2d_gt=use_2d_gt)

            features_gt_box = self.roi_head.roi_grid_pool(data_batch, use_gt_as_rois=True) # (B x N_gt, 6x6x6, C)
            grid_size = self.roi_head.model_cfg.ROI_GRID_POOL.GRID_SIZE
            batch_size_rcnn = features_gt_box.shape[0]
            features_gt_box = features_gt_box.permute(0, 2, 1).\
                contiguous().view(batch_size_rcnn, -1, grid_size, grid_size, grid_size)  # (BxN, C, 6, 6, 6)
            features_gt_box = self.roi_head.shared_fc_layer(features_gt_box.view(batch_size_rcnn, -1, 1)).squeeze(dim=2)

        cls_gt_box = data_batch['gt_boxes'][:,:,-1].view(-1)

        for cls_id in cls_feats:
            flag = cls_gt_box == cls_id
            cls_feats[cls_id].append(features_gt_box[flag])
            cls_nums[cls_id] += flag.sum().item()
    
        # delete empty cls and possible not cls in N_way
        for cls_id in range(1, len(support_class_names)+1):
            if cls_nums[cls_id]==0:
                cls_feats.pop(cls_id)  
                cls_nums.pop(cls_id)
            elif Nway_cons_cls:
                name_cur = support_class_names[cls_id-1]
                if (name_cur not in self.common_classes) and (name_cur not in Nway_cons_cls):
                    cls_feats.pop(cls_id)  
                    cls_nums.pop(cls_id)
        for cls_id in cls_feats:
            cls_feats[cls_id] = torch.cat(cls_feats[cls_id], dim=0)
            # K_shots constrain
            if (support_class_names[cls_id-1] not in self.common_classes) and (Kshots_cons):
                # print(cls_id, cls_nums[cls_id])
                if self.K_shot is not None:
                    if cls_nums[cls_id] > self.K_shot:
                        selected_idx = random.sample(range(cls_nums[cls_id]), self.K_shot)
                        cls_feats[cls_id] = cls_feats[cls_id][selected_idx]
                        cls_nums[cls_id] = self.K_shot
                    elif cls_nums[cls_id] < self.K_shot:
                        # print(self.K_shot-cls_nums[cls_id])
                        selected_idx = np.random.choice(range(cls_nums[cls_id]), (self.K_shot-cls_nums[cls_id]), replace=True)
                        cls_feats[cls_id] = torch.cat([
                            cls_feats[cls_id], 
                            cls_feats[cls_id][selected_idx].view(self.K_shot-cls_nums[cls_id], -1)], dim=0)
                        cls_nums[cls_id] = self.K_shot
            # print(cls_id, cls_feats[cls_id].shape, cls_nums[cls_id]) 
            cls_protos[cls_id] = torch.mean(cls_feats[cls_id], dim=0, keepdim=True)

        cls_prototypes = cls_protos # torch.cat([cls_protos[cls_id] for cls_id in cls_protos], dim=0)
        return cls_prototypes, cls_feats


    def sup_con_loss(self, support_features, support_labels, query_features, query_labels, use_support_anchor, temperature=0.07):
        """
        Args:
            features:  [N_obj, C].   Note, concat(support_features + query_features)    
            labels: [N_obj]
        Returns:
            A loss scalar.
        """
            
        labels = torch.cat([support_labels, query_labels], dim=0).contiguous().view(-1, 1)
        contrast_feature = torch.cat([support_features, query_features], dim=0)
        contrast_feature = torch.nn.functional.normalize(contrast_feature, p=2, dim=1)
        contrast_count = contrast_feature.shape[0]

        mask = torch.eq(labels, labels.T).float().cuda()

        if use_support_anchor:
            anchor_count = support_labels.shape[0]
            mask = mask[:anchor_count, :]
            anchor_feature = contrast_feature[:anchor_count, :]
        else:
            anchor_count = contrast_feature.shape[0]
            anchor_feature = contrast_feature

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # mask-out self-contrast cases
        logits_mask = torch.ones_like(mask)
        logits_mask[:,:anchor_count] -= torch.eye(anchor_count).cuda()
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        valid_flag = mask.sum(1) > 0 
        mean_log_prob_pos = (mask * log_prob).sum(1)[valid_flag] / mask.sum(1)[valid_flag]
        # print((mask * log_prob).sum(1), mask.sum(1))

        # loss
        loss = - temperature * mean_log_prob_pos
        loss = loss.mean()

        return loss
    
    def get_support_dataloader(self, training=True):
        from pcdet.datasets import build_dataloader
        support_set, support_loader, sampler = build_dataloader(
            dataset_cfg=self.dataset_support_cfg,
            class_names=self.dataset_support_cfg.CLASS_NAMES,
            batch_size=1, dist=False,  training=training,
            workers=0,
        )
        return support_loader

    def get_cls_prototype_whole_dataset(self, no_grad=False, use_2d_gt=False, output_add_roi=True):  
        # assert self.training == False  
        support_class_names = self.dataset_support_cfg.CLASS_NAMES
        cls_feats = {(support_class_names.index(cls_)+1):[] for cls_ in support_class_names}
        cls_nums = {(support_class_names.index(cls_)+1):0 for cls_ in support_class_names}
        cls_protos = {}
        
        support_loader = self.get_support_dataloader(training=True)
        from pcdet.models import load_data_to_gpu
        
        for i, data_batch in enumerate(support_loader):
            
            load_data_to_gpu(data_batch)
            if no_grad:
                with torch.no_grad():
                    data_batch = self.process_batch_dict_with_gt(batch_dict = data_batch, use_2d_gt=use_2d_gt)

                    features_gt_box = self.roi_head.roi_grid_pool(data_batch, use_gt_as_rois=True) # (B x N_gt, 6x6x6, C)
                    grid_size = self.roi_head.model_cfg.ROI_GRID_POOL.GRID_SIZE
                    batch_size_rcnn = features_gt_box.shape[0]
                    features_gt_box = features_gt_box.permute(0, 2, 1).\
                        contiguous().view(batch_size_rcnn, -1, grid_size, grid_size, grid_size)  # (BxN, C, 6, 6, 6)
                    features_gt_box = self.roi_head.shared_fc_layer(features_gt_box.view(batch_size_rcnn, -1, 1)).squeeze(dim=2)

            else:
                data_batch = self.process_batch_dict_with_gt(batch_dict = data_batch, use_2d_gt=use_2d_gt)

                features_gt_box = self.roi_head.roi_grid_pool(data_batch, use_gt_as_rois=True) # (B x N_gt, 6x6x6, C)
                grid_size = self.roi_head.model_cfg.ROI_GRID_POOL.GRID_SIZE
                batch_size_rcnn = features_gt_box.shape[0]
                features_gt_box = features_gt_box.permute(0, 2, 1).\
                    contiguous().view(batch_size_rcnn, -1, grid_size, grid_size, grid_size)  # (BxN, C, 6, 6, 6)
                features_gt_box = self.roi_head.shared_fc_layer(features_gt_box.view(batch_size_rcnn, -1, 1)).squeeze(dim=2)

            cls_gt_box = data_batch['gt_boxes'][:,:,-1].view(-1)

            for cls_id in cls_feats:
                flag = cls_gt_box == cls_id
                cls_feats[cls_id].append(features_gt_box[flag])
                cls_nums[cls_id] += flag.sum().item()
        
        # delete empty cls
        for cls_id in range(1, len(support_class_names)+1):
            if cls_nums[cls_id]==0:
                cls_feats.pop(cls_id)  
                cls_nums.pop(cls_id)
        
        for cls_id in cls_feats:
            cls_feats[cls_id] = torch.cat(cls_feats[cls_id], dim=0)      
            # print(cls_id, cls_feats[cls_id].shape, cls_nums[cls_id]) 
            cls_protos[cls_id] = torch.mean(cls_feats[cls_id], dim=0, keepdim=True)

        cls_prototypes = cls_protos #torch.cat([cls_protos[cls_id] for cls_id in cls_protos], dim=0)

        
        return cls_prototypes, cls_feats
    


class PVRCNNPlusPlus_COMMON_NwayNovel_CLS_PROTOTYPE_2DATA_LOADED_OverMetaClassLoading(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset, class_names):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset, class_names=class_names)
        self.module_list = self.build_networks()
        self.dataset_support_cfg = model_cfg.DATA_CONFIG_SUPPORT
        assert class_names == self.dataset_support_cfg.CLASS_NAMES
        self.K_shot = self.dataset_support_cfg.get('K_SHOTS', None)
        self.common_classes = self.dataset_support_cfg.COMMON_CLS

        self.cls_prototypes_for_test=None
        self.cls_feat_protos_for_test=None

        self.CL_domain_loss = model_cfg.PRE_SETTING.CL_FOR_DOMAIN_ALI
        self.use_support_anchor = model_cfg.PRE_SETTING.USE_SUPPORT_AS_ANCHOR
        self.cls_embedding_in_CL = model_cfg.PRE_SETTING.get('CLS_EMB_IN_CONTRAST', False)
        self.use_cls_embe_in_CL_only = model_cfg.PRE_SETTING.get('CLS_EMB_IN_CL_ONLY', False)
        self.use_all_sup_train = model_cfg.PRE_SETTING.get('ALL_SUP_DATA', False)
        self.use_all_cls_for_pro = model_cfg.PRE_SETTING.get('ALL_CLS_FOR_PRO', False)
        self.all_obj_as_pro = model_cfg.PRE_SETTING.get('ALL_OBJ_AS_PRO', False)
        self.use_siamese_refine = model_cfg.PRE_SETTING.get('USE_SIAMESE_REF', False)
        self.no_grad_cls_pro = model_cfg.PRE_SETTING.get('CLS_PRO_NO_GRAD', False)

        self.init_cls_prototype_before_training = model_cfg.PRE_SETTING.get('CLS_PRO_INIT_BEFORE_TRAIN', False)
        self.init_cls_prototype_flag = False

    
    def init_cls_prototypes(self):
        self.cls_prototypes_for_test=None
        self.cls_feat_protos_for_test=None
    
    def set_cls_prototypes_by_mean_features(self):
        cls_features, _ = self.get_cls_prototype_whole_dataset(no_grad=False, use_2d_gt=True)
        cls_features = torch.cat([cls_features[key] for key in cls_features], dim=0)
        # print(self.roi_head.cls_embedding.weight)
        # print(cls_features)
        self.roi_head.cls_embedding.weight.data.copy_(cls_features)
        # print(self.roi_head.cls_embedding.weight)
        print('Class prototypes are initialized with mean features of few-shot data')


    
    def add_additional_boxes_to_rois(self, batch_dict):     
        if batch_dict.get('add_roi_boxes', False):
            # raise NotImplementedError('ADDING BOXES PVRCNNPP, TO DO!')
            # print('add_roi_boxes')
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
            return batch_dict
        else:
            # print('no_roi_boxes')
            return batch_dict

    def forward(self, batch_dict, batch_dict_2=None):

        if self.training:

            if (not self.init_cls_prototype_flag) and (self.init_cls_prototype_before_training):
                self.set_cls_prototypes_by_mean_features()
                self.init_cls_prototype_flag = True

            assert batch_dict_2 is not None
            Nway_cons_cls = batch_dict['novel_classes'] if batch_dict.get('novel_classes', False) else None
            proto_selected_indexs = batch_dict['proto_selected_ids'] if batch_dict.get('proto_selected_ids',False) else None
            batch_dict['cls_prototypes'], support_features_dict = \
                self.get_cls_prototype(data_batch=batch_dict_2, Kshots_cons=True, Nway_cons_cls=Nway_cons_cls, no_grad=self.no_grad_cls_pro, use_2d_gt=True)
            

            batch_dict = self.vfe(batch_dict)
            if self.unetscn:
                batch_dict = self.unetscn(batch_dict)
            batch_dict = self.backbone_3d(batch_dict)
            if self.CL_domain_loss:
                if not self.use_cls_embe_in_CL_only:
                    raise NotImplementedError('Not implemented.')
                    # _, query_features_dict = self.get_cls_prototype(data_batch=batch_dict, Kshots_cons=False, use_2d_gt=True, output_add_roi=True)
                else:
                    query_features_dict = {}

            batch_dict = self.map_to_bev_module(batch_dict)
            batch_dict = self.backbone_2d(batch_dict)
            batch_dict = self.dense_head(batch_dict)

            batch_dict = self.roi_head.proposal_layer(
                batch_dict, nms_config=self.roi_head.model_cfg.NMS_CONFIG['TRAIN']
            )
            batch_dict = self.add_additional_boxes_to_rois(batch_dict)

            # roi assign targets for training only
            targets_dict = self.roi_head.assign_targets(batch_dict)
            batch_dict['rois'] = targets_dict['rois']
            batch_dict['roi_labels'] = targets_dict['roi_labels']
            batch_dict['roi_targets_dict'] = targets_dict
            num_rois_per_scene = targets_dict['rois'].shape[1]
            if 'roi_valid_num' in batch_dict:
                batch_dict['roi_valid_num'] = [num_rois_per_scene for _ in range(batch_dict['batch_size'])]

            batch_dict = self.pfe(batch_dict)
            batch_dict = self.point_head(batch_dict)
            batch_dict = self.roi_head(batch_dict)
            
            loss, tb_dict, disp_dict = self.get_training_loss()

            # ret_dict = {
            #     'loss': loss
            # }
            if self.CL_domain_loss: 
                if self.cls_embedding_in_CL:
                    cls_embedding_feats = self.roi_head.cls_embedding.weight #.clone()
                    common_cls_ids=[ self.class_names.index(cls_)+1 for cls_ in self.common_classes]  
                    if Nway_cons_cls is not None:
                        novel_labels = [self.class_names.index(cls_)+1 for cls_ in Nway_cons_cls]
                    else:
                        novel_labels = [self.class_names.index(item)+1 for item in self.dataset_support_cfg.NOVEL_CLS]
                        novel_labels.sort()
                    # print(novel_labels)
                    # print(Nway_cons_cls, novel_labels)
                    if not self.use_all_cls_for_pro:  
                        raise NotImplementedError('Not implemented.')                      
                        # assert cls_embedding_feats.shape[0] == len(novel_labels)           
                        # cls_embedding_labels = novel_labels
                    else:
                        cls_embedding_labels = common_cls_ids + novel_labels
                        if proto_selected_indexs is None:
                            assert cls_embedding_feats.shape[0] == len(common_cls_ids) + len(novel_labels), '%d, %d, %d' %(cls_embedding_feats.shape[0], len(common_cls_ids), len(novel_labels))
                            proto_selected_indexs = list(range(cls_embedding_feats.shape[0]))
                        else:
                            proto_selected_indexs = list(range(len(common_cls_ids))) + proto_selected_indexs

                    for i, label in enumerate(cls_embedding_labels):
                        # print(i, label)
                        cls_feat_embed = cls_embedding_feats[proto_selected_indexs[i]].view(1, -1)
                        if not self.use_cls_embe_in_CL_only:
                            raise NotImplementedError('Not implemented.')
                            # query_features_dict[label] = torch.cat((query_features_dict[label], cls_feat_embed), dim=0)  \
                            #     if label in query_features_dict else cls_feat_embed
                        else:
                            query_features_dict[label] = cls_feat_embed
                    # if self.use_cls_embe_in_CL_only:
                    #     for key in support_features_dict:
                    #         support_features_dict[key] = support_features_dict[key].detach()
                CL_loss = self.get_CL_domian_loss(support_features_dict, query_features_dict, self.use_support_anchor)
            else:
                CL_loss = torch.zeros(1, dtype=loss.dtype).cuda().mean()

            ret_dict = {
                'loss': {
                    'det_loss': loss,
                    'CL_loss': CL_loss,
                }
            }
            return ret_dict, tb_dict, disp_dict
        else:
            if self.cls_prototypes_for_test == None:
                # use gt-2d info of few-shot dataset for generate prototypes
                self.cls_prototypes_for_test, obj_feats = self.get_cls_prototype_whole_dataset(use_2d_gt=True)
                if self.all_obj_as_pro:
                    raise NotImplementedError('Not implemented.')
                    # self.cls_prototypes_for_test = obj_feats
                print('Acquire class prototypes!')
            batch_dict['cls_prototypes'] = self.cls_prototypes_for_test

            batch_dict = self.vfe(batch_dict)
            if self.unetscn:
                batch_dict = self.unetscn(batch_dict)
            batch_dict = self.backbone_3d(batch_dict)
            batch_dict = self.map_to_bev_module(batch_dict)
            batch_dict = self.backbone_2d(batch_dict)
            batch_dict = self.dense_head(batch_dict)
            batch_dict = self.roi_head.proposal_layer(
                batch_dict, nms_config=self.roi_head.model_cfg.NMS_CONFIG['TEST']
            )
            batch_dict = self.add_additional_boxes_to_rois(batch_dict)

            batch_dict = self.pfe(batch_dict)
            batch_dict = self.point_head(batch_dict)
            batch_dict = self.roi_head(batch_dict)

            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}
        loss_rpn, tb_dict = self.dense_head.get_loss()
        if self.point_head is not None:
            loss_point, tb_dict = self.point_head.get_loss(tb_dict)
        else:
            loss_point = 0
        loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)

        loss = loss_rpn + loss_point + loss_rcnn
        return loss, tb_dict, disp_dict

    def get_CL_domian_loss(self, support_features_dict, query_features_dict, use_support_anchor):

        support_features, support_labels, query_features, query_labels = [], [], [], []

        for cls_id in support_features_dict:
            N_temp = support_features_dict[cls_id].shape[0]
            support_features.append(support_features_dict[cls_id])
            support_labels += [cls_id]*N_temp
        support_features = torch.cat(support_features, dim=0)
        support_labels = torch.tensor(support_labels, dtype=float)

        for cls_id in query_features_dict:
            N_temp = query_features_dict[cls_id].shape[0]
            query_features.append(query_features_dict[cls_id])
            query_labels += [cls_id]*N_temp
        query_features = torch.cat(query_features, dim=0)
        query_labels = torch.tensor(query_labels, dtype=float)

        return self.sup_con_loss(support_features, support_labels, query_features, query_labels, use_support_anchor)

    def process_batch_dict_with_gt(self, batch_dict, use_2d_gt):
        batch_dict = self.vfe(batch_dict)
        if self.unetscn:
            batch_dict = self.unetscn(batch_dict)
        batch_dict = self.backbone_3d(batch_dict, output_add_roi=False, use_2d_gt=use_2d_gt)
        batch_dict = self.map_to_bev_module(batch_dict)
        batch_dict = self.backbone_2d(batch_dict)


        batch_dict['rois'] = batch_dict['gt_boxes'][:,:,:7]
        batch_dict['roi_labels'] = batch_dict['gt_boxes'][:,:,7]
        # batch_dict['roi_scores'] = batch_dict['roi_labels'].
        batch_dict['has_class_labels'] = True

        batch_dict = self.pfe(batch_dict)
        batch_dict = self.point_head(batch_dict)
        # batch_dict = self.roi_head(batch_dict)
        return batch_dict

    def get_cls_prototype(self, data_batch, Kshots_cons, Nway_cons_cls=None, no_grad=False, use_2d_gt=False, output_add_roi=True): 
        assert self.training == True  
        support_class_names = self.dataset_support_cfg.CLASS_NAMES
        cls_feats = {(support_class_names.index(cls_)+1):[] for cls_ in support_class_names}
        cls_nums = {(support_class_names.index(cls_)+1):0 for cls_ in support_class_names}
        cls_protos = {}

        if no_grad:
            with torch.no_grad():
                data_batch = self.process_batch_dict_with_gt(batch_dict = data_batch, use_2d_gt=use_2d_gt)

                features_gt_box = self.roi_head.roi_grid_pool(data_batch, use_gt_as_rois=True) # (B x N_gt, 6x6x6, C)
                grid_size = self.roi_head.model_cfg.ROI_GRID_POOL.GRID_SIZE
                batch_size_rcnn = features_gt_box.shape[0]
                features_gt_box = features_gt_box.permute(0, 2, 1).\
                    contiguous().view(batch_size_rcnn, -1, grid_size, grid_size, grid_size)  # (BxN, C, 6, 6, 6)
                features_gt_box = self.roi_head.shared_fc_layer(features_gt_box.view(batch_size_rcnn, -1, 1)).squeeze(dim=2)

        else:
            data_batch = self.process_batch_dict_with_gt(batch_dict = data_batch, use_2d_gt=use_2d_gt)

            features_gt_box = self.roi_head.roi_grid_pool(data_batch, use_gt_as_rois=True) # (B x N_gt, 6x6x6, C)
            grid_size = self.roi_head.model_cfg.ROI_GRID_POOL.GRID_SIZE
            batch_size_rcnn = features_gt_box.shape[0]
            features_gt_box = features_gt_box.permute(0, 2, 1).\
                contiguous().view(batch_size_rcnn, -1, grid_size, grid_size, grid_size)  # (BxN, C, 6, 6, 6)
            features_gt_box = self.roi_head.shared_fc_layer(features_gt_box.view(batch_size_rcnn, -1, 1)).squeeze(dim=2)

        cls_gt_box = data_batch['gt_boxes'][:,:,-1].view(-1)

        for cls_id in cls_feats:
            flag = cls_gt_box == cls_id
            cls_feats[cls_id].append(features_gt_box[flag])
            cls_nums[cls_id] += flag.sum().item()
    
        # delete empty cls and possible not cls in N_way
        for cls_id in range(1, len(support_class_names)+1):
            if cls_nums[cls_id]==0:
                cls_feats.pop(cls_id)  
                cls_nums.pop(cls_id)
            elif Nway_cons_cls:
                name_cur = support_class_names[cls_id-1]
                if (name_cur not in self.common_classes) and (name_cur not in Nway_cons_cls):
                    cls_feats.pop(cls_id)  
                    cls_nums.pop(cls_id)
        for cls_id in cls_feats:
            cls_feats[cls_id] = torch.cat(cls_feats[cls_id], dim=0)
            # K_shots constrain
            if (support_class_names[cls_id-1] not in self.common_classes) and (Kshots_cons):
                # print(cls_id, cls_nums[cls_id])
                if self.K_shot is not None:
                    if cls_nums[cls_id] > self.K_shot:
                        selected_idx = random.sample(range(cls_nums[cls_id]), self.K_shot)
                        cls_feats[cls_id] = cls_feats[cls_id][selected_idx]
                        cls_nums[cls_id] = self.K_shot
                    elif cls_nums[cls_id] < self.K_shot:
                        # print(self.K_shot-cls_nums[cls_id])
                        selected_idx = np.random.choice(range(cls_nums[cls_id]), (self.K_shot-cls_nums[cls_id]), replace=True)
                        cls_feats[cls_id] = torch.cat([
                            cls_feats[cls_id], 
                            cls_feats[cls_id][selected_idx].view(self.K_shot-cls_nums[cls_id], -1)], dim=0)
                        cls_nums[cls_id] = self.K_shot
            # print(cls_id, cls_feats[cls_id].shape, cls_nums[cls_id]) 
            cls_protos[cls_id] = torch.mean(cls_feats[cls_id], dim=0, keepdim=True)

        cls_prototypes = cls_protos # torch.cat([cls_protos[cls_id] for cls_id in cls_protos], dim=0)
        return cls_prototypes, cls_feats


    def sup_con_loss(self, support_features, support_labels, query_features, query_labels, use_support_anchor, temperature=0.07):
        """
        Args:
            features:  [N_obj, C].   Note, concat(support_features + query_features)    
            labels: [N_obj]
        Returns:
            A loss scalar.
        """
            
        labels = torch.cat([support_labels, query_labels], dim=0).contiguous().view(-1, 1)
        contrast_feature = torch.cat([support_features, query_features], dim=0)
        contrast_feature = torch.nn.functional.normalize(contrast_feature, p=2, dim=1)
        contrast_count = contrast_feature.shape[0]

        mask = torch.eq(labels, labels.T).float().cuda()

        if use_support_anchor:
            anchor_count = support_labels.shape[0]
            mask = mask[:anchor_count, :]
            anchor_feature = contrast_feature[:anchor_count, :]
        else:
            anchor_count = contrast_feature.shape[0]
            anchor_feature = contrast_feature

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # mask-out self-contrast cases
        logits_mask = torch.ones_like(mask)
        logits_mask[:,:anchor_count] -= torch.eye(anchor_count).cuda()
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        valid_flag = mask.sum(1) > 0 
        mean_log_prob_pos = (mask * log_prob).sum(1)[valid_flag] / mask.sum(1)[valid_flag]
        # print((mask * log_prob).sum(1), mask.sum(1))

        # loss
        loss = - temperature * mean_log_prob_pos
        loss = loss.mean()

        return loss
    
    def get_support_dataloader(self, training=True):
        from pcdet.datasets import build_dataloader
        support_set, support_loader, sampler = build_dataloader(
            dataset_cfg=self.dataset_support_cfg,
            class_names=self.dataset_support_cfg.CLASS_NAMES,
            batch_size=1, dist=False,  training=training,
            workers=0,
        )
        return support_loader

    def get_cls_prototype_whole_dataset(self, no_grad=False, use_2d_gt=False, output_add_roi=True):  
        # assert self.training == False  
        support_class_names = self.dataset_support_cfg.CLASS_NAMES
        cls_feats = {(support_class_names.index(cls_)+1):[] for cls_ in support_class_names}
        cls_nums = {(support_class_names.index(cls_)+1):0 for cls_ in support_class_names}
        cls_protos = {}
        
        support_loader = self.get_support_dataloader(training=True)
        from pcdet.models import load_data_to_gpu
        
        for i, data_batch in enumerate(support_loader):
            
            load_data_to_gpu(data_batch)
            if no_grad:
                with torch.no_grad():
                    data_batch = self.process_batch_dict_with_gt(batch_dict = data_batch, use_2d_gt=use_2d_gt)

                    features_gt_box = self.roi_head.roi_grid_pool(data_batch, use_gt_as_rois=True) # (B x N_gt, 6x6x6, C)
                    grid_size = self.roi_head.model_cfg.ROI_GRID_POOL.GRID_SIZE
                    batch_size_rcnn = features_gt_box.shape[0]
                    features_gt_box = features_gt_box.permute(0, 2, 1).\
                        contiguous().view(batch_size_rcnn, -1, grid_size, grid_size, grid_size)  # (BxN, C, 6, 6, 6)
                    features_gt_box = self.roi_head.shared_fc_layer(features_gt_box.view(batch_size_rcnn, -1, 1)).squeeze(dim=2)

            else:
                data_batch = self.process_batch_dict_with_gt(batch_dict = data_batch, use_2d_gt=use_2d_gt)

                features_gt_box = self.roi_head.roi_grid_pool(data_batch, use_gt_as_rois=True) # (B x N_gt, 6x6x6, C)
                grid_size = self.roi_head.model_cfg.ROI_GRID_POOL.GRID_SIZE
                batch_size_rcnn = features_gt_box.shape[0]
                features_gt_box = features_gt_box.permute(0, 2, 1).\
                    contiguous().view(batch_size_rcnn, -1, grid_size, grid_size, grid_size)  # (BxN, C, 6, 6, 6)
                features_gt_box = self.roi_head.shared_fc_layer(features_gt_box.view(batch_size_rcnn, -1, 1)).squeeze(dim=2)

            cls_gt_box = data_batch['gt_boxes'][:,:,-1].view(-1)

            for cls_id in cls_feats:
                flag = cls_gt_box == cls_id
                cls_feats[cls_id].append(features_gt_box[flag])
                cls_nums[cls_id] += flag.sum().item()
        
        # delete empty cls
        for cls_id in range(1, len(support_class_names)+1):
            if cls_nums[cls_id]==0:
                cls_feats.pop(cls_id)  
                cls_nums.pop(cls_id)
        
        for cls_id in cls_feats:
            cls_feats[cls_id] = torch.cat(cls_feats[cls_id], dim=0)      
            # print(cls_id, cls_feats[cls_id].shape, cls_nums[cls_id]) 
            cls_protos[cls_id] = torch.mean(cls_feats[cls_id], dim=0, keepdim=True)

        cls_prototypes = cls_protos #torch.cat([cls_protos[cls_id] for cls_id in cls_protos], dim=0)

        
        return cls_prototypes, cls_feats
    


