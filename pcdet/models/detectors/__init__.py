from .detector3d_template import Detector3DTemplate
from .detector3d_template_ada import ActiveDetector3DTemplate
from .detector3d_template_multi_db import Detector3DTemplate_M_DB
from .PartA2_net import PartA2Net
from .point_rcnn import PointRCNN
from .pointpillar import PointPillar
from .pv_rcnn import PVRCNN
from .pv_rcnn import PVRCNN_AE_TestTimeTraining
from .pv_rcnn import PVRCNN_M_DB
from .pv_rcnn import PVRCNN_M_DB_3
from .pv_rcnn import SemiPVRCNN
from .pv_rcnn import ActivePVRCNN_DUAL
from .pv_rcnn import PVRCNN_TQS
from .pv_rcnn import PVRCNN_CLUE
from .second_net import SECONDNet
from .second_net_iou import SECONDNetIoU
from .second_net_iou import ActiveSECONDNetIoU
from .caddn import CaDDN
from .voxel_rcnn import VoxelRCNN, VoxelRCNN_AE, VoxelRCNN_AE_2ChanDet, VoxelRCNN_AE_ChanOrg, VoxelRCNN_AE_TestTimeTraining
from .voxel_rcnn import VoxelRCNN_DG_2_Source_Domain, VoxelRCNN_DG_2_Source_Domain_FineTune, VoxelRCNN_DG_2_Source_Domain_Discriminator, VoxelRCNN_DG_2_Source_Domain_TestTime
from .voxel_rcnn import VoxelRCNN_DG_2_Source_Domain_Single_DSR_Encoder, VoxelRCNN_DG_2_Source_Domain_Single_DSR_Encoder_Orthognal_DIR_DSR, VoxelRCNN_DG_2_Source_Domain_Single_DSR_Encoder_Orthognal_DIR_DSR_ConLoss_DSR
from .voxel_rcnn import VoxelRCNN_DG_2_Source_Domain_DSRSubGen_DomClass_OrthognalDIRDSR
from .voxel_rcnn import VoxelRCNN_DG_2_Source_Domain_DSRSubGen_DSRDomClass_DIRAdvDomClass_OrthognalDIRDSR
from .voxel_rcnn import VoxelRCNN_adv
from .voxel_rcnn import VoxelRCNN_M_DB
from .voxel_rcnn import VoxelRCNN_M_DB_3
from .voxel_rcnn import ActiveDualVoxelRCNN
from .voxel_rcnn import VoxelRCNN_CLUE
from .voxel_rcnn import VoxelRCNN_TQS
from .centerpoint import CenterPoint
from .centerpoint import CenterPoint_M_DB
from .centerpoint import SemiCenterPoint
from .pv_rcnn_plusplus import PVRCNNPlusPlus
from .pv_rcnn_plusplus import PVRCNNPlusPlus_M_DB
from .pv_rcnn_plusplus import SemiPVRCNNPlusPlus
from .IASSD import IASSD
from .semi_second import SemiSECOND, SemiSECONDIoU
from .unsupervised_model.pvrcnn_plus_backbone import PVRCNN_PLUS_BACKBONE
# few-shot
from .voxel_rcnn_few_shot_clspro import VoxelRCNN_COMMON_NwayNovel_CLS_PROTOTYPE_2DATA_LOADED
from .voxel_rcnn_few_shot_clspro import VoxelRCNN_COMMON_NwayNovel_CLS_PROTOTYPE_2DATA_LOADED_OverMetaClassLoading

from .pv_rcnn_few_shot_2 import PVRCNN_COMMON_NwayNovel_CLS_PROTOTYPE_2DATA_LOADED
from .pv_rcnn_plusplus_fewshot import PVRCNNPlusPlus_COMMON_NwayNovel_CLS_PROTOTYPE_2DATA_LOADED
from .pv_rcnn_plusplus_fewshot import PVRCNNPlusPlus_COMMON_NwayNovel_CLS_PROTOTYPE_2DATA_LOADED_OverMetaClassLoading

__all__ = {
    'Detector3DTemplate': Detector3DTemplate,
    'Detector3DTemplate_ADA': ActiveDetector3DTemplate,
    'Detector3DTemplate_M_DB': Detector3DTemplate_M_DB,
    'SECONDNet': SECONDNet,
    'PartA2Net': PartA2Net,
    'PVRCNN': PVRCNN,
    'PVRCNN_AE_TestTimeTraining': PVRCNN_AE_TestTimeTraining,
    'PVRCNN_M_DB': PVRCNN_M_DB,
    'PVRCNN_M_DB_3': PVRCNN_M_DB_3,
    'SemiPVRCNN': SemiPVRCNN,
    'ActiveDualPVRCNN': ActivePVRCNN_DUAL,
    'PVRCNN_TQS': PVRCNN_TQS,
    'PVRCNN_CLUE': PVRCNN_CLUE,
    'PointPillar': PointPillar,
    'PointRCNN': PointRCNN,
    'SECONDNetIoU': SECONDNetIoU,
    'ActiveSECONDNetIoU': ActiveSECONDNetIoU,
    'CaDDN': CaDDN,
    'VoxelRCNN': VoxelRCNN,
    'VoxelRCNN_adv': VoxelRCNN_adv,
    'VoxelRCNN_AE': VoxelRCNN_AE,
    'VoxelRCNN_AE_2ChanDet': VoxelRCNN_AE_2ChanDet,
    'VoxelRCNN_AE_ChanOrg': VoxelRCNN_AE_ChanOrg,
    'VoxelRCNN_AE_TestTimeTraining': VoxelRCNN_AE_TestTimeTraining,
    'VoxelRCNN_DG_2_Source_Domain': VoxelRCNN_DG_2_Source_Domain,
    'VoxelRCNN_DG_2_Source_Domain_FineTune': VoxelRCNN_DG_2_Source_Domain_FineTune,
    'VoxelRCNN_DG_2_Source_Domain_Discriminator': VoxelRCNN_DG_2_Source_Domain_Discriminator,
    'VoxelRCNN_DG_2_Source_Domain_TestTime': VoxelRCNN_DG_2_Source_Domain_TestTime,
    'VoxelRCNN_DG_2_Source_Domain_Single_DSR_Encoder': VoxelRCNN_DG_2_Source_Domain_Single_DSR_Encoder,
    'VoxelRCNN_DG_2_Source_Domain_Single_DSR_Encoder_Orthognal_DIR_DSR': VoxelRCNN_DG_2_Source_Domain_Single_DSR_Encoder_Orthognal_DIR_DSR,
    'VoxelRCNN_DG_2_Source_Domain_Single_DSR_Encoder_Orthognal_DIR_DSR_ConLoss_DSR': VoxelRCNN_DG_2_Source_Domain_Single_DSR_Encoder_Orthognal_DIR_DSR_ConLoss_DSR,
    'VoxelRCNN_DG_2_Source_Domain_DSRSubGen_DomClass_OrthognalDIRDSR': VoxelRCNN_DG_2_Source_Domain_DSRSubGen_DomClass_OrthognalDIRDSR,
    'VoxelRCNN_DG_2_Source_Domain_DSRSubGen_DSRDomClass_DIRAdvDomClass_OrthognalDIRDSR': VoxelRCNN_DG_2_Source_Domain_DSRSubGen_DSRDomClass_DIRAdvDomClass_OrthognalDIRDSR,
    'VoxelRCNN_M_DB': VoxelRCNN_M_DB,
    'VoxelRCNN_M_DB_3': VoxelRCNN_M_DB_3,
    'ActiveDualVoxelRCNN': ActiveDualVoxelRCNN,
    'VoxelRCNN_CLUE': VoxelRCNN_CLUE,
    'VoxelRCNN_TQS': VoxelRCNN_TQS,
    'CenterPoint': CenterPoint,
    'CenterPoint_M_DB':CenterPoint_M_DB,
    'SemiCenterPoint': SemiCenterPoint,
    'PVRCNNPlusPlus': PVRCNNPlusPlus,
    'PVRCNNPlusPlus_M_DB': PVRCNNPlusPlus_M_DB,
    'SemiPVRCNNPlusPlus': SemiPVRCNNPlusPlus,
    'IASSD': IASSD,
    'ActiveDualPVRCNN': ActivePVRCNN_DUAL,
    'SemiSECOND': SemiSECOND,
    'SemiSECONDIoU': SemiSECONDIoU,
    'PVRCNN_PLUS_BACKBONE': PVRCNN_PLUS_BACKBONE,
    # few-shot
    'VoxelRCNN_COMMON_NwayNovel_CLS_PROTOTYPE_2DATA_LOADED': VoxelRCNN_COMMON_NwayNovel_CLS_PROTOTYPE_2DATA_LOADED,
    'VoxelRCNN_COMMON_NwayNovel_CLS_PROTOTYPE_2DATA_LOADED_OverMetaClassLoading': VoxelRCNN_COMMON_NwayNovel_CLS_PROTOTYPE_2DATA_LOADED_OverMetaClassLoading,

    'PVRCNN_COMMON_NwayNovel_CLS_PROTOTYPE_2DATA_LOADED': PVRCNN_COMMON_NwayNovel_CLS_PROTOTYPE_2DATA_LOADED,
    'PVRCNNPlusPlus_COMMON_NwayNovel_CLS_PROTOTYPE_2DATA_LOADED': PVRCNNPlusPlus_COMMON_NwayNovel_CLS_PROTOTYPE_2DATA_LOADED,
    'PVRCNNPlusPlus_COMMON_NwayNovel_CLS_PROTOTYPE_2DATA_LOADED_OverMetaClassLoading': PVRCNNPlusPlus_COMMON_NwayNovel_CLS_PROTOTYPE_2DATA_LOADED_OverMetaClassLoading,
}


def build_detector(model_cfg, num_class, dataset, class_names):
    model = __all__[model_cfg.NAME](
        model_cfg=model_cfg, num_class=num_class, dataset=dataset, class_names=class_names
    )

    return model

# def build_detector_multi_db_v2(model_cfg, num_class, dataset):
#     model = __all__[model_cfg.NAME](
#         model_cfg=model_cfg, num_class=num_class, dataset=dataset
#     )

#     return model

def build_detector_multi_db(model_cfg, num_class, num_class_s2, dataset, dataset_s2, source_one_name):
    model = __all__[model_cfg.NAME](
        model_cfg=model_cfg, num_class=num_class, num_class_s2=num_class_s2, dataset=dataset, 
        dataset_s2=dataset_s2, source_one_name=source_one_name
    )

    return model

def build_detector_multi_db_3(model_cfg, num_class, num_class_s2, num_class_s3, dataset, dataset_s2, dataset_s3, source_one_name, source_1):
    model = __all__[model_cfg.NAME](
        model_cfg=model_cfg, num_class=num_class, num_class_s2=num_class_s2, num_class_s3=num_class_s3, dataset=dataset, 
        dataset_s2=dataset_s2, dataset_s3=dataset_s3, source_one_name=source_one_name, source_1=source_1
    )

    return model