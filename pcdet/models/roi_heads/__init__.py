from .roi_head_template import RoIHeadTemplate
from .partA2_head import PartA2FCHead
from .pointrcnn_head import PointRCNNHead
from .pvrcnn_head import PVRCNNHead
from .pvrcnn_head_MoE import PVRCNNHeadMoE
from .pvrcnn_head import ActivePVRCNNHead
from .second_head import SECONDHead
from .second_head import ActiveSECONDHead
from .voxelrcnn_head import VoxelRCNNHead
from .voxelrcnn_head import ActiveVoxelRCNNHead
from .voxelrcnn_head import VoxelRCNNHead_ABL
from .voxelrcnn_head_cls_prototype import VoxelRCNNHead_CLS_PROTOTYPE
from .pvrcnn_head_semi import PVRCNNHeadSemi
from .pvrcnn_head_cls_prototype import PVRCNNHead_CLS_PROTOTYPE
from .pvrcnnpp_head_cls_prototype import PVRCNNPPHead_CLS_PROTOTYPE

__all__ = {
    'RoIHeadTemplate': RoIHeadTemplate,
    'PartA2FCHead': PartA2FCHead,
    'PointRCNNHead': PointRCNNHead,
    'PVRCNNHead': PVRCNNHead,
    'PVRCNNHeadMoE': PVRCNNHeadMoE,
    'ActivePVRCNNHead': ActivePVRCNNHead,
    'SECONDHead': SECONDHead,
    'ActiveSECONDHead': ActiveSECONDHead,
    'VoxelRCNNHead': VoxelRCNNHead,
    'ActiveVoxelRCNNHead': ActiveVoxelRCNNHead,
    'VoxelRCNNHead_ABL': VoxelRCNNHead_ABL,
    'VoxelRCNNHead_CLS_PROTOTYPE': VoxelRCNNHead_CLS_PROTOTYPE,
    'PVRCNNHeadSemi':PVRCNNHeadSemi,
    'PVRCNNHead_CLS_PROTOTYPE': PVRCNNHead_CLS_PROTOTYPE,
    'PVRCNNPPHead_CLS_PROTOTYPE': PVRCNNPPHead_CLS_PROTOTYPE,
}
