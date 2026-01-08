import torch
import cv2
import numpy as np

def remove_points_in_boxes_efficient(points, boxes):
    """
    高效地删除所有3D边界框内部的点（向量化实现）
    
    Args:
        points: 点云坐标，形状为(N, 3)的张量，N是点的数量
        boxes: 3D边界框参数，形状为(M, 7)的张量，每个边界框为[x, y, z, l, w, h, yaw]
        
    Returns:
        keep_points: 保留的点云坐标，形状为(K, 3)的张量，K是保留的点数量
        keep_indices: 保留点的索引，形状为(K,)的张量
    """
    num_points = points.shape[0]
    num_boxes = boxes.shape[0]
    device = points.device
    dtype = points.dtype
    
    mask = torch.ones(num_points, dtype=torch.bool, device=device)
    
    expanded_points = points.unsqueeze(1)  # (N, 1, 3)
    
    for i in range(num_boxes):
        cx, cy, cz, l, w, h, yaw = boxes[i]
        
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        rotation_matrix = torch.tensor([[cos_yaw, sin_yaw, 0],
                                       [-sin_yaw, cos_yaw, 0],
                                       [0, 0, 1]], dtype=dtype, device=device)
        
        centered_points = expanded_points - torch.tensor([cx, cy, cz], dtype=dtype, device=device)
        
        local_points = torch.matmul(centered_points.squeeze(1), rotation_matrix).unsqueeze(1)
        
        in_box_x = torch.abs(local_points[:, 0, 0]) <= l/2
        in_box_y = torch.abs(local_points[:, 0, 1]) <= w/2
        in_box_z = torch.abs(local_points[:, 0, 2]) <= h/2
        
        in_box = in_box_x & in_box_y & in_box_z
        
        mask = mask & (~in_box)
    
    keep_indices = torch.nonzero(mask, as_tuple=True)[0]
    keep_points = points[keep_indices]
    
    return keep_points, keep_indices

def project_3d_box_to_2d_torch(box3d, lidar2image, image_shape):
    device = box3d.device

    x, y, z, l, w, h, yaw = box3d
    
    corners = torch.tensor([
        [l/2, w/2, h/2],  
        [l/2, w/2, -h/2],  
        [l/2, -w/2, -h/2],  
        [l/2, -w/2, h/2],  
        [-l/2, w/2, h/2],  
        [-l/2, w/2, -h/2],  
        [-l/2, -w/2, -h/2],  
        [-l/2, -w/2, h/2]   
    ], dtype=box3d.dtype, device=device)
    
    rot_mat = torch.tensor([
        [torch.cos(yaw), -torch.sin(yaw), 0],
        [torch.sin(yaw), torch.cos(yaw), 0],
        [0, 0, 1]
    ], dtype=box3d.dtype, device=device)
    
    corners_rotated = torch.matmul(corners, rot_mat.transpose(0, 1))  
    corners_3d = corners_rotated + torch.tensor([x, y, z], dtype=box3d.dtype, device=device)

    ones = torch.ones((8, 1), dtype=box3d.dtype, device=device)
    corners_4d = torch.cat([corners_3d, ones], dim=1)

    try:
        corners_proj = torch.matmul(corners_4d, lidar2image.transpose(0, 1))
    except Exception as e:
        print(f"projection error: {e}")
        return torch.tensor([0, 0, 0, 0], dtype=box3d.dtype, device=device), torch.tensor(False, device=device)
    
    depths = corners_proj[:, 2]
    
    if torch.all(depths <= 0):
        return torch.tensor([0, 0, 0, 0], dtype=box3d.dtype, device=device), torch.tensor(False, device=device)
    
    epsilon = 1e-10
    safe_depths = torch.where(depths < epsilon, torch.tensor(epsilon, dtype=box3d.dtype, device=device), depths)
    
    corners_2d = corners_proj[:, :2] / safe_depths.reshape(-1, 1)
    
    valid_mask = torch.isfinite(corners_2d).all(dim=1) & (depths > 0)
    if not torch.any(valid_mask):
        return torch.tensor([0, 0, 0, 0], dtype=box3d.dtype, device=device), torch.tensor(False, device=device)
    
    valid_corners = corners_2d[valid_mask]
    
    x_min = torch.min(valid_corners[:, 0])
    y_min = torch.min(valid_corners[:, 1])
    x_max = torch.max(valid_corners[:, 0])
    y_max = torch.max(valid_corners[:, 1])
    
    x_min = torch.clamp(x_min, min=0, max=image_shape[1] - 1)
    y_min = torch.clamp(y_min, min=0, max=image_shape[0] - 1)
    x_max = torch.clamp(x_max, min=0, max=image_shape[1] - 1)
    y_max = torch.clamp(y_max, min=0, max=image_shape[0] - 1)
    
    if x_max <= x_min or y_max <= y_min:
        return torch.tensor([0, 0, 0, 0], dtype=box3d.dtype, device=device), torch.tensor(False, device=device)
    
    return torch.tensor([x_min, y_min, x_max, y_max], dtype=box3d.dtype, device=device), torch.tensor(True, device=device)

def project_3d_boxes_to_2d_torch(boxes3d, lidar2image, image_shape):
    device = boxes3d.device
    dtype = boxes3d.dtype
    num_boxes = boxes3d.shape[0]
    
    boxes2d = torch.zeros((num_boxes, 4), dtype=dtype, device=device)
    is_valid = torch.zeros(num_boxes, dtype=torch.bool, device=device)
    for i in range(num_boxes):
        box2d, valid = project_3d_box_to_2d_torch(boxes3d[i], lidar2image, image_shape)
        boxes2d[i] = box2d
        is_valid[i] = valid
    
    return boxes2d, is_valid


def project_lidar_to_image_torch(points_3d, lidar2image, image_shape):
    if not isinstance(points_3d, torch.Tensor):
        points_3d = torch.tensor(points_3d, dtype=torch.float32)
    
    if not isinstance(lidar2image, torch.Tensor):
        lidar2image = torch.tensor(lidar2image, dtype=torch.float32)
    
    device = points_3d.device
    lidar2image = lidar2image.to(device)
    
    n = points_3d.shape[0]
    ones = torch.ones((n, 1), device=device)
    points_4d = torch.cat([points_3d, ones], dim=1)
    
    points_proj = torch.matmul(points_4d, lidar2image.t())
    
    depths = points_proj[:, 2]
    points_2d = points_proj[:, :2] / depths.unsqueeze(1).clamp(min=1e-10)
    
    mask = (depths > 0)  
    mask = mask & (points_2d[:, 0] >= 0) & (points_2d[:, 0] < image_shape[1])
    mask = mask & (points_2d[:, 1] >= 0) & (points_2d[:, 1] < image_shape[0])
    
    return points_2d, mask



def shrink_mask(mask, shrink_ratio=0.8):
    mask = mask.astype(np.uint8)

    area = np.sum(mask)
    kernel_size = max(1, int(np.sqrt(area) * (1 - shrink_ratio)))
    
    if kernel_size % 2 == 0:
        kernel_size += 1

    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    shrunk_mask = cv2.erode(mask, kernel, iterations=1)
    
    return shrunk_mask

import torch
import torch.nn.functional as F

def shrink_mask_pure_torch(mask_tensor, shrink_ratio=0.8):
    """
    mask_tensor: (H, W) bool或0/1 Tensor
    """
    if mask_tensor.ndim == 2:
        mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

    h, w = mask_tensor.shape[-2:]
    area = torch.sum(mask_tensor).item()
    kernel_size = max(1, int((area**0.5) * (1 - shrink_ratio)))
    if kernel_size % 2 == 0:
        kernel_size += 1

    inverted = (~mask_tensor.bool()).float()  
    pooled = -F.max_pool2d(inverted, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
    shrunk_mask = (pooled > -1).squeeze(0).squeeze(0)

    return shrunk_mask

