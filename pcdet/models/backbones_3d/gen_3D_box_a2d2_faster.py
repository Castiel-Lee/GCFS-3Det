import numpy as np
from scipy.optimize import minimize
from pcdet.utils import box_utils, common_utils
import torch


# nusc_com3_kitti_nov4 = {1: [3.95169492, 1.61813559, 1.5379661 ], # Car
# 	2: [0.7208, 0.6348, 1.732],                # Pedestrian
# 	3: [8.255,      2.53333333, 3.35333333],   # Truck
# 	11: [5.348, 1.946, 2.326],                 # Van
# 	12: [1.008, 0.59,  1.264 ],                # Person_sitting
# 	13: [1.964, 0.588, 1.768],                 # Cyclist
# 	14: [14.202,  2.292,  3.568],              # Tram
# }

# 'Car', 'Pedestrian', 'Truck',  'Van', 'Person_sitting', 'Cyclist', 'Tram', 'Bicycle', 'UtilityVehicle', 'Bus'
kitti_com3_a2d2_nov3 = {
    1: [3.99, 1.99, 1.72 ],      # Car
	2: [0.72,  0.66, 1.70],      # Pedestrian
	3: [8.76,  3.14, 3.80],      # Truck
	8: [1.60,  0.73, 1.54],      # Bicycle
	9: [9.92, 2.94, 3.89],      # UtilityVehicle
	10: [9.24, 3.47, 3.39],     # Bus
}

# init_loc_nusc_com3_kitti_nov4 = {
#     1: 'origin', # Car
# 	2: 'obj_center',                # Pedestrian
# 	3: 'origin',   # Truck
# 	11: 'origin',                 # Van
# 	12: 'obj_center',                # Person_sitting
# 	13: 'obj_center',                 # Cyclist
# 	14: 'origin',              # Tram
# }

init_loc_kitti_com3_a2d2_nov3 = {
    1: 'origin',                      # Car
	2: 'obj_center',                     # Pedestrian
	3: 'origin',                           # Truck
	8: 'obj_center',                     # Bicycle
	9: 'obj_center',                      # UtilityVehicle
	10: 'origin',                      # Bus
}

# loss_cal_nusc_com3_kitti_nov4 = {
#     1: 'add', # Car
# 	2: 'add',                # Pedestrian
# 	3: 'add',               # Truck
# 	11: 'add',                 # Van
# 	12: 'add',                # Person_sitting
# 	13: 'add',                 # Cyclist
# 	14: 'origin',              # Tram
# }

loss_cal_kitti_com3_a2d2_nov3 = {
    1: 'add',                      # Car
	2: 'add',                     # Pedestrian
	3: 'add',                           # Truck
	8: 'add',                     # Bicycle
	9: 'add',                      # UtilityVehicle
	10: 'add',                      # Bus
}

size_mean_cls = {
    # 'nusc_com3_kitti_nov4': nusc_com3_kitti_nov4,
    'kitti_com3_a2d2_nov3': kitti_com3_a2d2_nov3,
}

init_loc = {
    # 'nusc_com3_kitti_nov4': init_loc_nusc_com3_kitti_nov4,
    'kitti_com3_a2d2_nov3': init_loc_kitti_com3_a2d2_nov3,
}

loss_cal = {
    # 'nusc_com3_kitti_nov4': loss_cal_nusc_com3_kitti_nov4,
    'kitti_com3_a2d2_nov3': loss_cal_kitti_com3_a2d2_nov3,
}

def create_rotation_matrix(theta):
    """创建2D旋转矩阵（优化计算效率）"""
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    
    # 使用float32提高计算效率
    return np.array([
        [cos_theta, -sin_theta, 0],
        [sin_theta, cos_theta, 0],
        [0, 0, 1]
    ], dtype=np.float32)

# 辅助函数 - 预处理点云，过滤异常点
def filter_outliers(points, threshold=2.0):
    """过滤异常点，提高拟合质量和速度"""
    if points.shape[0] <= 5:  # 点数太少不处理
        return points
    
    # 转换为float32提高效率
    if points.dtype != np.float32:
        points = points.astype(np.float32)
    
    # 仅对xy平面进行过滤
    point_xy_mean = np.mean(points[:, :2], 0)
    point_xy_std = np.std(points[:, :2], 0)
    
    # 避免除零
    point_xy_std = np.maximum(point_xy_std, 1e-6)
    
    # 创建有效点掩码
    valid_mask = np.all(np.abs(points[:, :2] - point_xy_mean) < threshold * point_xy_std, axis=1)
    
    return points[valid_mask]

# 辅助函数 - 估计初始姿态
def estimate_initial_pose(points, label, box_fit_config):
    """优化初始姿态估计，减少迭代次数"""
    # 中心位置估计
    center = np.mean(points, axis=0)
    theta = 0.01
    
    # 检查是否需要进行PCA方向估计
    if points.shape[0] > 5:
        # 计算主方向（PCA）
        points_xy = points[:, :2]
        points_centered = points_xy - np.mean(points_xy, axis=0)
        
        # 计算协方差矩阵
        cov = np.cov(points_centered.T)
        
        try:
            # 计算特征值和特征向量
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            
            # 使用最大特征值对应的特征向量作为主方向
            main_direction = eigenvectors[:, np.argmax(eigenvalues)]
            theta = np.arctan2(main_direction[1], main_direction[0])
        except:
            # 如果计算失败，使用默认值
            theta = 0.01
    else:
        # 点太少，使用默认角度
        theta = 0.01
    
    return center, theta

def generate_box_from_points(obj_points_dicts, scale_info=None, box_fit_config=None):
    '''
    Args:    
        obj_points_dicts: list of dicts (batch_size)
            'points_objs': list (num_obj)
                points: numpy.array (N,3)
            'labels': list (num_obj)
                label_objs: int or float
    Return:
        box_dicts: list of dicts (batch_size)
            'boxes': tensor (num_obj, 7)
            'labels': tensor (num_obj)
    '''
    batch_size = obj_points_dicts.__len__()
    box_dicts = []

    for b in range(batch_size):
        points_objs = obj_points_dicts[b]['points_objs']
        labels_objs = obj_points_dicts[b]['label_objs']
        num_obj = len(points_objs)

        box_dict = {
            'boxes': torch.zeros((num_obj, 7), dtype=torch.float32),  # 使用float32提高效率
            'labels': torch.zeros(num_obj, dtype=torch.float32),
        }
        scale = scale_info[b].item() if scale_info is not None else 1

        for i_obj in range(num_obj):
            points = points_objs[i_obj]
            
            if points.shape[0] == 0:
                continue

            label = labels_objs[i_obj]
            
            # 确保points是float32类型，提高计算效率
            if points.dtype != np.float32:
                points = points.astype(np.float32)
            
            # 获取预定义尺寸并应用缩放
            size_ = [item * scale for item in size_mean_cls[box_fit_config][int(label)]]
            box_dx, box_dy, box_dz = size_[0], size_[1], size_[2]
            
            # 预处理点云，过滤异常点
            points = filter_outliers(points)
            
            # 如果过滤后点太少，简单估计并跳过优化
            if points.shape[0] <= 3:
                center = np.mean(points, axis=0) if points.shape[0] > 0 else np.zeros(3, dtype=np.float32)
                box_numpy = np.array([center[0], center[1], center[2], box_dx, box_dy, box_dz, 0.0], dtype=np.float32)
                box_dict['boxes'][i_obj, :] = torch.from_numpy(box_numpy)
                box_dict['labels'][i_obj] = label
                continue
            
            # 优化目标函数
            def objective(params):
                translation = params[:3]
                theta = params[3]
                
                # 使用缓存的旋转矩阵
                rotation_matrix = create_rotation_matrix(theta)
                
                # 计算视角方向
                center_points_xy = np.mean(points[:, :2], 0)
                alpha = np.arctan2(center_points_xy[1], center_points_xy[0])
                view_dir = (alpha - theta) % (2*np.pi)
                
                # 确定符号
                if 0 <= view_dir <= np.pi/2:
                    sign_x, sign_y = -1, -1
                elif np.pi/2 <= view_dir <= np.pi:
                    sign_x, sign_y = 1, -1
                elif np.pi <= view_dir <= np.pi/2*3:
                    sign_x, sign_y = 1, 1
                else:
                    sign_x, sign_y = -1, 1
                
                # 应用旋转和平移（向量化）
                translated_points = points - translation
                rotated_points = np.dot(translated_points, rotation_matrix)
                
                # 定义边界框
                half_size = np.array([box_dx, box_dy, box_dz], dtype=np.float32) / 2
                box_min = -half_size
                box_max = half_size
                
                # 过滤异常点 - 只计算一次，提高效率
                point_xy_mean = np.mean(rotated_points[:, :2], 0)
                point_xy_std = np.std(rotated_points[:, :2], 0)
                valid_mask = np.all(np.abs(rotated_points[:, :2] - point_xy_mean) < 2.0 * point_xy_std, axis=1)
                filtered_points = rotated_points[valid_mask]
                
                # 避免过滤导致点太少
                if filtered_points.shape[0] < 5 and rotated_points.shape[0] > 0:
                    filtered_points = rotated_points
                
                # 计算点到边界框的距离（向量化计算）
                outside_distances = np.maximum(filtered_points - box_max, 0) + np.maximum(box_min - filtered_points, 0)
                total_distance = np.sum(outside_distances)
                
                # 根据损失配置应用额外惩罚
                if loss_cal[box_fit_config][label] == 'add':
                    if init_loc[box_fit_config][label] == 'obj_center':
                        # 已经过滤过点，无需重复过滤
                        center_distance = np.sum(np.abs(filtered_points[:, :2]))
                        total_distance += 0.5 * center_distance
                        
                    elif init_loc[box_fit_config][label] == 'origin':
                        # 已经过滤过点，无需重复过滤
                        target_boundaries = np.array([sign_x*box_dx/2, sign_y*box_dy/2], dtype=np.float32)
                        close_boundary = np.sum(np.mean(np.abs(filtered_points[:, :2]-target_boundaries), axis=1))
                        total_distance += 0.1 * close_boundary
                        
                    else:
                        raise NotImplementedError('Not Implemented initial loc: %s' % init_loc[box_fit_config][label])
                
                return total_distance
            
            # 计算初始猜测
            center, initial_theta = estimate_initial_pose(points, label, box_fit_config)
            initial_guess = np.array([center[0], center[1], center[2], initial_theta], dtype=np.float32)
            
            # 动态调整优化参数
            try:
                # 根据点云密度动态调整迭代次数
                max_iter = min(40, max(10, int(points.shape[0] / 10)))
                
                # 优化
                result = minimize(objective, 
                                initial_guess, 
                                method='L-BFGS-B',
                                options={'maxiter': max_iter, 'ftol': 1e-3, 'gtol': 1e-3, 'disp': False},
                )
                
                # 提取优化参数
                optimized_translation = result.x[:3]
                optimized_theta = result.x[3]
            except:
                # 优化失败时使用初始猜测
                optimized_translation = initial_guess[:3]
                optimized_theta = initial_guess[3]
            
            # 创建边界框参数
            box_numpy = np.array([
                optimized_translation[0], 
                optimized_translation[1], 
                optimized_translation[2], 
                box_dx, box_dy, box_dz, 
                optimized_theta
            ], dtype=np.float32)
            
            box_dict['boxes'][i_obj, :] = torch.from_numpy(box_numpy)
            box_dict['labels'][i_obj] = label

        box_dicts.append(box_dict)

    return box_dicts