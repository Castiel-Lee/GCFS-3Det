import numpy as np
from scipy.optimize import minimize
from pcdet.utils import box_utils, common_utils
import torch


kitti_com3_argo2_nov7 = {
    1: [4.70, 1.95, 1.75 ],      # Car
    2: [0.69, 0.73, 1.78],      # Pedestrian  
    3: [9.01, 2.82, 3.40],      # Truck
    8: [1.70, 0.64, 1.29],      # bicycle
    9: [11.75, 2.98, 3.33],                 # bus
    10: [1.97, 0.71, 1.32],                 # motorcycle
    11: [0.38, 0.38, 0.94],              # traffic_cone
    12: [6.69, 2.62, 3.06],      # large_vehicle
    13: [0.70, 0.70, 1.10],     # construction_barrel
    14: [0.41, 1.10, 3.13],                 # sign
}



init_loc_kitti_com3_argo2_nov7 = {
    1: 'origin',      # Car
    2: 'obj_center',      # Pedestrian  
    3: 'origin',      # Truck
    8: 'obj_center',      # bicycle
    9: 'origin',                 # bus
    10: 'obj_center',                 # motorcycle
    11: 'obj_center',              # traffic_cone
    12: 'origin',      # large_vehicle
    13: 'origin',     # construction_barrel
    14: 'origin',                 # sign
}

loss_cal_kitti_com3_argo2_nov7 = {
    1: 'add',                # Car
    2: 'add',                # Pedestrian
    3: 'add',                # Truck
    8: 'add',                # bicycle
    9: 'add',                # 
    10: 'add',               # 
    11: 'add',               # 
    12: 'add',               # 
    13: 'add',               #
    14: 'add',               # 
}


size_mean_cls = {
    # 'kitti_com3_nusc_nov7': kitti_com3_nusc_nov7,
    # 'waymo_com3_nusc_nov4': waymo_com3_nusc_nov4,
    'kitti_com3_argo2_nov7': kitti_com3_argo2_nov7,
}

init_loc = {
    # 'kitti_com3_nusc_nov7': init_loc_kitti_com3_nusc_nov7,
    # 'waymo_com3_nusc_nov4': init_loc_waymo_com3_nusc_nov4,
    'kitti_com3_argo2_nov7': init_loc_kitti_com3_argo2_nov7,
}

loss_cal = {
    # 'kitti_com3_nusc_nov7': loss_cal_kitti_com3_nusc_nov7,
    # 'waymo_com3_nusc_nov4': loss_cal_waymo_com3_nusc_nov4,
    'kitti_com3_argo2_nov7': loss_cal_kitti_com3_argo2_nov7,
}

def create_rotation_matrix(theta):
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    
    return np.array([
        [cos_theta, -sin_theta, 0],
        [sin_theta, cos_theta, 0],
        [0, 0, 1]
    ], dtype=np.float32)

def filter_outliers(points, threshold=2.0):
    if points.shape[0] <= 5: 
        return points
    
    if points.dtype != np.float32:
        points = points.astype(np.float32)
    
    point_xy_mean = np.mean(points[:, :2], 0)
    point_xy_std = np.std(points[:, :2], 0)
    point_xy_std = np.maximum(point_xy_std, 1e-6)
    
    valid_mask = np.all(np.abs(points[:, :2] - point_xy_mean) < threshold * point_xy_std, axis=1)
    
    return points[valid_mask]

def estimate_initial_pose(points, label, box_fit_config):
    center = np.mean(points, axis=0)
    theta = 0.01
    
    if points.shape[0] > 5:
        points_xy = points[:, :2]
        points_centered = points_xy - np.mean(points_xy, axis=0)
        
        cov = np.cov(points_centered.T)
        
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            
            main_direction = eigenvectors[:, np.argmax(eigenvalues)]
            theta = np.arctan2(main_direction[1], main_direction[0])
        except:
            theta = 0.01
    else:
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
            'boxes': torch.zeros((num_obj, 7), dtype=torch.float32), 
            'labels': torch.zeros(num_obj, dtype=torch.float32),
        }
        scale = scale_info[b].item() if scale_info is not None else 1

        for i_obj in range(num_obj):
            points = points_objs[i_obj]
            
            if points.shape[0] == 0:
                continue

            label = labels_objs[i_obj]

            if points.dtype != np.float32:
                points = points.astype(np.float32)

            size_ = [item * scale for item in size_mean_cls[box_fit_config][int(label)]]
            box_dx, box_dy, box_dz = size_[0], size_[1], size_[2]

            points = filter_outliers(points)

            if points.shape[0] <= 3:
                center = np.mean(points, axis=0) if points.shape[0] > 0 else np.zeros(3, dtype=np.float32)
                box_numpy = np.array([center[0], center[1], center[2], box_dx, box_dy, box_dz, 0.0], dtype=np.float32)
                box_dict['boxes'][i_obj, :] = torch.from_numpy(box_numpy)
                box_dict['labels'][i_obj] = label
                continue
            
            def objective(params):
                translation = params[:3]
                theta = params[3]
                
                rotation_matrix = create_rotation_matrix(theta)
                
                center_points_xy = np.mean(points[:, :2], 0)
                alpha = np.arctan2(center_points_xy[1], center_points_xy[0])
                view_dir = (alpha - theta) % (2*np.pi)

                if 0 <= view_dir <= np.pi/2:
                    sign_x, sign_y = -1, -1
                elif np.pi/2 <= view_dir <= np.pi:
                    sign_x, sign_y = 1, -1
                elif np.pi <= view_dir <= np.pi/2*3:
                    sign_x, sign_y = 1, 1
                else:
                    sign_x, sign_y = -1, 1
                
                translated_points = points - translation
                rotated_points = np.dot(translated_points, rotation_matrix)
                
                half_size = np.array([box_dx, box_dy, box_dz], dtype=np.float32) / 2
                box_min = -half_size
                box_max = half_size
                
                point_xy_mean = np.mean(rotated_points[:, :2], 0)
                point_xy_std = np.std(rotated_points[:, :2], 0)
                valid_mask = np.all(np.abs(rotated_points[:, :2] - point_xy_mean) < 2.0 * point_xy_std, axis=1)
                filtered_points = rotated_points[valid_mask]
                
                if filtered_points.shape[0] < 5 and rotated_points.shape[0] > 0:
                    filtered_points = rotated_points
                
                outside_distances = np.maximum(filtered_points - box_max, 0) + np.maximum(box_min - filtered_points, 0)
                total_distance = np.sum(outside_distances)
                
                if loss_cal[box_fit_config][label] == 'add':
                    if init_loc[box_fit_config][label] == 'obj_center':
                        center_distance = np.sum(np.abs(filtered_points[:, :2]))
                        total_distance += 0.5 * center_distance
                        
                    elif init_loc[box_fit_config][label] == 'origin':
                        target_boundaries = np.array([sign_x*box_dx/2, sign_y*box_dy/2], dtype=np.float32)
                        close_boundary = np.sum(np.mean(np.abs(filtered_points[:, :2]-target_boundaries), axis=1))
                        total_distance += 0.1 * close_boundary
                        
                    else:
                        raise NotImplementedError('Not Implemented initial loc: %s' % init_loc[box_fit_config][label])
                
                return total_distance
            
            center, initial_theta = estimate_initial_pose(points, label, box_fit_config)
            initial_guess = np.array([center[0], center[1], center[2], initial_theta], dtype=np.float32)
            
            try:
                max_iter = min(40, max(10, int(points.shape[0] / 10)))
                
                result = minimize(objective, 
                                initial_guess, 
                                method='L-BFGS-B',
                                options={'maxiter': max_iter, 'ftol': 1e-3, 'gtol': 1e-3, 'disp': False},
                )

                optimized_translation = result.x[:3]
                optimized_theta = result.x[3]
            except:
                optimized_translation = initial_guess[:3]
                optimized_theta = initial_guess[3]
            
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