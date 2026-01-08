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

        box_dict={
            'boxes': torch.zeros((num_obj, 7), dtype=float),
            'labels': torch.zeros(num_obj, dtype=float),
        }
        scale = scale_info[b].item() if scale_info is not None else 1

        for i_obj in range(num_obj):
            points = points_objs[i_obj]
            
            if points.shape[0] ==0:
                continue

            label = labels_objs[i_obj]
            size_ = [item * scale for item in size_mean_cls[box_fit_config][int(label)]]
            box_dx, box_dy, box_dz = size_[0], size_[1], size_[2]

            def objective(params):
                translation = params[:3]
                theta = params[3]
                
                # Rotation matrix for the given azimuth angle
                rotation_matrix = np.array([
                    [np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta), np.cos(theta), 0],
                    [0, 0, 1]
                ])

                center_points_xy = np.mean(points[:,:2],0)
                alpha = np.arctan2(center_points_xy[1], center_points_xy[0])
                view_dir = alpha - theta
                view_dir %= 2*np.pi

                if 0 <= view_dir <= np.pi/2:
                    sign_x, sign_y = -1 , -1
                elif np.pi/2 <= view_dir <= np.pi:
                    sign_x, sign_y = 1 , -1
                elif np.pi <= view_dir <= np.pi/2*3:
                    sign_x, sign_y = 1 , 1
                else:
                    sign_x, sign_y = -1 , 1

                # Apply rotation and translation to points
                translated_points = points - translation
                rotated_points = np.dot(translated_points, rotation_matrix)
            
                reversed_points = rotated_points


                
                # Define the box boundaries
                half_size = np.array([box_dx, box_dy, box_dz]) / 2
                box_min = -half_size
                box_max = half_size


                # Calculate distances for points outside the box
                outside_distances = np.maximum(reversed_points - box_max, 0) + np.maximum(box_min - reversed_points, 0)
                total_distance = np.sum(outside_distances)

                if loss_cal[box_fit_config][label] == 'add':
                    if init_loc[box_fit_config][label] == 'obj_center':
                        point_xy_mean = np.mean(reversed_points[:,:2], 0)
                        point_xy_stnd = np.std(reversed_points[:,:2], 0)
                        valid_flag = np.abs(reversed_points[:,:2] - point_xy_mean) < 2 * point_xy_stnd #### 
                        reversed_points = reversed_points[valid_flag[:,1]&valid_flag[:,0]]            
                        
                        center_distance = np.sum(np.abs(reversed_points[:,:2]))
                        total_distance += 0.5*center_distance
                    
                    elif init_loc[box_fit_config][label] == 'origin':
                        point_xy_mean = np.mean(reversed_points[:,:2], 0)
                        point_xy_stnd = np.std(reversed_points[:,:2], 0)
                        valid_flag = np.abs(reversed_points[:,:2] - point_xy_mean) < 2.5 * point_xy_stnd #### 
                        reversed_points = reversed_points[valid_flag[:,1]&valid_flag[:,0]]
                        
                        target_boundaries = np.array([sign_x*box_dx/2, sign_y*box_dy/2])
                        close_boundary = np.sum(np.mean(np.abs(reversed_points[:,:2]-target_boundaries), axis=1))
                        total_distance += 0.1*close_boundary
                        
                    else:
                        raise NotImplementedError('Not Implemented initial loc: %s' % init_loc[box_fit_config][label])

                return total_distance
            
            # Initial guess for translation and azimuth angle
            initial_guess = [0, 0, 0, 0]  # [tx, ty, tz, theta]
            # if init_loc[box_fit_config][label] == 'obj_center':
            #     initial_guess[:3] = np.mean(points, axis=0) 
            initial_guess[:3] = np.mean(points, axis=0) 
            initial_guess[3] = 0.01
            
            # Optimize
            # result = minimize(objective, initial_guess, method='BFGS', options={'disp':True, 'maxiter':1000})
            result = minimize(objective, initial_guess, method='BFGS')

            # Extract optimized parameters
            optimized_translation = result.x[:3]
            optimized_theta = result.x[3]

            box_numpy = np.concatenate([optimized_translation, np.array([box_dx, box_dy, box_dz, optimized_theta])], axis=0)
            box_dict['boxes'][i_obj, :] = torch.from_numpy(box_numpy)
            box_dict['labels'][i_obj] = label

        box_dicts.append(box_dict)

    return box_dicts
