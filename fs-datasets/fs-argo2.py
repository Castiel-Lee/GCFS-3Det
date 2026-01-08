# LICENSE: This script is designed for use with the Argoverse 2 (AV2) dataset.
# The dataset is licensed under CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike).
# This script is for research purposes only; commercial use is strictly prohibited.
# Users must download the original data from https://www.argoverse.org.

import pickle
import numpy as np
from pcdet.utils import box_utils  
           
# 5-shot
save_map = [{'0453065': [2, 3, 4, 5]}, 
            {'0065073': [0, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 79, 120]}, 
            {'0369068': [3, 4, 5, 6, 7, 8, 9, 39, 92, 94, 95]}, {'0138066': [8, 14, 69]}, {'0595114': [0, 2, 3, 4, 5, 6, 10]}, {'0585016': [3, 47, 51]}, 
            {'0556033': [1, 2, 4, 7, 8, 63]}, {'0467129': [1, 2, 48, 50, 51, 52, 53]}, {'0563063': [2, 3, 4, 5, 6, 13, 53, 80]}, {'0687010': [3, 5, 6, 95]}, 
            {'0219078': [1, 2, 3, 8]}, {'0123130': [8, 22, 31, 32, 121, 122, 123, 124]}, {'0683054': [1, 45, 46]}, 
            {'0283093': [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 51, 78, 79, 80, 128, 129]}, 
            {'0433106': [3, 24, 34, 47, 50, 92, 93, 95, 96]}, {'0603125': [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 25, 36, 67, 68]}, 
            {'0024105': [1, 27]}, {'0452051': [0, 5, 46, 69]}, {'0104054': [3, 12, 49]}, {'0649152': [2]}, {'0639102': [1, 2, 3, 4, 5, 6, 8, 26, 39, 40, 41, 42]}, 
            {'0447123': [3, 5, 43]}, {'0568141': [0, 15, 82, 85]}, {'0558139': [1, 2, 3, 64, 65, 66, 67, 68]}, {'0441093': [2, 3, 4, 5, 91]}]


# Directories
source_lidar_dir = 'xxx/velodyne'  # original lidar dir
target_lidar_dir = './lidar_fs'  # output lidar dir

# Load pkl
with open('argo2_infos_train.pkl', 'rb') as f: # original pkl file by openpcdet
    p2_data = pickle.load(f)

# Convert save_map to dict format for easier lookup: {lidar_id: [index_list]}
save_map_dict = {}
for item in save_map:
    for lidar_id, index_list in item.items():
        save_map_dict[lidar_id] = index_list

print(f"save_map contains {len(save_map_dict)} point cloud IDs")
print(f"pkl contains {len(p2_data)} point clouds")

# Filter pkl based on save_map
filtered_data = []

for element in p2_data:
    # Get point cloud id
    lidar_idx = element['sample_idx']
    if lidar_idx.endswith('.bin'):
        lidar_idx = lidar_idx[:-4]
    
    # Check if this point cloud is in save_map
    if lidar_idx in save_map_dict:
        target_indices = save_map_dict[lidar_idx]

        # Create del_flag: mark objects to be deleted (not in target_indices)
        original_indices = element['annos']['index']
        del_flag = ~np.isin(original_indices, target_indices)
        del_idx = np.arange(del_flag.shape[0])[del_flag]
        
        # Load original point cloud
        pc_org = np.fromfile(
            source_lidar_dir + '/' + element['sample_idx'] + '.bin',
            dtype=np.float32,
            count=-1
        ).reshape([-1, 4])

        # Get corners of boxes to be deleted
        loc, dims, rots = element['annos']['location'], element['annos']['dimensions'], element['annos']['rotation_y']
        gt_bboxes_3d = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1).astype(np.float32)
        corners_lidar_gt = box_utils.boxes_to_corners_3d(gt_bboxes_3d)
        flag_del_pts = np.zeros(pc_org.shape[0], dtype=bool)
        
        for idx in del_idx:
            if idx < corners_lidar_gt.shape[0]:
                flag_del_pts |= box_utils.in_hull(pc_org[:, :3], corners_lidar_gt[idx])
        
        # Keep only points not in deleted boxes
        pc_new = pc_org[~flag_del_pts]
        
        # Save filtered point cloud
        pc_new = pc_new.reshape(-1,).astype(np.float32)
        pc_new.tofile(target_lidar_dir + '/' + element['sample_idx'] + '.bin')
        
        # Create new element with filtered annotations
        new_element = {
            'point_cloud': element['point_cloud'].copy(),
            'uuid': element['uuid'], 
            'sample_idx': element['sample_idx'], 
            'pose': element['pose'].copy(), 
            'sweeps': element['sweeps'].copy(), 
            'cams': element['cams'].copy(),
            'image': element['image'].copy(),
            'calib': element['calib'].copy(),
            'annos': {}
        }
        
        # Filter annotations (keep only target_indices)
        mask = np.isin(original_indices, target_indices)

        none_keys = ['bbox', 'group_ids', 'camera_id', 'difficulty',]
        for key in element['annos'].keys():
            if key in none_keys:
                new_element['annos'][key] = None
            else:
                original_value = element['annos'][key]
                new_element['annos'][key] = original_value[mask]
        
        filtered_data.append(new_element)
        
        print(f"Processing point cloud {lidar_idx}: original objects={len(original_indices)}, "
              f"filtered objects={np.sum(mask)}, deleted points={np.sum(flag_del_pts)}")

print(f"\nTotal filtered out {len(filtered_data)} point clouds")

# Save results
with open('fs-argo2_infos_train.pkl', 'wb') as f:
    pickle.dump(filtered_data, f)

print("Results saved to fs-argo2_infos_train.pkl")

# import pickle
# info_file = 'fs-argo2_infos_train.pkl'
# with open(info_file, 'rb') as f:
#     infos = pickle.load(f)
#     print(infos[0].keys())
#     print(infos[0]['annos'].keys())
#     print(infos[0]['sample_idx'])

# # for key in infos[0]['annos'].keys():
# #     print(key, infos[2]['annos'][key].shape)

# cls_stat = {}
# for info in infos:
#     for obj in info['annos']['name']:
#         if obj not in cls_stat:
#             cls_stat[obj] = 1
#         else:
#             cls_stat[obj] += 1
# print(cls_stat)