# LICENSE: This script is for use with the Audi A2D2 dataset.
# All original data is subject to the CC BY-ND 4.0 license.
# Per the 'NoDerivatives' clause, you may use this script to generate data 
# for internal use, but distributing the modified/derived data is restricted.
# Please download original data from https://www.a2d2.audi.

import pickle
import numpy as np
from pcdet.utils import box_utils

# 5-shot
save_map = [{'20180925124435_frontcenter_000024735': [0, 1, 3]}, {'20180807145028_frontcenter_000023951': [3]}, 
            {'20180925124435_frontcenter_000092819': [2, 3]}, {'20180807145028_frontcenter_000033246': [0, 2]}, 
            {'20180925124435_frontcenter_000072300': [1, 5]}, {'20181107132730_frontcenter_000008152': [3, 5]}, 
            {'20180925124435_frontcenter_000036213': [2, 3, 6]}, {'20180807145028_frontcenter_000015108': [1, 2]}, 
            {'20181107132730_frontcenter_000007470': [0, 1, 2]}, {'20181107132730_frontcenter_000008225': [0, 3]}, 
            {'20181107132730_frontcenter_000005372': [0, 1, 3]}, {'20181107132730_frontcenter_000005411': [0, 2]}, 
            {'20180807145028_frontcenter_000031268': [1, 2, 3]}, {'20181107132730_frontcenter_000004230': [0, 2]}, 
            {'20181107132730_frontcenter_000005373': [0, 1]}]

def get_lidar(lidar_file):
    point_features = np.load(lidar_file)['points'].astype(np.float32)
    return point_features
    
def save_lidar(lidar_file, points):
    np.savez(lidar_file, points=points.astype(np.float32))

# Directories
source_lidar_dir = 'xxx/a2d2/training/lidar'  # original lidar dir
target_lidar_dir = './lidar_fs'  # output dir

# Load pkl
with open('a2d2_infos_train.pkl', 'rb') as f: # original pkl file by openpcdet
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
    lidar_idx = element['point_cloud']['lidar_idx']
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
        pc_org = get_lidar(source_lidar_dir+'/'+lidar_idx+'.npz')
        
        # Get corners of boxes to be deleted
        corners_lidar_gt = box_utils.boxes_to_corners_3d(element['annos']['gt_boxes_lidar'])
        flag_del_pts = np.zeros(pc_org.shape[0], dtype=bool)
        
        for idx in del_idx:
            if idx < corners_lidar_gt.shape[0]:
                flag_del_pts |= box_utils.in_hull(pc_org[:, :3], corners_lidar_gt[idx])
        
        # Keep only points not in deleted boxes
        pc_new = pc_org[~flag_del_pts]
        
        # Save filtered point cloud
        pc_new = pc_new.astype(np.float32)
        save_lidar(target_lidar_dir+'/'+lidar_idx+'.npz', pc_new)
        
        # Create new element with filtered annotations
        new_element = {
            'point_cloud': element['point_cloud'].copy(),
            'image': element['image'].copy(),
            'annos': {}
        }
        
        # Filter annotations (keep only target_indices)
        mask = np.isin(original_indices, target_indices)
        
        for key in element['annos'].keys():
            original_value = element['annos'][key]
            
            if key != 'gt_boxes_lidar':
                new_element['annos'][key] = original_value[mask]
            else:
                new_element['annos'][key] = original_value[mask[:original_value.shape[0]]]
        
        filtered_data.append(new_element)
        
        print(f"Processing point cloud {lidar_idx}: original objects={len(original_indices)}, "
              f"filtered objects={np.sum(mask)}, deleted points={np.sum(flag_del_pts)}")

print(f"\nTotal filtered out {len(filtered_data)} point clouds")

# Save results
with open('fs-a2d2_infos_train.pkl', 'wb') as f:
    pickle.dump(filtered_data, f)

print("Results saved to fs-a2d2_infos_train.pkl")

# import pickle
# info_file = 'fs-a2d2_infos_train.pkl'
# with open(info_file, 'rb') as f:
#     infos = pickle.load(f)
#     print(infos[0].keys())
#     print(infos[0]['annos'].keys())
#     print(infos[0]['point_cloud']['lidar_idx'])

# for key in infos[0]['annos'].keys():
#     print(key, infos[2]['annos'][key].shape)

# cls_stat = {}
# for info in infos:
#     for obj in info['annos']['name']:
#         if obj not in cls_stat:
#             cls_stat[obj] = 1
#         else:
#             cls_stat[obj] += 1
# print(cls_stat)