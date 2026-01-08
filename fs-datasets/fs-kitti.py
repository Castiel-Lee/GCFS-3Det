# LICENSE: This script is for use with the KITTI Vision Benchmark Suite.
# The dataset is licensed under CC BY-NC-SA 3.0 (Attribution-NonCommercial-ShareAlike).
# For research and non-commercial use only. Commercial use is not permitted.
# Please cite the original KITTI papers and download data from http://www.cvlibs.net/datasets/kitti/.

import pickle
import numpy as np
from pcdet.utils import box_utils  

# 1-shot
save_map = [{'002105': [0, 2, 11, 12]}, {'007286': [2, 4]}, {'003518': [4]}, {'001518': [1]}]
# 3-shot
save_map = [{'002105': [0, 2, 11, 12]}, {'005626': [3]}, {'001262': [2, 13]}, {'000518': [0, 1, 2, 3]}, {'005278': [3]}, {'000364': [2]}, 
            {'000157': [1, 2, 3, 4, 6, 7, 10]}, {'003518': [4]}, {'007338': [0]}, {'002464': [2]}, {'001518': [1]}, {'004872': [1]}]
# 10-shot
save_map = [{'004653': [0, 1, 4, 5, 8, 9]}, {'003991': [6]}, {'005667': [2, 4]}, {'007394': [2, 5, 8, 9, 12]}, {'007151': [0, 4, 6, 10]}, 
            {'001939': [0, 1, 5, 6, 9, 10]}, {'004090': [2, 3]}, {'005134': [7, 8]}, {'003704': [4, 14]}, {'001836': [0, 6]}, {'004712': [2, 4]},
            {'003009': [0, 1, 3, 9]}, {'000364': [2]}, {'001319': [2, 4]}, {'003084': [4]}, {'003438': [3]}, {'003149': [2]}, {'005604': [1, 5]},
            {'005846': [5]}, {'003765': [2, 3, 4, 5, 8]}, {'004908': [1]}, {'002444': [8]}, {'003585': [8]}, {'002227': [4]}, {'005039': [4]}, 
            {'004218': [0, 2, 4]}, {'002211': [0, 4]}, {'006492': [4]}, {'002571': [1, 4]}, {'005504': [1]}, {'006690': [0]}, {'006313': [1]}, 
            {'004671': [0, 1]}, {'001518': [1]}, {'002950': [1]}, {'006428': [2]}, {'004067': [2]}, {'002217': [1]}, {'003363': [1]}, {'001541': [3]}]
# 20-shot
save_map = [{'000012': [1]}, {'004269': [2]}, {'001638': [0, 2, 7, 8, 11, 12]}, {'000785': [3, 4, 9, 10, 13]}, {'007414': [5, 6]}, 
            {'003441': [1, 2, 6, 8]}, {'007317': [1, 4]}, {'002918': [3]}, {'005780': [6]}, {'004212': [0]}, {'003227': [0, 2, 3]}, {'006788': [3]}, 
            {'006329': [0, 1, 4]}, {'002067': [11, 13]}, {'001060': [3, 4, 5]}, {'001568': [1]}, {'006746': [1, 3, 7]}, {'001085': [3]}, 
            {'005134': [1, 7, 8]}, {'001796': [3]}, {'003149': [2]}, {'001988': [4, 7, 8]}, {'000277': [2, 5]}, {'001644': [3]}, {'000364': [2]},
            {'000518': [3]}, {'005604': [1, 5]}, {'003009': [0, 1, 3, 9]}, {'001319': [2, 4]}, {'004712': [4]}, {'001870': [3]}, {'005278': [3, 5]}, 
            {'005846': [5]}, {'003390': [6]}, {'003765': [2, 3, 4, 5, 8]}, {'003438': [2, 3]}, {'002005': [2]}, {'000177': [5]}, {'003380': [3, 6]}, 
            {'003084': [4]}, {'005778': [1, 2]}, {'003157': [3]}, {'003516': [0]}, {'000339': [1]}, {'000325': [7]}, {'003235': [0, 3]}, {'002494': [2]},
            {'005693': [5]}, {'002480': [0]}, {'006776': [1]}, {'000906': [2, 5]}, {'004371': [1, 2]}, {'001720': [1, 4]}, {'000271': [4]}, 
            {'003626': [3]}, {'005539': [1, 2]}, {'003589': [0, 2, 5, 6]}, {'001748': [1, 2]}, {'002943': [3]}, {'001788': [1, 6, 7]}, {'003362': [2]}, 
            {'002791': [2]}, {'006428': [2]}, {'000071': [1]}, {'004845': [0]}, {'002268': [1]}, {'002217': [1]}, {'002167': [0, 3]}, {'006983': [0]}, 
            {'001532': [2]}, {'004671': [1]}, {'001847': [1]}, {'003533': [0]}, {'002309': [0, 1]}, {'001316': [0, 1]}, {'001430': [2]}, {'002650': [1]},
            {'004099': [0, 1]}, {'006271': [0]}, {'005118': [0, 2]}, {'006224': [8]}, {'001883': [0]}, {'002664': [0]}, {'004179': [7]}, 
            {'003163': [6, 10]}, {'006554': [1]}, {'001056': [4]}, {'002731': [8, 11]}]
# 40-shot
save_map = [{'001685': [0, 3, 5, 7]}, {'001595': [3, 4]}, {'001895': [2]}, {'005488': [0, 1, 3, 12]}, {'004090': [3]}, {'005648': [8, 12]}, 
            {'004607': [3]}, {'002697': [0, 1, 3, 5, 11]}, {'002912': [5, 6, 13, 14, 17, 18]}, {'000755': [0, 2]}, {'003374': [0]}, 
            {'004387': [0, 2, 6, 8, 9]}, {'006363': [3]}, {'000713': []}, {'001430': [1, 2]}, {'002623': [4]}, {'002972': [0, 4, 5, 6, 16]}, 
            {'004653': [0, 1, 2, 4, 5, 6, 8, 9]}, {'007356': [3]}, {'007101': [1, 2, 4, 5, 11, 14]}, {'001568': [0, 1]}, {'000924': []}, 
            {'002915': [0, 1, 2, 3, 4]}, {'003390': [0, 2, 3, 6, 7]}, {'005846': [0, 1, 5, 6]}, {'000518': [0, 1, 2, 3, 4, 5, 7]}, {'000364': [2]}, 
            {'003149': [2]}, {'002005': [0, 2]}, {'001870': [0, 1, 3, 4, 7]}, {'001319': [0, 1, 2, 3]}, {'003380': [0, 1, 2, 3, 4, 6]}, 
            {'007286': [0, 1, 2, 3, 4, 5]}, {'000177': [0, 1, 2, 5, 6]}, {'003765': [1, 2, 3, 4, 5, 6, 7]}, {'004712': [0, 1, 2, 4, 5]}, 
            {'005604': [0, 1, 4, 5]}, {'006528': [0, 3, 5, 9]}, {'003438': [0, 1, 2, 3, 4]}, {'000277': [0, 2, 4, 9]}, {'001988': [0, 1, 4, 5, 7, 8, 9]},
            {'005278': [2, 3, 4]}, {'003084': [0, 1, 4, 5]}, {'003009': [0, 1, 2, 3, 5, 6, 7, 9, 10, 11]}, {'005825': [0, 1, 2]}, {'006842': [3, 4]}, 
            {'005719': [6]}, {'004371': [0, 1, 3, 4, 6]}, {'002535': []}, {'003740': [0, 1, 2, 3, 5, 6]}, {'002211': [0, 2, 3, 4]}, {'004382': [3]}, 
            {'005436': [5, 12]}, {'004144': [0]}, {'004558': [0, 1]}, {'001748': [0, 2]}, {'006160': [5]}, {'005928': [1]}, {'000461': [0]}, 
            {'006912': [1]}, {'007338': [0]}, {'001360': [0, 1]}, {'005579': [1, 2, 3, 4]}, {'001788': [3, 6, 7]}, {'002269': []}, {'000217': [3, 5, 6]},
            {'005521': [0, 1, 2, 3]}, {'006545': [0, 1, 2, 3]}, {'002791': [1, 2]}, {'005497': [1]}, {'002650': [1, 2, 3, 4]}, {'007020': [4]}, 
            {'003363': [0, 1, 2, 3]}, {'004286': [1, 3, 5]}, {'001677': [0]}, {'004099': [0]}, {'006983': [0]}, {'006428': [1, 2, 3, 4]}, 
            {'002335': [3]}, {'003209': [1, 2, 3]}, {'001791': [1, 2, 3]}, {'002309': [1, 2, 3, 4]}, {'005009': [0, 1]}, {'003862': [0, 1]}, 
            {'006271': [0, 1, 2, 3]}, {'001769': [0]}, {'005509': [0, 1]}, {'000296': [5]}, {'002700': [1]}, {'006599': [0]}, {'002731': [8]}, 
            {'000423': [7, 10]}, {'006243': [0]}, {'005243': [0]}, {'002116': [1]}, {'001676': [10]}, {'002510': [7, 10]}, {'003345': [3]}, 
            {'001735': [0]}, {'003163': [6, 10]}, {'005442': [0]}, {'005772': [0]}, {'004179': [7, 9]}, {'001302': [0]}, {'000539': [0]}, 
            {'006639': [6, 10]}, {'006557': [4]}, {'007245': [0]}, {'006896': [0]}, {'002716': [10]}]
# 5-shot
save_map = [{'002105': [0, 2, 11, 12]}, {'006143': [0, 3, 13]}, {'005626': [3]}, {'006838': [0]}, {'001262': [2, 13]}, {'003765': [1, 5]}, 
            {'000518': [0, 1, 2, 3]}, {'007286': [2, 4]}, {'005278': [3]}, {'000364': [2]}, {'000157': [1, 2, 3, 4, 6, 7, 10]}, {'003157': [3]}, 
            {'003518': [4]}, {'000424': [1, 2, 4, 6, 7, 10]}, {'007338': [0]}, {'002464': [2]}, {'006119': [1]}, {'001518': [1]}, {'005554': [0]}, 
            {'004872': [1]}]


# Directories
source_lidar_dir = 'xxx/kitti/training/velodyne' # original lidar dir
target_lidar_dir = './lidar_fs'  # output lidar dir

# Load pkl
with open('kitti_infos_train.pkl', 'rb') as f: # original pkl file by openpcdet
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
        pc_org = np.fromfile(
            source_lidar_dir + '/' + element['point_cloud']['lidar_idx'] + '.bin',
            dtype=np.float32,
            count=-1
        ).reshape([-1, 4])
        
        # Get corners of boxes to be deleted
        corners_lidar_gt = box_utils.boxes_to_corners_3d(element['annos']['gt_boxes_lidar'])
        flag_del_pts = np.zeros(pc_org.shape[0], dtype=bool)
        
        for idx in del_idx:
            if idx < corners_lidar_gt.shape[0]:
                flag_del_pts |= box_utils.in_hull(pc_org[:, :3], corners_lidar_gt[idx])
        
        # Keep only points not in deleted boxes
        pc_new = pc_org[~flag_del_pts]
        
        # Save filtered point cloud
        pc_new = pc_new.reshape(-1,).astype(np.float32)
        pc_new.tofile(target_lidar_dir + '/' + element['point_cloud']['lidar_idx'] + '.bin')
        
        # Create new element with filtered annotations
        new_element = {
            'point_cloud': element['point_cloud'].copy(),
            'image': element['image'].copy(),
            'calib': element['calib'].copy(),
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
with open('fs-kitti_infos_train.pkl', 'wb') as f:
    pickle.dump(filtered_data, f)

print("Results saved to fs-kitti_infos_train.pkl")

# import pickle
# info_file = 'fs-kitti_infos_train.pkl'
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