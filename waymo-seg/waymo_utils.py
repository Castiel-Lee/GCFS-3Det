import os, math
import pickle
import numpy as np
# from ...utils import common_utils
import tensorflow as tf
from waymo_open_dataset.utils import frame_utils, transform_utils, range_image_utils
from waymo_open_dataset import dataset_pb2
from collections import Counter
import zlib




try:
    tf.enable_eager_execution()
except:
    pass

WAYMO_CLASSES = ['unknown', 'Vehicle', 'Pedestrian', 'Sign', 'Cyclist']
# unwantted_labels = set([15, 18])
wantted_labels = {
    "unknown": set([9, 10, 11, 12, 13]),
    "Vehicle": set([1, 2, 3, 12, 13])
}
seg_label_dict = {
    0: "TYPE_UNDEFINED",
    1: "Car",
    2: "Truck",
    3: "Bus",
    4: "TYPE_OTHER_VEHICLE",
    5: "TYPE_MOTORCYCLIST",
    6: "TYPE_BICYCLIST",
    7: "TYPE_PEDESTRIAN",
    8: "TYPE_SIGN",
    9: "Trafficlight",
    10: "Pole",
    11: "Constructioncone",
    12: "Bicycle",
    13: "Motorcycle",
    14: "TYPE_BUILDING",
    15: "TYPE_VEGETATION",
    16: "TYPE_TREE_TRUNK",
    17: "TYPE_CURB",
    18: "TYPE_ROAD",
    19: "TYPE_LANE_MARKER",
    20: "TYPE_OTHER_GROUND",
    21: "TYPE_WALKABLE",
    22: "TYPE_SIDEWALK",
}


def convert_range_image_to_point_cloud_labels(frame, range_images, segmentation_labels, ri_index=0):
    """Convert segmentation labels from range images to point clouds.

    Args:
    frame: open dataset frame
    range_images: A dict of {laser_name, [range_image_first_return,
        range_image_second_return]}.
    segmentation_labels: A dict of {laser_name, [range_image_first_return,
        range_image_second_return]}.
    ri_index: 0 for the first return, 1 for the second return.

    Returns:
    point_labels: {[N, 2]} list of 3d lidar points's segmentation labels. 0 for
        points that are not labeled.
    """
    calibrations = sorted(frame.context.laser_calibrations, key=lambda c: c.name)
    point_labels = []
    for c in calibrations:
        range_image = range_images[c.name][ri_index]
        range_image_tensor = tf.reshape(
            tf.convert_to_tensor(range_image.data), range_image.shape.dims)
        range_image_mask = range_image_tensor[..., 0] > 0

        if c.name in segmentation_labels:
            sl = segmentation_labels[c.name][ri_index]
            sl_tensor = tf.reshape(tf.convert_to_tensor(sl.data), sl.shape.dims)
            sl_points_tensor = tf.gather_nd(sl_tensor, tf.where(range_image_mask))
        else:
            num_valid_point = tf.math.reduce_sum(tf.cast(range_image_mask, tf.int32))
            sl_points_tensor = tf.zeros([num_valid_point, 2], dtype=tf.int32)
            
        point_labels.append(sl_points_tensor.numpy())
    return point_labels

def is_point_in_box(point, box_center, box_size, box_heading):
    # Rotate the point to the box's coordinate system
    rotation_matrix = np.array([
        [math.cos(box_heading), -math.sin(box_heading)],
        [math.sin(box_heading), math.cos(box_heading)],
    ])
    
    # Translate point to the box's local coordinates
    point_relative = np.array([point[0] - box_center[0], point[1] - box_center[1]])

    # Rotate the translated point
    point_rotated = np.dot(rotation_matrix, point_relative)

    # Check if the point lies within the box bounds (considering only x and y)
    if -box_size[0] / 2 <= point_rotated[0] <= box_size[0] / 2 and -box_size[1] / 2 <= point_rotated[1] <= box_size[1] / 2:
        return True
    return False

def is_point_in_box_vectorized(points, box_center, box_size, box_heading):
    """
    :param points: (N, 3) numpy array of point coordinates (x, y, z)
    :param box_center: (3,) numpy array, center of the bounding box (x, y, z)
    :param box_size: (3,) numpy array, size of the box (width, length, height)
    :param box_heading: float, rotation angle of the box in radians (around the z-axis)
    :return: (N,) boolean numpy array indicating whether each point is inside the box
    """
    # Extract only the (x, y, z) components
    points_xyz = points[:, :3]  # (N, 3)
    box_center_xyz = np.array(box_center)  # Ensure it's an array (3,)
    
    # Create 2D rotation matrix (for x, y plane rotation)
    cos_theta = np.cos(-box_heading)  # Negative angle for inverse rotation
    sin_theta = np.sin(-box_heading)
    rotation_matrix = np.array([[cos_theta, -sin_theta, 0], [sin_theta, cos_theta, 0], [0, 0, 1]])  # (3, 3)
    
    # Translate points to the box's local coordinates
    points_relative = points_xyz - box_center_xyz  # (N, 3)
    
    # Rotate points to align with box axes
    points_rotated = points_relative @ rotation_matrix.T  # (N, 3)
    
    # Check if points are within box bounds (width along x, length along y, height along z)
    half_width, half_length, half_height = box_size[0] / 2, box_size[1] / 2, box_size[2] / 2
    inside_x = (-half_width <= points_rotated[:, 0]) & (points_rotated[:, 0] <= half_width)
    inside_y = (-half_length <= points_rotated[:, 1]) & (points_rotated[:, 1] <= half_length)
    inside_z = (-half_height <= points_rotated[:, 2]) & (points_rotated[:, 2] <= half_height)
    
    return inside_x & inside_y & inside_z  # (N,) boolean array

def generate_labels(frame, pose, frame_with_seg_label):
    seg_obj_name = []
    obj_name, difficulty, dimensions, locations, heading_angles = [], [], [], [], []
    tracking_difficulty, speeds, accelerations, obj_ids = [], [], [], []
    num_points_in_gt = []
    laser_labels = frame.laser_labels
    debug_boxes, debug_mask = [], []

    (range_images, camera_projections, segmentation_labels, range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(frame_with_seg_label)
    points, cp_points = frame_utils.convert_range_image_to_point_cloud(frame_with_seg_label, range_images, camera_projections, range_image_top_pose)
    # points_ri2, cp_points_ri2 = frame_utils.convert_range_image_to_point_cloud(frame, range_images, camera_projections, range_image_top_pose, ri_index=1)
    
    point_labels = convert_range_image_to_point_cloud_labels(frame, range_images, segmentation_labels)
    # point_labels_ri2 = convert_range_image_to_point_cloud_labels(frame, range_images, segmentation_labels, ri_index=1)

    # seg_labels[1][0].data
    # only top lidar has seg annotation
    points_all = points[0] # TOP 154450
    seg_labels_all = point_labels[0] # TOP 154450
    # points_all = np.concatenate(points, axis=0)  # Shape: (N, 3) [x, y, z]
    # seg_labels_all = np.concatenate(point_labels, axis=0)

    # summary = {
    #     "Car": 0,
    #     "Truck": 0,
    #     "Bus": 0,
    #     "Bicycle":0,
    #     "Motorcycle": 0,
    #     "None": 0
    # }

    for i in range(len(laser_labels)):
        box = laser_labels[i].box
        box_center = np.array([box.center_x, box.center_y, box.center_z])  # [x, y, z]
        box_size = np.array([box.width, box.length, box.height])  # [width, length, height]
        box_heading = box.heading

        class_ind = laser_labels[i].type
        loc = [box.center_x, box.center_y, box.center_z]
        heading_angles.append(box.heading)
        obj_name.append(WAYMO_CLASSES[class_ind])
        # print(WAYMO_CLASSES[class_ind])
        # if WAYMO_CLASSES[class_ind] == "unknown":
        #     breakpoint()

        final_label = WAYMO_CLASSES[class_ind]
        mask = is_point_in_box_vectorized(points_all, box_center, box_size, box_heading + math.pi / 2)
        labels_box = np.array(seg_labels_all)[mask, 1].astype(int)
        frequency = np.bincount(labels_box)
        valid_indices = np.where(frequency > 0)[0]
        sorted_labels = valid_indices[np.argsort(-frequency[valid_indices])]

        # debug_boxes.append(np.array([box.center_x, box.center_y, box.center_z, box.width, box.length, box.height, box.heading, class_ind]))
        # debug_mask.append(mask)
        # if final_label in wantted_labels.keys():            
        #     target_labels = wantted_labels[final_label]
        #     final_label = "None"
        #     for fl in sorted_labels:
        #         if fl in target_labels:
        #             final_label = seg_label_dict[fl]
        #             break
        if final_label in wantted_labels:
            for fl in sorted_labels:
                if fl in wantted_labels[final_label]:
                    final_label = seg_label_dict[fl]
                    break
            else:
                final_label = "None"

            # if laser_labels[i].type == 1:
            #     summary[final_label] += 1
        seg_obj_name.append(final_label)

        difficulty.append(laser_labels[i].detection_difficulty_level)
        tracking_difficulty.append(laser_labels[i].tracking_difficulty_level)
        dimensions.append([box.length, box.width, box.height])  # lwh in unified coordinate of OpenPCDet
        locations.append(loc)
        obj_ids.append(laser_labels[i].id)
        num_points_in_gt.append(laser_labels[i].num_lidar_points_in_box)
        speeds.append([laser_labels[i].metadata.speed_x, laser_labels[i].metadata.speed_y])
        accelerations.append([laser_labels[i].metadata.accel_x, laser_labels[i].metadata.accel_y])

    annotations = {}
    annotations['name'] = np.array(obj_name)
    annotations['difficulty'] = np.array(difficulty)
    annotations['dimensions'] = np.array(dimensions)
    annotations['location'] = np.array(locations)
    annotations['heading_angles'] = np.array(heading_angles)

    annotations["seg_obj_name"] = np.array(seg_obj_name)
    # annotations["summary"] = np.array(list(summary.values()))

    # points boxes seg_labels;
    # annotations["debug_points"] = np.array(points_all)
    # annotations["debug_seg_labels"] = np.array(seg_labels_all)
    # annotations["debug_boxes"] = np.array(debug_boxes)
    # annotations["debug_mask"] = np.array(debug_mask)


    annotations['obj_ids'] = np.array(obj_ids)
    annotations['tracking_difficulty'] = np.array(tracking_difficulty)
    annotations['num_points_in_gt'] = np.array(num_points_in_gt)
    annotations['speed_global'] = np.array(speeds)
    annotations['accel_global'] = np.array(accelerations)

    # annotations = common_utils.drop_info_with_name(annotations, name='unknown')
    # annotations = drop_info_with_name(annotations, name='unknown')
    if annotations['name'].__len__() > 0:
        global_speed = np.pad(annotations['speed_global'], ((0, 0), (0, 1)), mode='constant', constant_values=0)  # (N, 3)
        speed = np.dot(global_speed, np.linalg.inv(pose[:3, :3].T))
        speed = speed[:, :2]
        
        gt_boxes_lidar = np.concatenate([
            annotations['location'], annotations['dimensions'], annotations['heading_angles'][..., np.newaxis], speed],
            axis=1
        )
    else:
        gt_boxes_lidar = np.zeros((0, 9))
    annotations['gt_boxes_lidar'] = gt_boxes_lidar
    return annotations

def convert_range_image_to_point_cloud(frame, range_images, camera_projections, range_image_top_pose, ri_index=(0, 1)):
    """
    Modified from the codes of Waymo Open Dataset.
    Convert range images to point cloud.
    Args:
        frame: open dataset frame
        range_images: A dict of {laser_name, [range_image_first_return, range_image_second_return]}.
        camera_projections: A dict of {laser_name,
            [camera_projection_from_first_return, camera_projection_from_second_return]}.
        range_image_top_pose: range image pixel pose for top lidar.
        ri_index: 0 for the first return, 1 for the second return.

    Returns:
        points: {[N, 3]} list of 3d lidar points of length 5 (number of lidars).
        cp_points: {[N, 6]} list of camera projections of length 5 (number of lidars).
    """
    calibrations = sorted(frame.context.laser_calibrations, key=lambda c: c.name)
    points = []
    cp_points = []
    points_NLZ = []
    points_intensity = []
    points_elongation = []

    frame_pose = tf.convert_to_tensor(np.reshape(np.array(frame.pose.transform), [4, 4]))
    # [H, W, 6]
    range_image_top_pose_tensor = tf.reshape(
        tf.convert_to_tensor(range_image_top_pose.data), range_image_top_pose.shape.dims
    )
    # [H, W, 3, 3]
    range_image_top_pose_tensor_rotation = transform_utils.get_rotation_matrix(
        range_image_top_pose_tensor[..., 0], range_image_top_pose_tensor[..., 1],
        range_image_top_pose_tensor[..., 2])
    range_image_top_pose_tensor_translation = range_image_top_pose_tensor[..., 3:]
    range_image_top_pose_tensor = transform_utils.get_transform(
        range_image_top_pose_tensor_rotation,
        range_image_top_pose_tensor_translation)

    for c in calibrations:
        points_single, cp_points_single, points_NLZ_single, points_intensity_single, points_elongation_single \
            = [], [], [], [], []
        for cur_ri_index in ri_index:
            range_image = range_images[c.name][cur_ri_index]
            if len(c.beam_inclinations) == 0:  # pylint: disable=g-explicit-length-test
                beam_inclinations = range_image_utils.compute_inclination(
                    tf.constant([c.beam_inclination_min, c.beam_inclination_max]),
                    height=range_image.shape.dims[0])
            else:
                beam_inclinations = tf.constant(c.beam_inclinations)

            beam_inclinations = tf.reverse(beam_inclinations, axis=[-1])
            extrinsic = np.reshape(np.array(c.extrinsic.transform), [4, 4])

            range_image_tensor = tf.reshape(
                tf.convert_to_tensor(range_image.data), range_image.shape.dims)
            print(c.name,'range_image_tensor:',range_image_tensor.shape)
            pixel_pose_local = None
            frame_pose_local = None
            if c.name == dataset_pb2.LaserName.TOP:
                pixel_pose_local = range_image_top_pose_tensor
                pixel_pose_local = tf.expand_dims(pixel_pose_local, axis=0)
                frame_pose_local = tf.expand_dims(frame_pose, axis=0)
            range_image_mask = range_image_tensor[..., 0] > 0
            range_image_NLZ = range_image_tensor[..., 3]
            range_image_intensity = range_image_tensor[..., 1]
            range_image_elongation = range_image_tensor[..., 2]
            range_image_cartesian = range_image_utils.extract_point_cloud_from_range_image(
                tf.expand_dims(range_image_tensor[..., 0], axis=0),
                tf.expand_dims(extrinsic, axis=0),
                tf.expand_dims(tf.convert_to_tensor(beam_inclinations), axis=0),
                pixel_pose=pixel_pose_local,
                frame_pose=frame_pose_local)

            range_image_cartesian = tf.squeeze(range_image_cartesian, axis=0)
            print(c.name,'range_image_cartesian:',range_image_cartesian.shape)
            points_tensor = tf.gather_nd(range_image_cartesian,
                                         tf.where(range_image_mask))
            points_NLZ_tensor = tf.gather_nd(range_image_NLZ, tf.compat.v1.where(range_image_mask))
            points_intensity_tensor = tf.gather_nd(range_image_intensity, tf.compat.v1.where(range_image_mask))
            points_elongation_tensor = tf.gather_nd(range_image_elongation, tf.compat.v1.where(range_image_mask))
            cp = camera_projections[c.name][0]
            cp_tensor = tf.reshape(tf.convert_to_tensor(cp.data), cp.shape.dims)
            cp_points_tensor = tf.gather_nd(cp_tensor, tf.where(range_image_mask))

            points_single.append(points_tensor.numpy())
            cp_points_single.append(cp_points_tensor.numpy())
            points_NLZ_single.append(points_NLZ_tensor.numpy())
            points_intensity_single.append(points_intensity_tensor.numpy())
            points_elongation_single.append(points_elongation_tensor.numpy())

        points.append(np.concatenate(points_single, axis=0))
        cp_points.append(np.concatenate(cp_points_single, axis=0))
        points_NLZ.append(np.concatenate(points_NLZ_single, axis=0))
        points_intensity.append(np.concatenate(points_intensity_single, axis=0))
        points_elongation.append(np.concatenate(points_elongation_single, axis=0))

    return points, cp_points, points_NLZ, points_intensity, points_elongation


def save_lidar_points(frame, cur_save_path, use_two_returns=True):
    ret_outputs = frame_utils.parse_range_image_and_camera_projection(frame)
    if len(ret_outputs) == 4:
        range_images, camera_projections, seg_labels, range_image_top_pose = ret_outputs
        # print(seg_labels.keys())
    else:
        assert len(ret_outputs) == 3
        range_images, camera_projections, range_image_top_pose = ret_outputs

    points, cp_points, points_in_NLZ_flag, points_intensity, points_elongation = convert_range_image_to_point_cloud(
        frame, range_images, camera_projections, range_image_top_pose, ri_index=(0, 1) if use_two_returns else (0,)
    )

    # if 1 in list(seg_labels.keys()):
    #     print(type(seg_labels[1][0]))
    #     print(seg_labels[1][0].shape)
    #     print(range_images[1][0].shape)
    #     print(points[0].shape)
    #     raise NotFoundError('!!!')

    # 3d points in vehicle frame.
    points_all = np.concatenate(points, axis=0)
    points_in_NLZ_flag = np.concatenate(points_in_NLZ_flag, axis=0).reshape(-1, 1)
    points_intensity = np.concatenate(points_intensity, axis=0).reshape(-1, 1)
    points_elongation = np.concatenate(points_elongation, axis=0).reshape(-1, 1)

    num_points_of_each_lidar = [point.shape[0] for point in points]
    save_points = np.concatenate([
        points_all, points_intensity, points_elongation, points_in_NLZ_flag
    ], axis=-1).astype(np.float32)

    np.save(cur_save_path, save_points)
    # print('saving to ', cur_save_path)
    return num_points_of_each_lidar


def process_single_sequence(sequence_file, save_path, sampled_interval, has_label=True, use_two_returns=True, update_info_only=False):
    sequence_name = os.path.splitext(os.path.basename(sequence_file))[0]

    print('Load record (sampled_interval=%d): %s' % (sampled_interval, sequence_name))
    if not sequence_file.exists():
        print('NotFoundError: %s' % sequence_file)
        return []

    dataset = tf.data.TFRecordDataset(str(sequence_file), compression_type='')
    cur_save_dir = save_path / sequence_name
    cur_save_dir.mkdir(parents=True, exist_ok=True)
    pkl_file = cur_save_dir / ('%s.pkl' % sequence_name)

    sequence_infos = []
    # if pkl_file.exists():
    #     sequence_infos = pickle.load(open(pkl_file, 'rb'))
    #     sequence_infos_old = None
    #     if not update_info_only:
    #         print('Skip sequence since it has been processed before: %s' % pkl_file)
    #         return sequence_infos
    #     else:
    #         sequence_infos_old = sequence_infos
    #         sequence_infos = []
    
    # NOTE: perpare for sample un seg lab frame
    # for cnt, data in enumerate(dataset):
    #     if cnt % sampled_interval != 0:
    #         continue
    #     frame = dataset_pb2.Frame()
    #     frame.ParseFromString(bytearray(data.numpy()))
    #     if frame.lasers[0].ri_return1.segmentation_label_compressed:
    #         frame_with_seg_label = frame
    #         break

    for cnt, data in enumerate(dataset):
        if cnt % sampled_interval != 0:
            continue
        frame = dataset_pb2.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        if not frame.lasers[0].ri_return1.segmentation_label_compressed:
            continue
        frame_with_seg_label = frame

        info = {}
        pc_info = {'num_features': 5, 'lidar_sequence': sequence_name, 'sample_idx': cnt}
        info['point_cloud'] = pc_info

        info['frame_id'] = sequence_name + ('_%03d' % cnt)
        info['metadata'] = {
            'context_name': frame.context.name,
            'timestamp_micros': frame.timestamp_micros
        }
        image_info = {}
        for j in range(5):
            width = frame.context.camera_calibrations[j].width
            height = frame.context.camera_calibrations[j].height
            image_info.update({'image_shape_%d' % j: (height, width)})
        info['image'] = image_info

        pose = np.array(frame.pose.transform, dtype=np.float32).reshape(4, 4)
        info['pose'] = pose

        if has_label:
            annotations = generate_labels(frame, pose, frame_with_seg_label)
            info['annos'] = annotations

        # if update_info_only and sequence_infos_old is not None:
        #     assert info['frame_id'] == sequence_infos_old[cnt]['frame_id']
        #     num_points_of_each_lidar = sequence_infos_old[cnt]['num_points_of_each_lidar']
        # else:
        #     num_points_of_each_lidar = save_lidar_points(
        #         frame, cur_save_dir / ('%04d.npy' % cnt), use_two_returns=use_two_returns
        #     )
        # info['num_points_of_each_lidar'] = num_points_of_each_lidar

        sequence_infos.append(info)
    # with open(f"./data/{sequence_name}.pkl", "wb") as f:
    #     pickle.dump(sequence_infos, f)
    # breakpoint()    
    try:
        with open(pkl_file, 'wb') as f:
            pickle.dump(sequence_infos, f)
    except Exception as e:
        breakpoint()
        
    print('Infos are saved to (sampled_interval=%d): %s' % (sampled_interval, pkl_file))
    return sequence_infos

def drop_info_with_name(info, name):
    ret_info = {}
    keep_indices = [i for i, x in enumerate(info['name']) if x != name]
    for key in info.keys():
        if key in ["summary", "debug_points", "debug_seg_labels", "debug_boxes", "debug_mask"]:
            ret_info[key] =  info[key]
        else:
            ret_info[key] = info[key][keep_indices]
    return ret_info