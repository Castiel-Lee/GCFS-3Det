import copy
import pickle
import os
from skimage import io

import numpy as np
import json

from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import box_utils, common_utils
from ..dataset import DatasetTemplate
from .a2d2_utils import Calibration as calib_obj

cls_map ={
    'Car': 'Car',
    'Pedestrian': 'Pedestrian',
    'Truck': 'Truck',
    'Bicycle': 'Bicycle', 
    'UtilityVehicle': 'UtilityVehicle', 
    'Bus': 'Bus',
}


class A2D2Dataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]

        split_dir = os.path.join(self.root_path, 'ImageSets', (self.split + '.txt'))
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if os.path.exists(split_dir) else None

        self.a2d2_infos = []
        self.include_data(self.mode)
        self.class_names = class_names

    def include_data(self, mode):
        if self.logger is not None:
            self.logger.info('Loading a2d2 dataset.')
        a2d2_infos = []

        for info_path in self.dataset_cfg.INFO_PATH[mode]:
            info_path = self.root_path / info_path
            if not info_path.exists():
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                a2d2_infos.extend(infos)
                
        self.a2d2_infos.extend(a2d2_infos)
        if self.logger is not None:
            self.logger.info('Total samples for A2D2 dataset: %d' % (len(a2d2_infos)))

    def get_label(self, idx):
        label_file = self.root_path / 'training' / 'label3D' / ('%s.json' % idx)
        assert label_file.exists()
        with open(label_file, 'r') as f:
            config = json.load(f)

        # [N, 8]: (x y z dx dy dz heading_angle category_id)
        gt_boxes = []
        gt_names = []
        bboxes = []
        for box_name in config:
            box = config[box_name]
            # print(box['center'],box['size'],[box['rot_angle']])

            gt_boxes.append(box['center']+box['size']+[box['rot_angle']])
            name = cls_map[box['class']] if box['class'] in cls_map else'DontCare'
            gt_names.append(name)
            bbox = box['2d_bbox']
            bboxes.append([bbox[0], bbox[1], bbox[2], bbox[3]]) 

        return np.array(gt_boxes, dtype=np.float32), np.array(gt_names), np.array(bboxes, dtype=np.float32)

    def get_lidar(self, idx):
        lidar_file = self.root_path / 'training' / 'lidar' / ('%s.npz' % idx)
        assert lidar_file.exists()
        point_features = np.load(lidar_file)['points'].astype(np.float32)
        return point_features
    
    def get_image(self, idx):
        """
        Loads image for a sample
        Args:
            idx: int, Sample index
        Returns:
            image: (H, W, 3), RGB Image
        """
        img_file = self.root_path / 'training' / 'img' / ('%s.png' % idx)
        assert img_file.exists()
        image = io.imread(img_file)
        image = image.astype(np.float32)
        image /= 255.0
        return image
    
    def get_image_shape(self, idx):
        from skimage import io
        img_file = self.root_path / 'training' / 'img' / ('%s.png' % idx)
        assert img_file.exists()
        return np.array(io.imread(img_file).shape[:2], dtype=np.int32)
    
    def get_calib(self):
        calib_file = os.path.join(self.root_path, 'cams_lidars.json')
        return calib_obj(calib_file)

    def set_split(self, split):
        super().__init__(
            dataset_cfg=self.dataset_cfg, class_names=self.class_names, training=self.training,
            root_path=self.root_path, logger=self.logger
        )
        self.split = split

        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.sample_id_list) * self.total_epochs

        return len(self.a2d2_infos)

    def __getitem__(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.a2d2_infos)

        info = copy.deepcopy(self.a2d2_infos[index])

        sample_idx = info['point_cloud']['lidar_idx']
        img_shape = info['image']['image_shape']
        calib = self.get_calib()
        get_item_list = self.dataset_cfg.get('GET_ITEM_LIST', ['points'])

        input_dict = {
            'db_flag': "a2d2",
            'frame_id': sample_idx,
            'calib': calib,
        }

        points = self.get_lidar(sample_idx)
        if self.dataset_cfg.get('SHIFT_COOR', None):
            points[:, 0:3] += np.array(self.dataset_cfg.SHIFT_COOR, dtype=np.float32)
            input_dict['shift_coor'] = np.array(self.dataset_cfg.SHIFT_COOR, dtype=np.float32)
        
        input_dict['points'] = points

        if 'annos' in info:
            annos = info['annos']
            annos = common_utils.drop_info_with_name(annos, name='DontCare')
            gt_names = annos['name']
            gt_boxes_lidar = annos['gt_boxes_lidar']
            input_dict.update({
                'gt_names': gt_names,
                'gt_boxes': gt_boxes_lidar
            })

            if self.dataset_cfg.get('SHIFT_COOR', None):
                gt_boxes_lidar[:, 0:3] += self.dataset_cfg.SHIFT_COOR
            
            if "gt_boxes2d" in get_item_list:
                input_dict['gt_boxes2d'] = annos['bbox']
        
        if "images" in get_item_list:
            input_dict['images'] = self.get_image(sample_idx)
            if "gt_boxes2d" in get_item_list:
                input_dict['images'], input_dict['gt_boxes2d'] = calib.undistort_image_and_boxes(
                    input_dict['images'], input_dict['gt_boxes2d'], img_normed=True
                )
            else:
                input_dict['images'], _ = calib.undistort_image_and_boxes(input_dict['images'], img_normed=True)



        data_dict = self.prepare_data(data_dict=input_dict)
        data_dict['image_shape'] = img_shape
        return data_dict

    #@staticmethod
    def generate_prediction_dicts(self, batch_dict, pred_dicts, class_names, output_path=None):
        """
        Args:
            batch_dict:
                frame_id:
            pred_dicts: list of pred_dicts
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path:
        Returns:
        """
        def get_template_prediction(num_samples):
            ret_dict = {
                'name': np.zeros(num_samples), 'score': np.zeros(num_samples),
                'boxes_lidar': np.zeros([num_samples, 7]), 'pred_labels': np.zeros(num_samples)
            }
            return ret_dict

        def generate_single_sample_dict(box_dict):
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()
            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict

            if self.dataset_cfg.get('SHIFT_COOR', None):
                #print ("*******WARNING FOR SHIFT_COOR:", self.dataset_cfg.SHIFT_COOR)
                pred_boxes[:, 0:3] -= self.dataset_cfg.SHIFT_COOR

            pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            pred_dict['score'] = pred_scores
            pred_dict['boxes_lidar'] = pred_boxes
            pred_dict['pred_labels'] = pred_labels

            return pred_dict

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            single_pred_dict = generate_single_sample_dict(box_dict)
            single_pred_dict['frame_id'] = batch_dict['frame_id'][index]
            # single_pred_dict['metadata'] = batch_dict['metadata'][index]
            annos.append(single_pred_dict)

        return annos

    def evaluation(self, det_annos, class_names, **kwargs):
        if 'annos' not in self.a2d2_infos[0].keys():
            return 'No ground-truth boxes for evaluation', {}

        def kitti_eval(eval_det_annos, eval_gt_annos):
            from ..kitti.kitti_object_eval_python import eval as kitti_eval
            from ..kitti import kitti_utils

            map_name_to_kitti = {
                'Car': 'Car',
                'Pedestrian': 'Pedestrian',
                'Truck': 'Truck',
                'Bus': 'bus',
                'Bicycle': 'bicycle',
                'UtilityVehicle': 'UtilityVehicle',
                'H': 'Tram'
            }

            kitti_utils.transform_annotations_to_kitti_format(eval_det_annos, map_name_to_kitti=map_name_to_kitti)
            for info in eval_gt_annos:
                # info.keys: 'name', 'gt_boxes_lidar'
                cls_sav_flag = np.zeros(info['name'].shape[0], dtype=bool)
                for cls_int in map_name_to_kitti:
                    cls_sav_flag |= info['name'] == cls_int
                info['name'] = info['name'][cls_sav_flag]
                info['gt_boxes_lidar'] = info['gt_boxes_lidar'][cls_sav_flag]
                if info.get('gt_boxes_2d', False):
                    info['gt_boxes_2d'] = info['gt_boxes_2d'][cls_sav_flag]

                
            kitti_utils.transform_annotations_to_kitti_format(
                eval_gt_annos, map_name_to_kitti=map_name_to_kitti,
                info_with_fakelidar=self.dataset_cfg.get('INFO_WITH_FAKELIDAR', False)
            )
            kitti_class_names = [map_name_to_kitti[x] for x in class_names if x in map_name_to_kitti]
            ap_result_str, ap_dict = kitti_eval.get_official_eval_result(
                gt_annos=eval_gt_annos, dt_annos=eval_det_annos, current_classes=kitti_class_names
            )
            return ap_result_str, ap_dict

        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = [copy.deepcopy(info['annos']) for info in self.a2d2_infos]

        if kwargs['eval_metric'] == 'kitti':
            ap_result_str, ap_dict = kitti_eval(eval_det_annos, eval_gt_annos)
        else:
            raise NotImplementedError

        return ap_result_str, ap_dict

    def get_infos(self, class_names, num_workers=4, has_label=True, sample_id_list=None, num_features=3, count_inside_pts=True):
        import concurrent.futures as futures

        def process_single_scene(sample_idx):
            print('%s sample_idx: %s' % (self.split, sample_idx))
            info = {}
            pc_info = {'num_features': num_features, 'lidar_idx': sample_idx}
            info['point_cloud'] = pc_info

            image_info = {'image_idx': sample_idx, 'image_shape': self.get_image_shape(sample_idx)}
            info['image'] = image_info

            # calib = self.get_calib()
            # calib_info = {}
            # info['calib'] = calib_info

            if has_label:
                annotations = {}
                gt_boxes_lidar, name, gt_boxes_2d = self.get_label(sample_idx)
                annotations['name'] = name
                annotations['gt_boxes_lidar'] = gt_boxes_lidar[:, :7]
                annotations['bbox'] = gt_boxes_2d
                annotations['index'] = np.array(list(range(gt_boxes_lidar.shape[0])), dtype=np.int32)

                info['annos'] = annotations

                if count_inside_pts:
                    num_objects = gt_boxes_lidar.shape[0]
                    points = self.get_lidar(sample_idx)
                    corners_lidar = box_utils.boxes_to_corners_3d(gt_boxes_lidar)
                    num_points_in_gt = -np.ones(num_objects, dtype=np.int32)

                    for k in range(num_objects):
                        flag = box_utils.in_hull(points[:, 0:3], corners_lidar[k])
                        num_points_in_gt[k] = flag.sum()
                    annotations['num_points_in_gt'] = num_points_in_gt
                

            return info

        sample_id_list = sample_id_list if sample_id_list is not None else self.sample_id_list

        # create a thread pool to improve the velocity
        with futures.ThreadPoolExecutor(num_workers) as executor:
            infos = executor.map(process_single_scene, sample_id_list)
        return list(infos)

    def create_groundtruth_database(self, info_path=None, used_classes=None, split='train'):
        import torch

        database_save_path = Path(self.root_path) / ('gt_database' if split == 'train' else ('gt_database_%s' % split))
        db_info_save_path = Path(self.root_path) / ('a2d2_dbinfos_%s.pkl' % split)

        database_save_path.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}

        with open(info_path, 'rb') as f:
            infos = pickle.load(f)

        for k in range(len(infos)):
            print('gt_database sample: %d/%d' % (k + 1, len(infos)))
            info = infos[k]
            sample_idx = info['point_cloud']['lidar_idx']
            points = self.get_lidar(sample_idx)
            annos = info['annos']
            names = annos['name']
            bbox = annos['bbox']
            gt_boxes = annos['gt_boxes_lidar']

            num_obj = gt_boxes.shape[0]
            point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                torch.from_numpy(points[:, 0:3]), torch.from_numpy(gt_boxes)
            ).numpy()  # (nboxes, npoints)

            for i in range(num_obj):
                filename = '%s_%s_%d.bin' % (sample_idx, names[i], i)
                filepath = database_save_path / filename
                gt_points = points[point_indices[i] > 0]

                gt_points[:, :3] -= gt_boxes[i, :3]
                with open(filepath, 'w') as f:
                    gt_points.tofile(f)

                if (used_classes is None) or names[i] in used_classes:
                    db_path = str(filepath.relative_to(self.root_path))  # gt_database/xxxxx.bin
                    db_info = {'name': names[i], 'path': db_path, 'gt_idx': i, 'image_idx': sample_idx,
                               'box3d_lidar': gt_boxes[i], 'bbox': bbox[i], 'num_points_in_gt': gt_points.shape[0]}
                    if names[i] in all_db_infos:
                        all_db_infos[names[i]].append(db_info)
                    else:
                        all_db_infos[names[i]] = [db_info]

        # Output the num of all classes in database
        for k, v in all_db_infos.items():
            print('Database %s: %d' % (k, len(v)))

        with open(db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f)

    # @staticmethod
    # def create_label_file_with_name_and_box(class_names, gt_names, gt_boxes, save_label_path):
    #     with open(save_label_path, 'w') as f:
    #         for idx in range(gt_boxes.shape[0]):
    #             boxes = gt_boxes[idx]
    #             name = gt_names[idx]
    #             if name not in class_names:
    #                 continue
    #             line = "{x} {y} {z} {l} {w} {h} {angle} {name}\n".format(
    #                 x=boxes[0], y=boxes[1], z=(boxes[2]), l=boxes[3],
    #                 w=boxes[4], h=boxes[5], angle=boxes[6], name=name
    #             )
    #             f.write(line)


def create_a2d2_infos(dataset_cfg, class_names, data_path, save_path, workers=4):
    dataset = A2D2Dataset(
        dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path,
        training=False, logger=common_utils.create_logger()
    )
    train_split, val_split = 'train', 'val'
    num_features = len(dataset_cfg.POINT_FEATURE_ENCODING.src_feature_list)

    train_filename = save_path / ('a2d2_infos_%s.pkl' % train_split)
    val_filename = save_path / ('a2d2_infos_%s.pkl' % val_split)

    print('------------------------Start to generate data infos------------------------')

    dataset.set_split(train_split)
    a2d2_infos_train = dataset.get_infos(
        class_names, num_workers=workers, has_label=True, num_features=num_features
    )
    with open(train_filename, 'wb') as f:
        pickle.dump(a2d2_infos_train, f)
    print('A2D2 info train file is saved to %s' % train_filename)

    dataset.set_split(val_split)
    a2d2_infos_val = dataset.get_infos(
        class_names, num_workers=workers, has_label=True, num_features=num_features
    )
    with open(val_filename, 'wb') as f:
        pickle.dump(a2d2_infos_val, f)
    print('A2D2 info val file is saved to %s' % val_filename)

    print('------------------------Start create groundtruth database for data augmentation------------------------')
    dataset.set_split(train_split)
    dataset.create_groundtruth_database(train_filename, split=train_split)
    print('------------------------Data preparation done------------------------')


if __name__ == '__main__':
    import sys

    if sys.argv.__len__() > 1 and sys.argv[1] == 'create_a2d2_infos':
        import yaml
        from pathlib import Path
        from easydict import EasyDict

        dataset_cfg = EasyDict(yaml.safe_load(open(sys.argv[2])))
        # ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
        ROOT_DIR = Path('/xxx')
        create_a2d2_infos(
            dataset_cfg=dataset_cfg,
            class_names=['Car', 'Pedestrian', 'Cyclist'],
            data_path=ROOT_DIR / 'data' / 'a2d2',
            save_path=ROOT_DIR / 'data' / 'a2d2',
        )