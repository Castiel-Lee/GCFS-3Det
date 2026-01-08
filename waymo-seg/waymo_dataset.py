import os
import pickle
import copy
import numpy as np
import multiprocessing
import SharedArray
from tqdm import tqdm
from pathlib import Path
from functools import partial
import logging

# from .utils import  common_utils

class WaymoDataset():
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        self.root_path = root_path
        self.dataset_cfg = dataset_cfg
        self.training = training
        self.data_path = self.root_path / self.dataset_cfg.PROCESSED_DATA_TAG
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_sequence_list = [x.strip() for x in open(split_dir).readlines()]
        self.infos = []

    @property
    def mode(self):
        return 'train' if self.training else 'test'


    def set_split(self, split):
        self.split = split
        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_sequence_list = [x.strip() for x in open(split_dir).readlines()]
        self.infos = []
    
    
    @staticmethod
    def check_sequence_name_with_all_version(sequence_file):
        if not sequence_file.exists():
            found_sequence_file = sequence_file
            for pre_text in ['training', 'validation', 'testing']:
                if not sequence_file.exists():
                    temp_sequence_file = Path(str(sequence_file).replace('segment', pre_text + '_segment'))
                    if temp_sequence_file.exists():
                        found_sequence_file = temp_sequence_file
                        break
            if not found_sequence_file.exists():
                found_sequence_file = Path(str(sequence_file).replace('_with_camera_labels', ''))
            if found_sequence_file.exists():
                sequence_file = found_sequence_file
        return sequence_file

    def get_infos(self, raw_data_path, save_path, num_workers=multiprocessing.cpu_count(), has_label=True, sampled_interval=1, update_info_only=False):
        import waymo_utils
        print('---------------The waymo sample interval is %d, total sequecnes is %d-----------------'
              % (sampled_interval, len(self.sample_sequence_list)))

        process_single_sequence = partial(
            waymo_utils.process_single_sequence,
            save_path=save_path, sampled_interval=sampled_interval, has_label=has_label, update_info_only=update_info_only
        )
        sample_sequence_file_list = [
            self.check_sequence_name_with_all_version(raw_data_path / sequence_file)
            for sequence_file in self.sample_sequence_list
        ]

        num_workers = 1
        # print(num_workers)
        if num_workers > 1:
            with multiprocessing.Pool(num_workers) as p:
                sequence_infos = list(tqdm(p.imap(process_single_sequence, sample_sequence_file_list),
                                           total=len(sample_sequence_file_list)))
        elif num_workers == 1:
            sequence_infos = [process_single_sequence(seq) for seq in tqdm(sample_sequence_file_list)]
            # sequence_infos = [process_single_sequence(sample_sequence_file_list[0])] 

        
        # all_sequences_infos = [item for infos in sequence_infos for item in infos]
        # for info in all_sequences_infos[:3]:
        #     # Save np.ndarray debug message & drop
        #     with open(f'./data/{info["frame_id"]}.pkl', "wb") as f:
        #         pickle.dump({
        #             "debug_points": info["annos"]["debug_points"],
        #             "debug_seg_labels": info["annos"]["debug_seg_labels"],
        #             "debug_boxes": info["annos"]["debug_boxes"]
        #         }, f)
        # all_sequences_infos.append(summary)
        # return all_sequences_infos

def create_logger(log_file=None, rank=0, log_level=logging.INFO):
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level if rank == 0 else 'ERROR')
    formatter = logging.Formatter('%(asctime)s  %(levelname)5s  %(message)s')
    console = logging.StreamHandler()
    console.setLevel(log_level if rank == 0 else 'ERROR')
    console.setFormatter(formatter)
    logger.addHandler(console)
    if log_file is not None:
        file_handler = logging.FileHandler(filename=log_file)
        file_handler.setLevel(log_level if rank == 0 else 'ERROR')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    logger.propagate = False
    return logger
    
def create_waymo_infos(dataset_cfg, class_names, data_path, save_path,
                       raw_data_tag='raw_data', processed_data_tag='waymo_processed_data',
                       workers=min(16, multiprocessing.cpu_count()), update_info_only=False):
    dataset = WaymoDataset(
        dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path,
        # training=False, logger=common_utils.create_logger()
        training=False, logger=create_logger()
    )
    train_split, val_split = 'train_org', 'val'

    train_filename = save_path / ('%s_infos_%s.pkl' % (processed_data_tag, train_split))
    val_filename = save_path / ('%s_infos_%s.pkl' % (processed_data_tag, val_split))

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    print('---------------Start to generate data infos---------------')

    dataset.set_split(train_split)
    waymo_infos_train = dataset.get_infos(
        raw_data_path=data_path / raw_data_tag,
        save_path=save_path / processed_data_tag, num_workers=workers, has_label=True,
        sampled_interval=1, update_info_only=update_info_only
    )
    # with open(train_filename, 'wb') as f:
    #     pickle.dump(waymo_infos_train, f)
    print('----------------Waymo info train file is saved to %s----------------' % train_filename)

    # dataset.set_split(val_split)
    # waymo_infos_val = dataset.get_infos(
    #     raw_data_path=data_path / raw_data_tag,
    #     save_path=save_path / processed_data_tag, num_workers=workers, has_label=True,
    #     sampled_interval=1, update_info_only=update_info_only
    # )
    # with open(val_filename, 'wb') as f:
    #     pickle.dump(waymo_infos_val, f)
    # print('----------------Waymo info val file is saved to %s----------------' % val_filename)

    if update_info_only:
        return

    # print('---------------Start create groundtruth database for data augmentation---------------')
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # dataset.set_split(train_split)
    # dataset.create_groundtruth_database(
    #     info_path=train_filename, save_path=save_path, split='train', sampled_interval=1,
    #     used_classes=['Vehicle', 'Pedestrian', 'Cyclist'], processed_data_tag=processed_data_tag
    # )
    print('---------------Data preparation Done---------------')



# python -m waymo_dataset --cfg_file waymo_dataset_gt_D_p.yaml 
if __name__ == '__main__':
    import argparse
    import yaml
    from easydict import EasyDict

    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config of dataset')
    parser.add_argument('--func', type=str, default='create_waymo_infos', help='')
    parser.add_argument('--processed_data_tag', type=str, default='waymo_processed_data_v0_5_0', help='')
    parser.add_argument('--update_info_only', action='store_true', default=False, help='')
    parser.add_argument('--use_parallel', action='store_true', default=False, help='')
    parser.add_argument('--wo_crop_gt_with_tail', action='store_true', default=False, help='')

    args = parser.parse_args()

    ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()

    try:
        yaml_config = yaml.safe_load(open(args.cfg_file), Loader=yaml.FullLoader)
    except:
        yaml_config = yaml.safe_load(open(args.cfg_file))
    dataset_cfg = EasyDict(yaml_config)
    dataset_cfg.PROCESSED_DATA_TAG = args.processed_data_tag
    create_waymo_infos(
        dataset_cfg=dataset_cfg,
        class_names=['Vehicle', 'Pedestrian', 'Cyclist'],
        data_path=ROOT_DIR / 'data' / 'waymo-seg',
        save_path=ROOT_DIR / 'data' / 'waymo-seg',
        raw_data_tag='raw_data',
        processed_data_tag=args.processed_data_tag,
        update_info_only=args.update_info_only,
        workers=16,
    )
