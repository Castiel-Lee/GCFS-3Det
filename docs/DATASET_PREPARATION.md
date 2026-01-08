# Getting Started
The dataset configs are located within [tools/cfgs/dataset_configs](../tools/cfgs/dataset_configs), 
and the model configs are located within [tools/cfgs/GCFS](../tools/cfgs/GCFS) for different transfering settings. 

## Few-shot Datasets 
Due to constrains of the liscenes adopted by KITTI, Argo2, and A2D2, we provide *scripts* to generate FS-datasets in [fs-datasets](../fs-datasets). Note that you need to preprocess the raw data and obtain train pkl files before generating the fs-datasets.

## Full-shot (Source Training and Target Evaluation) Datasets Preparation

### KITTI & NuScenes Datasets

For KITTI and NuScenes dataset preparation, please follow official instructions in [OpenPCDet](https://github.com/open-mmlab/OpenPCDet/blob/master/docs/GETTING_STARTED.md) to acquire data info files.

### A2D2 Dataset
* Please download the official [A2D2: 3D Bounding Boxes](https://www.a2d2.audi/en/download/) dataset and reorganize the downloaded files as follows:

```
data
├── A2D2
│   │── ImageSets
|   |   ├──train.txt & val.txt 
│   │── training
│   │   ├──img (xxx.png) & lidar (xxx.npz) & label3D (xxx.json)
│   │── cams_lidars.json
```

* Generate the data infos by running the following command: 
```bash 
python -m pcdet.datasets.a2d2.a2d2_dataset create_a2d2_infos tools/cfgs/dataset_configs/a2d2/FS-DA/a2d2_dataset_gt_D_p.yaml.yaml
```

We review datasets and find specific samples with misaligned annotations given cross-dataset experiments. After info file generation, please delete these val samples according to [data/a2d2/ImageSets/del_val.txt](../data/a2d2/ImageSets/del_val.txt) for fair model evaluation.

### Waymo Open Dataset
* Please download the official [Waymo Open Dataset](https://waymo.com/open/download/), 
including the training data `training_0000.tar~training_0031.tar` and the validation 
data `validation_0000.tar~validation_0007.tar`, and organize files as below:

```
data
├── waymo
│   │── ImageSets
│   │── raw_data
│   │   │── segment-xxxxxxxx.tfrecord
|   |   |── ...
|   |── waymo_processed_data_v0_5_0
│   │   │── segment-xxxxxxxx/
|   |   |── ...
```

* Install the official `waymo-open-dataset` and run the following command to process dataset:
```bash 
python -m pcdet.datasets.waymo.waymo_dataset --func create_waymo_infos \
    --cfg_file tools/cfgs/dataset_configs/waymo/FS-DA/waymo_dataset_gt_D_p.yaml
```
* For the Waymo dataset, we leverage the subset of **23,750 frames** containing point-level object annotations (segmentation). This allows us to subdivide the original coarse-grained `Vehicle` class into fine-grained categories: `{Car, Truck, Bus, Motorcycle, Bicycle}`. Please run the following command to preprocess the data and generate the necessary info files:

```bash
cd waymo-seg
python -m waymo_dataset --cfg_file waymo_dataset.yaml
```

### Argoverse 2 (Argo2) Dataset
* Download the **Argoverse 2 Sensor Dataset** from the [official website](https://www.argoverse.org/av2.html#download-link), and then extract and organize them as follow:

```
data
├── argo2
│   │── Images
|   |   ├── seq tokens (e.g., 02678d04-cc9f-3148-9f95-1ba66347dff9)
|   |   |   ├── camera positions (e.g., ring_front_center)
│   │── train
│   |   │── seq tokens (e.g., 02678d04-cc9f-3148-9f95-1ba66347dff9)
│   |   |   │── calibration
│   |   |   │── map
│   |   |   │── sensors
│   |   |   |   │── cameras
|   |   |   |   |   ├── camera positions (e.g., ring_front_center)
│   |   |   |   │── lidar
│   |   |   │── annotations.feather
│   |   |   │── city_SE3_egovehicle.feather
│   │── val
│   |   │── ......
│   │── test
│   |   │── ......
```

* Process the dataset and generate the data infos by running the following command: 
```bash 
python pcdet.datasets.argo2.argo2_dataset.py
```

* After the info file generation, please run the following command to organize the class labels of dataset:
``` bash
python pcdet.datasets.argo2.reorg_argo2_obj_name.py
```