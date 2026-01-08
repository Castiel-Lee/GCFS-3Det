Run to process waymo relabeling, to get waymo with more detailed class annotations
```shell
python -m waymo_dataset --func create_waymo_infos \
    --cfg_file waymo_dataset.yaml
```