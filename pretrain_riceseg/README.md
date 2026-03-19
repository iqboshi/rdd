# RiceSEG Leaf/Non-Leaf Pretraining

This folder is an independent semantic-pretraining branch for RiceSEG.

## Files

- `dataset.py`: RiceSEG data scan, train/val split, 6-class -> leaf/non-leaf mapping, preview export.
- `transforms.py`: train/val semantic-seg transforms.
- `model.py`: Swin-Tiny + FPN semantic pretrain model.
- `train.py`: training loop, logs, visualization, checkpoints, backbone export.
- `configs/default_leaf_pretrain.json`: default hyperparameter snapshot.

## Label Mapping

- leaf (`1`): class ids `{1, 2}` = green vegetation + senescent vegetation
- non-leaf (`0`): class ids `{0, 3, 4, 5}` = background + panicle + weeds + duckweed

If your class ids differ, override `--leaf_class_ids`.

## Run

```powershell
& 'C:\Users\23581\Documents\some_scientific_ideas\.conda\envs\rdd-dev\python.exe' pretrain_riceseg\train.py --data_root data\external_data --device cuda
```

## Output

Each run creates: `pretrain_riceseg/outputs/exp_YYYYMMDD_HHMMSS/`

- `checkpoints/latest.pth`
- `checkpoints/best.pth`
- `checkpoints/best_backbone_for_instance.pth` (keys aligned to instance model backbone prefix)
- `checkpoints/best_backbone_module.pth` (direct `model.backbone` state dict)
- `logs/train_log.csv`
- `vis/train` and `vis/val`
