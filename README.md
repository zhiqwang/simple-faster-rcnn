# Object detection reference training scripts

Note: This repo is a copy of [pytorch/torchvision](http://github.com/pytorch/vision/blob/master/torchvision/models/detection).

This folder contains reference training scripts for object detection.
They serve as a log of how to train specific models, as provide baseline
training and evaluation scripts to quickly bootstrap research.

Except otherwise noted, all models have been trained on 8x V100 GPUs.

## Supported Model Structure

### Faster R-CNN

```bash
python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py\
    --dataset coco --model fasterrcnn_resnet50_fpn --epochs 26\
    --lr-steps 16 22 --aspect-ratio-group-factor 3
```

### Mask R-CNN

```bash
python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py\
    --dataset coco --model maskrcnn_resnet50_fpn --epochs 26\
    --lr-steps 16 22 --aspect-ratio-group-factor 3
```

### Keypoint R-CNN

```bash
python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py\
    --dataset coco_kp --model keypointrcnn_resnet50_fpn --epochs 46\
    --lr-steps 36 43 --aspect-ratio-group-factor 3
```
