# YOLOv8n Semi-Supervised Object Detection

This project uses pseudo-labeling for an eraser object detection dataset.

## Workflow

1. Prepare labeled images into `dataset/images` and `dataset/labels`.
2. Randomly split 60 labeled images as human-labeled training data.
3. Treat 30 images as unlabeled by ignoring their original labels.
4. Keep 15 labeled images as an independent validation set.
5. Train stage1 YOLOv8n on the 60 labeled training images.
6. Use stage1 `best.pt` to create pseudo labels for the unlabeled images.
7. Keep pseudo labels with confidence >= 0.3.
8. Merge human labels and pseudo labels, then train stage2.
9. Export Precision, Recall, mAP50, and mAP50-95 on the independent validation set.

## Dataset Layout

The dataset should be placed locally like this:

```txt
dataset/images/   image files
dataset/labels/   YOLO txt labels
```

Each image needs a same-name label file:

```txt
dataset/images/example.jpg
dataset/labels/example.txt
```

The repository does not include generated training outputs, virtual environments, or model weights. Those files are recreated locally when you run training.

## Commands

```powershell
pip install -r requirements.txt
python train.py
python predict.py --weights final_results/stage2_best.pt --source dataset/images
```

## Outputs

All training and prediction outputs are saved under `final_results/`.

- Best stage1 model: `final_results/stage1_best.pt`
- Best stage2 model: `final_results/stage2_best.pt`
- Final metrics: `final_results/metrics.csv`
- Prediction outputs: `final_results/predict/predict/`

## Current Local Experiment

The local experiment used 105 image-label pairs:

- 60 human-labeled training images
- 30 pseudo-labeled training images
- 15 independent validation images

The final local validation metrics were computed on the independent validation set, not on duplicated training images.
