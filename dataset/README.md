# Dataset

Put local training data here before running `train.py`.

Expected layout:

```txt
dataset/images/   image files such as .jpg, .jpeg, .png
dataset/labels/   YOLO label files with matching names
```

Example:

```txt
dataset/images/example.jpg
dataset/labels/example.txt
```

YOLO label rows use this format:

```txt
class_id x_center y_center width height
```

The current local experiment used 105 image-label pairs. Generated training outputs are written to `final_results/` and are intentionally not committed.
