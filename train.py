from __future__ import annotations

import argparse
import csv
import os
import random
import shutil
import stat
from pathlib import Path

YOLO_CONFIG_DIR = Path.cwd() / "final_results" / "ultralytics_config"
YOLO_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(YOLO_CONFIG_DIR.resolve()))

import yaml
from ultralytics import YOLO


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
RESULTS_DIR = Path("final_results")


def load_config() -> dict:
    with Path("semi_config.yaml").open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def reset_dir(path: Path) -> None:
    if path.exists():
        def remove_readonly(func, failed_path, _exc_info):
            os.chmod(failed_path, stat.S_IWRITE)
            func(failed_path)

        shutil.rmtree(path, onexc=remove_readonly)
    path.mkdir(parents=True, exist_ok=True)


def copy_pair(image_path: Path, label_path: Path, images_dir: Path, labels_dir: Path) -> None:
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image_path, images_dir / image_path.name)
    shutil.copy2(label_path, labels_dir / f"{image_path.stem}.txt")


def write_yaml(path: Path, train_images: Path, val_images: Path, names: list[str]) -> None:
    data = {
        "path": str(path.parent.resolve()),
        "train": str(train_images.resolve()),
        "val": str(val_images.resolve()),
        "test": str(val_images.resolve()),
        "nc": len(names),
        "names": names,
    }
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def list_labeled_pairs(dataset_dir: Path) -> list[tuple[Path, Path]]:
    pairs = []
    for image_path in sorted((dataset_dir / "images").glob("*")):
        if image_path.suffix.lower() not in IMAGE_EXTS:
            continue
        label_path = dataset_dir / "labels" / f"{image_path.stem}.txt"
        if label_path.exists() and label_path.read_text(encoding="utf-8").strip():
            pairs.append((image_path, label_path))
    return pairs


def make_empty_labels(images_dir: Path, labels_dir: Path) -> None:
    labels_dir.mkdir(parents=True, exist_ok=True)
    for image_path in images_dir.glob("*"):
        if image_path.suffix.lower() in IMAGE_EXTS:
            (labels_dir / f"{image_path.stem}.txt").write_text("", encoding="utf-8")


def find_best(run_dir: Path) -> Path:
    search_roots = [run_dir]
    if not run_dir.is_absolute():
        search_roots.append(Path("runs") / "detect" / run_dir)
    candidates = []
    for search_root in search_roots:
        candidates.extend(search_root.glob("**/weights/best.pt"))
    candidates = sorted(candidates, key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No best.pt found under {run_dir}")
    return candidates[-1]


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def export_final_artifacts(
    root: Path,
    stage1_best: Path,
    stage2_best: Path,
    metrics: dict,
    kept: int,
    split_counts: dict[str, int],
    config: dict,
) -> None:
    copy_if_exists(stage1_best, root / "stage1_best.pt")
    copy_if_exists(stage2_best, root / "stage2_best.pt")
    copy_if_exists(stage2_best.parent.parent / "results.csv", root / "stage2_results.csv")

    summary = "\n".join(
        [
            "YOLOv8n pseudo-labeling experiment",
            "",
            "Settings:",
            f"- human-labeled train images: {split_counts['labeled_train']}",
            f"- pseudo-labeled train images: {split_counts['unlabeled_train']}",
            f"- independent validation images: {split_counts['validation']}",
            f"- pseudo-label confidence threshold: {config['confidence_threshold']}",
            f"- prediction display confidence threshold: {config.get('prediction_confidence_threshold', config['confidence_threshold'])}",
            f"- model: {config['base_model']}",
            f"- epochs: {config['epochs']}",
            f"- imgsz: {config['imgsz']}",
            f"- device: GPU {config.get('device', 0)}",
            "",
            "Pseudo labels:",
            f"- pseudo label files: {split_counts['unlabeled_train']}",
            f"- boxes kept at confidence >= {config['confidence_threshold']}: {kept}",
            "",
            "Final stage2 metrics on independent validation set:",
            f"- Precision: {metrics['precision']}",
            f"- Recall: {metrics['recall']}",
            f"- mAP50: {metrics['mAP50']}",
            f"- mAP50-95: {metrics['mAP50-95']}",
            "",
        ]
    )
    (root / "summary.txt").write_text(summary, encoding="utf-8")


def model_path(config: dict) -> str:
    base = Path(config["base_model"])
    return str(base) if base.exists() else config["fallback_model"]


def write_pseudo_labels(model: YOLO, unlabeled_images: Path, pseudo_labels: Path, conf: float, imgsz: int) -> int:
    pseudo_labels.mkdir(parents=True, exist_ok=True)
    kept = 0
    results = model.predict(
        source=str(unlabeled_images),
        conf=conf,
        imgsz=imgsz,
        device=0,
        save=False,
        verbose=False,
    )
    for result in results:
        image_stem = Path(result.path).stem
        lines = []
        for box in result.boxes:
            confidence = float(box.conf[0])
            if confidence < conf:
                continue
            cls = int(box.cls[0])
            x, y, w, h = [float(v) for v in box.xywhn[0].tolist()]
            lines.append(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
            kept += 1
        (pseudo_labels / f"{image_stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return kept


def export_metrics(model: YOLO, data_yaml: Path, output_csv: Path, imgsz: int, device: int | str) -> dict:
    metrics = model.val(
        data=str(data_yaml),
        imgsz=imgsz,
        split="val",
        device=device,
        project=str((output_csv.parent / "validation").resolve()),
        name="metrics",
        exist_ok=True,
    )
    row = {
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "mAP50": float(metrics.box.map50),
        "mAP50-95": float(metrics.box.map),
    }
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        writer.writeheader()
        writer.writerow(row)
    print(row)
    return row


def run(require_exact: bool) -> None:
    config = load_config()
    random.seed(config["seed"])
    device = config.get("device", 0)

    dataset_dir = Path("dataset")
    pairs = list_labeled_pairs(dataset_dir)
    validation_count = int(config.get("validation_count", max(1, round(len(pairs) * 0.15))))
    requested = config["labeled_count"] + config["unlabeled_count"] + validation_count
    if len(pairs) < requested:
        message = (
            f"Need {requested} labeled pairs for the requested split, "
            f"but found {len(pairs)} in {dataset_dir}."
        )
        if require_exact:
            raise RuntimeError(message + " Restore the missing images/labels, then rerun prepare_dataset.py.")
        print("WARNING:", message, "Using all available data for a dry run split.")

    random.shuffle(pairs)
    validation_count = min(validation_count, max(0, len(pairs) - 1))
    train_pool = pairs[:-validation_count] if validation_count else pairs
    validation = pairs[-validation_count:] if validation_count else []
    labeled_count = min(config["labeled_count"], len(train_pool))
    unlabeled_count = min(config["unlabeled_count"], max(0, len(train_pool) - labeled_count))
    labeled = train_pool[:labeled_count]
    unlabeled = train_pool[labeled_count : labeled_count + unlabeled_count]
    split_counts = {
        "labeled_train": len(labeled),
        "unlabeled_train": len(unlabeled),
        "validation": len(validation),
    }

    root = RESULTS_DIR
    root.mkdir(parents=True, exist_ok=True)
    reset_dir(root / "stage1_dataset")
    reset_dir(root / "unlabeled")
    reset_dir(root / "pseudo_labels")
    reset_dir(root / "stage2_dataset")
    if not config.get("reuse_existing_models", False):
        reset_dir(root / "stage1")
        reset_dir(root / "stage2")

    for image_path, label_path in labeled:
        copy_pair(image_path, label_path, root / "stage1_dataset" / "images" / "train", root / "stage1_dataset" / "labels" / "train")

    for image_path, label_path in validation:
        copy_pair(image_path, label_path, root / "stage1_dataset" / "images" / "val", root / "stage1_dataset" / "labels" / "val")

    for image_path, _label_path in unlabeled:
        (root / "unlabeled" / "images").mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_path, root / "unlabeled" / "images" / image_path.name)
    make_empty_labels(root / "unlabeled" / "images", root / "unlabeled" / "labels")

    stage1_yaml = root / "stage1_dataset" / "data.yaml"
    write_yaml(stage1_yaml, root / "stage1_dataset" / "images" / "train", root / "stage1_dataset" / "images" / "val", config["names"])

    try:
        stage1_best = find_best(root / "stage1")
        print(f"Using existing stage1 model: {stage1_best}")
    except FileNotFoundError:
        stage1 = YOLO(model_path(config))
        stage1.train(
            data=str(stage1_yaml),
            epochs=config["epochs"],
            imgsz=config["imgsz"],
            device=device,
            project=str(root.resolve()),
            name="stage1",
            exist_ok=True,
            workers=0,
        )
        stage1_best = find_best(root / "stage1")
    stage1_model = YOLO(str(stage1_best))
    kept = write_pseudo_labels(
        stage1_model,
        root / "unlabeled" / "images",
        root / "pseudo_labels",
        config["confidence_threshold"],
        config["imgsz"],
    )
    print(f"Pseudo labels kept: {kept} boxes with confidence >= {config['confidence_threshold']}")

    for image_path, label_path in labeled:
        copy_pair(image_path, label_path, root / "stage2_dataset" / "images" / "train", root / "stage2_dataset" / "labels" / "train")

    for image_path, label_path in validation:
        copy_pair(image_path, label_path, root / "stage2_dataset" / "images" / "val", root / "stage2_dataset" / "labels" / "val")

    for image_path, _label_path in unlabeled:
        pseudo_path = root / "pseudo_labels" / f"{image_path.stem}.txt"
        copy_pair(image_path, pseudo_path, root / "stage2_dataset" / "images" / "train", root / "stage2_dataset" / "labels" / "train")

    stage2_yaml = root / "stage2_dataset" / "data.yaml"
    write_yaml(stage2_yaml, root / "stage2_dataset" / "images" / "train", root / "stage2_dataset" / "images" / "val", config["names"])

    stage2 = YOLO(model_path(config))
    stage2.train(
        data=str(stage2_yaml),
        epochs=config["epochs"],
        imgsz=config["imgsz"],
        device=device,
        project=str(root.resolve()),
        name="stage2",
        exist_ok=True,
        workers=0,
    )

    stage2_best = find_best(root / "stage2")
    metrics = export_metrics(YOLO(str(stage2_best)), stage2_yaml, root / "metrics.csv", config["imgsz"], device)
    export_final_artifacts(root, stage1_best, stage2_best, metrics, kept, split_counts, config)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-partial", action="store_true", help="Run even if fewer than 105 labeled pairs are available.")
    args = parser.parse_args()
    run(require_exact=not args.allow_partial)


if __name__ == "__main__":
    main()
