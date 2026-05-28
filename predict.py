from __future__ import annotations

import argparse
import os
from pathlib import Path

YOLO_CONFIG_DIR = Path.cwd() / "final_results" / "ultralytics_config"
YOLO_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(YOLO_CONFIG_DIR.resolve()))

import yaml
from ultralytics import YOLO


def load_config() -> dict:
    with Path("semi_config.yaml").open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    config = load_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="final_results/stage2_best.pt")
    parser.add_argument("--source", default="dataset/images")
    parser.add_argument("--conf", type=float, default=config.get("prediction_confidence_threshold", config["confidence_threshold"]))
    args = parser.parse_args()

    model = YOLO(args.weights)
    model.predict(
        source=args.source,
        conf=args.conf,
        imgsz=config["imgsz"],
        device=config.get("device", 0),
        save=True,
        save_txt=True,
        project=str((Path("final_results") / "predict").resolve()),
        name="predict",
        exist_ok=True,
    )


if __name__ == "__main__":
    main()
