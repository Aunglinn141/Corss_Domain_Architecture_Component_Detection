"""Run OBB prediction with a trained checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict with a YOLO P2+C2PSA OBB checkpoint.")
    parser.add_argument("--weights", required=True, help="Path to best.pt or another checkpoint.")
    parser.add_argument("--source", required=True, help="Image, directory, video, or glob.")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default="runs/predict")
    parser.add_argument("--name", default="p2_c2psa")
    parser.add_argument("--save-txt", action="store_true")
    parser.add_argument("--save-conf", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from ultralytics import YOLO

    weights = Path(args.weights).resolve()
    if not weights.is_file():
        raise FileNotFoundError(f"Weights not found: {weights}")

    model = YOLO(str(weights))
    results = model.predict(
        source=args.source,
        task="obb",
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        project=str(Path(args.project).resolve()),
        name=args.name,
        exist_ok=True,
        save=True,
        save_txt=args.save_txt,
        save_conf=args.save_conf,
        verbose=True,
    )
    if results:
        print(f"Saved predictions under: {getattr(results[0], 'save_dir', 'the Ultralytics run directory')}")


if __name__ == "__main__":
    main()

