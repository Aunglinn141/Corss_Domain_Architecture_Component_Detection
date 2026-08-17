"""Train either YOLO11m-P2+C2PSA or YOLO26m-P2+C2PSA.

Examples:
    python scripts/train.py --model configs/yolo11m-p2-c2psa.yaml \
        --data configs/data.example.yaml --weights yolo11m-obb.pt

    python scripts/train.py --model configs/yolo26m-p2-c2psa.yaml \
        --data configs/data.example.yaml --weights yolo26m-obb.pt \
        --remap-obb26-head
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a P2+C2PSA Ultralytics OBB detector.")
    parser.add_argument("--model", required=True, help="Model YAML path.")
    parser.add_argument("--data", required=True, help="Ultralytics OBB dataset YAML path.")
    parser.add_argument("--weights", default=None, help="Optional pretrained or previous checkpoint.")
    parser.add_argument(
        "--remap-obb26-head",
        action="store_true",
        help="Use the explicit YOLO26 official-to-P2 tensor remapping before training.",
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="0", help="CUDA index, e.g. 0, or cpu.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--project", default="runs")
    parser.add_argument("--name", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--close-mosaic", type=int, default=20)
    parser.add_argument("--optimizer", default="AdamW")
    parser.add_argument("--lr0", type=float, default=0.001)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--warmup-epochs", type=float, default=10.0)
    parser.add_argument("--cos-lr", action="store_true")
    parser.add_argument("--mosaic", type=float, default=0.5)
    parser.add_argument("--mixup", type=float, default=0.0)
    parser.add_argument("--cutmix", type=float, default=0.1)
    parser.add_argument("--copy-paste", type=float, default=0.15)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from ultralytics import YOLO

    model_path = Path(args.model).resolve()
    data_path = Path(args.data).resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"Model YAML not found: {model_path}")
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {data_path}")

    model = YOLO(str(model_path))
    transfer_report = None
    if args.weights:
        weights_path = Path(args.weights).resolve()
        if not weights_path.is_file():
            raise FileNotFoundError(f"Weights not found: {weights_path}")
        if args.remap_obb26_head:
            from yolo_p2_c2psa import load_obb26_p2_weights

            transfer_report = load_obb26_p2_weights(model, weights_path)
            print(json.dumps(transfer_report, indent=2))
        else:
            model.load(str(weights_path))

    name = args.name or model_path.stem
    train_kwargs = {
        "data": str(data_path),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "workers": args.workers,
        "project": str(Path(args.project).resolve()),
        "name": name,
        "exist_ok": True,
        "seed": args.seed,
        "task": "obb",
        "patience": args.patience,
        "close_mosaic": args.close_mosaic,
        "optimizer": args.optimizer,
        "lr0": args.lr0,
        "lrf": args.lrf,
        "warmup_epochs": args.warmup_epochs,
        "cos_lr": args.cos_lr,
        "mosaic": args.mosaic,
        "mixup": args.mixup,
        "cutmix": args.cutmix,
        "copy_paste": args.copy_paste,
        "conf": args.conf,
        "iou": args.iou,
        "max_det": args.max_det,
    }
    results = model.train(**train_kwargs)
    save_dir = Path(getattr(getattr(model, "trainer", None), "save_dir", Path(args.project) / name))
    save_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"model": str(model_path), "data": str(data_path), "train": vars(args)}
    if transfer_report is not None:
        manifest["transfer"] = transfer_report
    metrics = getattr(results, "results_dict", None)
    if metrics:
        manifest["metrics"] = dict(metrics)
    (save_dir / "open_source_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(f"Training finished. Results: {save_dir}")


if __name__ == "__main__":
    main()

