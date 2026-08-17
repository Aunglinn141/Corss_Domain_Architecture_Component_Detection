"""Instantiate a model YAML and print its parsed layer graph."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a P2+C2PSA model YAML.")
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    from ultralytics import YOLO

    path = Path(args.model).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    model = YOLO(str(path))
    model.info(verbose=True)
    print("\nParsed modules:")
    for index, module in enumerate(model.model.model):
        print(f"{index:>3}: {type(module).__name__}")


if __name__ == "__main__":
    main()

