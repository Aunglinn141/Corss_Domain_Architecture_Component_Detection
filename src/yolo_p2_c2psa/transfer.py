"""Pretrained-weight transfer helpers for the YOLO26m P2 graph.

Adding the P2 path inserts neck layers before the original YOLO26 P3-P5 neck
and shifts the OBB26 head. A blind ``model.load(weights)`` would therefore
reuse tensors by numeric layer index incorrectly. This module copies only
shape-compatible tensors and explicitly remaps the shifted layers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _find_single_layer(model: Any, class_name: str) -> int:
    modules = getattr(model, "model", None)
    if modules is None:
        raise RuntimeError("The Ultralytics model graph is not available.")
    indices = [i for i, module in enumerate(modules) if type(module).__name__ == class_name]
    if len(indices) != 1:
        raise RuntimeError(f"Expected one {class_name} layer, found {indices}.")
    return indices[0]


def load_obb26_p2_weights(model: Any, weights: str | Path) -> dict[str, int]:
    """Load compatible YOLO26 OBB weights into the P2+C2PSA graph.

    Parameters
    ----------
    model:
        An instantiated ``ultralytics.YOLO`` object built from
        ``configs/yolo26m-p2-c2psa.yaml``.
    weights:
        Official or previously trained YOLO26 OBB checkpoint.

    Returns
    -------
    dict
        Counts for transferred, skipped, and missing tensors.
    """

    from ultralytics import YOLO

    source_model = YOLO(str(weights))
    source_head = _find_single_layer(source_model.model, "OBB26")
    target_head = _find_single_layer(model.model, "OBB26")

    # Official YOLO26 P3-P5 graph -> this repository's P2-P5 graph.
    layer_map = {source_head: target_head}
    if source_head == 23 and target_head == 29:
        layer_map.update({17: 23, 19: 25, 20: 26, 22: 28})

    source_state = source_model.model.state_dict()
    target_state = model.model.state_dict()
    transferred: dict[str, Any] = {}
    scale_lists = {
        "cv2",
        "cv3",
        "cv4",
        "one2one_cv2",
        "one2one_cv3",
        "one2one_cv4",
    }

    for source_key, value in source_state.items():
        target_key = source_key
        for source_index, target_index in layer_map.items():
            prefix = f"model.{source_index}."
            if not source_key.startswith(prefix):
                continue

            relative_key = source_key[len(prefix) :]
            # The target head has a new P2 scale at index 0. Shift the
            # official P3/P4/P5 prediction lists by one position.
            if source_index == source_head and source_head == 23 and target_head == 29:
                parts = relative_key.split(".")
                if len(parts) > 1 and parts[0] in scale_lists and parts[1].isdigit():
                    parts[1] = str(int(parts[1]) + 1)
                    relative_key = ".".join(parts)
            target_key = f"model.{target_index}." + relative_key
            break

        target_value = target_state.get(target_key)
        if target_value is not None and getattr(target_value, "shape", None) == getattr(value, "shape", None):
            transferred[target_key] = value

    incompatible = model.model.load_state_dict(transferred, strict=False)
    # Ultralytics rebuilds the graph during training when no checkpoint marker
    # exists. Keep the remapped in-memory model for stage-2 or fine-tuning runs.
    model.ckpt = {"model": model.model}
    return {
        "transferred": len(transferred),
        "target_tensors": len(target_state),
        "missing": len(incompatible.missing_keys),
        "skipped": len(source_state) - len(transferred),
        "source_head": source_head,
        "target_head": target_head,
    }

