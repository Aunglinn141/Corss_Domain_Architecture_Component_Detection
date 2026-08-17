# Model notes

## YOLO11m-P2+C2PSA

The YOLO11 graph keeps the medium compound scale and exposes four OBB scales.
The high-resolution P2 branch is fused top-down and bottom-up with P3, P4 and
P5. C2PSA is placed after SPPF at P5/32, where the spatial resolution is low
enough for attention while the P2 branch preserves small-component detail.

## YOLO26m-P2+C2PSA

The YOLO26 graph uses `end2end: True`, `reg_max: 1`, the YOLO26 SPPF form, and
the `OBB26` head. The added P2 path is kept at 256 channels so the four scale
heads retain the intended YOLO26 width. The P2/P3/P4/P5 feature tensors are
fed to one OBB26 head at layer 29.

## What is included

The two YAML files are the complete architecture definitions for the custom
graphs. The basic operators (`Conv`, `C3k2`, `SPPF`, `C2PSA`, `OBB`, and
`OBB26`) are loaded from the compatible Ultralytics runtime rather than copied
into this repository. The only model-specific Python implementation here is
the YOLO26 pretrained-weight remapper needed after inserting P2.

## Label format

Each OBB label line contains nine values:

```text
class x1 y1 x2 y2 x3 y3 x4 y4
```

Coordinates are normalized to `[0, 1]`. The four points should describe the
oriented rectangle in the format expected by the installed Ultralytics OBB
dataset loader.

