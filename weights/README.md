# Weights

Large checkpoint files are intentionally not committed to the source
repository. Put pretrained or released checkpoints in this directory locally,
or download them from the GitHub Releases page you create.

Suggested names:

```text
weights/yolo11m-obb.pt
weights/yolo26m-obb.pt
weights/yolo11m-p2-c2psa-best.pt
weights/yolo26m-p2-c2psa-best.pt
```

The stock YOLO26m-OBB checkpoint must be loaded with
`--remap-obb26-head` when initializing `yolo26m-p2-c2psa.yaml`.

