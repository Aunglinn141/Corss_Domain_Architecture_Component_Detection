# YOLO11m-P2+C2PSA & YOLO26m-P2+C2PSA

基于 Ultralytics 的两个四尺度旋转目标检测模型：

- **YOLO11m-P2+C2PSA**：YOLO11m 主干，加入 P2/4 高分辨率检测分支，在 P5/32 使用 C2PSA。
- **YOLO26m-P2+C2PSA**：YOLO26m 主干，保留 YOLO26 的 SPPF/C2PSA 与端到端 OBB26 检测头，加入 P2/4 分支。

这里的 `P2` 指增加了 stride=4 的特征尺度，`C2PSA` 指注意力增强模块；两个模型都输出 P2、P3、P4、P5 四个 OBB 尺度。它们不是普通水平框检测模型，标签应使用四点旋转框格式。

## 目录

```text
YOLO_P2_C2PSA_OpenSource/
├── configs/
│   ├── yolo11m-p2-c2psa.yaml
│   ├── yolo26m-p2-c2psa.yaml
│   └── data.example.yaml
├── scripts/
│   ├── inspect_model.py
│   ├── predict.py
│   └── train.py
├── src/yolo_p2_c2psa/
│   └── transfer.py
├── docs/model_notes.md
├── examples/
├── weights/README.md
├── requirements.txt
├── LICENSE
└── NOTICE.md
```

## 安装

建议使用 Python 3.10–3.12，并在干净环境中安装依赖：

```bash
pip install -r requirements.txt
```

`yolo26m-p2-c2psa.yaml` 需要安装一个提供 `YOLO26`/`OBB26` 的 Ultralytics 构建版本。发布正式复现实验时，请把你实际使用的 Ultralytics、PyTorch、CUDA 版本写入仓库的 release notes 或锁定文件。

## 数据格式

复制 `configs/data.example.yaml`，修改 `path` 和类别名称。数据目录应类似：

```text
obb_dataset/
├── images/train/
├── images/val/
├── images/test/
├── labels/train/
├── labels/val/
└── labels/test/
```

每个标签文件使用：

```text
class x1 y1 x2 y2 x3 y3 x4 y4
```

其中坐标是归一化后的四个多边形点，类别编号从 `0` 开始。模型 YAML 中默认 `nc: 6`；如果类别数不同，请同步修改对应 YAML 的 `nc`。

## 先检查模型结构

```bash
python scripts/inspect_model.py --model configs/yolo11m-p2-c2psa.yaml
python scripts/inspect_model.py --model configs/yolo26m-p2-c2psa.yaml
```

如果第二条命令提示找不到 `OBB26`，说明当前 Ultralytics 版本不兼容，需要换成包含 YOLO26/OBB26 的构建版本。

## 训练

### YOLO11m-P2+C2PSA

```bash
python scripts/train.py \
  --model configs/yolo11m-p2-c2psa.yaml \
  --data /path/to/obb_dataset.yaml \
  --weights /path/to/yolo11m-obb.pt \
  --epochs 300 \
  --imgsz 960 \
  --batch 4 \
  --device 0 \
  --name yolo11m-p2-c2psa
```

### YOLO26m-P2+C2PSA

```bash
python scripts/train.py \
  --model configs/yolo26m-p2-c2psa.yaml \
  --data /path/to/obb_dataset.yaml \
  --weights /path/to/yolo26m-obb.pt \
  --remap-obb26-head \
  --epochs 300 \
  --imgsz 960 \
  --batch 3 \
  --device 0 \
  --name yolo26m-p2-c2psa
```

YOLO26 的 `--remap-obb26-head` 不能省略：P2 分支插入后，原 YOLO26 P3–P5 颈部和 OBB26 检测头的层索引发生变化。迁移脚本只复制形状兼容的权重，并把官方 P3/P4/P5 预测分支平移到目标模型的 P3/P4/P5 分支；新增加的 P2 分支保持随机初始化。

Windows 用户也可以直接运行 `examples/train_yolo11m.bat` 和 `examples/train_yolo26m.bat`。

## 推理

```bash
python scripts/predict.py \
  --weights runs/yolo11m-p2-c2psa/weights/best.pt \
  --source /path/to/images \
  --imgsz 1280 \
  --conf 0.25 \
  --device 0 \
  --save-txt
```

## 开源说明

本仓库包含两个自定义模型图、训练/推理入口、YOLO26 P2 权重迁移逻辑和数据格式说明；不会把数据集、实验输出或大体积权重混入 Git 历史。基础算子由兼容的 Ultralytics runtime 提供，具体第三方许可证请见 `NOTICE.md` 并遵守所安装版本的许可条款。

## Citation

如果你在论文或项目中使用这两个结构，请补充你的论文、数据集和权重发布链接后再提交 GitHub。当前仓库不虚构论文 DOI、作者信息或评测结果。

