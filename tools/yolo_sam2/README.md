# YOLO Box Prompt Generator for SAM2

This folder contains a small pipeline for turning existing SAM2 mask outputs into
a YOLO detector, then using YOLO boxes as automatic SAM2 prompts on new videos.

## Dependencies

Install the SAM2 repo dependencies first, then add the YOLO/runtime packages:

```bash
conda run -n sam2 python -m pip install ultralytics opencv-python-headless pillow numpy pyyaml
```

For training, install a CUDA-enabled PyTorch build if you want GPU acceleration.

## 1. Convert SAM2 Masks to YOLO Detect Data

```bash
conda run -n sam2 python tools/yolo_sam2/convert_sam2_masks_to_yolo.py \
  --input outputs/task1 \
  --output datasets/task1_yolo_detect \
  --task detect \
  --split-mode random \
  --train-ratio 0.8 \
  --seed 42 \
  --overwrite
```

The converter reads:

- `outputs/task1/labels.txt`
- `outputs/task1/file-*_sam2_masks/sam2_prompts.json`
- `outputs/task1/file-*_sam2_masks/object_mask_frames/*_obj*.png`

It extracts clean images from the original videos listed in `sam2_prompts.json`;
it does not train on `overlay_frames`.

The generated dataset is:

```text
datasets/task1_yolo_detect/
  images/train/*.jpg
  images/val/*.jpg
  labels/train/*.txt
  labels/val/*.txt
  data.yaml
  manifest.json
```

For a later segmentation baseline, generate polygon labels with:

```bash
conda run -n sam2 python tools/yolo_sam2/convert_sam2_masks_to_yolo.py \
  --input outputs/task1 \
  --output datasets/task1_yolo_segment \
  --task segment \
  --overwrite
```

## 2. Train YOLO Detect

```bash
conda run -n sam2 python tools/yolo_sam2/train_yolo.py \
  --data datasets/task1_yolo_detect/data.yaml \
  --model yolo26n.pt \
  --epochs 100 \
  --imgsz 640 \
  --batch auto \
  --patience 20 \
  --project runs/yolo \
  --name task1_detect
```

Use `--model yolo11n.pt` or `--model yolo11s.pt` if your installed Ultralytics
version does not provide YOLO26 weights.

Smoke test:

```bash
conda run -n sam2 python tools/yolo_sam2/train_yolo.py \
  --data datasets/task1_yolo_detect/data.yaml \
  --model yolo26n.pt \
  --epochs 1 \
  --imgsz 640 \
  --batch 2 \
  --name task1_detect_smoke \
  --exist-ok
```

## 3. Use YOLO Boxes as SAM2 Prompts

```bash
conda run -n sam2 python tools/yolo_sam2/yolo_to_sam2.py \
  --video /path/to/new_video.mp4 \
  --yolo-weights runs/yolo/task1_detect/weights/best.pt \
  --sam2-checkpoint checkpoints/sam2.1_hiera_base_plus.pt \
  --sam2-config configs/sam2.1/sam2.1_hiera_b+.yaml \
  --labels outputs/task1/labels.txt \
  --output outputs/new_video_yolo_sam2 \
  --conf 0.25 \
  --overwrite
```

Add `--add-center-point` to include one positive point at each YOLO box center
in addition to the box prompt.

The output mirrors the existing SAM2 output style:

```text
outputs/new_video_yolo_sam2/
  mask_frames/*.png
  object_mask_frames/*_obj*.png
  overlay_frames/*.jpg
  overlay.mp4
  sam2_prompts.json
```

## Gradio App Integration

The same flow is also available in `apps/sam2_video_gradio_app.py`.

```bash
conda run -n sam2 python apps/sam2_video_gradio_app.py
```

After loading a video/task, open **YOLO 自动 Prompt**:

- `YOLO 权重`: usually `runs/yolo/task1_detect/weights/best.pt`
- `YOLO 置信度阈值`: start with `0.25`; lower it if a class is missed
- `同时添加 box 中心正点`: adds one positive click at each detected box center
- `覆盖当前标注`: replace current prompts with YOLO-generated prompts
- `YOLO 生成框 Prompt`: generate prompts only, so you can inspect/refine them
- `YOLO 生成 Prompt 并分割导出`: generate prompts, run SAM2, and export masks/overlay

For unattended processing, open **批量自动分割** and click
`分割当前目录剩余视频`. The app scans the current video directory, skips videos
that already have an overlay and `mask_frames` under the selected task, then
processes the remaining videos in order. For each video it creates YOLO box
prompts at the 3/5 and 4/5 frame positions and then runs SAM2 propagation.
