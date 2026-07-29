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
/home/qihan/miniconda3/envs/sam2/bin/python \
  tools/yolo_sam2/convert_sam2_masks_to_yolo.py \
  --input /home/qihan/data/sam2/segdata/temp/newdata_3object/sam2_masks \
  --output /home/qihan/data/sam2/segdata/temp/newdata_3object/yolo_detect_random \
  --labels /home/qihan/data/sam2/segdata/origin/newdata_3object/labels.txt \
  --task detect \
  --split-mode random \
  --train-ratio 0.8 \
  --seed 42 \
  --frame-stride 3 \
  --min-mask-pixels 20 \
  --min-box-size 2 \
  --quality-filter \
  --overwrite
```

The converter recursively discovers all `file-*_sam2_masks` directories under
the input root, including both front and side views. It reads:

- the class name before `:` or `：` on each line of `labels.txt`
- `file-*_sam2_masks/sam2_prompts.json`
- `file-*_sam2_masks/object_mask_frames/*_obj*.png`

It extracts clean images from the original videos listed in `sam2_prompts.json`;
it does not train on `mask_visual_frames`.

Frames are sampled every three frames and then globally shuffled with seed 42.
The train/validation split is not grouped by task or camera view. The default
quality filter rejects fragmented masks, masks covering most of the image, and
large global or temporal area spikes. If any object mask is rejected, the whole
sampled frame is removed. Review exact decisions in `manifest.json`.

The generated dataset is:

```text
/home/qihan/data/sam2/segdata/temp/newdata_3object/yolo_detect_random/
  images/train/*.jpg
  images/val/*.jpg
  labels/train/*.txt
  labels/val/*.txt
  data.yaml
  labels.txt
  manifest.json
```

The current detector uses four flat classes: `occluder`, `object`, `region`, and
`tool`. Cube, paper ball, and screw are all task-specific appearances of
`object`; hierarchical labels are not needed for generating SAM2 prompts.

## 2. Train YOLO Detect

```bash
/home/qihan/miniconda3/envs/sam2/bin/python tools/yolo_sam2/train_yolo.py \
  --data /home/qihan/data/sam2/segdata/temp/newdata_3object/yolo_detect_random/data.yaml \
  --model yolo11n.pt \
  --epochs 100 \
  --imgsz 640 \
  --batch 64 \
  --patience 20 \
  --device 0 \
  --workers 8 \
  --seed 42 \
  --project /home/qihan/data/sam2/segdata/temp/newdata_3object/runs/yolo \
  --name newdata_3object_detect
```

The best checkpoint is written to
`runs/yolo/newdata_3object_detect/weights/best.pt` under the selected project.

## 3. Segment the Full LeRobot Dataset

Generate three temporally separated YOLO box prompts per detected class for all
front and side videos:

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    -u http_proxy -u https_proxy -u all_proxy \
  /home/qihan/miniconda3/envs/sam2/bin/python \
  tools/yolo_sam2/generate_yolo_prompts.py \
  --dataset-root /home/qihan/data/sam2/segdata/origin/newdata_3object \
  --video-key observation.images.front \
  --video-key observation.images.side \
  --yolo-weights /home/qihan/data/sam2/segdata/temp/newdata_3object/runs/yolo/newdata_3object_detect/weights/best.pt \
  --labels /home/qihan/data/sam2/segdata/origin/newdata_3object/labels.txt \
  --output-root /home/qihan/data/sam2/segdata/temp/newdata_3object/final_work/prompts \
  --device 0 \
  --sample-stride 5 \
  --prompts-per-class 3 \
  --conf 0.25 \
  --side-tool-conf 0.50
```

The higher side-view tool threshold reduces false prompts on the static black
background. Prompt generation skips existing JSON files unless `--overwrite`
is supplied.

Validate and propagate all prompts with one SAM2 model load:

```bash
python tools/sam2_prompts_to_masks.py \
  --prompt-root /home/qihan/data/sam2/segdata/temp/newdata_3object/final_work/prompts \
  --output-root /home/qihan/data/sam2/segdata/temp/newdata_3object/final_work/sam2_masks \
  --validate-only

python tools/sam2_prompts_to_masks.py \
  --prompt-root /home/qihan/data/sam2/segdata/temp/newdata_3object/final_work/prompts \
  --output-root /home/qihan/data/sam2/segdata/temp/newdata_3object/final_work/sam2_masks \
  --sam2-checkpoint checkpoints/sam2.1_hiera_base_plus.pt \
  --sam2-config configs/sam2.1/sam2.1_hiera_b+.yaml \
  --device cuda:0
```

Completed videos are skipped automatically, so the propagation command is safe
to resume after interruption. Do not add `--overwrite` for a normal resume.

Confirm that both stages contain 600 completed videos:

```bash
find /home/qihan/data/sam2/segdata/temp/newdata_3object/final_work/prompts \
  -name sam2_prompts.json | wc -l
find /home/qihan/data/sam2/segdata/temp/newdata_3object/final_work/sam2_masks \
  -name complete.json | wc -l
```

After both counts reach 600, create the final LeRobot dataset copy:

```bash
python tools/yolo_sam2/convert_sam2_masks_to_lerobot.py \
  --source-root /home/qihan/data/sam2/segdata/origin/newdata_3object \
  --seg-root /home/qihan/data/sam2/segdata/temp/newdata_3object/final_work/sam2_masks \
  --labels /home/qihan/data/sam2/segdata/origin/newdata_3object/labels.txt \
  --output-root /home/qihan/data/sam2/segdata/final/newdata_3object \
  --video-key observation.images.front \
  --video-key observation.images.side \
  --crf 18
```

The final copy preserves the original LeRobot dataset and adds eight video
features: four binary masks for each of the front and side views.
