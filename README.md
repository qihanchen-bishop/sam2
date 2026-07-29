# SAM2-YOLO Video Annotation Pipeline

This project extends [SAM 2](https://github.com/facebookresearch/sam2) for semi-automatic video segmentation annotation.

The workflow is designed for datasets where fully manual video mask annotation is too expensive:

1. Manually add a small number of prompts on several seed videos.
2. Use SAM2 to propagate the prompts and export segmentation masks.
3. Convert these SAM2 masks into a YOLO detection dataset.
4. Train a YOLO detector on the seed annotations.
5. Use YOLO to batch-generate box prompts for the remaining videos.
6. Run SAM2 again to export segmentation masks for the full video set.

The original SAM2 model code, configs, checkpoints, and license come from Meta's official repository:
<https://github.com/facebookresearch/sam2>

## Environment

If the server has proxy variables configured and downloads are slow or unstable, clear them before installing packages:

```bash
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY=localhost,127.0.0.1,::1
export no_proxy=localhost,127.0.0.1,::1
```

Create and activate a conda environment:

```bash
conda create -n sam2 python=3.10 pip -y
conda activate sam2
```

Install CUDA PyTorch. Choose the wheel index that matches your CUDA/driver setup; CUDA 12.8 is used here as an example:

```bash
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

Install this repository and the command-line runtime packages:

```bash
conda install -c conda-forge ffmpeg -y
python -m pip install --no-build-isolation -e .
python -m pip install matplotlib eva-decord opencv-python-headless pillow numpy pyyaml ultralytics
```

Download the required SAM2 checkpoint as described in
[Segment Reviewed Prompt Videos](#segment-reviewed-prompt-videos).

## Seed Annotations

Start from a small set of seed videos that already have SAM2 segmentation outputs. The expected task directory is:

```text
outputs/task1/
  labels.txt
  file-*_sam2_masks/
    sam2_prompts.json
    object_mask_frames/*.png
    mask_frames/*.png
    mask_visual_frames/*.png
    mask_visual.mp4
```

`labels.txt` defines the object classes, one class per line. `sam2_prompts.json` records the source video path and object metadata. `object_mask_frames` stores per-object mask PNGs used to build the YOLO dataset.

Seed annotations can be produced by any SAM2 prompting workflow as long as this output structure is preserved.

For targets that disappear or closely resemble a static background, an object may
define inclusive `visible_ranges`, for example `"visible_ranges": [[170, 440],
[900, 1100]]`. Frames outside those ranges are exported as empty object masks.

## Segment Reviewed Prompt Videos

The reviewed prompt files under `temp/<dataset>/prompts/json` already contain
the source video path, object IDs, and box prompts on selected frames. Validate
them first:

```bash
python tools/sam2_prompts_to_masks.py \
  --prompt-root /home/qihan/data/sam2/segdata/temp/newdata_3object/prompts/json \
  --output-root /home/qihan/data/sam2/segdata/temp/newdata_3object/sam2_masks \
  --validate-only
```

Download only the SAM2.1 Base+ checkpoint without using proxy variables:

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    -u http_proxy -u https_proxy -u all_proxy \
  wget -P checkpoints \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt
```

Run SAM2 on all prompt JSON files without Gradio:

```bash
python tools/sam2_prompts_to_masks.py \
  --prompt-root /home/qihan/data/sam2/segdata/temp/newdata_3object/prompts/json \
  --output-root /home/qihan/data/sam2/segdata/temp/newdata_3object/sam2_masks \
  --sam2-checkpoint checkpoints/sam2.1_hiera_base_plus.pt \
  --sam2-config configs/sam2.1/sam2.1_hiera_b+.yaml
```

Add `--limit 1` for a one-video test. Add `--offload-state-to-cpu` if GPU
memory is insufficient. Existing completed outputs are skipped; use
`--overwrite` to regenerate them.

If the optional `sam2._C` CUDA extension is unavailable, the command-line tool
disables only small-hole filling and continues with the remaining SAM2
post-processing. This does not normally require rebuilding the environment.

Each video output contains:

```text
file-*_sam2_masks/
  sam2_prompts.json
  mask_frames/             # class-ID PNG: 0=background, 1/2/...=object ID
  mask_visual_frames/      # colored masks for inspection
  mask_visual.mp4          # colored-mask video without the source image
  object_mask_frames/      # one 0/255 binary mask per object and frame
  complete.json
```

## Train YOLO From SAM2 Masks

Convert exported SAM2 object masks into a YOLO detection dataset:

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
  --quality-filter \
  --overwrite
```

The converter recursively combines front/side videos from all tasks, then
globally shuffles frames instead of splitting by task. Its default quality
filter removes fragmented masks and abnormal mask-area jumps. The full split
and rejection log is stored in `manifest.json`.

This dataset uses four flat classes: `occluder`, `object`, `region`, and `tool`.
Cube, paper ball, and screw remain one `object` class because the detector is
used to generate class-agnostic SAM2 prompts. Use separate flat YOLO classes
only if a later application must distinguish those object types.

Train YOLO:

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

The trained detector is usually saved at:

```text
/home/qihan/data/sam2/segdata/temp/newdata_3object/runs/yolo/newdata_3object_detect/weights/best.pt
```

## Generate the Final LeRobot Dataset

Generate YOLO prompts for every front and side video:

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
  --side-tool-conf 0.50
```

Run SAM2 propagation. Completed videos are skipped automatically:

```bash
python tools/sam2_prompts_to_masks.py \
  --prompt-root /home/qihan/data/sam2/segdata/temp/newdata_3object/final_work/prompts \
  --output-root /home/qihan/data/sam2/segdata/temp/newdata_3object/final_work/sam2_masks \
  --sam2-checkpoint checkpoints/sam2.1_hiera_base_plus.pt \
  --sam2-config configs/sam2.1/sam2.1_hiera_b+.yaml \
  --device cuda:0
```

After all 600 videos contain `complete.json`, export a complete LeRobot dataset:

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

The final dataset adds `front_occluder`, `front_object`, `front_region`,
`front_tool`, and the corresponding four side-view video features.

## Key Files

- `tools/yolo_sam2/convert_sam2_masks_to_yolo.py`: convert SAM2 masks to YOLO labels.
- `tools/yolo_sam2/train_yolo.py`: train an Ultralytics YOLO detector.
- `tools/yolo_sam2/generate_yolo_prompts.py`: generate multi-frame prompts for all dataset videos.
- `tools/yolo_sam2/yolo_to_sam2.py`: run YOLO prompt generation and SAM2 segmentation for one video.
- `tools/yolo_sam2/convert_sam2_masks_to_lerobot.py`: add front/side mask videos to a LeRobot dataset copy.
- `tools/yolo_sam2/README.md`: more detailed command examples.

## Citation

If you use the SAM2 model or checkpoints, cite the original SAM2 work and follow the original license:
<https://github.com/facebookresearch/sam2>
