## SAM 2 toolkits

This directory provides toolkits for additional SAM 2 use cases.

### Semi-supervised VOS inference

The `vos_inference.py` script can be used to generate predictions for semi-supervised video object segmentation (VOS) evaluation on datasets such as [DAVIS](https://davischallenge.org/index.html), [MOSE](https://henghuiding.github.io/MOSE/) or the SA-V dataset.

After installing SAM 2 and its dependencies, it can be used as follows ([DAVIS 2017 dataset](https://davischallenge.org/davis2017/code.html) as an example). This script saves the prediction PNG files to the `--output_mask_dir`.
```bash
python ./tools/vos_inference.py \
  --sam2_cfg configs/sam2.1/sam2.1_hiera_b+.yaml \
  --sam2_checkpoint ./checkpoints/sam2.1_hiera_base_plus.pt \
  --base_video_dir /path-to-davis-2017/JPEGImages/480p \
  --input_mask_dir /path-to-davis-2017/Annotations/480p \
  --video_list_file /path-to-davis-2017/ImageSets/2017/val.txt \
  --output_mask_dir ./outputs/davis_2017_pred_pngs
```
(replace `/path-to-davis-2017` with the path to DAVIS 2017 dataset)

To evaluate on the SA-V dataset with per-object PNG files for the object masks, we need to **add the `--per_obj_png_file` flag** as follows (using SA-V val as an example). This script will also save per-object PNG files for the output masks under the `--per_obj_png_file` flag.
```bash
python ./tools/vos_inference.py \
  --sam2_cfg configs/sam2.1/sam2.1_hiera_b+.yaml \
  --sam2_checkpoint ./checkpoints/sam2.1_hiera_base_plus.pt \
  --base_video_dir /path-to-sav-val/JPEGImages_24fps \
  --input_mask_dir /path-to-sav-val/Annotations_6fps \
  --video_list_file /path-to-sav-val/sav_val.txt \
  --per_obj_png_file \
  --output_mask_dir ./outputs/sav_val_pred_pngs
```
(replace `/path-to-sav-val` with the path to SA-V val)

Then, we can use the evaluation tools or servers for each dataset to get the performance of the prediction PNG files above.

Note: by default, the `vos_inference.py` script above assumes that all objects to track already appear on frame 0 in each video (as is the case in DAVIS, MOSE or SA-V). **For VOS datasets that don't have all objects to track appearing in the first frame (such as LVOS or YouTube-VOS), please add the `--track_object_appearing_later_in_video` flag when using `vos_inference.py`**.

### Lightweight semantic segmentation

`train_light_semseg.py` trains a small Tiny-UNet semantic segmentation model.
For the local `newdata_3object` dataset, it first prepares a jpg/png cache under
the corresponding `segdata/temp` folder, then trains only from those image
files. Missing cache files are generated from the LeRobot videos before
training starts:

```bash
conda run -n sam2 python tools/train_light_semseg.py \
  --dataset-root /home/qihan/data/sam2/segdata/final/newdata_3object \
  --epochs 100 \
  --frame-stride 10 \
  --batch-size 32 \
  --workers 4
```

The default classes are `background`, `occluder`, `object`, `region`, and
`tool`. Training monitors validation mIoU and stops early by default after 12
epochs without a meaningful improvement, while keeping the best checkpoint. The
usual cache and checkpoint paths are:

```text
/home/qihan/data/sam2/segdata/temp/newdata_3object/semantic_seg_light_cache/
/home/qihan/data/sam2/segdata/temp/newdata_3object/semantic_seg_light/best.pt
/home/qihan/data/sam2/segdata/temp/newdata_3object/semantic_seg_light/latest.pt
```

To only prepare or repair the image cache without training:

```bash
conda run -n sam2 python tools/train_light_semseg.py \
  --dataset-root /home/qihan/data/sam2/segdata/final/newdata_3object \
  --frame-stride 10 \
  --prepare-cache-only
```
