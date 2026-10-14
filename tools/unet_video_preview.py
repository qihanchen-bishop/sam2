#!/usr/bin/env python3
"""Produce independent U-Net class PNGs and browser-compatible preview videos."""
import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from infer_light_semseg import Predictor
from sam2_prompts_to_masks import encode_class_mask_visual_video, find_executable
from sam2_direction_review import filter_mask_components, FILTER_VERSION, component_filter_policy


def run(model_dir, video_path, output, device, source_output=None):
    output.mkdir(parents=True, exist_ok=False)
    masks = output / 'mask_frames'
    masks.mkdir()
    raw_masks = output / 'raw_mask_frames'
    raw_masks.mkdir()
    started = time.time()
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    source = json.loads((source_output/'complete.json').read_text()) if source_output else None
    predictor = None if source else Predictor(model_dir, device)
    config = json.loads((model_dir/'model_config.json').read_text())
    objects = source['objects'] if source else [
        {'id': i, 'name': name, 'color': '#%02x%02x%02x' % tuple(config['palette_rgb'][i])}
        for i, name in enumerate(config['labels']) if i]
    palette = np.zeros((256,3), np.uint8)
    for obj in objects:
        palette[obj['id']] = tuple(int(obj['color'][i:i+2],16) for i in (1,3,5))
    source_masks = None
    if source:
        source_masks = source_output/'raw_mask_frames'
        if not source_masks.is_dir():
            if source.get('component_filter'):
                raise ValueError('原始预测不存在，不能从已过滤版本再次过滤')
            source_masks = source_output/'mask_frames'
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError('Cannot open input video')
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.
    encoder = subprocess.Popen([find_executable('ffmpeg'), '-hide_banner', '-loglevel', 'error',
        '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{width}x{height}', '-r', str(fps),
        '-i', '-', '-an', '-c:v', 'libx264', '-threads', '2', '-preset', 'fast', '-crf', '20',
        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(output/'overlay.mp4')], stdin=subprocess.PIPE)
    count = 0
    edits = []
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            raw = cv2.imread(str(source_masks/f'{count:05d}.png'), cv2.IMREAD_UNCHANGED) if source else predictor.predict(bgr)
            if raw is None or raw.shape != bgr.shape[:2] or raw.dtype != np.uint8:
                raise ValueError(f'Invalid raw mask frame {count}')
            if not cv2.imwrite(str(raw_masks/f'{count:05d}.png'), raw):
                raise RuntimeError('Raw mask write failed')
            ids, details = filter_mask_components(raw, objects, return_details=True)
            if details:
                edits.append({'frame': count, 'removed_pixels': int(np.count_nonzero(raw != ids)), 'components': details})
            if not cv2.imwrite(str(masks/f'{count:05d}.png'), ids):
                raise RuntimeError('Mask write failed')
            color = cv2.cvtColor(palette[ids], cv2.COLOR_RGB2BGR)
            overlay = bgr.copy()
            foreground = ids != 0
            overlay[foreground] = cv2.addWeighted(bgr, .5, color, .5, 0)[foreground]
            encoder.stdin.write(overlay.tobytes())
            count += 1
            if count % 100 == 0:
                print(f'U-Net {"filtered saved masks" if source else "segmented"} {count} frames', flush=True)
        encoder.stdin.close()
        if encoder.wait() != 0 or count == 0:
            raise RuntimeError('Overlay encoding failed or empty video')
    finally:
        cap.release()
        if encoder.poll() is None:
            encoder.terminate()
            encoder.wait()
    if source and count != source['frame_count']:
        raise ValueError('Source mask/video frame counts differ')
    encode_class_mask_visual_video(masks, output/'mask_visual.mp4', video_path, objects)
    completion = {'method': 'unet_direct', 'model_dir': str(model_dir), 'video_path': str(video_path),
                  'model_sha256': source['model_sha256'] if source else hashlib.sha256((model_dir/config['torchscript']).read_bytes()).hexdigest(),
                  'objects': objects, 'frame_count': count, 'width': width, 'height': height, 'fps': fps,
                  'seconds': time.time()-started, 'input_size_wh': source['input_size_wh'] if source else config['input_size_wh'],
                  'component_filter': component_filter_policy(),
                  'filter_changed_frames': len(edits), 'filter_removed_pixels': sum(e['removed_pixels'] for e in edits),
                  'source_output': str(source_output) if source_output else None,
                  'note': 'Independent predictions followed by SAM2-identical component filtering; raw predictions retained.'}
    (output/'filter_edits.json').write_text(json.dumps(edits, ensure_ascii=False, indent=2))
    (output/'complete.json').write_text(json.dumps(completion, ensure_ascii=False, indent=2))
    print(f'COMPLETE: {count} frames in {completion["seconds"]:.1f}s', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model-dir', type=Path, required=True)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda:1')
    p.add_argument('--source-output', type=Path, help='Filter saved raw predictions without loading the model')
    args = p.parse_args()
    run(args.model_dir, args.input, args.output, args.device, args.source_output)
