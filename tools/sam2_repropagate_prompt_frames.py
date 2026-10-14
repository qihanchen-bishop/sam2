#!/usr/bin/env python3
"""Produce true propagated predictions on prompt frames from adjacent mask seeds.

The target frame receives no prompt. Each candidate uses a fresh two-frame
state, so its original conditioning prediction cannot bypass inference.
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from sam2.build_sam import build_sam2_video_predictor
from sam2_prompts_to_masks import (
    CLASS_PRIORITY, disable_unavailable_hole_filling,
    encode_class_mask_visual_video, object_is_visible,
)
from semantic_point_find_occluder_gaps import enclosed_background
from semantic_point_mask_audit import overlay


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:1')
    args = parser.parse_args()
    src, dst = args.source.resolve(), args.output.resolve()
    if dst.exists():
        raise FileExistsError(dst)
    data = json.loads((src / 'sam2_prompts.json').read_text())
    complete = json.loads((src / 'complete.json').read_text())
    total = int(data['frame_count'])
    events = sorted(set(data['prompt_frames']))
    if total < 2 or any(t < 0 or t >= total for t in events):
        raise ValueError('Invalid video or prompt frames')
    if any((t - 1 in events and t + 1 in events) for t in events):
        raise ValueError('Both adjacent frames are prompts; choose reviewed seeds explicitly')
    wanted = {i for t in events for i in (t-1, t, t+1) if 0 <= i < total}
    cap = cv2.VideoCapture(data['video_path'])
    frames = {}
    for i in range(total):
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f'Video ended at {i}, expected {total}')
        if i in wanted:
            frames[i] = frame
    cap.release()
    print(f'Decoded {total} source frames; retained {len(frames)}', flush=True)
    dst.mkdir(parents=True)
    shutil.copytree(src / 'mask_frames', dst / 'mask_frames')
    shutil.copy2(src / 'sam2_prompts.json', dst / 'sam2_prompts.json')
    for sub in ('review', 'directional_mask_frames/forward', 'directional_mask_frames/reverse'):
        (dst / sub).mkdir(parents=True)
    predictor = build_sam2_video_predictor(
        complete['sam2_config'], complete['sam2_checkpoint'], device=args.device,
        hydra_overrides_extra=['++model.clear_non_cond_mem_around_input=false'],
    )
    disable_unavailable_hole_filling(predictor)
    ordered = sorted(data['objects'], key=lambda o: (
        CLASS_PRIORITY.index(o['name']) if o['name'] in CLASS_PRIORITY else len(CLASS_PRIORITY),
        o['id']))
    rows = []
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        for target in events:
            candidates = {}
            seeds = {}
            for direction, seed in [('forward', target-1), ('reverse', target+1)]:
                if seed < 0 or seed >= total or seed in events:
                    continue
                with tempfile.TemporaryDirectory(prefix='sam2_prompt_reprop_') as tmp:
                    lo, hi = sorted([target, seed])
                    for idx, original in enumerate([lo, hi]):
                        if not cv2.imwrite(str(Path(tmp) / f'{idx:05d}.jpg'), frames[original],
                                           [cv2.IMWRITE_JPEG_QUALITY, 95]):
                            raise RuntimeError('Failed to write temporary source image')
                    state = predictor.init_state(video_path=tmp, offload_video_to_cpu=True)
                    mask = np.asarray(Image.open(src / 'mask_frames' / f'{seed:05d}.png'))
                    for obj in ordered:
                        predictor.add_new_mask(state, seed-lo, int(obj['id']), mask == obj['id'])
                    result = None
                    for index, ids, logits in predictor.propagate_in_video(
                        state, start_frame_idx=seed-lo, max_frame_num_to_track=1,
                        reverse=direction == 'reverse',
                    ):
                        if index != target-lo:
                            continue
                        result = np.zeros(mask.shape, dtype=np.uint8)
                        by_id = {int(k): logits[j, 0].float().cpu().numpy() for j, k in enumerate(ids)}
                        for obj in ordered:
                            if object_is_visible(obj, target):
                                result[by_id[int(obj['id'])] > complete['mask_threshold']] = obj['id']
                    if result is None:
                        raise RuntimeError(f'No prediction for {target} {direction}')
                    predictor.reset_state(state)
                    del state
                candidates[direction] = result
                seeds[direction] = seed
                Image.fromarray(result).save(dst / 'directional_mask_frames' / direction / f'{target:05d}.png')
            selected = 'forward' if 'forward' in candidates else 'reverse'
            result = candidates[selected]
            Image.fromarray(result).save(dst / 'mask_frames' / f'{target:05d}.png')
            old = np.asarray(Image.open(src / 'mask_frames' / f'{target:05d}.png'))
            row = {'frame': target, 'selected_direction': selected, 'seed_frames': seeds,
                   'changed_pixels': int((old != result).sum()),
                   'old_largest_occluder_hole': enclosed_background(old)[1],
                   'new_largest_occluder_hole': enclosed_background(result)[1],
                   'class_pixels_before': {o['name']: int((old == o['id']).sum()) for o in ordered},
                   'class_pixels_after': {o['name']: int((result == o['id']).sum()) for o in ordered}}
            if len(candidates) == 2:
                a, b = candidates['forward'], candidates['reverse']
                row['directional_iou'] = {o['name']: float(((a == o['id']) & (b == o['id'])).sum() /
                    max(1, ((a == o['id']) | (b == o['id'])).sum())) for o in ordered}
            rows.append(row)
            tiles = [overlay(frames[target], old, f'f{target} original')]
            tiles += [overlay(frames[target], m, f'f{target} {direction}' + (' SELECTED' if direction == selected else ''))
                      for direction, m in candidates.items()]
            sheet = np.hstack([cv2.resize(tile, (600, 456)) for tile in tiles])
            cv2.imwrite(str(dst / 'review' / f'compare_f{target:05d}.jpg'), sheet)
            print(json.dumps(row), flush=True)
    # Verify every frame, not just replaced prompts. Unchanged frames must be identical.
    valid = {0, *(int(o['id']) for o in ordered)}
    for i in range(total):
        new = np.asarray(Image.open(dst / 'mask_frames' / f'{i:05d}.png'))
        old = np.asarray(Image.open(src / 'mask_frames' / f'{i:05d}.png'))
        if new.shape != old.shape or not set(np.unique(new)).issubset(valid):
            raise ValueError(f'Invalid mask {i}')
        if i not in events and not np.array_equal(new, old):
            raise ValueError(f'Non-prompt frame unexpectedly changed: {i}')
    encode_class_mask_visual_video(dst / 'mask_frames', dst / 'mask_visual.mp4',
                                   Path(data['video_path']), data['objects'])
    report = {'source': str(src), 'method': 'adjacent_mask_seed_one_frame_propagation',
              'target_frame_prompts_used': False, 'selection': 'forward when available, reverse otherwise',
              'frame_count': total, 'replaced_frames': events, 'frames': rows,
              'validation': 'all masks valid; all non-prompt masks unchanged'}
    (dst / 'repropagation_report.json').write_text(json.dumps(report, indent=2) + '\n')
    complete['source_output'] = str(src)
    complete['prompt_frame_repropagation'] = report
    complete['review_status'] = 'candidate_requires_visual_review'
    (dst / 'complete.json').write_text(json.dumps(complete, indent=2) + '\n')
    print(f'DONE {dst}', flush=True)


if __name__ == '__main__':
    main()
