"""Apply contact filtering to saved U-Net masks, then whole-video diagnosis."""
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from sam2_direction_review import FILTER_VERSION, component_filter_policy, write_json
from unet_video_preview import run as run_unet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--round-root', type=Path, required=True)
    args = parser.parse_args()
    root = args.round_root.resolve()
    out = root / 'contact_filter_r1'
    out.mkdir(exist_ok=True)
    source = json.loads((root / 'manifest.json').read_text())
    if (out / 'manifest.json').exists():
        shutil.copy2(out / 'manifest.json', out / f'manifest.previous.{time.time_ns()}.json')
    records = []
    manifest = dict(workflow='unet_filter_then_global_temporal',
                    model_dir=source['model_dir'], filter=component_filter_policy(),
                    records=records, training_selection='NONE')
    for original in source['records']:
        ep, item = original['id'], original['item']
        folder = out / ep
        folder.mkdir(exist_ok=True)
        unet = folder / 'unet'
        if not (unet / 'complete.json').exists():
            if unet.exists():
                unet.rename(folder / f'unet.incomplete.{time.time_ns()}')
            run_unet(Path(source['model_dir']), Path(item['video_path']), unet, 'cpu', root / ep / 'unet')
        meta = json.loads((unet / 'complete.json').read_text())
        if meta['component_filter']['version'] != FILTER_VERSION:
            raise ValueError(f'{ep}: incompatible filter version')
        records.append(dict(id=ep, item=item, status='complete', media={'unet': {
            'overlay': str((unet / 'overlay.mp4').relative_to(root)),
            'segmentation': str((unet / 'mask_visual.mp4').relative_to(root))}}))
        write_json(out / 'manifest.json', manifest)
        print(ep, 'U-Net filtered masks ready', flush=True)
    for script, name in [('unet_temporal_scores.py', 'unet_temporal_r2'),
                         ('unet_global_temporal_scores.py', 'unet_global_r2')]:
        subprocess.run([sys.executable, str(Path(__file__).with_name(script)),
                        '--round-root', str(out), '--output-name', name,
                        '--web-prefix', 'contact_filter_r1/'], check=True)
    write_json(out / 'complete.json', dict(workflow=manifest['workflow'],
               videos=len(records), frames=sum(r['item']['frame_count'] for r in records),
               filter_version=FILTER_VERSION, training_selection='NONE'))


if __name__ == '__main__':
    main()
