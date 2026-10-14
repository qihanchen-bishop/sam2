"""Publish lightweight batch progress and audit outputs once all workers finish."""
import argparse
import json
import time
from pathlib import Path

from sam2_direction_review import write_json
from validate_unet_sam2_round import validate


def main(root):
    while True:
        manifest=json.loads((root/'manifest.json').read_text())
        records=[]
        for r in manifest['records']:
            output=root/r['id']/'sam2'/f'{Path(r["item"]["video_path"]).stem}_sam2_masks'
            if not output.exists():output=output.with_name(output.name+'.partial')
            counts={d:sum(1 for _ in (output/'directional_mask_frames'/d).glob('*.png')) for d in ('forward','reverse')}
            records.append({'id':r['id'],'status':r['status'],'frames':r['item']['frame_count'],**counts})
        write_json(root/'progress.json',{'updated_at':time.time(),'records':records})
        if all(r['status'] in ('complete','failed') for r in manifest['records']):
            result=validate(root)
            write_json(root/'validation.json',result)
            return int(bool(result['errors']))
        time.sleep(15)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);args=parser.parse_args()
    raise SystemExit(main(args.root))
