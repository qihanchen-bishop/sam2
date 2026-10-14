"""Read-only integrity audit of a completed expansion round (not semantic acceptance)."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def validate(root):
    manifest=json.loads((root/'manifest.json').read_text())
    errors=[];total=0;prompt_count=0;seed_count=0
    def check(condition,message):
        if not condition:errors.append(message)
    for seed in json.loads((root/'seed_reference.json').read_text()):
        folder=Path(seed['final_root'])/'mask_frames'
        check({p.name for p in folder.glob('*.png')}==set(seed['mask_hashes']),f'Seed file set changed: {seed["episode"]}')
        for name,digest in seed['mask_hashes'].items():
            path=folder/name
            check(path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest()==digest,f'Seed changed: {path}')
            seed_count+=1
    for r in manifest['records']:
        name=r['id'];n=r['item']['frame_count'];total+=n
        check(r['status']=='complete',f'{name}: incomplete {r["status"]}')
        if r['status']!='complete':continue
        data=json.loads((root/r['scores']).read_text());rows=data['frames']
        check(len(rows)==n and [x['frame'] for x in rows]==list(range(n)),f'{name}: score frame alignment')
        check(all(np.isfinite(x['score']) and 0<=x['score']<=100 for x in rows),f'{name}: invalid score')
        with (root/r['csv']).open() as handle:
            check(len(list(csv.DictReader(handle)))==n,f'{name}: CSV length')
        prompt=json.loads((root/r['prompt_json']).read_text());frames=prompt['prompt_frames'];prompt_count+=len(frames)
        check(frames==np.rint(np.linspace(0,n-1,10)).astype(int).tolist(),f'{name}: uniform prompts')
        audit=json.loads((root/name/'prompts/prompt_audit.json').read_text())
        for a in audit:
            if a['box'] is None:continue
            x0,y0,x1,y1=a['box']
            for point in a['positive']+a['negative']:
                check(x0<=point['x']<=x1 and y0<=point['y']<=y1 and point['confidence']>=.9,f'{name}: point bounds/confidence')
        unet=root/name/'unet'
        sam2=root/name/'sam2'/f'{Path(r["item"]["video_path"]).stem}_sam2_masks'
        review=json.loads((sam2/'direction_review.json').read_text())
        filtered=sam2/review['candidate_subdir']
        check(review['candidate_filter']['version']==manifest['filter_version'],f'{name}: filter version')
        expected={f'{i:05d}.png' for i in range(n)}
        for folder in [unet/'mask_frames',filtered/'directional_mask_frames/forward',filtered/'directional_mask_frames/reverse']:
            check({p.name for p in folder.glob('*.png')}==expected,f'{name}: mask count {folder}')
        for source,types in r['media'].items():
            for kind,path in types.items():
                cap=cv2.VideoCapture(str(root/path));count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=cap.get(cv2.CAP_PROP_FPS)
                check(cap.isOpened() and count==n and abs(fps-r['item']['fps'])<.01,f'{name}: media {source}/{kind} count/fps')
                cap.release()
        print(f'{name}: {n} frames checked',flush=True)
    result={'videos':len(manifest['records']),'frames':total,'prompt_frames':prompt_count,
            'seed_masks_verified_unchanged':seed_count,'errors':errors,'training_selection':manifest['training_selection']}
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);args=parser.parse_args()
    raise SystemExit(bool(validate(args.root)['errors']))
