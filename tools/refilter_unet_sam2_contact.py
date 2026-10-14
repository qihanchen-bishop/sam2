"""Re-filter saved raw U-Net/SAM2 outputs into an isolated contact-rule revision."""
import argparse
import copy
import json
import shutil
import subprocess
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2

from sam2_direction_review import FILTER_VERSION, component_filter_policy, filter_candidates, write_json
from unet_video_preview import run as run_unet
from unet_sam2_round import render_overlays
from unet_sam2_scores import score_video


def main():
    p=argparse.ArgumentParser();p.add_argument('--round-root',type=Path,required=True);p.add_argument('--output-name',default='contact_filter_r1')
    args=p.parse_args();root=args.round_root.resolve();out=root/args.output_name
    if out.parent!=root:raise ValueError('Output must be a child of source round')
    out.mkdir(exist_ok=True);source=json.loads((root/'manifest.json').read_text())
    records=[{'id':r['id'],'item':r['item'],'prompt_json':r['prompt_json'],'prompt_frames':r['prompt_frames'],'status':'pending'} for r in source['records']]
    manifest={'source_round':str(root),'model_dir':source['model_dir'],'filter':component_filter_policy(),
              'records':records,'training_selection':'NONE','seed_episodes':source['seed_episodes']}
    write_json(out/'manifest.json',manifest)
    shutil.copy2(Path(__file__).with_name('unet_sam2_review.html'),out/'index.html')
    html=(out/'index.html').read_text().replace('name="round-manifest" content=""',f'name="round-manifest" content="{args.output_name}/manifest.json"')
    (out/'index.html').write_text(html)
    for name in ['labels.txt','seed_reference.json']:shutil.copy2(root/name,out/name)
    relative=lambda path:str(path.relative_to(root))
    lock=threading.Lock()
    def publish(record,status):
        with lock:
            record['status']=status
            records[:]=[copy.deepcopy(record) if r['id']==record['id'] else r for r in records]
            write_json(out/'manifest.json',manifest)
    def process(original):
        ep=original['id'];item=original['item'];folder=out/ep;folder.mkdir(exist_ok=True)
        record={'id':ep,'item':item,'prompt_json':original['prompt_json'],'prompt_frames':original['prompt_frames'],'status':'refiltering_unet'}
        publish(record,'refiltering_unet')
        unet=folder/'unet'
        if not (unet/'complete.json').exists():
            if unet.exists():unet.rename(folder/f'unet.incomplete.{time.time_ns()}')
            run_unet(Path(source['model_dir']),Path(item['video_path']),unet,'cpu',root/ep/'unet')
        if json.loads((unet/'complete.json').read_text())['component_filter']['version']!=FILTER_VERSION:raise ValueError('Unexpected U-Net filter revision')
        record['media']={'unet':{'overlay':relative(unet/'overlay.mp4'),'segmentation':relative(unet/'mask_visual.mp4')}}
        publish(record,'refiltering_sam2')
        source_sam=root/ep/'sam2'/f'{Path(item["video_path"]).stem}_sam2_masks'
        sam=folder/'sam2'/source_sam.name;sam.mkdir(parents=True,exist_ok=True)
        if not (sam/'direction_review.json').exists():
            for name in ['complete.json','sam2_prompts.json','direction_conflicts.json']:shutil.copy2(source_sam/name,sam/name)
            for name in ['directional_mask_frames','mask_frames']:
                if not (sam/name).exists():(sam/name).symlink_to(source_sam/name,target_is_directory=True)
            write_json(sam/'direction_review.json',{'status':'pending','revision':0,'choices':{},'default_direction':'forward'})
        state=filter_candidates(sam)
        meta=json.loads((sam/'complete.json').read_text());meta.update(candidate_filter=state['candidate_filter'],refilter_of=str(source_sam),propagation_reused=True)
        write_json(sam/'complete.json',meta)
        publish(record,'scoring')
        objects=json.loads((unet/'complete.json').read_text())['objects']
        filtered=score_video(unet,sam,folder/'scores',objects,item['frame_count'],item['fps'],record['prompt_frames'])
        if not all((folder/f'{d}_overlay.mp4').exists() for d in ['forward','reverse']):render_overlays(item['video_path'],filtered,folder,objects)
        for d in ['forward','reverse']:
            record['media'][d]={'overlay':relative(folder/f'{d}_overlay.mp4'),'segmentation':relative(filtered/f'mask_visual_{d}.mp4')}
        record.update(status='complete',scores=relative(folder/'scores/frame_scores.json'),csv=relative(folder/'scores/frame_scores.csv'),
                      summary=json.loads((folder/'scores/summary.json').read_text()))
        publish(record,'complete');print(ep,'REFILTERED',flush=True)
    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(process,source['records']))
    for script in ['unet_temporal_scores.py','unet_global_temporal_scores.py']:
        command=[sys.executable,str(Path(__file__).with_name(script)),'--round-root',str(out),'--web-prefix',args.output_name+'/']
        if script=='unet_temporal_scores.py':command+=['--output-name','unet_temporal_r2']
        subprocess.run(command,check=True)
    write_json(out/'complete.json',{'videos':len(records),'frames':sum(r['item']['frame_count'] for r in records),
                                  'filter_version':FILTER_VERSION,'propagation_reused':True,'training_selection':'NONE'})


if __name__=='__main__':
    cv2.setNumThreads(1)
    main()
