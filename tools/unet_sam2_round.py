#!/usr/bin/env python3
"""Independent reviewed-seed expansion: uniform prompts, bidirectional masks, per-frame scores.

Does not select training samples or mutate seed/web-app annotation versions.
"""
import argparse
import fcntl
import hashlib
import json
import random
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import torch

from infer_light_semseg import Predictor
from unet_video_preview import run as run_unet
from sam2_direction_review import FILTER_VERSION, write_json
from sam2_prompts_to_masks import find_executable, load_and_validate_prompt
from unet_sam2_scores import score_video

REPO=Path(__file__).resolve().parents[1]
SEEDS=[0,20,40,60]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def spread_points(mask, confidence, count, threshold=.9):
    eligible=(mask & (confidence>=threshold)).astype(np.uint8)
    interior=cv2.distanceTransform(eligible,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
    margin=max(1.,min(8.,float(interior.max())*.15))
    core=(interior>margin).astype(np.uint8)
    used_boundary=not bool(core.any())
    y,x=np.nonzero(eligible if used_boundary else core)
    if len(x)>12000:
        indices=np.linspace(0,len(x)-1,12000,dtype=int);x,y=x[indices],y[indices]
    if not len(x):return [],used_boundary
    coords=np.stack([x,y],axis=1);probs=confidence[y,x];chosen=[]
    nearest=np.full(len(coords),np.inf)
    for _ in range(min(count,len(coords))):
        initial=probs*(1+.1*interior[y,x]/max(float(interior.max()),1.))
        idx=int(np.argmax(initial if not chosen else nearest*(.8+.2*probs)))
        if chosen and nearest[idx]<4:break
        chosen.append({'x':int(x[idx]),'y':int(y[idx]),'confidence':float(probs[idx])})
        nearest=np.minimum(nearest,((coords-coords[idx])**2).sum(axis=1));nearest[idx]=-1
    return chosen,used_boundary


@torch.inference_mode()
def generate_prompts(item, unet, target, model_dir, device):
    target.mkdir(parents=True,exist_ok=True)
    previews=target/'previews';previews.mkdir(exist_ok=True)
    probabilities=target/'probabilities';probabilities.mkdir(exist_ok=True)
    predictor=Predictor(model_dir,device)
    completion=json.loads((unet/'complete.json').read_text())
    objects=[{**o,'boxes':[],'points':[]} for o in completion['objects']]
    frames=np.unique(np.rint(np.linspace(0,item['frame_count']-1,10)).astype(int)).tolist()
    cap=cv2.VideoCapture(item['video_path']);audit=[]
    for frame in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,bgr=cap.read()
        if not ok:raise RuntimeError(f'Cannot read prompt frame {frame}')
        h,w=bgr.shape[:2]
        filtered=cv2.imread(str(unet/'mask_frames'/f'{frame:05d}.png'),0)
        raw=cv2.imread(str(unet/'raw_mask_frames'/f'{frame:05d}.png'),0)
        rgb=cv2.cvtColor(cv2.resize(bgr,(predictor.width,predictor.height),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2RGB)
        normalized=(rgb.astype(np.float32)/255-predictor.mean)/predictor.std
        tensor=torch.from_numpy(normalized.transpose(2,0,1)).unsqueeze(0).to(predictor.device)
        probs=predictor.model(tensor).softmax(1)[0].cpu().numpy()
        np.savez_compressed(probabilities/f'{frame:05d}.npz',probabilities=probs.astype(np.float16))
        full=np.stack([cv2.resize(p,(w,h),interpolation=cv2.INTER_LINEAR) for p in probs])
        combined=bgr.copy()
        for obj in objects:
            cid=obj['id'];mask=filtered==cid
            entry={'frame':frame,'class_id':cid,'class':obj['name'],'box':None,'positive':[],'negative':[],
                   'confidence_threshold':.9,'status':'absent_after_filter'}
            if mask.any():
                x,y,bw,bh=cv2.boundingRect(mask.astype(np.uint8))
                box=[x,y,min(w-1,x+bw),min(h-1,y+bh)]
                if box[0]==box[2]:box[0]=max(0,box[0]-1)
                if box[1]==box[3]:box[1]=max(0,box[1]-1)
                positive,boundary=spread_points(mask,full[cid],5)
                negative=[]
                inside=np.zeros_like(mask);inside[y:y+bh,x:x+bw]=True
                for other in range(len(full)):
                    if other==cid:continue
                    # Removed fragments are not treated as confident background.
                    other_mask=(raw==0) if other==0 else (filtered==other)
                    points,_=spread_points(inside&other_mask,full[other],2)
                    negative += [{**pt,'source_class_id':other} for pt in points]
                obj['boxes'].append({'frame':frame,'box':box})
                obj['points'] += [{'frame':frame,'x':p['x'],'y':p['y'],'label':1} for p in positive]
                obj['points'] += [{'frame':frame,'x':p['x'],'y':p['y'],'label':0} for p in negative]
                entry.update(box=box,positive=positive,negative=negative,used_boundary_fallback=boundary,
                             status='generated' if len(positive)>=3 else 'few_high_confidence_positives')
                color=tuple(int(obj['color'][i:i+2],16) for i in (5,3,1))
                cv2.rectangle(combined,tuple(box[:2]),tuple(box[2:]),color,2)
                canvas=bgr.copy();cv2.rectangle(canvas,tuple(box[:2]),tuple(box[2:]),color,2)
                for p in positive:cv2.circle(canvas,(p['x'],p['y']),6,(0,255,0),2)
                for p in negative:cv2.drawMarker(canvas,(p['x'],p['y']),(0,0,255),cv2.MARKER_TILTED_CROSS,12,2)
                cv2.putText(canvas,f'{frame} {obj["name"]} +{len(positive)} -{len(negative)}',(12,30),cv2.FONT_HERSHEY_SIMPLEX,.8,(255,255,255),2)
                cv2.imwrite(str(previews/f'{frame:05d}_{obj["name"]}.jpg'),canvas,[cv2.IMWRITE_JPEG_QUALITY,94])
            audit.append(entry)
        cv2.putText(combined,f'frame {frame} (uniform 10 prompts)',(12,30),cv2.FONT_HERSHEY_SIMPLEX,.8,(255,255,255),2)
        cv2.imwrite(str(previews/f'{frame:05d}_boxes.jpg'),combined,[cv2.IMWRITE_JPEG_QUALITY,94])
    cap.release()
    missing=[o['name'] for o in objects if not o['boxes']]
    if missing:raise RuntimeError(f'No visible prompt for classes {missing}; requires inspection, no invented prompts')
    prompt={'video_path':item['video_path'],'frame_count':item['frame_count'],'objects':objects,
            'method':'uniform10_unet_confidence_spread_v2','prompt_frames':frames,
            'point_policy':{'confidence_threshold':.9,'positive_target':5,'negative_per_other_class':2,
                            'negative_background':2,'interior_margin':'15% max interior distance, clamped 1..8 original pixels'}}
    write_json(target/'sam2_prompts.json',prompt);write_json(target/'prompt_audit.json',audit)
    load_and_validate_prompt(target/'sam2_prompts.json')
    del predictor
    if device.startswith('cuda'):torch.cuda.empty_cache()
    return frames


def render_overlays(video, filtered, output, objects):
    palette=np.zeros((256,3),np.uint8)
    for o in objects:palette[o['id']]=tuple(int(o['color'][i:i+2],16) for i in (5,3,1))
    cap=cv2.VideoCapture(str(video));w,h=int(cap.get(3)),int(cap.get(4));fps=cap.get(5)
    encoders={}
    for direction in ('forward','reverse'):
        encoders[direction]=subprocess.Popen([find_executable('ffmpeg'),'-hide_banner','-loglevel','error','-y',
            '-f','rawvideo','-pix_fmt','bgr24','-s',f'{w}x{h}','-r',str(fps),'-i','-','-an',
            '-c:v','libx264','-threads','2','-preset','fast','-crf','20','-pix_fmt','yuv420p',
            '-movflags','+faststart',str(output/f'{direction}_overlay.mp4')],stdin=subprocess.PIPE)
    n=0
    try:
        while True:
            ok,bgr=cap.read()
            if not ok:break
            for direction,encoder in encoders.items():
                mask=cv2.imread(str(filtered/'directional_mask_frames'/direction/f'{n:05d}.png'),0)
                if mask is None:raise RuntimeError('Missing directional frame')
                overlay=bgr.copy();fg=mask!=0
                overlay[fg]=cv2.addWeighted(bgr,.5,palette[mask],.5,0)[fg]
                encoder.stdin.write(overlay.tobytes())
            n+=1
        for encoder in encoders.values():
            encoder.stdin.close()
            if encoder.wait()!=0:raise RuntimeError('Overlay encode failed')
    finally:
        cap.release()
        for encoder in encoders.values():
            if encoder.poll() is None:encoder.terminate();encoder.wait()


def initialize(args):
    root=args.work_dir;root.mkdir(parents=True,exist_ok=True)
    if (root/'manifest.json').exists():
        manifest=json.loads((root/'manifest.json').read_text())
        if manifest['model_sha256']!=sha(args.model_dir/'model.ts'):raise RuntimeError('Model changed; use a new round directory')
        return manifest
    def get(p):
        with urllib.request.urlopen(args.api+p,timeout=30) as response:return json.load(response)
    bootstrap=get('/api/bootstrap');items=[i for i in bootstrap['items'] if i['view']=='front']
    rng=random.Random(args.seed);selected=[];populations={}
    for task in sorted({i['task'] for i in items if i['episode_index'] in SEEDS}):
        pool=sorted([i for i in items if i['task']==task and i['episode_index'] not in SEEDS],key=lambda i:i['episode_index'])
        populations[task]=[i['episode_index'] for i in pool]
        selected += rng.sample(pool,2)
    seeds=[]
    for ep in SEEDS:
        item=next(i for i in items if i['episode_index']==ep)
        preview=get('/api/propagation?workspace_id='+bootstrap['workspace_id']+'&id='+item['item_id'])['preview']
        if preview['stale'] or preview['direction_review']['status']!='finalized':raise RuntimeError('Seed not finalized')
        final=Path(preview['output_path'])/preview['direction_review']['final_subdir']
        seeds.append({'episode':ep,'task':item['task'],'final_root':str(final),'run_id':preview['run_id'],
                      'mask_hashes':{p.name:sha(p) for p in sorted((final/'mask_frames').glob('*.png'))},
                      'role':'human-reviewed reference; never overwritten by this round'})
    write_json(root/'seed_reference.json',seeds)
    shutil.copy2(Path(bootstrap['dataset'])/'labels.txt',root/'labels.txt')
    records=[{'id':f'ep{i["episode_index"]:06d}','item':i,'status':'pending'} for i in sorted(selected,key=lambda i:i['episode_index'])]
    manifest={'created_at':time.time(),'dataset':bootstrap['dataset'],'seed':args.seed,'sampling':'2 random videos per seed task excluding ep0,20,40,60',
        'candidate_populations':populations,'model_dir':str(args.model_dir),'model_sha256':sha(args.model_dir/'model.ts'),
        'filter_version':FILTER_VERSION,'seed_episodes':SEEDS,'records':records,'training_selection':'NONE'}
    write_json(root/'manifest.json',manifest)
    return manifest


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--work-dir',type=Path,required=True)
    p.add_argument('--model-dir',type=Path,default=REPO/'segdata/temp/newur/unet_resnet18_front_seed4_r1')
    p.add_argument('--api',default='http://127.0.0.1:7861')
    p.add_argument('--device',default='cuda:1');p.add_argument('--seed',type=int,default=42)
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--refresh-prompts',action='store_true',help='Preserve initial prompts in history and regenerate before propagation')
    p.add_argument('--num-shards',type=int,default=1);p.add_argument('--shard-index',type=int,default=0)
    p.add_argument('--offload-state-to-cpu',action='store_true')
    args=p.parse_args();args.work_dir=args.work_dir.resolve();args.model_dir=args.model_dir.resolve()
    if not 0<=args.shard_index<args.num_shards:raise ValueError('Invalid shard index')
    torch.set_num_threads(4);cv2.setNumThreads(1)
    manifest=initialize(args)
    shutil.copy2(REPO/'tools/unet_sam2_review.html',args.work_dir/'index.html')
    def update(record,status):
        if status!='failed':record.pop('error',None)
        record.update(status=status,updated_at=time.time())
        with (args.work_dir/'.manifest.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            current=json.loads((args.work_dir/'manifest.json').read_text())
            current['records']=[record if r['id']==record['id'] else r for r in current['records']]
            write_json(args.work_dir/'manifest.json',current)
        print(record['id'],status,flush=True)
    for record in manifest['records'][args.shard_index::args.num_shards]:
        if record['status']=='complete':continue
        item=record['item'];root=args.work_dir/record['id'];root.mkdir(exist_ok=True)
        try:
            unet=root/'unet'
            if not (unet/'complete.json').is_file():
                update(record,'unet')
                if unet.exists():unet.rename(root/f'unet.incomplete.{int(time.time())}')
                run_unet(args.model_dir,Path(item['video_path']),unet,args.device)
            prompt_root=root/'prompts'
            if args.refresh_prompts and (prompt_root/'sam2_prompts.json').exists():
                if (root/'sam2').exists():raise RuntimeError('Cannot change prompts after SAM2 outputs exist; use a new round')
                history=root/'prompt_history';history.mkdir(exist_ok=True)
                prompt_root.rename(history/f'prompts_{time.time_ns()}')
            if not (prompt_root/'sam2_prompts.json').is_file():
                update(record,'prompts')
                frames=generate_prompts(item,unet,prompt_root,args.model_dir,args.device)
            else:frames=json.loads((prompt_root/'sam2_prompts.json').read_text())['prompt_frames']
            record['prompt_frames']=frames
            record['prompt_json']=str((prompt_root/'sam2_prompts.json').relative_to(args.work_dir))
            update(record,'prompts_ready')
            if args.prepare_only:continue
            output=root/'sam2'/f'{Path(item["video_path"]).stem}_sam2_masks'
            if not (output/'complete.json').is_file():
                update(record,'sam2_bidirectional')
                partial=output.with_name(output.name+'.partial')
                if partial.exists():
                    partial.rename(partial.with_name(partial.name+f'.interrupted.{time.time_ns()}'))
                with (root/'sam2.log').open('a') as log:
                    command=[sys.executable,str(REPO/'tools/sam2_prompts_to_masks.py'),'--prompt-json',str(prompt_root/'sam2_prompts.json'),
                        '--output-root',str(root/'sam2'),'--device',args.device,'--save-bidirectional','--propagate-prompt-frames',
                        '--composition-mode','priority','--compile-mode','none']
                    if args.offload_state_to_cpu:command.append('--offload-state-to-cpu')
                    result=subprocess.run(command,cwd=REPO,stdout=log,stderr=subprocess.STDOUT)
                    if result.returncode:raise RuntimeError(f'SAM2 failed: see {root}/sam2.log')
            update(record,'scoring')
            completion=json.loads((unet/'complete.json').read_text());objects=completion['objects']
            filtered=score_video(unet,output,root/'scores',objects,item['frame_count'],item['fps'],frames)
            if not all((root/f'{d}_overlay.mp4').is_file() for d in ('forward','reverse')):
                render_overlays(item['video_path'],filtered,root,objects)
            relative=lambda path:str(path.relative_to(args.work_dir))
            record['media']={'unet':{'overlay':relative(unet/'overlay.mp4'),'segmentation':relative(unet/'mask_visual.mp4')},
                **{d:{'overlay':relative(root/f'{d}_overlay.mp4'),'segmentation':relative(filtered/f'mask_visual_{d}.mp4')} for d in ('forward','reverse')}}
            record['scores']=relative(root/'scores/frame_scores.json');record['csv']=relative(root/'scores/frame_scores.csv')
            record['summary']=json.loads((root/'scores/summary.json').read_text())
            update(record,'complete')
        except Exception as exc:
            record['error']=str(exc);update(record,'failed');print(repr(exc),flush=True)
    manifest=json.loads((args.work_dir/'manifest.json').read_text())
    if all(r['status']=='complete' for r in manifest['records']):
        write_json(args.work_dir/'complete.json',{'finished_at':time.time(),'videos':len(manifest['records']),
            'frames':sum(r['item']['frame_count'] for r in manifest['records']),'training_selection':'NONE'})
    elif not args.prepare_only and any(r['status']=='failed' for r in manifest['records'][args.shard_index::args.num_shards]):
        raise SystemExit(1)


if __name__=='__main__':main()
