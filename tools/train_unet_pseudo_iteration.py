"""Eight-video U-Net pseudo-label iteration with held-out reviewed seed tests."""
import argparse
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
from torch.utils.data import DataLoader

from sam2_direction_review import component_filter_policy, write_json
from train_reviewed_resnet_unet import LABELS, ResNetUNet
from train_light_semseg import ImageSemSegDataset, prebuilt_image_samples, evaluate
from unet_video_preview import run as segment

REPO = Path(__file__).resolve().parents[1]
SEEDS = [0, 20, 40, 60]


def get(path):
    return json.load(urllib.request.urlopen('http://127.0.0.1:7861'+path, timeout=60))


def initialize(root, previous):
    if (root/'manifest.json').exists():
        return json.loads((root/'manifest.json').read_text())
    root.mkdir(parents=True, exist_ok=True)
    b = get('/api/bootstrap')
    old = json.loads((previous/'manifest.json').read_text())
    selected = [r['item'] for r in old['records']]
    items = [i for i in b['items'] if i['view']=='front']
    rng = random.Random(43)
    # Two candidate videos per reviewed seed: o_n has two seeds, hence four candidates.
    for task in sorted({i['task'] for i in items if i['episode_index'] in SEEDS}):
        target = 2*sum(i['task']==task and i['episode_index'] in SEEDS for i in items)
        need = target-sum(i['task']==task for i in selected)
        pool = sorted([i for i in items if i['task']==task and i['episode_index'] not in SEEDS
                       and i['episode_index'] not in {j['episode_index'] for j in selected}], key=lambda i:i['episode_index'])
        selected += rng.sample(pool, need)
    assert len(selected)==8 and not ({i['episode_index'] for i in selected}&set(SEEDS))
    seeds=[]
    for ep in SEEDS:
        item=next(i for i in items if i['episode_index']==ep)
        preview=get('/api/propagation?workspace_id='+b['workspace_id']+'&id='+item['item_id'])['preview']
        review=preview['direction_review']
        assert not preview['stale'] and review['status']=='finalized' and review['unresolved']==0
        final=Path(preview['output_path'])/review['final_subdir']
        seeds.append({'episode':ep,'item':item,'final_root':str(final),'run_id':preview['run_id'],
                      'hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((final/'mask_frames').glob('*.png'))}})
    write_json(root/'seed_test_snapshot.json',seeds)
    records=[{'id':f'ep{i["episode_index"]:06d}','item':i,'status':'pending'} for i in sorted(selected,key=lambda i:i['episode_index'])]
    m={'workflow':'unet_only_pseudo8_training_seed4_test','model_dir':str(REPO/'segdata/temp/newur/unet_resnet18_front_seed4_r1'),
       'teacher_trained_on_test_seeds':True,'sampling_seed':43,'split_seed':42,'filter':component_filter_policy(),
       'test_episodes':SEEDS,'records':records,'model_selection':'pseudo_label_validation_only',
       'exclusion':'Any suspected class OR any low-confidence IoU boundary review class excludes the entire frame.'}
    write_json(root/'manifest.json',m)
    return m


def build_cache(root, model, prior_cache=None):
    cache=model/'cache';manifest_path=cache/'manifest.json'
    if manifest_path.exists():return cache
    m=json.loads((root/'manifest.json').read_text());accepted=[];excluded=[]
    for r in m['records']:
        scores=json.loads((root/'unet_global_r2'/r['id']/'frame_scores.json').read_text())
        for f in scores['frames']:
            bad=f['suspected_classes'];review=f.get('uncertain_classes',[])
            if bad or review:
                excluded.append({'episode':r['item']['episode_index'],'frame':f['frame'],'suspected_classes':bad,'review_classes':review})
            else:accepted.append((r['item']['episode_index'],f['frame']))
    previous={}
    if prior_cache is not None:
        prior=json.loads((prior_cache/'manifest.json').read_text())
        previous={(s['episode'],s['frame']):s for s in prior['samples']}
        prior_episodes={ep for ep,_ in previous}
        assert {p for p in accepted if p[0] in prior_episodes}==set(previous), 'Previous accepted frames changed'
    new=[pair for pair in accepted if pair not in previous]
    random.Random(42).shuffle(new);new_train_count=int(.8*len(new))
    split={pair:('train' if index<new_train_count else 'val') for index,pair in enumerate(new)}
    split.update({pair:s['split'] for pair,s in previous.items()})
    train_count=sum(s=='train' for s in split.values())
    assert len(split)==len(accepted) and not ({ep for ep,_ in split}&set(SEEDS))
    write_json(root/'frame_selection.json',{'rule':m['exclusion'],'accepted':len(accepted),'excluded':len(excluded),
               'excluded_frames':excluded,'train':train_count,'val':len(accepted)-train_count})
    counts={s:np.zeros(len(LABELS),np.int64) for s in ['train','val']};records=[]
    for r in m['records']:
        item=r['item'];ep=item['episode_index']
        if prior_cache is not None and ep in prior_episodes:
            for pair,s in previous.items():
                if pair[0]!=ep:continue
                rel=Path(s['split'])/item['video_key']/f'file-{ep:03d}'/f'{pair[1]:05d}'
                for folder,suffix in [('images','.jpg'),('masks','.png')]:
                    target=cache/folder/rel.with_suffix(suffix);target.parent.mkdir(parents=True,exist_ok=True)
                    shutil.copy2(prior_cache/folder/rel.with_suffix(suffix),target)
                resized=cv2.imread(str(cache/'masks'/rel.with_suffix('.png')),0)
                assert resized is not None and resized.shape==(456,600)
                counts[s['split']]+=np.bincount(resized.ravel(),minlength=len(LABELS))
                records.append(s.copy())
            print('CACHED FROZEN PRIOR',r['id'],flush=True)
            continue
        cap=cv2.VideoCapture(item['video_path'])
        for frame in range(item['frame_count']):
            ok,bgr=cap.read();assert ok,(ep,frame)
            if (ep,frame) not in split:continue
            mask_path=root/r['id']/'unet/mask_frames'/f'{frame:05d}.png'
            mask=cv2.imread(str(mask_path),0);assert mask is not None and mask.shape==bgr.shape[:2] and mask.max()<len(LABELS)
            name=split[(ep,frame)];rel=Path(name)/item['video_key']/f'file-{ep:03d}'/f'{frame:05d}'
            image=cache/'images'/rel.with_suffix('.jpg');target=cache/'masks'/rel.with_suffix('.png')
            image.parent.mkdir(parents=True,exist_ok=True);target.parent.mkdir(parents=True,exist_ok=True)
            resized=cv2.resize(mask,(600,456),interpolation=cv2.INTER_NEAREST)
            assert cv2.imwrite(str(image),cv2.resize(bgr,(600,456),interpolation=cv2.INTER_AREA),[cv2.IMWRITE_JPEG_QUALITY,95])
            assert cv2.imwrite(str(target),resized)
            counts[name]+=np.bincount(resized.ravel(),minlength=len(LABELS))
            records.append({'episode':ep,'frame':frame,'split':name,'source_mask_sha256':hashlib.sha256(mask_path.read_bytes()).hexdigest()})
        cap.release();print('CACHED',r['id'],flush=True)
    write_json(manifest_path,{'labels':LABELS,'seed':42,'image_size':[600,456],
               'split_method':'frozen_prior_split_plus_new_random_frame_80_20' if prior_cache is not None else 'global_random_frame_80_20_after_exclusion',
               'prior_cache':str(prior_cache) if prior_cache is not None else None,
               'test_episodes':SEEDS,'samples':records,'train':train_count,'val':len(accepted)-train_count,
               'pixel_counts':{s:c.tolist() for s,c in counts.items()},'source_round':str(root),
               'teacher_trained_on_test_seeds':m.get('teacher_trained_on_test_seeds',True),
               'ancestor_teacher_trained_on_test_seeds':m.get('ancestor_teacher_trained_on_test_seeds',False)})
    return cache


def seed_test(root, model, device):
    # Called only after training, checkpoint selection and export are finished.
    result=model/'test_seed4.json'
    if result.exists():return
    sources=json.loads((root/'seed_test_snapshot.json').read_text());cache=model/'test_cache'
    for source in sources:
        ep=source['episode'];item=source['item'];cap=cv2.VideoCapture(item['video_path'])
        for frame in range(item['frame_count']):
            ok,bgr=cap.read();assert ok
            original=Path(source['final_root'])/'mask_frames'/f'{frame:05d}.png'
            assert hashlib.sha256(original.read_bytes()).hexdigest()==source['hashes'][original.name], 'Seed reference changed'
            mask=cv2.imread(str(original),0);assert mask is not None and mask.shape==bgr.shape[:2]
            rel=Path('test')/item['video_key']/f'file-{ep:03d}'/f'{frame:05d}'
            image=cache/'images'/rel.with_suffix('.jpg');target=cache/'masks'/rel.with_suffix('.png')
            image.parent.mkdir(parents=True,exist_ok=True);target.parent.mkdir(parents=True,exist_ok=True)
            assert cv2.imwrite(str(image),cv2.resize(bgr,(600,456),interpolation=cv2.INTER_AREA),[cv2.IMWRITE_JPEG_QUALITY,95])
            assert cv2.imwrite(str(target),cv2.resize(mask,(600,456),interpolation=cv2.INTER_NEAREST))
        cap.release();print('TEST CACHED',ep,flush=True)
    samples=prebuilt_image_samples(cache,'test');net=ResNetUNet(pretrained=False).to(device)
    ck=torch.load(model/'best.pt',map_location=device,weights_only=False);net.load_state_dict(ck['model']);net.eval()
    def assess(group):
        data=ImageSemSegDataset(group,(600,456),False,0.)
        loader=DataLoader(data,batch_size=16,num_workers=4,pin_memory=True)
        with torch.inference_mode():metrics=evaluate(net,loader,torch.device(device),len(LABELS))
        return {'frames':len(group),'foreground_miou':float(np.mean([metrics[f'iou_{i}'] for i in range(1,len(LABELS))])),
                'metrics':metrics,'per_class_iou':{name:metrics[f'iou_{i}'] for i,name in enumerate(LABELS)}}
    total=assess(samples);per_episode={}
    for ep in SEEDS:per_episode[str(ep)]=assess([s for s in samples if s.file_index==ep])
    write_json(result,{'best_epoch':ck['epoch'],'test_episodes':SEEDS,'evaluation_resolution':[600,456],
               'test_used_for_training_or_selection':False,'teacher_trained_on_test_seeds':True,
               'interpretation':'Student-training-held-out reference test; teacher learned these seeds, so not a fully independent generalization test.',
               'all':total,'per_episode':per_episode})
    print('TEST COMPLETE',total,flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--work-dir',type=Path,required=True);p.add_argument('--model-dir',type=Path,required=True)
    p.add_argument('--device',default='cuda:1');args=p.parse_args();root=args.work_dir.resolve();model=args.model_dir.resolve()
    torch.set_num_threads(4);cv2.setNumThreads(1);model.mkdir(parents=True,exist_ok=True)
    previous=REPO/'segdata/temp/newur/unet_sam2_round1/contact_filter_r1'
    m=initialize(root,previous)
    for r in m['records']:
        folder=root/r['id'];folder.mkdir(exist_ok=True);unet=folder/'unet'
        if (previous/r['id']/'unet/complete.json').exists() and not unet.exists():
            unet.symlink_to(previous/r['id']/'unet',target_is_directory=True)
        if not (unet/'complete.json').exists():
            if unet.exists():unet.rename(folder/f'unet.incomplete.{time.time_ns()}')
            segment(Path(m['model_dir']),Path(r['item']['video_path']),unet,args.device)
        r.update(status='complete',media={'unet':{'overlay':str((unet/'overlay.mp4').relative_to(root)),
                                                  'segmentation':str((unet/'mask_visual.mp4').relative_to(root))}})
        write_json(root/'manifest.json',m);print('ANNOTATED',r['id'],flush=True)
    for script,name in [('unet_temporal_scores.py','unet_temporal_r2'),('unet_global_temporal_scores.py','unet_global_r2')]:
        if not (root/name/'complete.json').exists():
            subprocess.run([sys.executable,str(REPO/'tools'/script),'--round-root',str(root),'--output-name',name],check=True)
    cache=build_cache(root,model)
    write_json(root/'status.json',{'stage':'training','model_dir':str(model),'updated_at':time.time()})
    if not (model/'complete.json').exists():
        command=[sys.executable,str(REPO/'tools/train_reviewed_resnet_unet.py'),'--work-dir',str(model),'--prepared-cache',str(cache),
                 '--episodes',*[str(r['item']['episode_index']) for r in m['records']], '--device',args.device,
                 '--batch-size','16','--workers','6','--epochs','100','--min-epochs','25','--patience','12']
        if (model/'latest.pt').exists():command.append('--resume')
        with (model/'train.log').open('a') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
    write_json(root/'status.json',{'stage':'seed_test','model_dir':str(model),'updated_at':time.time()})
    seed_test(root,model,args.device)
    write_json(root/'complete.json',{'videos':8,'model_dir':str(model),'test_episodes':SEEDS,'finished_at':time.time()})
    write_json(root/'status.json',{'stage':'complete','model_dir':str(model),'updated_at':time.time()})


if __name__=='__main__':main()
