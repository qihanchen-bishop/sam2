"""Add sixteen unseen videos, train on their accepted frames plus frozen prior eight."""
import argparse
import copy
import hashlib
import json
import random
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from sam2_direction_review import FILTER_VERSION, component_filter_policy, write_json
from train_unet_pseudo_iteration import REPO, SEEDS, get, build_cache
from train_reviewed_resnet_unet import LABELS, ResNetUNet
from train_light_semseg import ImageSemSegDataset, prebuilt_image_samples, evaluate
from unet_temporal_scores import score_video
from unet_global_temporal_scores import score_record, POLICY


def initialize(root, previous, teacher):
    if (root/'manifest.json').exists():
        m=json.loads((root/'manifest.json').read_text())
        assert m['teacher_sha256']==hashlib.sha256((teacher/'model.ts').read_bytes()).hexdigest()
        return m
    root.mkdir(parents=True,exist_ok=True)
    old=json.loads((previous/'manifest.json').read_text());b=get('/api/bootstrap')
    items=[i for i in b['items'] if i['view']=='front'];old_ids={r['item']['episode_index'] for r in old['records']}
    assert len(old_ids)==8 and not old_ids&set(SEEDS)
    rng=random.Random(44);selected=[]
    for task in sorted({r['item']['task'] for r in old['records']}):
        count=2*sum(r['item']['task']==task for r in old['records'])
        pool=sorted([i for i in items if i['task']==task and i['episode_index'] not in old_ids|set(SEEDS)],key=lambda i:i['episode_index'])
        selected+=rng.sample(pool,count)
    assert len(selected)==16 and len({i['episode_index'] for i in selected})==16
    records=[{'id':r['id'],'item':r['item'],'cohort':'prior8','status':'pending'} for r in old['records']]
    records += [{'id':f'ep{i["episode_index"]:06d}','item':i,'cohort':'new16','status':'pending'} for i in selected]
    records.sort(key=lambda r:r['item']['episode_index'])
    m={'workflow':'unet_only_prior8_plus_new16','records':records,'sampling_seed':44,'split_seed':42,
       'model_dir':str(teacher),'teacher_sha256':hashlib.sha256((teacher/'model.ts').read_bytes()).hexdigest(),
       'teacher_trained_on_test_seeds':False,'ancestor_teacher_trained_on_test_seeds':True,
       'prior_round':str(previous),'filter':component_filter_policy(),
       'exclusion':old['exclusion'],'test_episodes':SEEDS,
       'training_initialization':'fresh ImageNet ResNet18 encoder and new decoder',
       'split_method':'prior train/val assignments frozen; new16 accepted frames random 80/20'}
    write_json(root/'manifest.json',m)
    shutil.copy2(previous/'seed_test_snapshot.json',root/'seed_test_snapshot.json')
    return m


def test_and_compare(model, previous_model):
    result=model/'test_seed4.json'
    if result.exists():return
    # Use byte-identical cached test inputs/labels from r2, only after best model is frozen.
    cache=previous_model/'test_cache';samples=prebuilt_image_samples(cache,'test')
    assert len(samples)==4990 and {s.file_index for s in samples}==set(SEEDS)
    device=torch.device(ARGS.device);net=ResNetUNet(pretrained=False).to(device)
    ck=torch.load(model/'best.pt',map_location=device,weights_only=False);net.load_state_dict(ck['model']);net.eval()
    def assess(group):
        ds=ImageSemSegDataset(group,(600,456),False,0.)
        loader=DataLoader(ds,batch_size=16,num_workers=4,pin_memory=True)
        with torch.inference_mode():metrics=evaluate(net,loader,device,len(LABELS))
        return {'frames':len(group),'foreground_miou':float(np.mean([metrics[f'iou_{i}'] for i in range(1,len(LABELS))])),
                'metrics':metrics,'per_class_iou':{name:metrics[f'iou_{i}'] for i,name in enumerate(LABELS)}}
    total=assess(samples);per={str(ep):assess([s for s in samples if s.file_index==ep]) for ep in SEEDS}
    write_json(result,{'best_epoch':ck['epoch'],'test_episodes':SEEDS,'evaluation_resolution':[600,456],
                      'test_used_for_training_or_selection':False,'teacher_trained_on_test_seeds':False,
                      'ancestor_teacher_trained_on_test_seeds':True,
                      'test_cache':str(cache),'test_cache_identical_to_previous':True,'all':total,'per_episode':per})
    prior=json.loads((previous_model/'test_seed4.json').read_text())
    write_json(model/'comparison_r2_r3.json',{
        'test_frames':len(samples),'test_cache_identical':True,'baseline_model':str(previous_model),
        'foreground_miou':{'prior8':prior['all']['foreground_miou'],'prior8_plus_new16':total['foreground_miou'],
                           'delta_percentage_points':100*(total['foreground_miou']-prior['all']['foreground_miou'])},
        'per_class_iou':{name:{'prior8':prior['all']['per_class_iou'][name],'prior8_plus_new16':value,
                              'delta_percentage_points':100*(value-prior['all']['per_class_iou'][name])}
                         for name,value in total['per_class_iou'].items()},
        'teacher_seed_exposure':'Seeds held out from both student trainings, but first teacher learned these seed videos.'})
    print('TEST COMPLETE',total,flush=True)


def main():
    root=ARGS.work_dir.resolve();model=ARGS.model_dir.resolve();previous=REPO/'segdata/temp/newur/unet_pseudo8_r2'
    teacher=REPO/'segdata/temp/newur/unet_resnet18_front_pseudo8_r2';model.mkdir(parents=True,exist_ok=True)
    m=initialize(root,previous,teacher);records=m['records'];lock=threading.Lock()
    global_root=root/'unet_global_r2';global_root.mkdir(exist_ok=True)
    shutil.copy2(REPO/'tools/unet_global_temporal_review.html',global_root/'index.html')
    review={'policy':POLICY,'records':[],'training_selection':'pending','source_features':'unet_temporal_r2'}
    def publish(r,status):
        with lock:
            r['status']=status
            write_json(root/'manifest.json',m)
            review['records']=[{'id':v['id'],'item':v['item'],'status':v['status'],
                               'media':v.get('media',{}).get('unet',{}),'scores':f'unet_global_r2/{v["id"]}/frame_scores.json',
                               'csv':f'unet_global_r2/{v["id"]}/frame_scores.csv',**({'summary':v['summary']} if 'summary' in v else {})}
                              for v in records]
            write_json(global_root/'manifest.json',review)
    def process(r):
        ep=r['id'];folder=root/ep;folder.mkdir(exist_ok=True);unet=folder/'unet'
        if r['cohort']=='prior8':
            if not unet.exists():unet.symlink_to(previous/ep/'unet',target_is_directory=True)
            for name in ['unet_temporal_r2','unet_global_r2']:
                target=root/name/ep
                if not target.exists():target.parent.mkdir(exist_ok=True);shutil.copytree(previous/name/ep,target)
        elif not (unet/'complete.json').exists():
            if unet.exists():unet.rename(folder/f'unet.incomplete.{time.time_ns()}')
            publish(r,'segmenting')
            cmd=[sys.executable,str(REPO/'tools/unet_video_preview.py'),'--model-dir',str(teacher),
                 '--input',r['item']['video_path'],'--output',str(unet),'--device',ARGS.device]
            with (folder/'segment.log').open('a') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
        meta=json.loads((unet/'complete.json').read_text());assert meta['component_filter']['version']==FILTER_VERSION
        assert meta['frame_count']==r['item']['frame_count']
        if r['cohort']=='new16':assert meta['model_sha256']==m['teacher_sha256']
        r['media']={'unet':{'overlay':f'{ep}/unet/overlay.mp4','segmentation':f'{ep}/unet/mask_visual.mp4'}}
        if r['cohort']=='new16' and not (global_root/ep/'summary.json').exists():
            publish(r,'scoring')
            score_video(unet,root/'unet_temporal_r2'/ep,r['item']['fps'])
            score_record(root,r,global_root/ep,POLICY)
        r['summary']=json.loads((global_root/ep/'summary.json').read_text())
        publish(r,'complete');print('READY',ep,r['cohort'],r['summary']['frame_count'],flush=True)
    write_json(root/'status.json',{'stage':'annotation_and_scoring','updated_at':time.time()})
    for r in records:
        if r['cohort']=='prior8':process(r)
    with ThreadPoolExecutor(max_workers=2) as executor:list(executor.map(process,[r for r in records if r['cohort']=='new16']))
    write_json(global_root/'complete.json',{'videos':24,'frames':sum(r['item']['frame_count'] for r in records)})
    write_json(root/'status.json',{'stage':'cache_building','updated_at':time.time()})
    cache=build_cache(root,model,teacher/'cache')
    selection=json.loads((root/'frame_selection.json').read_text())
    print('SELECTION',{k:v for k,v in selection.items() if k!='excluded_frames'},flush=True)
    write_json(model/'data_protocol.json',{**{k:v for k,v in m.items() if k!='records'},
               'training_episodes':[r['item']['episode_index'] for r in records],
               'frame_counts':{k:selection[k] for k in ['accepted','excluded','train','val']},
               'test_cache':str(teacher/'test_cache'),'test_used_for_training_or_selection':False})
    write_json(root/'status.json',{'stage':'training','model_dir':str(model),'updated_at':time.time()})
    if not (model/'complete.json').exists():
        cmd=[sys.executable,str(REPO/'tools/train_reviewed_resnet_unet.py'),'--work-dir',str(model),'--prepared-cache',str(cache),
             '--episodes',*[str(r['item']['episode_index']) for r in records],'--device',ARGS.device,
             '--batch-size','32','--workers','6','--epochs','100','--min-epochs','25','--patience','12']
        if (model/'latest.pt').exists():cmd.append('--resume')
        with (model/'train.log').open('a') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
    write_json(root/'status.json',{'stage':'seed_test','model_dir':str(model),'updated_at':time.time()})
    test_and_compare(model,teacher)
    shutil.copy2(model/'comparison_r2_r3.json',root/'comparison_r2_r3.json')
    write_json(root/'complete.json',{'videos':24,'new_videos':16,'model_dir':str(model),'finished_at':time.time()})
    write_json(root/'status.json',{'stage':'complete','model_dir':str(model),'updated_at':time.time()})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--work-dir',type=Path,required=True)
    parser.add_argument('--model-dir',type=Path,required=True);parser.add_argument('--device',default='cuda:1')
    ARGS=parser.parse_args();torch.set_num_threads(4);cv2.setNumThreads(1)
    try:main()
    except Exception as exc:
        if ARGS.work_dir.exists():write_json(ARGS.work_dir/'status.json',{'stage':'failed','error':repr(exc),'updated_at':time.time()})
        raise
