"""Transparent consistency and change-return diagnostics, never automatic training admission."""
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from sam2_direction_review import compare_components, write_json, FILTER_VERSION


def feature(mask):
    ys, xs = np.nonzero(mask)
    if not len(xs):return np.zeros(4, np.float64)
    crop=mask[ys.min():ys.max()+1,xs.min():xs.max()+1].astype(np.uint8)
    count=cv2.connectedComponents(crop,connectivity=8)[0]-1
    return np.array([len(xs), xs.mean(), ys.mean(), count], np.float64)


def distance(a, b):
    """Relative area and centroid displacement; finite and symmetric, supports batches."""
    a, b = np.asarray(a), np.asarray(b)
    area_a, area_b = a[..., 0], b[..., 0]
    area = np.abs(area_a-area_b)/np.maximum(np.maximum(area_a,area_b),20.)
    center = np.linalg.norm(a[...,1:3]-b[...,1:3],axis=-1)/np.maximum(24.,2*np.sqrt(np.maximum(area_a,area_b)))
    d = np.minimum(1.,np.maximum(area,center))
    if a.shape[-1]>=4 and b.shape[-1]>=4:
        d=np.maximum(d,np.where(a[...,3]!=b[...,3],.75,0.))
    return np.where((area_a==0)&(area_b==0),0.,np.where((area_a==0)^(area_b==0),1.,d))


def transient_scores(features, fps, window_seconds=.5):
    """Only penalize bounded excursions with matching pre/post baselines, not persistent steps.

    Up to 0.5 s long; stable 0.1 s baselines on both sides. The robust interior must
    differ from both baselines, with >=80% of its frames deviating. This is a
    diagnostic (fast real motion can also return), not proof of a label error.
    """
    f=np.asarray(features,np.float64);n=len(f);b=max(2,int(round(fps*.1)));w=max(1,int(round(fps*window_seconds)))
    penalties=np.zeros(n,np.float64);events=[]
    for start in range(b,n-b):
        pre=np.median(f[start-b:start],axis=0)
        if distance(f[start],pre)<.5:continue
        for length in range(1,min(w,n-b-start)+1):
            end=start+length
            post=np.median(f[end:end+b],axis=0)
            baseline=float(distance(pre,post))
            if baseline>.15:continue
            if np.max(distance(f[start-b:start],pre))>.2 or np.max(distance(f[end:end+b],post))>.2:continue
            body=f[start:end]
            departures=np.minimum(distance(body,pre),distance(body,post))
            if np.mean(departures>=.5)<.8 or distance(f[end-1],post)<.5:continue
            severity=float(np.median(departures)*(1-baseline))
            penalties[start:end]=np.maximum(penalties[start:end],severity)
            events.append({'start':start,'end':end-1,'severity':severity,'baseline_distance':baseline,
                           'pre_features':pre.tolist(),'post_features':post.tolist(),
                           'excursion_features':np.median(body,axis=0).tolist()})
            break
    observable=np.ones(n,bool);observable[:b]=False;observable[max(0,n-b):]=False
    return penalties,observable,events


def nullable_iou(a,b):
    union=np.count_nonzero(a|b)
    return float(np.count_nonzero(a&b)/union) if union else None


def score_video(unet_root, sam2_root, output, objects, frame_count, fps, prompt_frames):
    output.mkdir(parents=True,exist_ok=True)
    review=json.loads((sam2_root/'direction_review.json').read_text())
    unet_meta=json.loads((unet_root/'complete.json').read_text())
    if unet_meta.get('component_filter',{}).get('version')!=FILTER_VERSION or review.get('candidate_filter',{}).get('version')!=FILTER_VERSION:
        raise ValueError('All three streams must use the same current component filter')
    filtered=sam2_root/review['candidate_subdir']
    roots={'unet':unet_root/'mask_frames', 'forward':filtered/'directional_mask_frames/forward',
           'reverse':filtered/'directional_mask_frames/reverse'}
    features={s:{o['name']:[] for o in objects} for s in roots};rows=[]
    for frame in range(frame_count):
        masks={s:cv2.imread(str(p/f'{frame:05d}.png'),0) for s,p in roots.items()}
        if any(m is None for m in masks.values()) or len({m.shape for m in masks.values()})!=1:
            raise ValueError(f'Invalid/misaligned masks at {frame}')
        record={'frame':frame,'time_seconds':frame/fps,'prompt_frame':frame in prompt_frames,'classes':{}}
        for o in objects:
            name=o['name'];binary={s:m==o['id'] for s,m in masks.items()}
            for s,m in binary.items():features[s][name].append(feature(m))
            u,f,r=(binary[s] for s in ('unet','forward','reverse'))
            comparison=compare_components(f,r)
            ious={'fr':nullable_iou(f,r),'fu':nullable_iou(f,u),'ru':nullable_iou(r,u)}
            valid=[x for x in ious.values() if x is not None]
            record['classes'][name]={'iou':ious,'consistency':min(valid) if valid else None,
                'areas':{s:int(np.count_nonzero(m)) for s,m in binary.items()},
                'component_counts':comparison['component_counts'],
                'conflict_iou':comparison['A'],'conflict_components':comparison['B'],
                'different_pixels_fr':comparison['different_pixels'],
                'all_empty':not bool(valid)}
        rows.append(record)
        if frame%300==0:print(f'scoring {frame}/{frame_count}',flush=True)
    temporal={};all_events=[]
    for s in roots:
        temporal[s]={}
        for o in objects:
            name=o['name'];penalty,observable,events=transient_scores(features[s][name],fps)
            temporal[s][name]=(penalty,observable)
            all_events.extend({'source':s,'class':name,**e} for e in events)
    for row in rows:
        frame=row['frame'];scores=[];consistencies=[];penalties=[]
        for name,c in row['classes'].items():
            c['transient']={s:float(temporal[s][name][0][frame]) for s in roots}
            c['temporal_observable']=bool(temporal['unet'][name][1][frame])
            p=max(c['transient'].values());penalties.append(p)
            c['score']=None if c['consistency'] is None else 100*c['consistency']*(1-p)
            if c['score'] is not None:scores.append(c['score']);consistencies.append(c['consistency'])
        row.update(score=min(scores) if scores else 0.,consistency_min=min(consistencies) if consistencies else None,
            consistency_mean=float(np.mean(consistencies)) if consistencies else None,
            transient_max=max(penalties), temporal_observable=all(c['temporal_observable'] for c in row['classes'].values()),
            directional_conflict=any(c['conflict_iou'] or c['conflict_components'] for c in row['classes'].values()),
            all_empty_classes=[n for n,c in row['classes'].items() if c['all_empty']])
    policy={'version':'pairwise_iou_and_bounded_return_v1','score':'min_class(100 * min_valid(FR,FU,RU) * (1-max_source_transient))',
        'absent_class':'All three empty => IoU/score=null, not proof of correct invisibility; excluded from min.',
        'temporal_window_seconds':.5,'baseline_seconds':.1,'excursion_threshold':.5,'return_threshold':.15,
        'sustained_changes':'No return in window => not flagged by transient detector',
        'limitations':'Diagnostics, not calibrated accuracy. Real short actions may be flagged; shared errors can score high. Endpoints lack full temporal context.',
        'training_selection':'NONE; user selects later','spatial_conflict':'FR IoU<0.85 AND diff>16 OR component counts differ'}
    write_json(output/'frame_scores.json',{'policy':policy,'frames':rows})
    write_json(output/'transient_events.json',all_events)
    header=['frame','time_seconds','prompt_frame','score','consistency_min','consistency_mean','transient_max','temporal_observable','directional_conflict','all_empty_classes']
    for o in objects:
        header += [o['name']+'_'+k for k in ('score','iou_fr','iou_fu','iou_ru','transient_unet','transient_forward','transient_reverse','area_unet','area_forward','area_reverse')]
    with (output/'frame_scores.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,header);writer.writeheader()
        for row in rows:
            flat={k:row[k] for k in header if k in row};flat['all_empty_classes']=';'.join(row['all_empty_classes'])
            for name,c in row['classes'].items():
                flat[name+'_score']=c['score']
                flat.update({name+'_iou_'+k:v for k,v in c['iou'].items()})
                flat.update({name+'_transient_'+k:v for k,v in c['transient'].items()})
                flat.update({name+'_area_'+k:v for k,v in c['areas'].items()})
            writer.writerow(flat)
    write_json(output/'summary.json',{'frame_count':frame_count,'score_quantiles':dict(zip(['min','p10','median','p90','max'],np.quantile([r['score'] for r in rows],[0,.1,.5,.9,1]).tolist())),
        'directional_conflict_frames':sum(r['directional_conflict'] for r in rows),'transient_flagged_frames':sum(r['transient_max']>=.5 for r in rows),
        'prompt_frames':prompt_frames,'selection':'not selected for training'})
    return filtered
