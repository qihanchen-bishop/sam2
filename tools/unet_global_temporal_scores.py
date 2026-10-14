"""Whole-video robust trends for existing U-Net features and mask previews."""
import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np

from sam2_direction_review import write_json

POLICY = {
    'version': 'whole_video_robust_area_count_iou_v2',
    'iou_absolute_tolerance': .12,
    'component_transition_cost_seconds': .5,
    'curve_smooth_seconds': .35,
    'robust_initial_median_radius_seconds': .8,
    'area_relative_tolerance': .08,
    'shape_change_tolerance': .12,
    'iterations': 8,
    'area_persistent_step_baseline_seconds': .5,
    'area_persistent_step_relative_threshold': .15,
    'area_persistent_step_reference_mad_max': .05,
    'score_meaning': 'Whole-video temporal support, not segmentation accuracy.',
    'training_selection': 'NONE',
}


def rolling_median(values, radius):
    values=np.asarray(values,float)
    if not len(values):return values.copy()
    radius=min(max(1,int(radius)),max(1,len(values)-1))
    padded=np.pad(values,(radius,radius),mode='edge')
    return np.median(np.lib.stride_tricks.sliding_window_view(padded,2*radius+1),axis=1)


def count_states(counts, transition_cost):
    """Exact Potts dynamic program over the whole sequence; persistent states stay."""
    counts=np.asarray(counts,int);states=np.unique(counts);n=len(counts)
    if not n:return counts.copy()
    k=len(states);back=np.zeros((n,k),np.int16);cost=(states!=counts[0]).astype(float)
    for t in range(1,n):
        candidates=cost[:,None]+transition_cost*(states[:,None]!=states[None,:])
        back[t]=np.argmin(candidates,axis=0)
        cost=candidates[back[t],np.arange(k)]+(states!=counts[t])
    state=int(np.argmin(cost));out=np.empty(n,int)
    for t in range(n-1,-1,-1):
        out[t]=states[state]
        if t:state=int(back[t,state])
    return out


def penalized_curve(values, weights, strength):
    """Solve W + lambda D2.T D2 by band Cholesky, linear time and memory."""
    y=np.asarray(values,float);w=np.maximum(np.asarray(weights,float),1e-4);n=len(y)
    if n<3:return y.copy()
    diag=np.zeros(n);off1=np.zeros(n-1);off2=np.full(n-2,strength)
    diag[:-2]+=strength;diag[1:-1]+=4*strength;diag[2:]+=strength;diag+=w
    off1[:-1]-=2*strength;off1[1:]-=2*strength
    l0=np.zeros(n);l1=np.zeros(n);l2=np.zeros(n);rhs=w*y;forward=np.zeros(n)
    for i in range(n):
        if i>=2:l2[i]=off2[i-2]/l0[i-2]
        if i>=1:l1[i]=(off1[i-1]-l2[i]*l1[i-1])/l0[i-1]
        l0[i]=np.sqrt(max(1e-12,diag[i]-l1[i]**2-l2[i]**2))
        v=rhs[i]
        if i>=1:v-=l1[i]*forward[i-1]
        if i>=2:v-=l2[i]*forward[i-2]
        forward[i]=v/l0[i]
    out=np.zeros(n)
    for i in range(n-1,-1,-1):
        v=forward[i]
        if i+1<n:v-=l1[i+1]*out[i+1]
        if i+2<n:v-=l2[i+2]*out[i+2]
        out[i]=v/l0[i]
    return out


def robust_curve(values, base_weights, fps, relative_floor, policy):
    values=np.asarray(values,float);radius=max(1,round(fps*policy['robust_initial_median_radius_seconds']))
    curve=rolling_median(values,radius)
    strength=max(1.,(fps*policy['curve_smooth_seconds'])**4)
    weights=np.asarray(base_weights,float).copy()
    for _ in range(policy['iterations']):
        residual=values-curve
        noise=rolling_median(np.abs(residual),radius)*1.4826
        scale=np.maximum(noise,relative_floor*np.maximum(np.maximum(np.abs(curve),np.abs(values)),20. if relative_floor>.001 else 1.))
        ratio=np.abs(residual)/np.maximum(4.685*scale,1e-5)
        robust=np.where(ratio<1,(1-ratio**2)**2,.01)
        weights=np.maximum(.001,np.asarray(base_weights)*np.maximum(.01,robust))
        curve=penalized_curve(values,weights,strength)
    noise=rolling_median(np.abs(values-curve),radius)*1.4826
    return np.maximum(curve,0),noise,weights


def persistent_area_steps(areas, fps, policy):
    """Only permit a discontinuity with stable pre/post levels lasting >=1 s."""
    areas=np.asarray(areas,float);n=len(areas);w=max(3,round(fps*policy['area_persistent_step_baseline_seconds']))
    out=[]
    for t in range(w,n-2*w+1):
        if abs(areas[t]-areas[t-1])/max(areas[t],areas[t-1],20.) < .12:continue
        pre=areas[t-w:t];post=areas[t:t+2*w]
        a=float(np.median(pre));b=float(np.median(post));scale=max(a,b,20.)
        if abs(a-b)/scale<policy['area_persistent_step_relative_threshold']:continue
        if max(np.median(np.abs(pre-a)),np.median(np.abs(post-b)))/scale>policy['area_persistent_step_reference_mad_max']:continue
        # The immediate new frames must actually belong to the new level.
        if abs(areas[t]-b)/scale>.10 or abs(areas[t-1]-a)/scale>.10:continue
        if np.mean(np.abs(post-b)/scale<=.10)<.9:continue
        if out and t-out[-1]<w:continue
        out.append(t)
    return out


def assess_global(areas, counts, ious, differences, fps, policy=None, local=None):
    policy=dict(POLICY if policy is None else policy)
    a=np.asarray(areas,float);c=np.asarray(counts,int);n=len(a)
    expected_count=count_states(c,max(1.,fps*policy['component_transition_cost_seconds']))
    count_outlier=c!=expected_count
    boundaries=sorted(set([0,n]+(np.flatnonzero(np.diff(expected_count))+1).tolist()+persistent_area_steps(a,fps,policy)))
    expected_area=np.zeros(n);noise=np.zeros(n);weights=np.zeros(n)
    for start,end in zip(boundaries[:-1],boundaries[1:]):
        reference=(~count_outlier[start:end]).astype(float)*.95+.05
        curve,mad,w=robust_curve(a[start:end],reference,fps,.005,policy)
        expected_area[start:end]=curve;noise[start:end]=mad;weights[start:end]=w
    scale=np.maximum(np.maximum(a,expected_area),20.)
    area_error=np.abs(a-expected_area)/scale
    # Estimate noise from fit residuals after suppressing suspect observations.
    # Keep a floor so small, harmless boundary jitter does not become an outlier.
    noise_relative=noise/scale
    tolerance=np.maximum(policy['area_relative_tolerance'],4*noise_relative)
    area_outlier=area_error>tolerance
    progressive=np.zeros(n,bool)
    if local is not None:
        for t,row in enumerate(local):
            progressive[t]=any(d.get('area',{}).get('progressive_support',False)
                               and d.get('components',{}).get('reliable',False)
                               and d.get('components',{}).get('expected')==int(expected_count[t])
                               for d in [row['past'],row['future']])
        area_outlier &= ~progressive
    trusted=~(count_outlier|area_outlier)
    signed=np.zeros(n);signed[1:]=np.diff(a)/np.maximum(np.maximum(a[:-1],a[1:]),20.)
    iou=np.asarray([np.nan if v is None else v for v in ious],float)
    edge_trusted=np.zeros(n,bool);edge_trusted[1:]=trusted[:-1]&trusted[1:]
    valid=np.isfinite(iou)
    valid[0]=False
    # Fit the actual adjacent-IoU sequence, without subtracting area changes.
    # Missing edges have almost no fit weight and cannot be outliers.
    filled=np.interp(np.arange(n),np.flatnonzero(valid),iou[valid]) if valid.any() else np.ones(n)
    iou_reference,iou_noise,_=robust_curve(filled,np.where(valid,1.,.001),fps,.0002,policy)
    iou_reference=np.clip(iou_reference,0,1)
    iou_tol=np.maximum(policy['iou_absolute_tolerance'],4*iou_noise)
    edge_bad=valid & ((iou_reference-filled)>iou_tol) & (np.asarray(differences)>16)
    shape=1-filled;shape_reference=1-iou_reference;shape_tol=iou_tol
    edges=[]
    for t in range(1,n):
        if edge_bad[t]:
            edges.append({'previous_frame':t-1,'frame':t,'iou':float(iou[t]),
                          'expected_iou':float(iou_reference[t]),'tolerance':float(iou_tol[t]),
                          'difference_pixels':int(differences[t]),
                          'endpoints_area_count_normal':bool(edge_trusted[t])})
    records=[]
    for t in range(n):
        # A normal frame surrounded by rejected frames must not inherit their
        # bad IoUs. A shape-only spike still has two trusted, discordant edges.
        incoming=t>0 and bool(edge_trusted[t]);outgoing=t+1<n and bool(edge_trusted[t+1])
        previous_bad=bool(t>0 and edge_bad[t]);next_bad=bool(t+1<n and edge_bad[t+1])
        shape_outlier=bool(incoming and outgoing and previous_bad and next_bad)
        boundary_review=bool((previous_bad or next_bad) and not shape_outlier)
        reasons=[]
        if count_outlier[t]:reasons.append('global_component_state')
        if area_outlier[t]:reasons.append('global_area_curve')
        if shape_outlier:reasons.append('two_sided_iou_outlier')
        score=100.
        if count_outlier[t]:score=0.
        if area_outlier[t]:score=min(score,100*(1-float(np.clip((area_error[t]-tolerance[t])/tolerance[t],0,1))))
        if shape_outlier:
            severity=min((shape[t]-shape_reference[t]-shape_tol[t])/shape_tol[t],
                         (shape[t+1]-shape_reference[t+1]-shape_tol[t+1])/shape_tol[t+1])
            score=min(score,100*(1-float(np.clip(severity,0,1))))
        absent=bool(a[t]==0 and expected_count[t]==0)
        records.append({
            'expected_area':float(expected_area[t]),'expected_component_count':int(expected_count[t]),
            'area_relative_error':float(area_error[t]),'area_tolerance':float(tolerance[t]),
            'fit_weight':float(weights[t]),'progressive_area_support':bool(progressive[t]),
            'component_outlier':bool(count_outlier[t]),'area_outlier':bool(area_outlier[t]),
            'shape_change':float(shape[t]),'expected_shape_change':float(shape_reference[t]),
            'expected_iou_previous':float(iou_reference[t]) if valid[t] else None,
            'expected_iou_next':float(iou_reference[t+1]) if t+1<n and valid[t+1] else None,
            'iou_tolerance_previous':float(iou_tol[t]) if valid[t] else None,
            'iou_tolerance_next':float(iou_tol[t+1]) if t+1<n and valid[t+1] else None,
            'iou_previous_outlier':previous_bad,'iou_next_outlier':next_bad,
            'iou_boundary_review':boundary_review,
            'iou_frame_confidence':'high' if shape_outlier else 'corroborated' if boundary_review and (count_outlier[t] or area_outlier[t]) else 'low' if boundary_review else 'none',
            'shape_tolerance':float(shape_tol[t]),'incoming_reference_valid':bool(incoming),
            'outgoing_reference_valid':bool(outgoing),'shape_outlier':shape_outlier,
            'reasons':reasons,'status':'suspected_abrupt' if reasons else 'absent_in_window' if absent else 'stable_or_gradual',
            'score':None if absent and not reasons else score,'suspected':bool(reasons),
        })
    return records,{'curve_segment_boundaries':boundaries,'component_transition_cost':max(1.,fps*policy['component_transition_cost_seconds']), 'iou_outlier_edges':edges}


def score_record(root, source, output, policy):
    ep=source['id'];cache=root/'unet_temporal_r2'/ep/'features.json'
    features=json.loads(cache.read_text())['features']
    local=json.loads((root/'unet_temporal_r2'/ep/'frame_scores.json').read_text())['frames']
    objects=json.loads((root/ep/'unet/complete.json').read_text())['objects'];fps=source['item']['fps'];n=len(local)
    model={}
    for o in objects:
        cid=str(o['id']);name=o['name']
        series,metadata=assess_global(*(features[k][cid] for k in ['areas','counts','ious','differences']),fps,policy,[r['classes'][name] for r in local])
        model[name]=metadata
        for t,s in enumerate(series):
            c=local[t]['classes'][name];c['local_status']=c['status'];c['local_score']=c['score']
            c['global']=s;c.update(status=s['status'],score=s['score'],suspected=s['suspected'])
    for r in local:
        values=[c['score'] for c in r['classes'].values() if c['score'] is not None]
        r['score']=min(values) if values else None;r['min_known_score']=r['score']
        r['suspected_classes']=[name for name,c in r['classes'].items() if c['suspected']]
        r['uncertain_classes']=[name for name,c in r['classes'].items() if c['global']['iou_boundary_review'] and not c['suspected']]
        r['absent_classes']=[name for name,c in r['classes'].items() if c['status']=='absent_in_window']
    output.mkdir(parents=True,exist_ok=True)
    write_json(output/'frame_scores.json',{'policy':policy,'global_models':model,'frames':local})
    write_json(output/'iou_outlier_edges.json',{name:meta['iou_outlier_edges'] for name,meta in model.items()})
    intervals=[]
    for row in local:
        names=sorted(set(row['suspected_classes']+row['uncertain_classes']))
        if not names:continue
        if intervals and intervals[-1]['end']==row['frame']-1 and intervals[-1]['classes']==names:intervals[-1]['end']=row['frame']
        else:intervals.append({'start':row['frame'],'end':row['frame'],'classes':names})
    write_json(output/'suspected_intervals.json',intervals)
    header=['frame','time_seconds','score','suspected_classes','uncertain_classes']
    basic=['score','status','area','component_count','iou_previous','iou_next','area_change_acceleration','component_change_acceleration','iou_change_acceleration']
    global_keys=['expected_area','expected_component_count','area_relative_error','area_tolerance','fit_weight','component_outlier','area_outlier','shape_outlier','expected_iou_previous','expected_iou_next','iou_tolerance_previous','iou_tolerance_next','iou_previous_outlier','iou_next_outlier','iou_boundary_review','iou_frame_confidence']
    for o in objects:header += [o['name']+'_'+key for key in basic+global_keys]
    with (output/'frame_scores.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,header);writer.writeheader()
        for r in local:
            flat={'frame':r['frame'],'time_seconds':r['time_seconds'],'score':r['score'],'suspected_classes':';'.join(r['suspected_classes']),'uncertain_classes':';'.join(r['uncertain_classes'])}
            for name,c in r['classes'].items():
                flat.update({name+'_'+k:c[k] for k in basic});flat.update({name+'_'+k:c['global'][k] for k in global_keys})
            writer.writerow(flat)
    summary={'frame_count':n,'suspected_frames':sum(bool(r['suspected_classes']) for r in local),
             'per_class_suspected':{name:sum(r['classes'][name]['suspected'] for r in local) for name in model},
             'iou_boundary_review_frames':sum(bool(r['uncertain_classes']) for r in local),
             'iou_outlier_edges':sum(len(meta['iou_outlier_edges']) for meta in model.values()),
             'suspected_interval_count':len(intervals),'training_selection':'NONE'}
    write_json(output/'summary.json',summary)
    return summary


def main():
    p=argparse.ArgumentParser();p.add_argument('--round-root',type=Path,required=True);p.add_argument('--output-name',default='unet_global_r1')
    p.add_argument('--web-prefix',default='')
    args=p.parse_args();root=args.round_root.resolve();output=root/args.output_name
    if output.parent!=root:raise ValueError('Output must be a child of round root')
    output.mkdir(exist_ok=True);source=json.loads((root/'manifest.json').read_text());records=[]
    shutil.copy2(Path(__file__).with_name('unet_global_temporal_review.html'),output/'index.html')
    manifest={'policy':POLICY,'records':records,'training_selection':'NONE','source_features':'unet_temporal_r2'}
    for r in source['records']:
        record={'id':r['id'],'item':r['item'],'media':r['media']['unet'],'status':'scoring',
                'scores':args.web_prefix+args.output_name+'/'+r['id']+'/frame_scores.json','csv':args.web_prefix+args.output_name+'/'+r['id']+'/frame_scores.csv'}
        records.append(record);write_json(output/'manifest.json',manifest)
        record['summary']=score_record(root,r,output/r['id'],POLICY);record['status']='complete'
        write_json(output/'manifest.json',manifest);print(r['id'],record['summary'],flush=True)
    write_json(output/'complete.json',{'videos':len(records),'frames':sum(r['summary']['frame_count'] for r in records),'training_selection':'NONE'})


if __name__=='__main__':main()
