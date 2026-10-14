"""Score existing U-Net masks using two-sided area/component/IoU trend checks.

No inference or mask modification. A stable past OR future supports a frame.
Scores measure temporal consistency, never calibrated semantic accuracy.
"""
import argparse
import csv
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from sam2_direction_review import FILTER_VERSION, write_json

POLICY = {
    'version': 'unet_area_components_iou_trends_v2',
    'window_seconds': .3,
    'min_context_frames': 3,
    'area_relative_tolerance': .05,
    'iou_change_tolerance': .12,
    'robust_deviation_multiplier': 4.,
    'reference_max_relative_mad': .10,
    'component_consensus_min': .67,
    'progressive_min_active_changes': 3,
    'progressive_max_opposite_variation_fraction': .10,
    'acceptance': 'Stable or gradual in past OR future; suspect only when both directions have measured deviations.',
    'area': 'Robust linear extrapolation or sustained same-direction gradual change; normalized by max(current, predicted, 20 pixels).',
    'iou': 'Compare adjacent shape-change rate to its one-sided median trend; subtract the change explainable by area.',
    'components': 'Compare current count to the one-sided modal count, only with >=67% agreement.',
    'score': '100 * (1 - min(past_deviation, future_deviation)); any stable direction gives 100. Missing/uncertain context => null.',
    'limitations': 'Temporal consistency only. Persistent wrong masks, sustained wrong plateaus, normal rapid reversals, occlusion and hidden objects need review.',
    'training_selection': 'NONE',
}


def robust_line(x, y):
    """Median pair slope and intercept; robust to short islands in the window."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    i, j = np.triu_indices(len(x), 1)
    slope = float(np.median((y[j]-y[i])/(x[j]-x[i])))
    intercept = float(np.median(y-slope*x))
    mad = float(np.median(np.abs(y-(intercept+slope*x))))
    return intercept, slope, mad


def excess(value, tolerance):
    return float(np.clip((value-tolerance)/max(tolerance, .01), 0., 1.))


def direction_check(areas, counts, shape_rates, frame, step, window, policy):
    n = len(areas)
    indices = [frame+step*k for k in range(1, window+1) if 0 <= frame+step*k < n]
    if len(indices) < policy['min_context_frames']:
        return {'available': False, 'stable_or_gradual': False, 'deviation': None,
                'reason': 'insufficient_context', 'context_frames': len(indices)}
    near = indices[0]
    x = np.arange(1, len(indices)+1, dtype=float)
    expected, slope, mad = robust_line(x, areas[indices])
    expected = max(0., expected)
    scale = max(float(areas[frame]), expected, 20.)
    area_error = abs(float(areas[frame])-expected)/scale
    area_mad = mad/scale
    area_tolerance = max(policy['area_relative_tolerance'], policy['robust_deviation_multiplier']*area_mad)
    area_reliable = area_mad <= policy['reference_max_relative_mad']
    area_deviation = excess(area_error, area_tolerance) if area_reliable else None
    # Accelerating exposure can be progressive without fitting a straight line.
    # Require several active changes: one sudden jump followed by a plateau is
    # not a progressive reference, nor is a transient spike then recovery.
    trajectory = np.r_[areas[frame], areas[indices]]
    changes = np.diff(trajectory)
    active = np.abs(changes) > max(1., .01*max(float(trajectory.max()),20.))
    variation = float(np.abs(changes).sum())
    total_change = float(trajectory[-1]-trajectory[0])
    opposite = float(np.abs(changes[changes*np.sign(total_change)<0]).sum())
    progressive = bool(np.count_nonzero(active) >= policy['progressive_min_active_changes'] and
                       variation > 0 and abs(total_change) > .02*max(float(trajectory.max()),20.) and
                       opposite/variation <= policy['progressive_max_opposite_variation_fraction'])
    if progressive:
        area_reliable = True;area_deviation = 0.

    values, frequencies = np.unique(counts[indices], return_counts=True)
    # If tied, prefer the nearer observed count; reference remains unreliable.
    modes = values[frequencies == frequencies.max()]
    expected_count = int(counts[near] if counts[near] in modes else modes[0])
    agreement = float(frequencies.max()/len(indices))
    count_reliable = agreement >= policy['component_consensus_min']
    count_deviation = float(int(counts[frame]) != expected_count) if count_reliable else None

    edge = max(frame, near)
    internal_edges = [max(a, b) for a, b in zip(indices[:-1], indices[1:])]
    reference = shape_rates[internal_edges]
    baseline = float(np.median(reference))
    shape_mad = float(np.median(np.abs(reference-baseline)))
    shape_tolerance = max(policy['iou_change_tolerance'], policy['robust_deviation_multiplier']*shape_mad)
    shape_error = max(0., float(shape_rates[edge])-baseline)
    shape_reliable = shape_mad <= policy['reference_max_relative_mad']
    shape_deviation = excess(shape_error, shape_tolerance) if shape_reliable else None
    known = [v for v in (area_deviation, count_deviation, shape_deviation) if v is not None]
    deviation = max(known) if known else None
    stable = all(v is not None and v == 0. for v in (area_deviation, count_deviation, shape_deviation))
    reasons = [name for name, v in [('area_trend', area_deviation), ('component_count', count_deviation), ('iou_change', shape_deviation)] if v is not None and v > 0]
    return {
        'available': True, 'context_frames': len(indices), 'stable_or_gradual': stable,
        'deviation': deviation, 'reasons': reasons,
        'area': {'predicted': expected, 'slope_pixels_per_frame_away': slope,
                 'relative_error': area_error, 'tolerance': area_tolerance,
                 'reference_mad_relative': area_mad, 'reliable': bool(area_reliable),
                 'progressive_support': progressive, 'active_changes': int(np.count_nonzero(active)),
                 'opposite_variation_fraction': opposite/variation if variation else 0.},
        'components': {'expected': expected_count, 'agreement': agreement, 'reliable': bool(count_reliable)},
        'iou': {'shape_change': float(shape_rates[edge]), 'reference_shape_change': baseline,
                'excess_change': shape_error, 'tolerance': shape_tolerance, 'reliable': bool(shape_reliable)},
    }


def evaluate_series(areas, counts, ious, differences, fps, policy=None):
    policy = dict(POLICY if policy is None else policy)
    areas = np.asarray(areas, float);counts = np.asarray(counts, int)
    ious = np.asarray([np.nan if x is None else x for x in ious], float)
    differences = np.asarray(differences, int)
    n = len(areas);window = max(policy['min_context_frames'], int(round(fps*policy['window_seconds'])))
    signed_area_change = np.zeros(n)
    signed_area_change[1:] = np.diff(areas)/np.maximum(np.maximum(areas[:-1], areas[1:]), 20.)
    component_change = np.r_[0, np.diff(counts)]
    shape_rates = np.nan_to_num(1-ious, nan=0.)
    shape_rates = np.maximum(0., shape_rates-np.abs(signed_area_change))
    shape_rates[differences <= 16] = 0.
    out = []
    for frame in range(n):
        past = direction_check(areas, counts, shape_rates, frame, -1, window, policy)
        future = direction_check(areas, counts, shape_rates, frame, 1, window, policy)
        available = [d for d in (past, future) if d['available']]
        absent = not np.any(areas[max(0,frame-window):min(n,frame+window+1)])
        stable = any(d['stable_or_gradual'] for d in available)
        suspect = len(available) == 2 and all(d['deviation'] is not None and d['deviation'] > 0 for d in available)
        if absent:status, score = 'absent_in_window', None
        elif stable:status, score = 'stable_or_gradual', 100.
        elif suspect:status, score = 'suspected_abrupt', 100*(1-min(d['deviation'] for d in available))
        else:status, score = 'uncertain_context', None
        adjacent_iou = lambda k: None if k < 1 or k >= n or np.isnan(ious[k]) else float(ious[k])
        iou_rate_change = None
        if frame >= 2 and not np.isnan(ious[frame]) and not np.isnan(ious[frame-1]):
            iou_rate_change = float(ious[frame]-ious[frame-1])
        out.append({
            'area': int(areas[frame]), 'component_count': int(counts[frame]),
            'iou_previous': adjacent_iou(frame), 'iou_next': adjacent_iou(frame+1),
            'area_change_previous': float(signed_area_change[frame]) if frame else None,
            'area_change_next': float(signed_area_change[frame+1]) if frame+1<n else None,
            'area_change_acceleration': float(signed_area_change[frame]-signed_area_change[frame-1]) if frame>=2 else None,
            'component_change_previous': int(component_change[frame]) if frame else None,
            'component_change_acceleration': int(component_change[frame]-component_change[frame-1]) if frame>=2 else None,
            'iou_change_acceleration': iou_rate_change,
            'past': past, 'future': future, 'status': status, 'score': score,
            'suspected': bool(suspect and not absent),
        })
    return out


def extract_features(mask_dir, objects, n):
    ids = [o['id'] for o in objects];classes = max(ids)+1
    areas = {i: [] for i in ids};counts = {i: [] for i in ids}
    ious = {i: [None] for i in ids};differences = {i: [0] for i in ids}
    previous = None
    for frame in range(n):
        mask = cv2.imread(str(mask_dir/f'{frame:05d}.png'), cv2.IMREAD_GRAYSCALE)
        if mask is None:raise ValueError(f'Missing mask: {mask_dir}/{frame:05d}.png')
        if mask.max() >= classes:raise ValueError('Unexpected mask class ID')
        pixels = np.bincount(mask.ravel(), minlength=classes)
        for cid in ids:
            areas[cid].append(int(pixels[cid]))
            binary = (mask == cid).astype(np.uint8);x,y,w,h = cv2.boundingRect(binary)
            count = cv2.connectedComponents(binary[y:y+h,x:x+w], connectivity=8)[0]-1 if w and h else 0
            counts[cid].append(count)
        if previous is not None:
            if previous.shape != mask.shape:raise ValueError('Mask shapes changed')
            joint = np.bincount((previous.ravel().astype(np.int16)*classes+mask.ravel()).astype(np.int16), minlength=classes*classes).reshape(classes,classes)
            for cid in ids:
                inter = int(joint[cid,cid]);union = int(joint[cid,:].sum()+joint[:,cid].sum()-inter)
                ious[cid].append(inter/union if union else None)
                differences[cid].append(union-inter)
        previous = mask
        if frame % 500 == 0:print(f'features {frame}/{n}', flush=True)
    return {'areas': areas, 'counts': counts, 'ious': ious, 'differences': differences}


def score_video(mask_root, output, fps, policy=None):
    output.mkdir(parents=True, exist_ok=True)
    meta = json.loads((mask_root/'complete.json').read_text())
    if meta.get('component_filter', {}).get('version') != FILTER_VERSION:
        raise ValueError('Requires current uniformly filtered U-Net masks')
    objects = meta['objects'];n = meta['frame_count'];policy = dict(POLICY if policy is None else policy)
    cache = output/'features.json'
    if cache.exists():
        cached=json.loads(cache.read_text())
        if cached['source'] != str(mask_root.resolve()) or cached['frame_count'] != n or cached['source_metadata'] != meta:
            raise ValueError('Feature cache source changed; choose a new output directory')
        features = {k:{int(i):v for i,v in values.items()} for k,values in cached['features'].items()}
    else:
        features = extract_features(mask_root/'mask_frames', objects, n)
        write_json(cache, {'source': str(mask_root.resolve()), 'frame_count': n, 'source_metadata': meta, 'features': features})
    results = {}
    for obj in objects:
        cid = obj['id'];print('trends', obj['name'], flush=True)
        results[obj['name']] = evaluate_series(*(features[k][cid] for k in ('areas','counts','ious','differences')), fps, policy)
    rows = []
    for frame in range(n):
        cls = {name: values[frame] for name,values in results.items()}
        present = [c for c in cls.values() if c['status'] != 'absent_in_window']
        known = [c['score'] for c in present if c['score'] is not None]
        rows.append({'frame': frame, 'time_seconds': frame/fps, 'classes': cls,
                     'score': min(known) if known and len(known)==len(present) else None,
                     'min_known_score': min(known) if known else None,
                     'suspected_classes': [name for name,c in cls.items() if c['suspected']],
                     'uncertain_classes': [name for name,c in cls.items() if c['status']=='uncertain_context'],
                     'absent_classes': [name for name,c in cls.items() if c['status']=='absent_in_window']})
    intervals = []
    for row in rows:
        names = row['suspected_classes']
        if not names:continue
        if intervals and intervals[-1]['end'] == row['frame']-1 and intervals[-1]['classes'] == names:
            intervals[-1]['end'] = row['frame']
        else:intervals.append({'start': row['frame'], 'end': row['frame'], 'classes': names})
    write_json(output/'frame_scores.json', {'policy': policy, 'frames': rows})
    write_json(output/'suspected_intervals.json', intervals)
    metrics = ['score','status','area','component_count','iou_previous','iou_next',
               'area_change_previous','area_change_next','area_change_acceleration',
               'component_change_previous','component_change_acceleration','iou_change_acceleration']
    header = ['frame','time_seconds','score','min_known_score','suspected_classes','uncertain_classes','absent_classes']
    for obj in objects:
        header += [obj['name']+'_'+k for k in metrics+['past_deviation','future_deviation','past_stable','future_stable']]
    with (output/'frame_scores.csv').open('w',newline='') as handle:
        writer = csv.DictWriter(handle,header);writer.writeheader()
        for row in rows:
            flat = {k:row[k] for k in header if k in row}
            for k in ('suspected_classes','uncertain_classes','absent_classes'):flat[k]=';'.join(flat[k])
            for name,c in row['classes'].items():
                flat.update({name+'_'+k:c[k] for k in metrics})
                for direction in ('past','future'):
                    flat[name+'_'+direction+'_deviation']=c[direction]['deviation']
                    flat[name+'_'+direction+'_stable']=c[direction]['stable_or_gradual']
            writer.writerow(flat)
    summary = {'frame_count': n, 'suspected_frames': sum(bool(r['suspected_classes']) for r in rows),
               'uncertain_frames': sum(bool(r['uncertain_classes']) for r in rows),
               'per_class_suspected': {name:sum(c['suspected'] for c in cs) for name,cs in results.items()},
               'per_class_uncertain': {name:sum(c['status']=='uncertain_context' for c in cs) for name,cs in results.items()},
               'suspected_interval_count': len(intervals), 'training_selection': 'NONE'}
    write_json(output/'summary.json',summary)
    return summary


def main():
    parser = argparse.ArgumentParser();parser.add_argument('--round-root',type=Path,required=True)
    parser.add_argument('--output-name',default='unet_temporal_r1');parser.add_argument('--window-seconds',type=float,default=.3)
    parser.add_argument('--web-prefix',default='')
    args = parser.parse_args();root = args.round_root.resolve();out = root/args.output_name
    if out.parent != root or args.window_seconds <= 0:raise ValueError('Invalid output name or window')
    out.mkdir(exist_ok=True);policy = {**POLICY, 'window_seconds': args.window_seconds}
    source = json.loads((root/'manifest.json').read_text());records=[]
    manifest={'source_manifest':str(root/'manifest.json'),'policy':policy,'records':records,'training_selection':'NONE'}
    shutil.copy2(Path(__file__).with_name('unet_temporal_review.html'),out/'index.html')
    for r in source['records']:
        record={'id':r['id'],'item':r['item'],'status':'scoring',
                'media':r['media']['unet'], 'scores':args.web_prefix+args.output_name+'/'+r['id']+'/frame_scores.json',
                'csv':args.web_prefix+args.output_name+'/'+r['id']+'/frame_scores.csv'}
        records.append(record);write_json(out/'manifest.json',manifest)
        try:
            record['summary']=score_video(root/r['id']/'unet',out/r['id'],r['item']['fps'],policy)
            record['status']='complete'
        except Exception as exc:
            record['status']='failed';record['error']=str(exc);print(repr(exc),flush=True)
        write_json(out/'manifest.json',manifest)
        print(r['id'],record['status'],record.get('summary',{}),flush=True)
    if any(r['status']!='complete' for r in records):raise SystemExit(1)
    write_json(out/'complete.json',{'videos':len(records),'frames':sum(r['item']['frame_count'] for r in records),'training_selection':'NONE'})


if __name__=='__main__':
    cv2.setNumThreads(1)
    main()
