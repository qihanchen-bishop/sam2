"""Bidirectional mask disagreement reports and explicit human-selected exports."""
from __future__ import annotations

import json
import os
import shutil
import time
import uuid
from pathlib import Path

import cv2
import numpy as np


def write_json(path, data):
    path = Path(path)
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    os.replace(tmp, path)


RULE_VERSION = 'whole_iou085_count_auto50_v2'


def compare_components(x, y, main_iou=.85, min_difference=16, secondary_iou=None):
    """Compare whole per-class masks after class-specific component filtering."""
    x, y = x.astype(bool), y.astype(bool)
    union = int((x | y).sum())
    difference = int((x ^ y).sum())
    iou = float((x & y).sum() / union) if union else 1.0
    def areas(binary):
        _, _, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
        return sorted(stats[1:, cv2.CC_STAT_AREA].tolist(), reverse=True)
    aa, ba = areas(x), areas(y)
    ac, bc = len(aa), len(ba)
    return {'iou': iou, 'different_pixels': difference,
            'A': iou < main_iou and difference > min_difference,
            'B': ac != bc, 'C': ac != bc,
            'component_counts': {'forward': ac, 'reverse': bc},
            'component_areas': {'forward': aa, 'reverse': ba}}


def classify_conflicts(conflicts):
    if any(c['A'] for c in conflicts):
        return 'other', None
    directions = []
    for c in conflicts:
        counts, areas = c['component_counts'], c['component_areas']
        if counts['forward'] == 1 and counts['reverse'] > 1 and all(a < 50 for a in areas['reverse'][1:]):
            directions.append('forward')
        elif counts['reverse'] == 1 and counts['forward'] > 1 and all(a < 50 for a in areas['forward'][1:]):
            directions.append('reverse')
        else:
            return 'components', None
    return 'components', directions[0] if directions and len(set(directions)) == 1 else None


def analyze(root: Path, objects, frame_count: int, iou_threshold=.85, min_difference=16, initialize_review=True, output_root=None):
    rows, intervals = [], []
    for frame in range(frame_count):
        masks = [cv2.imread(str(root / 'directional_mask_frames' / d / f'{frame:05d}.png'), 0)
                 for d in ('forward', 'reverse')]
        if any(m is None for m in masks) or masks[0].shape != masks[1].shape:
            raise ValueError(f'Incomplete directional masks at frame {frame}')
        a, b = masks
        conflicts = []
        for obj in objects:
            x, y = a == obj['id'], b == obj['id']
            comparison = compare_components(x, y, iou_threshold, min_difference)
            if comparison['A'] or comparison['B']:
                conflicts.append({'class': obj['name'],
                                  'different_pixels': int((x ^ y).sum()), 'reasons': [k for k in ('A', 'B') if comparison[k]], **comparison})
        if conflicts:
            category, auto = classify_conflicts(conflicts)
            rows.append({'frame': frame, 'classes': conflicts, 'category': category, 'auto_direction': auto})
            names = sorted(c['class'] for c in conflicts)
            worst = min(c['iou'] for c in conflicts)
            structural_c = any(c['C'] for c in conflicts)
            if intervals and intervals[-1]['end'] == frame - 1 and intervals[-1]['category'] == category and intervals[-1]['auto_direction'] == auto:
                interval = intervals[-1]
                interval['end'] = frame
                interval['classes'] = sorted(set(interval['classes']) | set(names))
                interval['worst_iou'] = min(interval['worst_iou'], worst)
            else:
                intervals.append({'id': len(intervals), 'start': frame, 'end': frame,
                                  'classes': names, 'worst_iou': worst, 'structural_c': structural_c, 'category': category, 'auto_direction': auto})
    report = {'rule_version': RULE_VERSION, 'frame_count': frame_count, 'thresholds': {'iou': iou_threshold, 'min_difference_pixels': min_difference, 'difference_operator': '>', 'auto_secondary_area_lt': 50},
              'conflict_frame_count': len(rows), 'intervals': intervals, 'frames': rows,
              'meaning': 'directional disagreement, not a correctness judgment'}
    write_json(Path(output_root or root) / 'direction_conflicts.json', report)
    if initialize_review:
        write_json(root / 'direction_review.json', {'status': 'pending', 'revision': 0,
                   'default_direction': 'forward', 'choices': {str(i['id']): i['auto_direction'] for i in intervals if i['auto_direction']}})
    return report


def summary(root):
    root = Path(root)
    if not (root / 'direction_conflicts.json').is_file():
        return None
    review = json.loads((root / 'direction_review.json').read_text())
    report = json.loads(((root / review['report_subdir']) if review.get('report_subdir') else candidate_root(root, review)).joinpath('direction_conflicts.json').read_text())
    result = {k: v for k, v in report.items() if k != 'frames'}
    result['count_frame_details'] = {
        str(row['frame']): [{'class': c['class'], 'counts': c['component_counts'],
                             'areas': c.get('component_areas')} for c in row['classes'] if c.get('B')]
        for row in report.get('frames', []) if any(c.get('B') for c in row['classes'])}
    result.update(review)
    marks = root / 'manual_revision_frames.json'
    result['manual_revision_frames'] = json.loads(marks.read_text()).get('frames', []) if marks.is_file() else []
    result['unresolved'] = sum(str(i['id']) not in review['choices'] for i in report['intervals'])
    result['can_undo_correction'] = bool(review.get('candidate_subdir'))
    result['can_undo_final_correction'] = review.get('status') == 'finalized' and bool(review.get('final_edit'))
    return result


def reanalyze(root, filtered_source=None, review_updates=None):
    root = Path(root)
    old = json.loads((root / 'direction_review.json').read_text())
    old_summary = summary(root)
    prompt = json.loads((root / 'sam2_prompts.json').read_text())
    output = root / 'conflict_reports' / uuid.uuid4().hex
    output.mkdir(parents=True)
    report = analyze(filtered_source or candidate_root(root, old), prompt['objects'], old_summary['frame_count'], initialize_review=False, output_root=output)
    # Preserve explicit human decisions by frame, never by a renumbered interval ID.
    decisions = {}
    for interval in old_summary['intervals']:
        selected = old['choices'].get(str(interval['id']))
        if selected:
            for frame in range(interval['start'], interval['end']+1):
                decisions[frame] = selected
    intervals, choices = [], {}
    for row in report['frames']:
        frame = row['frame']; selected = decisions.get(frame, row['auto_direction'])
        names = sorted(c['class'] for c in row['classes']); worst = min(c['iou'] for c in row['classes'])
        structural_c = any(c['C'] for c in row['classes'])
        if intervals and intervals[-1]['end'] == frame-1 and choices.get(str(intervals[-1]['id'])) == selected and intervals[-1]['category'] == row['category'] and intervals[-1]['auto_direction'] == row['auto_direction']:
            intervals[-1]['end'] = frame
            intervals[-1]['classes'] = sorted(set(intervals[-1]['classes']) | set(names))
            intervals[-1]['worst_iou'] = min(intervals[-1]['worst_iou'], worst)
        else:
            interval = {'id': len(intervals), 'start': frame, 'end': frame, 'classes': names, 'worst_iou': worst, 'structural_c': structural_c, 'category': row['category'], 'auto_direction': row['auto_direction']}
            intervals.append(interval)
            if selected:
                choices[str(interval['id'])] = selected
    report['intervals'] = intervals
    write_json(output / 'direction_conflicts.json', report)
    write_json(output / 'previous_review.json', old)
    updated = {**old, **(review_updates or {}), 'choices': choices, 'revision': old['revision']+1, 'report_subdir': str(output.relative_to(root)),
               'conflict_report_updated': True}
    if old_summary.get('rule_version') != RULE_VERSION:
        updated['status'] = 'pending'
        for key in ('final_subdir', 'final_edit', 'last_final_correction', 'finalized_at'):
            updated.pop(key, None)
    write_json(root / 'direction_review.json', updated)
    return summary(root)


FILTER_VERSION = 'area20_distance_contact_limits_v7'
CONTACT_DISTANCE_PIXELS = 3


def component_filter_policy():
    return {'version': FILTER_VERSION, 'min_area': 20, 'area_applies_to_main': True,
            'distance_rule': 'nearest_pixel_center_euclidean_distance > component_area',
            'coordinate_space': 'original_video_pixels',
            'order': ['area_distance', 'righttool_object_contact', 'component_limits'],
            'component_limits': {'lefttool': 1, 'righttool': 1, 'occluder': 2, 'object': 1, 'region': 1, 'default': 1},
            'contact_component_limits': {'righttool': 2, 'object': 2},
            'contact_rule': {'classes': ['righttool', 'object'], 'max_distance_pixels': CONTACT_DISTANCE_PIXELS,
                             'measured_on': 'area_distance_cleaned_largest_two_candidates_before_final_limits'}}


def righttool_object_contact(mask, objects):
    ids = {o.get('name'): o['id'] for o in objects}
    if 'righttool' not in ids or 'object' not in ids:
        return False
    tool = (mask == ids['righttool']).astype(np.uint8)
    target = mask == ids['object']
    if not tool.any() or not target.any():
        return False
    radius = CONTACT_DISTANCE_PIXELS
    yy, xx = np.ogrid[-radius:radius+1, -radius:radius+1]
    kernel = (xx*xx+yy*yy <= radius*radius).astype(np.uint8)
    return bool(np.any(cv2.dilate(tool, kernel) & target))


def filter_mask_components(mask, objects, return_details=False):
    result = mask.copy()
    details = []
    cleaned = mask.copy()
    prepared = []
    for obj in objects:
        binary = (mask == obj['id']).astype(np.uint8)
        x, y, w, h = cv2.boundingRect(binary)
        if not w or not h:
            continue
        count, labels, stats, _ = cv2.connectedComponentsWithStats(binary[y:y+h, x:x+w], connectivity=8)
        ranked = sorted(range(1, count), key=lambda i: (-int(stats[i, cv2.CC_STAT_AREA]), i))
        main = ranked[0]
        distances = np.zeros(count, dtype=np.float64)
        if count > 2:
            distance_map = cv2.distanceTransform((labels != main).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
            distances[:] = np.inf
            foreground = labels > 0
            np.minimum.at(distances, labels[foreground], distance_map[foreground])
        reasons_by_id = {}
        eligible = []
        for component in ranked:
            area = int(stats[component, cv2.CC_STAT_AREA])
            reasons = ['area_lt20'] if area < 20 else []
            if component != main and distances[component] > area:
                reasons.append('distance_gt_area')
            reasons_by_id[component] = reasons
            if not reasons:
                eligible.append(component)
        if len(eligible) != count-1:
            cleaned[y:y+h, x:x+w][np.isin(labels, [i for i in range(1,count) if i not in eligible])] = 0
        if obj.get('name') in ('righttool', 'object') and len(eligible) > 2:
            cleaned[y:y+h, x:x+w][np.isin(labels, eligible[2:])] = 0
        prepared.append((obj,x,y,h,w,count,labels,stats,distances,reasons_by_id,eligible))
    contact = righttool_object_contact(cleaned, objects)
    for obj,x,y,h,w,count,labels,stats,distances,reasons_by_id,eligible in prepared:
        conditional = obj.get('name') in ('righttool', 'object')
        limit = 2 if obj.get('name') == 'occluder' or conditional and contact else 1
        keep = set(eligible[:limit])
        erase = []
        for component in range(1, count):
            if component in keep:
                continue
            area = int(stats[component, cv2.CC_STAT_AREA])
            reasons = reasons_by_id[component] or ['component_rank_limit']
            if conditional and not contact and reasons == ['component_rank_limit']:
                reasons = reasons + ['not_in_righttool_object_contact']
            if reasons:
                erase.append(component)
                details.append({'class_id': obj['id'], 'class': obj.get('name'), 'component': component,
                                'area': area, 'nearest_distance_pixels': float(distances[component]),
                                'keep_limit': limit, 'righttool_object_contact': contact, 'reasons': reasons})
        if erase:
            roi = result[y:y+h, x:x+w]
            roi[np.isin(labels, erase)] = 0
    return (result, details) if return_details else result


def filter_candidates(root):
    """Fresh experiment from raw directional predictions, never cumulative cleanup."""
    from sam2_prompts_to_masks import encode_class_mask_visual_video
    root = Path(root)
    old = json.loads((root / 'direction_review.json').read_text())
    if old.get('candidate_filter', {}).get('version') == FILTER_VERSION:
        current = summary(root)
        return current if current['rule_version'] == RULE_VERSION else reanalyze(root)
    prompt = json.loads((root / 'sam2_prompts.json').read_text())
    complete = json.loads((root / 'complete.json').read_text())
    video = Path(complete.get('video_path') or prompt['video_path'])
    frame_count = summary(root)['frame_count']
    previous_dirs = {name: list((root / name).iterdir()) if (root / name).exists() else []
                     for name in ('filtered_versions', 'reviewed_versions', 'conflict_reports')}
    output = root / 'filtered_versions' / uuid.uuid4().hex
    output.mkdir(parents=True)
    stats, edits = {}, []
    for direction in ('forward', 'reverse'):
        dst = output / 'directional_mask_frames' / direction
        dst.mkdir(parents=True)
        removed = changed = components = 0
        for frame in range(frame_count):
            mask = cv2.imread(str(root / 'directional_mask_frames' / direction / f'{frame:05d}.png'), 0)
            if mask is None:
                raise ValueError(f'缺少原始候选 {direction} 第 {frame} 帧')
            result, details = filter_mask_components(mask, prompt['objects'], True)
            pixels = int((result != mask).sum())
            removed += pixels; changed += bool(pixels); components += len(details)
            if details:
                edits.append({'source': direction, 'frame': frame, 'removed_components': details})
            if not cv2.imwrite(str(dst / f'{frame:05d}.png'), result):
                raise RuntimeError('保存过滤结果失败')
        stats[direction] = {'removed_pixels': removed, 'changed_frames': changed, 'removed_components': components}
        encode_class_mask_visual_video(dst, output / f'mask_visual_{direction}.mp4', video, prompt['objects'])
    policy = component_filter_policy()
    write_json(output / 'filter.json', {**policy, 'source_candidates': '.', 'stats': stats})
    write_json(output / 'filter_edits.json', {'frames': edits})
    report = analyze(output, prompt['objects'], frame_count, initialize_review=False)
    archive = root / 'retired_experiments' / uuid.uuid4().hex
    archive.mkdir(parents=True)
    write_json(archive / 'direction_review.json', old)
    shutil.copy2(root / 'direction_conflicts.json', archive / 'direction_conflicts.json')
    # Publish only after complete masks, videos and report; discard old choices/final pointer.
    write_json(root / 'direction_review.json', {
        'status': 'pending', 'revision': old.get('revision', 0)+1,
        'choices': {str(i['id']): i['auto_direction'] for i in report['intervals'] if i['auto_direction']},
        'default_direction': 'forward', 'candidate_subdir': str(output.relative_to(root)),
        'candidate_revision': output.name, 'candidate_filter': policy,
        'experiment_reset': True})
    for name, paths in previous_dirs.items():
        if paths:
            (archive / name).mkdir()
            for path in paths:
                shutil.move(str(path), str(archive / name / path.name))
    return summary(root)


def candidate_root(root, review=None):
    root = Path(root)
    if review is None:
        review = json.loads((root / 'direction_review.json').read_text())
    return root / review['candidate_subdir'] if review.get('candidate_subdir') else root


def repair_mask(mask, class_id, remove_area, fill_area, keep_largest=True):
    """8-connected foreground; 4-connected background holes. Other classes are protected."""
    result = mask.copy()
    removed = filled = 0
    if remove_area:
        count, labels, stats, _ = cv2.connectedComponentsWithStats((result == class_id).astype(np.uint8), connectivity=8)
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) if count > 1 else -1
        erase = [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] <= remove_area
                 and not (keep_largest and i == largest)]
        if erase:
            selected = np.isin(labels, erase)
            removed = int(selected.sum())
            result[selected] = 0
    if fill_area:
        count, labels, stats, _ = cv2.connectedComponentsWithStats((result != class_id).astype(np.uint8), connectivity=4)
        h, w = result.shape
        for i in range(1, count):
            x, y, cw, ch, area = map(int, stats[i])
            if area > fill_area or x == 0 or y == 0 or x + cw == w or y + ch == h:
                continue
            region = labels[y:y+ch, x:x+cw] == i
            roi = result[y:y+ch, x:x+cw]
            if np.all(roi[region] == 0):
                roi[region] = class_id
                filled += area
    return result, removed, filled


def edit_components(mask, class_id, operations):
    result = mask.copy()
    h, w = mask.shape
    for operation in operations:
        kind = operation.get('kind')
        if kind == 'fill_holes':
            result, _, _ = repair_mask(result, class_id, 0, result.size)
        elif kind == 'keep_largest':
            result, _, _ = repair_mask(result, class_id, result.size, 0, True)
        elif kind in ('cut', 'erase', 'paint'):
            points = operation.get('points', [])
            width = operation.get('width', 3)
            if not isinstance(width, int) or not 1 <= width <= 200 or not 1 <= len(points) <= 10000:
                raise ValueError('切割线宽或点数无效')
            line = np.zeros(mask.shape, np.uint8)
            coords = []
            for point in points:
                if len(point) != 2 or any(not isinstance(v, int) for v in point) or not (0 <= point[0] < w and 0 <= point[1] < h):
                    raise ValueError('切割坐标超出图像')
                coords.append(point)
            cv2.polylines(line, [np.array(coords, np.int32)], False, 1, width)
            for point in (coords[0], coords[-1]):
                cv2.circle(line, tuple(point), width // 2, 1, -1)
            if kind == 'paint':
                overwrite = operation.get('overwrite', False)
                if not isinstance(overwrite, bool):
                    raise ValueError('覆盖其他类别参数无效')
                result[(line != 0) & ((result == 0) | (result == class_id) | overwrite)] = class_id
            else:
                result[(line != 0) & (result == class_id)] = 0
        elif kind == 'delete':
            x, y = operation.get('x'), operation.get('y')
            if not isinstance(x, int) or not isinstance(y, int) or not (0 <= x < w and 0 <= y < h):
                raise ValueError('删除坐标无效')
            _, labels = cv2.connectedComponents((result == class_id).astype(np.uint8), connectivity=8)
            selected = labels[y, x]
            if selected:
                result[labels == selected] = 0
        elif kind == 'morphology':
            remove, fill = operation.get('remove_area'), operation.get('fill_area')
            if any(type(v) is not int or not 0 <= v <= 100_000_000 for v in (remove, fill)) or not (remove or fill):
                raise ValueError('请设置有效的去除或填洞面积，至少一项大于 0')
            if not isinstance(operation.get('keep_largest', True), bool):
                raise ValueError('保留最大分量参数无效')
            result, _, _ = repair_mask(result, class_id, remove, fill, operation.get('keep_largest', True))
        elif kind == 'small':
            area = operation.get('area')
            if not isinstance(area, int) or not 1 <= area <= 100_000_000:
                raise ValueError('面积无效')
            result, _, _ = repair_mask(result, class_id, area, 0, operation.get('keep_largest', True))
        else:
            raise ValueError('未知编辑操作')
    return result


def component_preview(mask):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    flat = labels.ravel()
    boundaries = np.r_[0, np.flatnonzero(flat[1:] != flat[:-1]) + 1, len(flat)]
    runs = [[int(a), int(b-a), int(flat[a])] for a, b in zip(boundaries[:-1], boundaries[1:]) if flat[a]]
    largest = sorted(range(1, count), key=lambda i: int(stats[i, cv2.CC_STAT_AREA]), reverse=True)[:5]
    components = []
    for i in largest:
        first = int(np.flatnonzero(flat == i)[0])
        components.append({'id': i, 'area': int(stats[i, cv2.CC_STAT_AREA]),
                           'box': stats[i, :4].tolist(), 'point': [first % mask.shape[1], first // mask.shape[1]]})
    return {'width': mask.shape[1], 'height': mask.shape[0], 'runs': runs, 'components': components, 'count': count-1}


def correction_options(root, payload):
    root = Path(root)
    prompt = json.loads((root / 'sam2_prompts.json').read_text())
    direction = payload.get('direction')
    if direction not in ('forward', 'reverse', 'final'):
        raise ValueError('无效的修正来源')
    try:
        options = {k: int(payload[k]) for k in ('class_id', 'start', 'end', 'remove_area', 'fill_area')}
        if any(isinstance(payload[k], bool) or float(payload[k]) != value for k, value in options.items()):
            raise ValueError('non-integer')
    except (ValueError, TypeError, KeyError):
        raise ValueError('类别、帧号和面积必须为整数')
    if options['class_id'] not in {int(o['id']) for o in prompt['objects']}:
        raise ValueError('未知类别')
    if not 0 <= options['start'] <= options['end'] < int(prompt['frame_count']):
        raise ValueError('修正帧范围无效')
    if min(options['remove_area'], options['fill_area']) < 0 or max(options['remove_area'], options['fill_area']) > 100_000_000:
        raise ValueError('面积阈值无效')
    operations = payload.get('operations', [])
    if not isinstance(operations, list) or len(operations) > 200 or any(not isinstance(o, dict) for o in operations):
        raise ValueError('编辑操作无效或超过 200 步')
    batch = payload.get('batch_keep_largest', False)
    batch_fill = payload.get('batch_fill_holes', False)
    if type(batch) is not bool or type(batch_fill) is not bool or (batch and batch_fill):
        raise ValueError('区间处理参数无效')
    if batch and (direction != 'final' or operations != [{'kind': 'keep_largest'}] or options['remove_area'] or options['fill_area']):
        raise ValueError('区间处理仅支持合并结果保留最大连通分量')
    if batch_fill and (direction != 'final' or operations != [{'kind': 'fill_holes'}] or options['remove_area'] or options['fill_area']):
        raise ValueError('区间填洞仅支持合并结果填充所选类别孔洞')
    if operations and options['start'] != options['end'] and not (batch or batch_fill):
        raise ValueError('鼠标编辑仅作用于当前单帧')
    if not (options['remove_area'] or options['fill_area'] or operations):
        raise ValueError('请至少启用一项修正：面积设为大于 0')
    if not isinstance(payload.get('keep_largest', True), bool):
        raise ValueError('保留最大分量参数无效')
    return {**options, 'operations': operations, 'direction': direction, 'keep_largest': payload.get('keep_largest', True), 'batch_keep_largest': batch, 'batch_fill_holes': batch_fill}


def correct(root, options):
    from sam2_prompts_to_masks import encode_class_mask_visual_video
    root = Path(root)
    old_review = json.loads((root / 'direction_review.json').read_text())
    review = summary(root)
    source = candidate_root(root, old_review)
    revision = root / 'correction_versions' / uuid.uuid4().hex
    revision.mkdir(parents=True)
    shutil.copytree(source / 'directional_mask_frames', revision / 'directional_mask_frames')
    prompt = json.loads((root / 'sam2_prompts.json').read_text())
    changed = []
    directory = revision / 'directional_mask_frames' / options['direction']
    for frame in range(options['start'], options['end'] + 1):
        path = directory / f'{frame:05d}.png'
        mask = cv2.imread(str(path), 0)
        if mask is None:
            raise ValueError(f'缺少第 {frame} 帧')
        repaired, removed, filled = repair_mask(mask, options['class_id'], options['remove_area'],
                                              options['fill_area'], options['keep_largest'])
        edited = edit_components(repaired, options['class_id'], options.get('operations', []))
        removed = int(((mask == options['class_id']) & (edited != options['class_id'])).sum())
        filled = int(((mask != options['class_id']) & (edited == options['class_id'])).sum())
        repaired = edited
        if removed or filled:
            if not cv2.imwrite(str(path), repaired):
                raise RuntimeError(f'修正掩码保存失败：{frame}')
            changed.append({'frame': frame, 'removed_pixels': removed, 'filled_pixels': filled})
    for direction in ('forward', 'reverse'):
        video = revision / f'mask_visual_{direction}.mp4'
        if direction == options['direction'] and changed:
            encode_class_mask_visual_video(directory, video, Path(prompt['video_path']), prompt['objects'])
        else:
            shutil.copy2(source / f'mask_visual_{direction}.mp4', video)
    thresholds = review['thresholds']
    analyze(revision, prompt['objects'], review['frame_count'], thresholds['iou'],
            thresholds['min_difference_pixels'], initialize_review=False)
    write_json(revision / 'previous_review.json', old_review)
    edit = {**options, 'changed_frames': len(changed),
            'removed_pixels': sum(c['removed_pixels'] for c in changed),
            'filled_pixels': sum(c['filled_pixels'] for c in changed), 'frames': changed}
    write_json(revision / 'correction.json', edit)
    new_review = {'status': 'pending', 'revision': old_review['revision'] + 1,
                  'default_direction': old_review['default_direction'], 'choices': {},
                  'candidate_subdir': str(revision.relative_to(root)), 'candidate_revision': revision.name,
                  'last_correction': {k: v for k, v in edit.items() if k != 'frames'}}
    if not changed:
        # A no-op must not discard existing review decisions or the final selection.
        new_review = {**old_review, 'revision': old_review['revision'] + 1,
                      'last_correction': new_review['last_correction']}
    write_json(root / 'direction_review.json', new_review)
    return summary(root)


def undo_correction(root):
    root = Path(root)
    current = json.loads((root / 'direction_review.json').read_text())
    if not current.get('candidate_subdir'):
        raise ValueError('没有可撤销的修正')
    previous = json.loads((candidate_root(root, current) / 'previous_review.json').read_text())
    previous['revision'] = current['revision'] + 1
    write_json(root / 'direction_review.json', previous)
    return summary(root)


def final_root(root, review=None):
    root = Path(root)
    review = review or summary(root)
    if not review or review.get('status') != 'finalized':
        raise ValueError('请先完成正反向冲突选择并生成合并结果')
    return root / review['final_subdir']


def multiple_correction_options(root, payload):
    entries = payload.get('edits')
    if not isinstance(entries, list) or not 1 <= len(entries) <= 1000:
        raise ValueError('批量草稿需包含1–1000个帧/类别条目')
    edits, seen = [], set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError('草稿条目无效')
        option = correction_options(root, {**entry, 'direction': 'final', 'batch_keep_largest': False})
        if option['start'] != option['end']:
            raise ValueError('每个草稿条目必须为单帧')
        key = (option['start'], option['class_id'])
        if key in seen:
            raise ValueError('重复的帧/类别草稿')
        seen.add(key); edits.append(option)
    return {'direction': 'final', 'start': min(e['start'] for e in edits),
            'end': max(e['end'] for e in edits), 'edits': edits}


def correct_final(root, options):
    """Edit the merged result only; preserve directional candidates and human choices."""
    from sam2_prompts_to_masks import encode_class_mask_visual_video
    root = Path(root)
    old = json.loads((root / 'direction_review.json').read_text())
    source = final_root(root, old)
    if options['direction'] != 'final' or (options['start'] != options['end'] and not options.get('batch_keep_largest') and not options.get('batch_fill_holes') and not options.get('edits')):
        raise ValueError('单帧修正只能作用于合并后的当前帧')
    output = None
    changes = []
    by_frame = {}
    for entry in options.get('edits', [options]):
        for frame in range(entry['start'], entry['end'] + 1):
            by_frame.setdefault(frame, []).append(entry)
    for frame, entries in sorted(by_frame.items()):
        mask = cv2.imread(str(source / 'mask_frames' / f'{frame:05d}.png'), 0)
        if mask is None:
            raise ValueError(f'无法读取合并掩码：{frame}')
        result = mask.copy()
        for entry in entries:
            result, _, _ = repair_mask(result, entry['class_id'], entry['remove_area'], entry['fill_area'], entry['keep_largest'])
            result = edit_components(result, entry['class_id'], entry.get('operations', []))
        if np.array_equal(mask, result):
            continue
        if output is None:
            output = root / 'reviewed_versions' / uuid.uuid4().hex
            output.mkdir(parents=True)
            shutil.copytree(source / 'mask_frames', output / 'mask_frames')
        if not cv2.imwrite(str(output / 'mask_frames' / f'{frame:05d}.png'), result):
            raise RuntimeError('修正保存失败')
        changes.append({'frame': frame,
                        'removed_pixels': int(((mask != 0) & (result != mask)).sum()),
                        'filled_pixels': int(((result != 0) & (result != mask)).sum())})
    if output is None:
        return summary(root)
    prompt = json.loads((root / 'sam2_prompts.json').read_text())
    shutil.copy2(root / 'sam2_prompts.json', output / 'sam2_prompts.json')
    encode_class_mask_visual_video(output / 'mask_frames', output / 'mask_visual.mp4', Path(prompt['video_path']), prompt['objects'])
    edit = {**options, 'changed_frames': len(changes), 'frames': changes,
            'removed_pixels': sum(c['removed_pixels'] for c in changes),
            'filled_pixels': sum(c['filled_pixels'] for c in changes)}
    manifest = {**old, 'revision': old['revision'] + 1, 'final_subdir': str(output.relative_to(root)),
                'final_edit': True, 'last_final_correction': edit, 'parent_final_subdir': old['final_subdir']}
    write_json(output / 'previous_review.json', old)
    write_json(output / 'selection.json', manifest)
    write_json(output / 'correction.json', edit)
    completion = json.loads((source / 'complete.json').read_text())
    completion.update(human_direction_selection=manifest, manual_final_correction=edit)
    write_json(output / 'complete.json', completion)
    write_json(root / 'direction_review.json', manifest)
    return summary(root)


def undo_final_correction(root):
    root = Path(root)
    current = json.loads((root / 'direction_review.json').read_text())
    source = final_root(root, current)
    if not current.get('final_edit'):
        raise ValueError('没有可撤销的合并结果修正')
    previous = json.loads((source / 'previous_review.json').read_text())
    previous['revision'] = current['revision'] + 1
    write_json(root / 'direction_review.json', previous)
    return summary(root)


def scan_final(source, class_id, frame_count, thresholds):
    """Heuristic flags only: multiple components/holes can be legitimate anatomy."""
    rows = []
    previous = None
    previous_area = 0
    previous_center = None
    for frame in range(frame_count):
        mask = cv2.imread(str(Path(source) / 'mask_frames' / f'{frame:05d}.png'), 0)
        if mask is None:
            raise ValueError(f'缺少合并结果第 {frame} 帧')
        binary = mask == class_id
        count, _, stats, centers = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
        area = int(binary.sum())
        center = np.average(centers[1:], axis=0, weights=stats[1:, cv2.CC_STAT_AREA]) if count > 1 else None
        filled, _, hole_area = repair_mask(mask, class_id, 0, mask.size)
        holes = cv2.connectedComponents(((filled == class_id) & ~binary).astype(np.uint8), connectivity=4)[0] - 1
        row = {'frame': frame, 'components': count-1, 'holes': holes, 'hole_pixels': hole_area,
               'area': area, 'flags': [], 'transitions': []}
        if count > 2:
            row['flags'].append('components')
        if holes:
            row['flags'].append('holes')
        if previous is not None:
            union = int((binary | previous).sum())
            difference = int((binary ^ previous).sum())
            iou = float((binary & previous).sum()/union) if union else 1.0
            area_change = abs(area-previous_area)/max(area, previous_area, 1)
            shift = float(np.linalg.norm(center-previous_center)/np.hypot(*mask.shape)) if center is not None and previous_center is not None else 0.0
            if difference >= thresholds['min_pixels'] and (iou < thresholds['iou'] or area_change > thresholds['area_change'] or shift > thresholds['centroid_shift']):
                transition = {'from': frame-1, 'to': frame, 'iou': round(iou, 4), 'area_change': round(area_change, 4), 'centroid_shift': round(shift, 4), 'different_pixels': difference}
                row['flags'].append('jump')
                row['transitions'].append(transition)
                if 'jump' not in rows[-1]['flags']:
                    rows[-1]['flags'].append('jump')
                rows[-1]['transitions'].append(transition)
        rows.append(row)
        previous, previous_area, previous_center = binary, area, center
    return {'class_id': class_id, 'frame_count': frame_count, 'thresholds': thresholds,
            'counts': {flag: sum(flag in r['flags'] for r in rows) for flag in ('components', 'holes', 'jump')},
            'frames': [r for r in rows if r['flags']], 'meaning': 'review hints, not automatic error labels'}


def choose(root, direction, interval_id):
    root = Path(root)
    if direction not in ('forward', 'reverse'):
        raise ValueError('请选择正向或反向')
    review = summary(root)
    if review is None:
        raise ValueError('此版本没有双向结果，请重新传播')
    if interval_id == 'all':
        review['default_direction'] = direction
        review['choices'] = {str(i['id']): direction for i in review['intervals']}
    else:
        if str(interval_id) not in {str(i['id']) for i in review['intervals']}:
            raise ValueError('无效的冲突片段')
        review['choices'][str(interval_id)] = direction
    saved = {k: review[k] for k in ('default_direction', 'choices')}
    saved.update({k: review[k] for k in ('candidate_subdir', 'candidate_revision', 'last_correction', 'report_subdir', 'candidate_filter') if k in review})
    saved.update(status='pending', revision=review['revision'] + 1)
    write_json(root / 'direction_review.json', saved)
    return summary(root)


def finalize(root):
    # Publish a new immutable revision only after all frames and video exist.
    from sam2_prompts_to_masks import encode_class_mask_visual_video
    root = Path(root)
    review = summary(root)
    if review is None or review['unresolved']:
        raise ValueError('请先为全部冲突片段选择方向')
    prompt = json.loads((root / 'sam2_prompts.json').read_text())
    directions = [review['default_direction']] * review['frame_count']
    for interval in review['intervals']:
        direction = review['choices'][str(interval['id'])]
        for frame in range(interval['start'], interval['end'] + 1):
            directions[frame] = direction
    revision = root / 'reviewed_versions' / uuid.uuid4().hex
    (revision / 'mask_frames').mkdir(parents=True)
    for frame, direction in enumerate(directions):
        shutil.copy2(candidate_root(root, review) / 'directional_mask_frames' / direction / f'{frame:05d}.png',
                     revision / 'mask_frames' / f'{frame:05d}.png')
    shutil.copy2(root / 'sam2_prompts.json', revision / 'sam2_prompts.json')
    encode_class_mask_visual_video(revision / 'mask_frames', revision / 'mask_visual.mp4',
                                   Path(prompt['video_path']), prompt['objects'])
    manifest = {'status': 'finalized', 'revision': review['revision'] + 1,
                'default_direction': review['default_direction'], 'choices': review['choices'],
                'final_subdir': str(revision.relative_to(root)), 'finalized_at': time.time()}
    manifest.update({k: review[k] for k in ('candidate_subdir', 'candidate_revision', 'last_correction', 'report_subdir', 'candidate_filter') if k in review})
    write_json(revision / 'selection.json', manifest)
    completion = json.loads((root / 'complete.json').read_text())
    completion.update(human_direction_selection=manifest, source_candidates=str(root))
    completion['review_status'] = 'human_selected'
    if 'bidirectional' in completion:
        completion['bidirectional']['review_status'] = 'human_selected'
    write_json(revision / 'complete.json', completion)
    write_json(root / 'direction_review.json', manifest)
    return manifest
