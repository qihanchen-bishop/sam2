import json
import tempfile
import unittest
import sys
import types
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

import sam2_direction_review as review
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'apps'))
from seed_bbox_web import EditorManager


class DirectionReviewTest(unittest.TestCase):
    def test_range_fill_holes_protects_other_classes_and_outside_range(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            objects = [{'id': 1, 'name': 'object'}, {'id': 2, 'name': 'region'}]
            mask = np.zeros((20, 20), np.uint8)
            mask[2:18, 2:18] = 1
            mask[4:6, 4:6] = 0
            mask[10:12, 10:12] = 2
            for direction in ('forward', 'reverse'):
                folder = root / 'directional_mask_frames' / direction
                folder.mkdir(parents=True)
                for frame in range(4):
                    cv2.imwrite(str(folder / f'{frame:05d}.png'), mask)
            review.write_json(root / 'sam2_prompts.json', {'objects': objects, 'frame_count': 4, 'video_path': 'unused'})
            review.write_json(root / 'complete.json', {'frame_count': 4})
            review.analyze(root, objects, 4)
            review.choose(root, 'forward', 'all')
            payload = dict(direction='final', class_id=1, start=1, end=2, remove_area=0, fill_area=0,
                           batch_fill_holes=True, operations=[{'kind': 'fill_holes'}])
            options = review.correction_options(root, payload)
            for bad in ({'direction': 'forward'}, {'batch_keep_largest': True}, {'operations': [{'kind': 'erase'}]}):
                with self.assertRaises(ValueError):
                    review.correction_options(root, {**payload, **bad})
            with patch('sam2_prompts_to_masks.encode_class_mask_visual_video', lambda m, o, *a: o.write_bytes(b'test')):
                original = review.finalize(root)
                updated = review.correct_final(root, options)
                self.assertEqual(updated['last_final_correction']['changed_frames'], 2)
                for frame in range(4):
                    expected = mask.copy()
                    if frame in (1, 2):
                        expected[4:6, 4:6] = 1
                    actual = cv2.imread(str(review.final_root(root) / 'mask_frames' / f'{frame:05d}.png'), 0)
                    np.testing.assert_array_equal(actual, expected)
                self.assertEqual(review.undo_final_correction(root)['final_subdir'], original['final_subdir'])

    def test_cross_video_jobs_are_isolated(self):
        preview = {'run_id': 'run40', 'stale': False, 'output_path': '/unused'}
        editor = types.SimpleNamespace(items={'ep40': {}, 'ep60': {}}, latest_preview=lambda _: preview, latest_unet=lambda _: None)
        manager = EditorManager(editor)
        manager.job = {'item_id': 'ep60', 'workspace_id': manager.workspace_id,
                       'run_id': 'run60', 'kind': 'propagation', 'state': 'running', 'started_at': 1}
        propagation = manager.job
        payload = {'workspace_id': manager.workspace_id, 'item_id': 'ep40',
                   'run_id': 'run40', 'direction': 'forward', 'interval_id': 0}
        report = {'unresolved': 0, 'revision': 1, 'candidate_filter': {'version': review.FILTER_VERSION}}
        status = manager.propagation_status(manager.workspace_id, 'ep40')
        self.assertIsNone(status['job'])
        self.assertEqual(status['propagation_job']['item_id'], 'ep60')
        with patch.object(review, 'summary', return_value=report), patch.object(review, 'choose', return_value=report) as choose:
            manager.review_action(payload)
            choose.assert_called_once()
            with patch('seed_bbox_web.threading.Thread') as thread:
                manager.review_action(payload, finalize=True)
                worker = thread.call_args.kwargs['target']
            self.assertIs(manager.job, propagation)
            self.assertEqual(manager.item_jobs['ep40']['kind'], 'finalize')
            with self.assertRaises(RuntimeError):
                manager.review_action(payload)
            with patch.object(review, 'finalize', side_effect=RuntimeError('fixture failure')):
                worker()
        self.assertEqual(manager.item_jobs['ep40']['state'], 'failed')
        self.assertEqual(propagation['state'], 'running')
        self.assertEqual(manager.propagation_status(manager.workspace_id, 'ep40')['job']['kind'], 'finalize')
        self.assertIs(manager.running_job(), propagation)
        self.assertIsNone(manager.running_job('ep40'))

    def test_final_batch_largest_inclusive_range_and_undo(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);objects=[{'id':1,'name':'object'},{'id':2,'name':'region'}]
            a=np.zeros((20,20),np.uint8);a[1:5,1:5]=1;a[10,10]=1;a[15,15]=2
            for direction in ('forward','reverse'):
                folder=root/'directional_mask_frames'/direction;folder.mkdir(parents=True)
                for f in range(4):cv2.imwrite(str(folder/f'{f:05d}.png'),a)
            review.write_json(root/'sam2_prompts.json',{'objects':objects,'frame_count':4,'video_path':'unused'})
            review.write_json(root/'complete.json',{'frame_count':4})
            review.analyze(root,objects,4);review.choose(root,'forward','all')
            encode=lambda masks,output,*args:output.write_bytes(b'fixture')
            with patch('sam2_prompts_to_masks.encode_class_mask_visual_video',encode):original=review.finalize(root)
            payload={'direction':'final','class_id':1,'start':1,'end':2,'remove_area':0,'fill_area':0,'batch_keep_largest':True,'operations':[{'kind':'keep_largest'}]}
            options=review.correction_options(root,payload)
            with self.assertRaises(ValueError):review.correction_options(root,{**payload,'operations':[{'kind':'fill_holes'}]})
            with self.assertRaises(ValueError):review.correction_options(root,{**payload,'batch_keep_largest':False})
            with patch('sam2_prompts_to_masks.encode_class_mask_visual_video',side_effect=RuntimeError('failed')):
                with self.assertRaises(RuntimeError):review.correct_final(root,options)
            self.assertEqual(review.summary(root)['final_subdir'],original['final_subdir'])
            with patch('sam2_prompts_to_masks.encode_class_mask_visual_video',encode):updated=review.correct_final(root,options)
            for f in range(4):
                actual=cv2.imread(str(review.final_root(root)/'mask_frames'/f'{f:05d}.png'),0)
                expected=a.copy()
                if f in (1,2):expected[10,10]=0
                np.testing.assert_array_equal(actual,expected)
            self.assertEqual(updated['last_final_correction']['changed_frames'],2)
            self.assertEqual(updated['last_final_correction']['removed_pixels'],2)
            with patch('sam2_prompts_to_masks.encode_class_mask_visual_video',encode):again=review.correct_final(root,options)
            self.assertEqual(again['revision'],updated['revision'])
            restored=review.undo_final_correction(root)
            self.assertEqual(restored['final_subdir'],original['final_subdir'])
            edits=[{**payload,'start':f,'end':f,'batch_keep_largest':False} for f in (0,3)]
            multi=review.multiple_correction_options(root,{'edits':edits})
            with self.assertRaises(ValueError):review.multiple_correction_options(root,{'edits':[edits[0],edits[0]]})
            before=(root/'direction_review.json').read_bytes()
            with patch('sam2_prompts_to_masks.encode_class_mask_visual_video',side_effect=RuntimeError('failed')):
                with self.assertRaises(RuntimeError):review.correct_final(root,multi)
            self.assertEqual(before,(root/'direction_review.json').read_bytes())
            with patch('sam2_prompts_to_masks.encode_class_mask_visual_video',side_effect=encode) as encoder:
                saved=review.correct_final(root,multi);self.assertEqual(encoder.call_count,1)
            for f in range(4):
                actual=cv2.imread(str(review.final_root(root)/'mask_frames'/f'{f:05d}.png'),0)
                expected=a.copy()
                if f in (0,3):expected[10,10]=0
                np.testing.assert_array_equal(actual,expected)
            self.assertEqual(saved['last_final_correction']['changed_frames'],2)
            self.assertEqual(review.undo_final_correction(root)['final_subdir'],original['final_subdir'])

    def test_manual_brush_paint_erase(self):
        mask=np.zeros((50,50),np.uint8);mask[20:30,20:30]=2
        op={'kind':'paint','points':[[5,25],[40,25]],'width':5}
        painted=review.edit_components(mask,1,[op])
        self.assertEqual(painted[25,10],1);self.assertEqual(painted[25,25],2)
        overwritten=review.edit_components(mask,1,[{**op,'overwrite':True}])
        self.assertEqual(overwritten[25,25],1)
        erased=review.edit_components(painted,1,[{**op,'kind':'erase'}])
        np.testing.assert_array_equal(erased,mask)
        small=review.edit_components(mask,1,[{**op,'points':[[5,5]],'width':1}])
        large=review.edit_components(mask,1,[{**op,'points':[[5,5]],'width':9}])
        self.assertGreater((large==1).sum(),(small==1).sum())

    def test_manual_marks_persist_independently_of_review(self):
        import threading
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            review.write_json(root/'direction_conflicts.json',{'frame_count':10,'frames':[],'intervals':[]})
            review.write_json(root/'direction_review.json',{'revision':0,'status':'pending','choices':{}})
            manager=EditorManager.__new__(EditorManager);manager.lock=threading.RLock();manager.workspace_id='ws'
            manager.editor=types.SimpleNamespace(latest_preview=lambda item:{'run_id':'run','stale':False,'output_path':str(root)})
            payload={'workspace_id':'ws','item_id':'item','run_id':'run','frame':3}
            before=(root/'direction_review.json').read_bytes()
            manager.mark_revision_frame(payload)
            self.assertEqual(review.summary(root)['manual_revision_frames'],[{'frame':3,'done':False}])
            manager.mark_revision_frame({**payload,'action':'done'})
            self.assertTrue(review.summary(root)['manual_revision_frames'][0]['done'])
            self.assertEqual(before,(root/'direction_review.json').read_bytes())
            with self.assertRaises(ValueError):manager.mark_revision_frame({**payload,'frame':10})
            manager.mark_revision_frame({**payload,'action':'remove'})
            self.assertEqual(review.summary(root)['manual_revision_frames'],[])

    def test_difference_frame_before_finalization(self):
        import base64
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for direction,mask in [('forward',[[1,0,1,2]]),('reverse',[[0,1,2,2]])]:
                folder=root/'directional_mask_frames'/direction;folder.mkdir(parents=True)
                cv2.imwrite(str(folder/'00000.png'),np.array(mask,np.uint8))
            editor=types.SimpleNamespace(items={'sample':{'frame_count':1}})
            editor.latest_preview=lambda _: {'run_id':'run','output_path':str(root),'frame_count':1,'direction_review':{'status':'pending'}}
            manager=EditorManager(editor)
            result=manager.difference_frame(manager.workspace_id,'sample',0,'run','original')
            self.assertEqual([result[k] for k in ['different_pixels','forward_only','reverse_only','class_mismatch']],[3,1,1,1])
            rgba=cv2.imdecode(np.frombuffer(base64.b64decode(result['image'].split(',')[1]),np.uint8),cv2.IMREAD_UNCHANGED)
            self.assertEqual(rgba[0,:,3].tolist(),[255,255,255,0])
            with self.assertRaises(RuntimeError):manager.difference_frame(manager.workspace_id,'sample',0,'run','stale')
    def test_fresh_filter_resets_and_archives(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);objects=[{'id':1,'name':'object'},{'id':2,'name':'occluder'},{'id':3,'name':'lefttool'}]
            mask=np.zeros((80,80),np.uint8)
            for i in (1,2,3):
                x=i*20
                mask[2:12,x:x+10]=i;mask[20:25,x:x+4]=i;mask[30,x]=i
            for direction in ('forward','reverse'):
                folder=root/'directional_mask_frames'/direction;folder.mkdir(parents=True)
                cv2.imwrite(str(folder/'00000.png'),mask)
            review.write_json(root/'sam2_prompts.json',{'objects':objects,'video_path':'unused'})
            review.write_json(root/'complete.json',{'frame_count':1})
            review.analyze(root,objects,1)
            review.choose(root,'reverse','all')
            stale=root/'filtered_versions/stale';stale.mkdir(parents=True)
            def encode(masks,output,*args):output.write_bytes(b'fixture')
            original=(root/'direction_review.json').read_bytes()
            with patch('sam2_prompts_to_masks.encode_class_mask_visual_video',side_effect=RuntimeError('failed')):
                with self.assertRaises(RuntimeError):review.filter_candidates(root)
            self.assertEqual(original,(root/'direction_review.json').read_bytes())
            with patch('sam2_prompts_to_masks.encode_class_mask_visual_video',encode):
                state=review.filter_candidates(root)
            self.assertEqual(state['choices'],{});self.assertEqual(state['status'],'pending')
            self.assertNotIn('final_subdir',state);self.assertFalse(stale.exists())
            actual=cv2.imread(str(review.candidate_root(root)/'directional_mask_frames/forward/00000.png'),0)
            for i,limit in [(1,1),(2,2),(3,1)]:
                self.assertEqual(cv2.connectedComponents((actual==i).astype(np.uint8),connectivity=8)[0]-1,limit)
            np.testing.assert_array_equal(cv2.imread(str(root/'directional_mask_frames/forward/00000.png'),0),mask)
            self.assertTrue(list((root/'retired_experiments').glob('*/filtered_versions/stale')))

    def test_area_distance_before_ranking(self):
        mask=np.zeros((180,180),np.uint8)
        mask[10:30,10:30]=1
        mask[130:135,130:136]=1  # 30 pixels, far; must not occupy second slot.
        mask[10:14,49:54]=1  # 20 pixels, nearest distance exactly20; keep.
        mask[40,10:29]=1  # 19 pixels, near; delete.
        mask[150,1:20]=2  # sole main19 also removed.
        result,details=review.filter_mask_components(mask,[{'id':1,'name':'occluder'},{'id':2,'name':'region'}],True)
        self.assertEqual(result[130,130],0);self.assertEqual(result[10,49],1)
        self.assertEqual(result[40,10],0);self.assertEqual(result[150,1],0)
        self.assertEqual(next(d for d in details if d['area']==30)['reasons'],['distance_gt_area'])
        np.testing.assert_array_equal(result,review.filter_mask_components(result,[{'id':1,'name':'occluder'},{'id':2,'name':'region'}]))

    def test_second_tool_and_object_components_require_clean_contact(self):
        objects=[{'id':1,'name':'object'},{'id':2,'name':'righttool'},{'id':3,'name':'occluder'}]
        mask=np.zeros((100,120),np.uint8)
        mask[10:30,10:30]=2;mask[40:45,10:20]=2
        mask[10:30,33:53]=1;mask[40:45,40:50]=1
        mask[60:80,80:100]=3;mask[85:90,80:90]=3
        filtered,details=review.filter_mask_components(mask,objects,True)
        self.assertFalse(review.righttool_object_contact(mask,objects))
        self.assertEqual(filtered[40,10],0);self.assertEqual(filtered[40,40],0)
        self.assertEqual(filtered[85,80],3)
        self.assertTrue(any('not_in_righttool_object_contact' in d['reasons'] for d in details))
        mask[10:30,33:53]=0;mask[10:30,32:52]=1  # nearest distance=3
        filtered=review.filter_mask_components(mask,objects)
        self.assertEqual(filtered[40,10],2);self.assertEqual(filtered[40,40],1)
        # A 1-pixel accidental contact must be removed before contact is tested.
        mask[10:30,32:52]=0;mask[10:30,70:90]=1;mask[10,30]=1
        filtered=review.filter_mask_components(mask,objects)
        self.assertEqual(filtered[40,10],0);self.assertEqual(filtered[40,40],0)
        # A third-ranked fragment that would be dropped cannot trigger contact.
        mask=np.zeros((100,120),np.uint8)
        mask[10:30,10:30]=2;mask[40:45,10:20]=2
        mask[10:30,70:90]=1;mask[40:45,70:80]=1;mask[10:18,32:38]=1
        filtered=review.filter_mask_components(mask,objects)
        self.assertEqual(filtered[40,10],0);self.assertEqual(filtered[40,70],0)

    def test_auto_candidate_and_conflict_categories(self):
        a=np.zeros((100,100),bool);a[1:31,1:31]=True
        b=a.copy();b[40:47,40:47]=True #49
        c=review.compare_components(a,b)
        self.assertEqual(review.classify_conflicts([c]),('components','forward'))
        self.assertEqual(review.classify_conflicts([review.compare_components(b,a)]),('components','reverse'))
        self.assertEqual(review.classify_conflicts([c,review.compare_components(b,a)]),('components',None))
        b[47,40]=True #50, strict boundary
        self.assertEqual(review.classify_conflicts([review.compare_components(a,b)]),('components',None))
        shifted=np.zeros_like(a);shifted[60:90,60:90]=True;shifted[40:47,40:47]=True
        self.assertEqual(review.classify_conflicts([review.compare_components(a,shifted)]),('other',None))
        empty=np.zeros_like(a)
        self.assertEqual(review.classify_conflicts([review.compare_components(empty,a)]),('other',None))

    def test_whole_iou_count_thresholds(self):
        a=np.zeros((40,40),bool);a[1:5,1:5]=True
        b=np.zeros_like(a);b[10:14,10:14]=True
        r=review.compare_components(a,b)
        self.assertTrue(r['A']);self.assertFalse(r['B'])
        self.assertEqual(r['component_areas'],{'forward':[16],'reverse':[16]})
        # Exactly 16 differing pixels, same component counts: no conflict.
        a[:]=False;b[:]=False;a[1:3,1:5]=True;b[10:12,10:14]=True
        self.assertFalse(review.compare_components(a,b)['A'])
        empty=np.zeros_like(a)
        self.assertTrue(review.compare_components(a,empty)['B'])
        self.assertFalse(review.compare_components(empty,empty)['A'])
        self.assertFalse(review.compare_components(empty,empty)['B'])
        # Exactly 0.85 IoU, difference >16: strict threshold must not trigger.
        a[:]=False;b[:]=False;a[1:21,1:11]=True;b[1:18,1:11]=True
        self.assertEqual(review.compare_components(a,b)['iou'],.85)
        self.assertFalse(review.compare_components(a,b)['A'])
        # Small secondary shift with equal count is not a separate criterion.
        a[:]=False;a[1:21,1:21]=True;b=a.copy();a[30,30]=True;b[31,31]=True
        r=review.compare_components(a,b);self.assertFalse(r['A']);self.assertFalse(r['B'])

    def test_reanalysis_preserves_frame_choices_and_final(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for direction in ('forward','reverse'):
                (root/'directional_mask_frames'/direction).mkdir(parents=True)
            a=np.zeros((80,80),np.uint8);a[5:40,5:40]=1
            b=a.copy();b[60:68,60:68]=1
            for frame in range(3):
                for direction,mask in [('forward',a),('reverse',b)]:
                    cv2.imwrite(str(root/'directional_mask_frames'/direction/f'{frame:05d}.png'),mask)
            review.write_json(root/'sam2_prompts.json',{'objects':[{'id':1,'name':'object'}]})
            review.write_json(root/'direction_conflicts.json',{'rule_version':review.RULE_VERSION,'frame_count':3,'thresholds':{'iou':.85,'min_difference_pixels':16},'conflict_frame_count':1,'intervals':[{'id':7,'start':1,'end':1,'classes':['object'],'worst_iou':.5}], 'frames':[]})
            old={'status':'finalized','revision':10,'default_direction':'forward','choices':{'7':'reverse'},'final_subdir':'reviewed_versions/existing'}
            review.write_json(root/'direction_review.json',old)
            updated=review.reanalyze(root)
            self.assertEqual(updated['choices'],{'1':'reverse'})
            self.assertEqual(updated['unresolved'],2)
            self.assertEqual(updated['final_subdir'],old['final_subdir'])
            self.assertEqual(updated['status'],'finalized')
            self.assertEqual([(i['start'],i['end']) for i in updated['intervals']],[(0,0),(1,1),(2,2)])
            review.choose(root,'forward','0')
            self.assertEqual(review.summary(root)['rule_version'],review.RULE_VERSION)

    def test_manual_cut_delete_and_rank(self):
        mask = np.zeros((30, 40), np.uint8)
        mask[5:20, 3:30] = 1
        mask[8:10, 34:36] = 1
        mask[2:5, 15:18] = 2
        cut = {'kind': 'cut', 'points': [[16, 1], [16, 23]], 'width': 3}
        separated = review.edit_components(mask, 1, [cut])
        info = review.component_preview(separated == 1)
        self.assertEqual(info['count'], 3)
        self.assertEqual(info['components'][-1]['area'], 4)
        result = review.edit_components(mask, 1, [cut, {'kind': 'delete', 'x': 23, 'y': 10}])
        self.assertEqual(result[10, 23], 0)
        self.assertEqual(result[10, 5], 1)
        self.assertEqual(result[3, 16], 2)
        self.assertEqual(mask[10, 23], 1)
        result = review.edit_components(result, 1, [{'kind': 'small', 'area': 4}])
        self.assertEqual(result[8, 34], 0)
        reconstructed = np.zeros(mask.size, np.int32)
        for start, length, label in info['runs']:
            reconstructed[start:start+length] = label
        np.testing.assert_array_equal(reconstructed.reshape(mask.shape) != 0, separated == 1)
        with self.assertRaises(ValueError):
            review.edit_components(mask, 1, [{'kind': 'delete', 'x': -1, 'y': 3}])

    def test_morphology_protects_other_classes_and_open_background(self):
        mask = np.zeros((20, 20), np.uint8)
        mask[3:15, 3:15] = 1
        mask[5:7, 5:7] = 0
        mask[10, 10] = 2
        mask[17, 17] = 1
        result, removed, filled = review.repair_mask(mask, 1, 1, 4)
        self.assertEqual((removed, filled), (1, 4))
        self.assertEqual(result[10, 10], 2)
        self.assertEqual(result[0, 0], 0)
        self.assertEqual(mask[5, 5], 0)
        self.assertEqual(review.repair_mask(mask, 1, 0, 3)[2], 0)
        edited = review.edit_components(mask, 1, [{'kind': 'morphology', 'remove_area': 1, 'fill_area': 4, 'keep_largest': True}])
        np.testing.assert_array_equal(edited, result)
        single = np.zeros((5, 5), np.uint8)
        single[2, 2] = 1
        self.assertEqual(review.repair_mask(single, 1, 10, 0)[1], 0)
        self.assertEqual(review.repair_mask(single, 1, 10, 0, False)[1], 1)

    def test_correction_version_export_noop_and_undo(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mask = np.zeros((20, 20), np.uint8)
            mask[3:15, 3:15] = 1
            mask[5:7, 5:7] = 0
            mask[17, 17] = 1
            for direction in ('forward', 'reverse'):
                directory = root / 'directional_mask_frames' / direction
                directory.mkdir(parents=True)
                cv2.imwrite(str(directory / '00000.png'), mask)
                (root / f'mask_visual_{direction}.mp4').write_bytes(b'original')
            objects = [{'id': 1, 'name': 'object'}]
            review.write_json(root / 'sam2_prompts.json', {'frame_count': 1, 'video_path': 'unused', 'objects': objects})
            review.write_json(root / 'complete.json', {})
            review.analyze(root, objects, 1)
            review.choose(root, 'reverse', 'all')
            options = dict(direction='forward', class_id=1, start=0, end=0, remove_area=1, fill_area=4, keep_largest=True)
            review.correction_options(root, options)
            with self.assertRaises(ValueError):
                review.correction_options(root, {**options, 'fill_area': 1.5})
            def fake_encode(masks, output, *args):
                output.write_bytes(b'corrected')
            with patch('sam2_prompts_to_masks.encode_class_mask_visual_video', fake_encode):
                changed = review.correct(root, options)
                self.assertEqual(changed['last_correction']['filled_pixels'], 4)
                self.assertEqual(changed['choices'], {})
                candidate = review.candidate_root(root)
                np.testing.assert_array_equal(cv2.imread(str(root / 'directional_mask_frames/forward/00000.png'), 0), mask)
                np.testing.assert_array_equal(cv2.imread(str(candidate / 'directional_mask_frames/reverse/00000.png'), 0), mask)
                review.choose(root, 'forward', 'all')
                final = review.finalize(root)
                exported = cv2.imread(str(root / final['final_subdir'] / 'mask_frames/00000.png'), 0)
                self.assertEqual(exported[5, 5], 1)
                self.assertEqual(exported[17, 17], 0)
                noop = review.correct(root, options)
                self.assertEqual(noop['status'], 'finalized')
                self.assertEqual(noop['last_correction']['changed_frames'], 0)
                self.assertEqual(review.candidate_root(root), candidate)
                manual = review.correct(root, {**options, 'remove_area': 0, 'fill_area': 0,
                    'operations': [{'kind': 'delete', 'x': 5, 'y': 5}]})
                self.assertEqual(manual['last_correction']['removed_pixels'], 144)
                review.choose(root, 'forward', 'all')
                manual_final = review.finalize(root)
                self.assertEqual(cv2.imread(str(root / manual_final['final_subdir'] / 'mask_frames/00000.png'), 0).sum(), 0)
                review.undo_correction(root)
                self.assertEqual(review.candidate_root(root), candidate)
            undone = review.undo_correction(root)
            self.assertEqual(review.candidate_root(root), root)
            self.assertEqual(undone['default_direction'], 'reverse')
            self.assertTrue((root / final['final_subdir']).exists())

    def test_conflicts_require_choices_and_export_selected_masks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for direction in ('forward', 'reverse'):
                (root / 'directional_mask_frames' / direction).mkdir(parents=True)
            for frame in range(4):
                a = np.zeros((32, 32), np.uint8)
                a[5:15, 5:15] = 1
                b = a.copy()
                if frame in (1, 2):
                    b[5:15, 5:15] = 0
                for d, m in [('forward', a), ('reverse', b)]:
                    cv2.imwrite(str(root / 'directional_mask_frames' / d / f'{frame:05d}.png'), m)
            objects = [{'id': 1, 'name': 'object'}]
            report = review.analyze(root, objects, 4)
            self.assertEqual(report['conflict_frame_count'], 2)
            self.assertEqual([(i['start'], i['end']) for i in report['intervals']], [(1, 2)])
            editor = types.SimpleNamespace(items={'sample': {'frame_count': 4}})
            editor.latest_preview = lambda item_id: {
                'run_id': 'test-run', 'output_path': str(root), 'stale': False,
                'frame_count': 4, 'direction_review': review.summary(root), 'settings': {}}
            manager = EditorManager(editor)
            with self.assertRaises(ValueError):
                manager.preview_file(manager.workspace_id, 'sample', 'video', source='final')
            self.assertEqual(manager.preview_file(manager.workspace_id, 'sample', 'frame',
                             frame=1, source='reverse'), root / 'directional_mask_frames/reverse/00001.png')
            with self.assertRaises(RuntimeError):
                manager.review_action({'workspace_id': manager.workspace_id, 'item_id': 'sample',
                                       'run_id': 'old-run', 'direction': 'reverse', 'interval_id': '0'})
            with self.assertRaises(ValueError):
                review.finalize(root)
            with self.assertRaises(ValueError):
                review.choose(root, 'reverse', '99')
            manager.review_action({'workspace_id': manager.workspace_id, 'item_id': 'sample',
                                   'run_id': 'test-run', 'direction': 'reverse', 'interval_id': '0'})
            self.assertEqual(review.summary(root)['unresolved'], 0)
            review.write_json(root / 'sam2_prompts.json', {'video_path': 'unused.mp4', 'objects': objects, 'frame_count': 4})
            payload = {'workspace_id': manager.workspace_id, 'item_id': 'sample', 'run_id': 'test-run',
                       'review_revision': review.summary(root)['revision'], 'class_id': 1,
                       'direction': 'final', 'start': 0, 'end': 0, 'remove_area': 0, 'fill_area': 0}
            with self.assertRaises(ValueError):
                manager.component_preview(payload)
            with self.assertRaises(RuntimeError):
                manager.component_preview({**payload, 'review_revision': -1})
            review.write_json(root / 'complete.json', {'frame_count': 4})
            def fake_encode(masks, output, *args):
                output.write_bytes(b'fixture')
            with patch('sam2_prompts_to_masks.encode_class_mask_visual_video', fake_encode):
                final = review.finalize(root)
            payload['review_revision'] = final['revision']
            self.assertEqual(manager.component_preview(payload)['components'][0]['area'], 100)
            self.assertEqual(manager.component_preview({**payload, 'operations': [{'kind': 'delete', 'x': 6, 'y': 6}]})['count'], 0)
            with self.assertRaises(ValueError):
                manager.component_preview({**payload, 'direction': 'forward'})
            self.assertEqual(review.summary(root)['status'], 'finalized')
            output = root / final['final_subdir']
            self.assertEqual(manager.preview_file(manager.workspace_id, 'sample', 'video'), output / 'mask_visual.mp4')
            for frame in range(4):
                m = cv2.imread(str(output / 'mask_frames' / f'{frame:05d}.png'), 0)
                self.assertEqual(int((m == 1).sum()), 0 if frame in (1, 2) else 100)
            options = review.correction_options(root, {**payload, 'operations': [{'kind': 'delete', 'x': 6, 'y': 6}]})
            with patch('sam2_prompts_to_masks.encode_class_mask_visual_video', fake_encode):
                edited = review.correct_final(root, options)
            self.assertEqual(edited['status'], 'finalized')
            self.assertEqual(edited['choices'], final['choices'])
            self.assertEqual(edited['last_final_correction']['removed_pixels'], 100)
            self.assertEqual(cv2.imread(str(root / edited['final_subdir'] / 'mask_frames/00000.png'), 0).sum(), 0)
            self.assertEqual(cv2.imread(str(root / 'directional_mask_frames/forward/00000.png'), 0).sum(), 100)
            self.assertEqual(cv2.imread(str(output / 'mask_frames/00000.png'), 0).sum(), 100)
            undone = review.undo_final_correction(root)
            self.assertEqual(undone['final_subdir'], final['final_subdir'])
            review.choose(root, 'forward', 'all')
            self.assertEqual(review.summary(root)['status'], 'pending')
            self.assertTrue((output / 'mask_visual.mp4').is_file())

    def test_scan_and_one_click_operations(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'mask_frames').mkdir()
            a=np.zeros((40,40),np.uint8)
            a[5:20,5:20]=1
            a[8:10,8:10]=0
            a[12,12]=2
            a[30:32,30:32]=1
            for frame,mask in enumerate([a,a,np.zeros_like(a)]):
                cv2.imwrite(str(root/'mask_frames'/f'{frame:05d}.png'),mask)
            report=review.scan_final(root,1,3,dict(iou=.65,area_change=.5,centroid_shift=.1,min_pixels=16))
            self.assertEqual(report['counts'],dict(components=2,holes=2,jump=2))
            self.assertEqual(report['frames'][0]['hole_pixels'],4)
            filled=review.edit_components(a,1,[{'kind':'fill_holes'}])
            self.assertEqual(filled[8,8],1)
            self.assertEqual(filled[12,12],2)
            largest=review.edit_components(a,1,[{'kind':'keep_largest'}])
            self.assertEqual(largest[30,30],0)
            self.assertEqual(largest[5,5],1)


if __name__ == '__main__':
    unittest.main()
