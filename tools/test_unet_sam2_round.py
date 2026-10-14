import unittest
import tempfile
import json
import csv
from pathlib import Path
import cv2
import numpy as np
from unet_sam2_scores import transient_scores, nullable_iou, score_video
from unet_sam2_round import spread_points


class RoundAlgorithmsTest(unittest.TestCase):
    def test_scores_every_frame_and_absence_is_not_perfect_iou(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);u=root/'unet';s=root/'sam2';filtered=s/'filtered'
            (u/'mask_frames').mkdir(parents=True)
            for d in ('forward','reverse'):(filtered/'directional_mask_frames'/d).mkdir(parents=True)
            from sam2_direction_review import FILTER_VERSION
            (s/'direction_review.json').write_text(json.dumps({'candidate_subdir':'filtered','candidate_filter':{'version':FILTER_VERSION}}))
            (u/'complete.json').write_text(json.dumps({'component_filter':{'version':FILTER_VERSION}}))
            for frame in range(40):
                mask=np.zeros((30,40),np.uint8);mask[2:8,2:8]=1
                if frame>=20:mask[15:23,15:23]=2
                for d in ('forward','reverse'):cv2.imwrite(str(filtered/'directional_mask_frames'/d/f'{frame:05d}.png'),mask)
                if frame==10:mask[2:8,2:8]=0;mask[2:8,20:26]=1
                cv2.imwrite(str(u/'mask_frames'/f'{frame:05d}.png'),mask)
            score_video(u,s,root/'scores',[{'id':1,'name':'lefttool'},{'id':2,'name':'object'}],40,30,[0,39])
            rows=json.loads((root/'scores/frame_scores.json').read_text())['frames']
            self.assertEqual(len(rows),40);self.assertEqual(rows[10]['score'],0)
            self.assertIsNone(rows[0]['classes']['object']['iou']['fr'])
            self.assertEqual(rows[25]['score'],100)
            self.assertEqual(rows[20]['classes']['object']['transient']['unet'],0)
            with (root/'scores/frame_scores.csv').open() as f:self.assertEqual(len(list(csv.DictReader(f))),40)

    def test_transient_not_sustained(self):
        stable=np.tile([100.,50.,50.],(90,1))
        persistent=stable.copy();persistent[30:,0]=300
        self.assertFalse(transient_scores(persistent,30)[0].any())
        pulse=stable.copy();pulse[30:34,0]=300
        penalties,visible,events=transient_scores(pulse,30)
        self.assertTrue((penalties[30:34]>=.5).all());self.assertEqual(np.count_nonzero(penalties),4)
        self.assertFalse(visible[0]);self.assertTrue(events)
        long=stable.copy();long[30:60,0]=300
        self.assertFalse(transient_scores(long,30)[0].any())
        empty=np.zeros((90,3));empty[30:32]=[100,50,50]
        self.assertTrue((transient_scores(empty,30)[0][30:32]==1).all())
        components=np.tile([100.,50.,50.,1.],(90,1));components[30:33,3]=2
        self.assertTrue((transient_scores(components,30)[0][30:33]>=.5).all())
        components[30:,3]=2
        self.assertFalse(transient_scores(components,30)[0].any())

    def test_uniform_high_confidence_points_and_empty_iou(self):
        mask=np.zeros((100,100),bool);mask[10:90,20:80]=True
        prob=np.full(mask.shape,.95,np.float32);prob[10:40]=.7
        points,_=spread_points(mask,prob,5)
        self.assertEqual(len(points),5)
        self.assertTrue(all(mask[p['y'],p['x']] and p['confidence']>=.9 for p in points))
        coords=np.array([[p['x'],p['y']] for p in points]);self.assertGreater(np.ptp(coords[:,0]),40);self.assertGreater(np.ptp(coords[:,1]),40)
        blank=np.zeros((5,5),bool);self.assertIsNone(nullable_iou(blank,blank))
        other=blank.copy();other[2,2]=1;self.assertEqual(nullable_iou(blank,other),0.)


if __name__=='__main__':unittest.main()
