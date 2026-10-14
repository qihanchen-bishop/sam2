import unittest

import cv2
import numpy as np

from unet_temporal_scores import evaluate_series


def evaluate(masks):
    areas=[int(m.sum()) for m in masks]
    counts=[cv2.connectedComponents(m.astype(np.uint8),connectivity=8)[0]-1 for m in masks]
    ious=[None];diffs=[0]
    for a,b in zip(masks[:-1],masks[1:]):
        union=int((a|b).sum());ious.append(float((a&b).sum()/union) if union else None)
        diffs.append(int((a^b).sum()))
    return evaluate_series(areas,counts,ious,diffs,30)


def rectangle(x=10,width=20):
    m=np.zeros((100,160),bool);m[20:40,x:x+width]=True
    return m


class TemporalTests(unittest.TestCase):
    def test_constant_and_gradual_motion_or_growth_are_supported(self):
        for masks in [[rectangle() for _ in range(60)],
                      [rectangle(x=10+i) for i in range(60)],
                      [rectangle(width=20+i) for i in range(60)]]:
            rows=evaluate(masks)
            self.assertFalse(any(r['suspected'] for r in rows))
            self.assertTrue(all(r['status']=='stable_or_gradual' for r in rows))

    def test_persistent_changes_have_one_stable_direction(self):
        masks=[rectangle() if i<30 else rectangle(x=80) for i in range(60)]
        self.assertFalse(any(r['suspected'] for r in evaluate(masks)))
        masks=[rectangle() for _ in range(60)]
        for m in masks[30:]:m[70:75,120:125]=True
        self.assertFalse(any(r['suspected'] for r in evaluate(masks)))

    def test_isolated_shape_error_same_area_and_count(self):
        masks=[rectangle() for _ in range(60)];masks[30]=rectangle(x=80)
        rows=evaluate(masks)
        self.assertTrue(rows[30]['suspected']);self.assertEqual(rows[30]['score'],0.)
        self.assertEqual(sum(r['suspected'] for r in rows),1)
        self.assertIn('iou_change',rows[30]['past']['reasons'])

    def test_brief_extra_component_and_missing_mask(self):
        masks=[rectangle() for _ in range(60)]
        for m in masks[30:33]:m[70:75,120:125]=True
        rows=evaluate(masks)
        self.assertEqual([i for i,r in enumerate(rows) if r['suspected']],[30,31,32])
        self.assertIn('component_count',rows[31]['past']['reasons'])
        masks=[rectangle() for _ in range(60)]
        for i in [30,31]:masks[i][:]=False
        rows=evaluate(masks)
        self.assertTrue(rows[30]['suspected'] and rows[31]['suspected'])

    def test_reversal_symmetry_and_absence(self):
        masks=[rectangle(width=20+i//3) for i in range(60)];masks[30]=rectangle(x=90)
        a=evaluate(masks);b=list(reversed(evaluate(list(reversed(masks)))))
        self.assertEqual([r['suspected'] for r in a],[r['suspected'] for r in b])
        self.assertEqual([r['score'] for r in a],[r['score'] for r in b])
        absent=evaluate([np.zeros((40,40),bool) for _ in range(30)])
        self.assertTrue(all(r['score'] is None and r['status']=='absent_in_window' for r in absent))

    def test_insufficient_context_is_not_accepted(self):
        rows=evaluate([rectangle() for _ in range(2)])
        self.assertTrue(all(r['score'] is None and r['status']=='uncertain_context' for r in rows))

    def test_accelerating_exposure_is_progressive_but_spike_is_not(self):
        areas=[0]*20+[20,24,168,188,220,228,244,256,316,324,328]+[328]*20
        counts=[int(a>0) for a in areas];ious=[None];diffs=[0]
        for a,b in zip(areas[:-1],areas[1:]):
            ious.append(min(a,b)/max(a,b) if a or b else None);diffs.append(abs(a-b))
        rows=evaluate_series(areas,counts,ious,diffs,30)
        self.assertFalse(any(r['suspected'] for r in rows))
        self.assertTrue(rows[21]['future']['area']['progressive_support'])
        areas=[400]*60;areas[30]=800;counts=[1]*60
        ious=[None]+[min(a,b)/max(a,b) for a,b in zip(areas[:-1],areas[1:])]
        diffs=[0]+[abs(a-b) for a,b in zip(areas[:-1],areas[1:])]
        self.assertTrue(evaluate_series(areas,counts,ious,diffs,30)[30]['suspected'])


if __name__=='__main__':unittest.main()
