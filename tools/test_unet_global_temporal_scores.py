import unittest

import numpy as np

from unet_global_temporal_scores import assess_global, count_states, penalized_curve


def evaluate(areas,counts,ious=None):
    a=np.asarray(areas,float);ious=ious if ious is not None else [None]+[min(x,y)/max(x,y) if x or y else None for x,y in zip(a[:-1],a[1:])]
    diff=[0]+[max(17,abs(x-y)) if ious[i+1] is not None and ious[i+1]<1 else 0 for i,(x,y) in enumerate(zip(a[:-1],a[1:]))]
    return assess_global(a,counts,ious,diff,30)[0]


class GlobalTests(unittest.TestCase):
    def test_band_solver_matches_dense(self):
        rng=np.random.default_rng(42);n=40;y=rng.normal(size=n);w=rng.uniform(.01,1,n);d=np.diff(np.eye(n),n=2,axis=0);strength=17.
        expected=np.linalg.solve(np.diag(w)+strength*d.T@d,w*y)
        np.testing.assert_allclose(penalized_curve(y,w,strength),expected,rtol=1e-9,atol=1e-9)

    def test_normal_frame_inside_wrong_interval_is_not_an_outlier(self):
        areas=np.full(120,8500.);counts=np.ones(120,int)
        areas[45:61]=9500;counts[45:61]=2;areas[53]=8500;counts[53]=1
        rows=evaluate(areas,counts)
        self.assertFalse(rows[53]['suspected']);self.assertEqual(rows[53]['score'],100.)
        self.assertTrue(all(rows[i]['component_outlier'] for i in range(45,61) if i!=53))
        self.assertFalse(rows[53]['incoming_reference_valid']);self.assertFalse(rows[53]['outgoing_reference_valid'])

    def test_persistent_component_and_area_states_are_retained(self):
        counts=np.ones(180,int);counts[60:120]=2
        np.testing.assert_array_equal(count_states(counts,15),counts)
        areas=np.full(180,400.);areas[60:]=800
        rows=evaluate(areas,np.ones(180,int));self.assertFalse(any(r['suspected'] for r in rows))

    def test_shape_only_spike_and_short_area_excursion(self):
        ious=[None]+[1.]*119;ious[60]=ious[61]=0.
        rows=evaluate([400.]*120,[1]*120,ious)
        self.assertTrue(rows[60]['shape_outlier']);self.assertFalse(rows[59]['suspected']);self.assertFalse(rows[61]['suspected'])
        areas=np.full(120,400.);areas[60:63]=800
        rows=evaluate(areas,[1]*120)
        self.assertTrue(all(rows[i]['area_outlier'] for i in [60,61,62]))

    def test_gradual_growth_and_empty_class(self):
        rows=evaluate(np.linspace(400,800,180),[1]*180)
        self.assertFalse(any(r['suspected'] for r in rows))
        rows=evaluate([0.]*120,[0]*120)
        self.assertTrue(all(r['score'] is None and not r['suspected'] for r in rows))

    def test_raw_iou_curve_marks_pair_and_single_sided_review(self):
        ious=[None]+[.95]*179;ious[90]=.3
        rows=evaluate([400.]*180,[1]*180,ious)
        self.assertTrue(rows[89]['iou_next_outlier'])
        self.assertTrue(rows[90]['iou_previous_outlier'])
        self.assertTrue(rows[89]['iou_boundary_review'])
        self.assertEqual(rows[90]['iou_frame_confidence'],'low')
        self.assertFalse(rows[89]['suspected'])
        self.assertFalse(rows[90]['shape_outlier'])

    def test_iou_gradual_trend_and_persistent_low_plateau(self):
        ious=[None]+np.linspace(.95,.5,179).tolist()
        rows=evaluate([400.]*180,[1]*180,ious)
        self.assertFalse(any(r['iou_previous_outlier'] for r in rows))
        ious=[None]+[.95]*59+[.5]*60+[.95]*60
        rows=evaluate([400.]*180,[1]*180,ious)
        self.assertFalse(rows[90]['iou_previous_outlier'])

    def test_iou_small_pixel_difference_ignored(self):
        ious=[None]+[1.]*119;ious[60]=.1
        rows,_=assess_global([400.]*120,[1]*120,ious,[0]*60+[16]+[0]*59,30)
        self.assertFalse(rows[60]['iou_previous_outlier'])


if __name__=='__main__':unittest.main()
