import tempfile
import unittest
import types
from pathlib import Path

import numpy as np
from PIL import Image

from sam2_prompts_to_masks import compose_priority_masks, export_class_mask_range, export_visual_outputs, save_propagated_class_masks


class OverlapPriorityTest(unittest.TestCase):
    def setUp(self):
        self.objects = [{'id': 1, 'name': 'occluder'}, {'id': 2, 'name': 'object'},
                        {'id': 3, 'name': 'region'}, {'id': 4, 'name': 'lefttool'},
                        {'id': 5, 'name': 'righttool'}]
        self.masks = {i: np.zeros((12, 12), bool) for i in range(1, 6)}
        self.masks[1][:] = True
        self.masks[3][1:11, 1:11] = True
        self.masks[5][1:5, 1:5] = True  # Main component: 16 pixels.
        self.masks[5][8:10, 8:10] = True  # Secondary: 4 pixels.
        self.masks[4][2, 2] = True
        self.masks[4][8, 8] = True
        self.masks[2][3, 3] = True

    def test_order_and_no_invented_left_pixels(self):
        a = compose_priority_masks(self.objects, self.masks, (12, 12))
        self.assertEqual(a[2, 2], 5)  # Main right wins against left.
        self.assertEqual(a[8, 8], 4)  # Left wins against secondary right.
        self.assertEqual(a[9, 9], 5)  # Secondary right still beats region.
        self.assertEqual(a[3, 3], 2)  # Object still beats tools.
        self.assertEqual(a[6, 6], 3)
        self.assertEqual(a[0, 0], 1)
        self.masks[5][:] = False
        self.assertEqual(compose_priority_masks(self.objects, self.masks, (12, 12))[2, 2], 4)

    def test_export_paths_share_rule(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / 'object_mask_frames'
            folder.mkdir()
            for i, m in self.masks.items():
                Image.fromarray(m.astype('uint8') * 255).save(folder / f'00000_obj{i:03d}.png')
            frame = root / 'rgb.png'
            Image.new('RGB', (12, 12)).save(frame)
            expected = compose_priority_masks(self.objects, self.masks, (12, 12))
            export_visual_outputs(root, [frame], self.objects)
            export_class_mask_range(folder, root / 'range', self.objects, range(1), (12, 12))
            for path in [root / 'mask_frames/00000.png', root / 'range/00000.png']:
                np.testing.assert_array_equal(np.asarray(Image.open(path)), expected)

    def test_forward_and_reverse_stream_use_rule(self):
        import torch
        logits = torch.tensor(np.stack([self.masks[i] for i in range(1, 6)]), dtype=torch.float32) * 2 - 1
        predictor = types.SimpleNamespace(propagate_in_video=lambda *a, **kw: iter([(0, list(range(1, 6)), logits)]))
        expected = compose_priority_masks(self.objects, self.masks, (12, 12))
        with tempfile.TemporaryDirectory() as tmp:
            for reverse in (False, True):
                folder = Path(tmp) / str(reverse)
                save_propagated_class_masks(predictor, {}, [(folder, None, None)], self.objects, 0, reverse, 0, 1)
                np.testing.assert_array_equal(np.asarray(Image.open(folder / '00000.png')), expected)


if __name__ == '__main__':
    unittest.main()
