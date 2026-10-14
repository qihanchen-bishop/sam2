"""Regression checks for output-only prompt-frame prediction (no GPU needed)."""
import types
import unittest

import torch

from sam2.sam2_video_predictor import SAM2VideoPredictor


class PromptFramePropagationTest(unittest.TestCase):
    def run_case(self, enabled, conditions=(0, 2)):
        original = {t: {'pred_masks': torch.tensor([[[[float(t + 10)]]]])}
                    for t in conditions}
        outputs = {'cond_frame_outputs': original.copy(), 'non_cond_frame_outputs': {}}
        state = {'obj_ids': [1], 'num_frames': 3, 'device': 'cpu',
                 'output_dict_per_obj': {0: outputs}, 'frames_tracked_per_obj': {0: {}}}
        calls = []

        def infer(**kwargs):
            calls.append(kwargs)
            pred = torch.tensor([[[[float(100 + kwargs['frame_idx'])]]]])
            return {'pred_masks': pred}, pred

        predictor = types.SimpleNamespace(
            propagate_in_video_preflight=lambda state: None,
            _get_obj_num=lambda state: 1,
            _run_single_frame_inference=infer,
            _get_orig_video_res_output=lambda state, masks: (None, masks),
            clear_non_cond_mem_around_input=False,
        )
        result = list(SAM2VideoPredictor.propagate_in_video(
            predictor, state, propagate_conditioning_outputs=enabled))
        return original, outputs, calls, result

    def test_output_changes_without_replacing_conditioning_memory(self):
        original, outputs, calls, result = self.run_case(True)
        self.assertEqual([int(x[2].item()) for x in result], [10, 101, 102])
        for t in (0, 2):
            self.assertIs(outputs['cond_frame_outputs'][t], original[t])
            self.assertEqual(original[t]['pred_masks'].item(), t + 10)
        self.assertEqual(set(outputs['non_cond_frame_outputs']), {1})
        correction = calls[-1]
        self.assertFalse(correction['run_mem_encoder'])
        self.assertFalse(correction['is_init_cond_frame'])
        self.assertIsNone(correction['point_inputs'])
        self.assertIsNone(correction['mask_inputs'])
        self.assertEqual(set(correction['output_dict']['cond_frame_outputs']), {0})

    def test_disabled_preserves_original_behavior(self):
        _, _, calls, result = self.run_case(False)
        self.assertEqual([int(x[2].item()) for x in result], [10, 101, 12])
        self.assertEqual(len(calls), 1)

    def test_sole_condition_still_gets_a_tracking_prediction(self):
        original, outputs, calls, result = self.run_case(True, conditions=(2,))
        self.assertEqual(int(result[0][2].item()), 102)
        self.assertIs(outputs['cond_frame_outputs'][2], original[2])
        self.assertFalse(calls[0]['run_mem_encoder'])
        self.assertEqual(set(calls[0]['output_dict']['cond_frame_outputs']), {2})


if __name__ == '__main__':
    unittest.main()
