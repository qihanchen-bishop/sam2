import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from unet_video_preview import run
from sam2_direction_review import filter_mask_components, FILTER_VERSION


class UnetFilterTest(unittest.TestCase):
    def test_raw_retained_filtered_outputs_and_refilter_without_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp);model = root/'model';model.mkdir()
            labels = ['background','occluder','object','region','lefttool','righttool']
            palette = [[0,0,0],[64,160,255],[255,105,97],[119,221,119],[255,209,102],[101,214,173]]
            config = {'labels':labels,'palette_rgb':palette,'torchscript':'model.ts','input_size_wh':[200,120]}
            (model/'model_config.json').write_text(json.dumps(config));(model/'model.ts').write_bytes(b'fixture')
            mask = np.zeros((120,200),np.uint8)
            mask[5:15,5:15]=4;mask[20:25,5:10]=4;mask[30:32,5:7]=4
            mask[5:25,40:60]=5;mask[30:35,40:45]=5;mask[40:45,40:45]=5
            mask[80:90,10:20]=2;mask[80:84,160:165]=2
            video=root/'input.mp4'
            writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),30,(200,120))
            for _ in range(2):writer.write(np.zeros((120,200,3),np.uint8))
            writer.release()
            with patch('unet_video_preview.Predictor') as predictor:
                predictor.return_value.predict.return_value=mask
                run(model,video,root/'first','cpu')
            with patch('unet_video_preview.Predictor',side_effect=AssertionError('Must reuse raw masks')):
                run(model,video,root/'second','cpu',root/'first')
            for name in ('first','second'):
                out=root/name;meta=json.loads((out/'complete.json').read_text())
                self.assertEqual(meta['component_filter']['version'],FILTER_VERSION)
                self.assertEqual(meta['filter_changed_frames'],2)
                expected=filter_mask_components(mask,meta['objects'])
                for frame in range(2):
                    np.testing.assert_array_equal(cv2.imread(str(out/'raw_mask_frames'/f'{frame:05d}.png'),0),mask)
                    np.testing.assert_array_equal(cv2.imread(str(out/'mask_frames'/f'{frame:05d}.png'),0),expected)
                self.assertGreater(meta['filter_removed_pixels'],0)
                for filename in ('mask_visual.mp4','overlay.mp4'):
                    cap=cv2.VideoCapture(str(out/filename));count=0
                    while cap.read()[0]:count+=1
                    cap.release();self.assertEqual(count,2)


if __name__ == '__main__':unittest.main()
