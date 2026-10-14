import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from train_unet_pseudo_iteration import build_cache


class FrozenCacheTests(unittest.TestCase):
    def fixture(self, parent):
        root=parent/'round';root.mkdir();model=parent/'model';prior=parent/'prior'
        old=[]
        for frame,split in [(0,'train'),(1,'val')]:
            for folder,suffix in [('images','.jpg'),('masks','.png')]:
                path=prior/folder/split/'front/file-002'/f'{frame:05d}{suffix}'
                path.parent.mkdir(parents=True,exist_ok=True)
                pixels=np.full((456,600,3) if folder=='images' else (456,600),80 if folder=='images' else 2,np.uint8)
                self.assertTrue(cv2.imwrite(str(path),pixels))
            old.append({'episode':2,'frame':frame,'split':split,'source_mask_sha256':f'prior-{frame}'})
        (prior/'manifest.json').write_text(json.dumps({'samples':old}))
        video=parent/'new.avi';writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'MJPG'),10,(16,16))
        self.assertTrue(writer.isOpened())
        for _ in range(4):writer.write(np.full((16,16,3),100,np.uint8))
        writer.release()
        records=[]
        for ep,count in [(2,2),(3,4)]:
            name=f'ep{ep:06d}';records.append({'id':name,'item':{'episode_index':ep,'frame_count':count,'video_key':'front','video_path':str(video)}})
            scores=root/'unet_global_r2'/name;scores.mkdir(parents=True)
            frames=[{'frame':f,'suspected_classes':['object'] if ep==3 and f==0 else [],'uncertain_classes':[]} for f in range(count)]
            (scores/'frame_scores.json').write_text(json.dumps({'frames':frames}))
            masks=root/name/'unet/mask_frames';masks.mkdir(parents=True)
            for f in range(count):self.assertTrue(cv2.imwrite(str(masks/f'{f:05d}.png'),np.full((16,16),2,np.uint8)))
        (root/'manifest.json').write_text(json.dumps({'records':records,'exclusion':'suspect OR review'}))
        return root,model,prior

    def test_previous_split_and_label_files_are_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            root,model,prior=self.fixture(Path(temp));cache=build_cache(root,model,prior)
            manifest=json.loads((cache/'manifest.json').read_text());samples={(s['episode'],s['frame']):s for s in manifest['samples']}
            self.assertEqual(samples[(2,0)]['split'],'train');self.assertEqual(samples[(2,1)]['split'],'val')
            self.assertNotIn((3,0),samples);self.assertEqual(manifest['train'],3);self.assertEqual(manifest['val'],2)
            self.assertEqual((cache/'masks/val/front/file-002/00001.png').read_bytes(),(prior/'masks/val/front/file-002/00001.png').read_bytes())

    def test_changed_prior_frame_selection_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root,model,prior=self.fixture(Path(temp));path=root/'unet_global_r2/ep000002/frame_scores.json'
            scores=json.loads(path.read_text());scores['frames'][1]['uncertain_classes']=['righttool'];path.write_text(json.dumps(scores))
            with self.assertRaisesRegex(AssertionError,'Previous accepted frames changed'):build_cache(root,model,prior)


if __name__=='__main__':unittest.main()
