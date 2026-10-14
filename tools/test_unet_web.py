import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'apps'))
from seed_bbox_web import Editor, EditorManager


class UnetWebTest(unittest.TestCase):
    def test_separate_versions_locks_and_media_guards(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / 'model';model.mkdir()
            (model / 'model_config.json').write_text(json.dumps({'labels': ['background', 'object']}))
            editor = Editor.__new__(Editor)
            editor.work_dir = root
            editor.items = {'ep40': {'view': 'front', 'video_path': '/fixture.mp4', 'frame_count': 2},
                            'top': {'view': 'top'}}
            editor.labels = lambda _: ([{'name': 'object'}], '')
            manager = EditorManager(editor)
            payload = {'workspace_id': manager.workspace_id, 'item_id': 'ep40', 'device': 'cpu'}
            sam2 = editor.preview_root('ep40');sam2.mkdir(parents=True)
            sentinel = sam2 / 'latest.json';sentinel.write_text('unchanged SAM2')
            with patch('seed_bbox_web.UNET_MODEL_DIR', model), patch.object(manager, 'propagation_options', return_value={'devices': [{'value': 'cpu'}]}):
                with self.assertRaises(ValueError):manager.start_unet({**payload, 'item_id': 'top'})
                with patch('seed_bbox_web.threading.Thread') as thread:
                    started = manager.start_unet(payload)
                    worker = thread.call_args.kwargs['target']
                self.assertEqual(manager.job['kind'], 'unet')
                with self.assertRaises(RuntimeError):manager.start_unet(payload)
                output = editor.unet_root('ep40') / started['run_id'] / 'output'
                def complete(*args, **kwargs):
                    output.mkdir()
                    (output/'complete.json').write_text(json.dumps({'frame_count': 2}))
                    for name in ('mask_visual.mp4','overlay.mp4'):(output/name).write_bytes(b'fixture')
                    return types.SimpleNamespace(returncode=0)
                with patch('seed_bbox_web.subprocess.run', side_effect=complete):worker()
                self.assertEqual(manager.job['state'], 'complete')
                self.assertEqual(sentinel.read_text(), 'unchanged SAM2')
                self.assertEqual(editor.latest_unet('ep40')['run_id'], started['run_id'])
                self.assertEqual(manager.unet_video(manager.workspace_id, 'ep40', started['run_id'], 'overlay'), output/'overlay.mp4')
                with self.assertRaises(RuntimeError):manager.unet_video('stale', 'ep40', started['run_id'], 'overlay')
                with self.assertRaises(RuntimeError):manager.unet_video(manager.workspace_id, 'ep40', 'old', 'overlay')
                with self.assertRaises(ValueError):manager.unet_video(manager.workspace_id, 'ep40', started['run_id'], '../latest.json')
                with patch('seed_bbox_web.threading.Thread') as thread:
                    manager.start_unet({**payload, 'filter_existing': True})
                    worker = thread.call_args.kwargs['target']
                with patch('seed_bbox_web.subprocess.run', return_value=types.SimpleNamespace(returncode=1)) as process:
                    worker()
                    self.assertIn('--source-output', process.call_args.args[0])
                self.assertEqual(manager.job['state'], 'failed')
                self.assertEqual(editor.latest_unet('ep40')['run_id'], started['run_id'])
                self.assertEqual(sentinel.read_text(), 'unchanged SAM2')


if __name__ == '__main__':unittest.main()
