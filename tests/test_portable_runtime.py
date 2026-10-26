import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from feedwatch.data import file_hash
from feedwatch.local_app import LocalService
from feedwatch.portable_runtime import load_predictor


class PortableRuntimeTest(unittest.TestCase):
    def test_missing_manifest_keeps_the_historical_loader(self):
        with tempfile.TemporaryDirectory() as folder:
            previous=Path.cwd()
            try:
                os.chdir(folder)
                with patch('feedwatch.portable_runtime.OperationalPredictor',return_value='historical') as loader:
                    self.assertEqual(load_predictor(Path('protocol'),'cpu'),'historical')
                    loader.assert_called_once_with(Path('protocol'),'cpu')
            finally:os.chdir(previous)

    def test_tampered_bundle_is_rejected_before_loading_a_model(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'artifacts').mkdir();(root/'head.pt').write_bytes(b'corrupted')
            (root/'artifacts/runtime.json').write_text(json.dumps({'head':{'path':'head.pt','sha256':'0'*64}}))
            previous=Path.cwd()
            try:
                os.chdir(root)
                with patch('feedwatch.portable_runtime.load_temporal_checkpoint') as loader:
                    with self.assertRaisesRegex(ValueError,'pacote de execução inválido'):load_predictor()
                    loader.assert_not_called()
            finally:os.chdir(previous)

    def test_bundle_cannot_reference_a_file_outside_its_root(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'checkout';(root/'artifacts').mkdir(parents=True)
            outside=Path(folder)/'outside.pt';outside.write_bytes(b'not a checkpoint')
            (root/'artifacts/runtime.json').write_text(json.dumps({'head':{'path':'../outside.pt','sha256':file_hash(outside)}}))
            previous=Path.cwd()
            try:
                os.chdir(root)
                with self.assertRaisesRegex(ValueError,'pacote de execução inválido'):load_predictor()
            finally:os.chdir(previous)

    def test_oversized_upload_fails_before_video_decode_and_model_load(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/'oversized.mp4'
            with source.open('wb') as stream:stream.truncate(100*1024*1024+1)
            with patch('feedwatch.local_app.open_capture') as decode:
                service=LocalService(root/'runs',runtime_factory=lambda *args: self.fail('Model must not load'))
                with self.assertRaisesRegex(ValueError,'100 MB'):service.analyze(source)
                decode.assert_not_called()
                self.assertEqual(service.model_load_count,0)


if __name__=='__main__':unittest.main()
