import json
import hashlib
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
import torch

from feedwatch.detection_evaluation import detection_metrics, identity_metrics, match_boxes
from feedwatch.dynamic_video import visual_timestamp
from feedwatch.fish_detector import decode_torchscript_boxes, letterbox_tensor, prepare_detector_weights
from feedwatch.association_tracking import NativeAssociationTracker


class DetectionEvaluationTests(unittest.TestCase):
    def test_matching_is_one_to_one_and_keeps_valid_matches_over_iou_sum(self):
        truth = [[0, 0, 20, 20], [15, 0, 35, 20]]
        predicted = [[0, 0, 20, 20], [0, 0, 20, 20]]
        self.assertEqual(len(match_boxes(truth, predicted, .5)), 1)
        self.assertEqual(match_boxes([], predicted), [])
        metrics = detection_metrics([{'frame': 'empty', 'reference': [{'xyxy': truth[0]}], 'predicted': []}])
        self.assertEqual(metrics['fn'], 1)
        self.assertEqual(metrics['recall'], 0)

    def test_identity_switch_and_miss_are_counted_separately(self):
        frames = [{'reference': [{'identity': 'fish', 'xyxy': [0, 0, 20, 20]}],
                   'predicted': [{'track_id': track, 'xyxy': [0, 0, 20, 20]}] if track else []}
                  for track in [1, None, 2, 2]]
        metrics = identity_metrics(frames)
        self.assertEqual(metrics['identity_switches'], 1)
        self.assertEqual(metrics['missed_observations'], 1)

    def test_torchscript_coordinates_remove_padding_once_on_non_square_input(self):
        image = np.zeros((100, 200, 3), np.uint8)
        tensor, geometry = letterbox_tensor(image, 640, 'cpu')
        self.assertEqual(tuple(tensor.shape), (1, 3, 640, 640))
        self.assertEqual(geometry, (3.2, 0, 160))
        output = torch.zeros((1, 5, 8400))
        output[0, :, 0] = torch.tensor([320, 320, 320, 160, .8])
        output[0, :, 1] = output[0, :, 0]  # duplicate removed
        boxes = decode_torchscript_boxes(output, image.shape[:2], geometry,
                                        {'conf': .15, 'iou': .45, 'max_det': 100})
        self.assertEqual(boxes.shape, (1, 6))
        np.testing.assert_allclose(boxes[0, :4], [50, 25, 150, 75])

    def test_torchscript_rejects_wrong_output_contract(self):
        with self.assertRaises(ValueError):
            decode_torchscript_boxes(torch.zeros((1, 300, 6)), (100, 200), (3.2, 0, 160), {})

    def test_explicit_prepare_verifies_and_extracts_both_registered_members(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def configuration(name, member, payload):
                parent = root / name; parent.mkdir()
                archive = parent / 'weights.zip'
                with zipfile.ZipFile(archive, 'w') as package:
                    package.writestr('folder/' + member, payload)
                    package.writestr('../unexpected.py', b'never extract this')
                return {'weights': str(parent / member),
                        'weights_sha256': hashlib.sha256(payload).hexdigest(),
                        'archive_filename': 'weights.zip', 'archive_member': member,
                        'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest()}
            primary = configuration('primary', 'model.pt', b'primary fixture')
            secondary = configuration('secondary', 'model.ts', b'secondary fixture')
            primary['additional_detectors'] = [secondary]
            result = prepare_detector_weights(primary)
            self.assertTrue(result['downloaded'])
            self.assertTrue(result['additional_detectors'][0]['downloaded'])
            self.assertEqual(Path(secondary['weights']).read_bytes(), b'secondary fixture')
            self.assertFalse(list(root.rglob('*.pending')))
            self.assertFalse(list(root.rglob('unexpected.py')))
            Path(secondary['weights']).write_bytes(b'corrupted')
            with self.assertRaises(ValueError):
                prepare_detector_weights(primary)

    def test_decoder_duplicate_is_repaired_but_large_clock_error_is_not_hidden(self):
        self.assertEqual(visual_timestamp(0., 0, 30.), (0., True, 'decoder'))
        self.assertEqual(visual_timestamp(0., 1, 30., 0.),
                         (1 / 30, True, 'frame_index_fallback_nonmonotonic_decoder'))
        first = visual_timestamp(1 / 30, 0, 30.)
        self.assertEqual(first, (0., True, 'frame_index_fallback_initial_decoder_offset'))
        self.assertEqual(visual_timestamp(1 / 30, 1, 30., first[0]), (1 / 30, True, 'decoder'))
        self.assertFalse(visual_timestamp(6., 1, 30., 0.)[1])
        self.assertFalse(visual_timestamp(float('nan'), 1, 30., 0.)[1])

    def test_native_association_never_changes_boxes_or_bridges_missed_trails(self):
        cfg = json.loads(Path('config/simple_video.json').read_text())['tracker']
        for kind in ['bytetrack', 'botsort']:
            tracker = NativeAssociationTracker({**cfg, 'kind': kind}, 30.)
            def detection(x):
                return {'xyxy': [x, 10., x + 50., 40.], 'score': .2,
                        'detector_class_id': 0, 'detector_class': 'Fish'}
            first = tracker.update([detection(10)], 0., 0)[0]
            second = tracker.update([detection(12)], 1 / 30, 1)[0]
            self.assertEqual(first['track_id'], second['track_id'])
            self.assertEqual(second['xyxy'], detection(12)['xyxy'])
            self.assertFalse(second['position_predicted_only'])
            self.assertEqual(tracker.update([], 2 / 30, 2), [])
            recovered = tracker.update([detection(16)], 3 / 30, 3)[0]
            self.assertEqual(recovered['track_id'], first['track_id'])
            self.assertEqual(recovered['trail_txy'], [])
            self.assertIsNone(recovered['speed_source_px_per_second'])

    def test_mixed_native_thresholds_preserve_original_detection_indices(self):
        base = json.loads(Path('config/simple_video.json').read_text())['tracker']
        for kind in ['bytetrack', 'botsort']:
            tracker = NativeAssociationTracker({**base, 'kind': kind,
                'track_high_thresh': .45, 'track_low_thresh': .15, 'new_track_thresh': .45}, 30.)
            detections = [{'xyxy': [x, 10., x + 50., 40.], 'score': score,
                           'detector_class_id': 0, 'detector_class': 'Fish'}
                          for x, score in [(10., .2), (300., .9)]]
            rows = [tracker.update(detections, index / 30, index) for index in range(3)]
            self.assertEqual(len({row[1]['track_id'] for row in rows}), 1)
            self.assertNotEqual(rows[0][0]['track_id'], rows[0][1]['track_id'])


if __name__ == '__main__':
    unittest.main()
