import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import cv2
from torchvision.models import MobileNet_V3_Small_Weights

from feedwatch.data import file_hash
from feedwatch.tracking_replay import _configuration, _geometry_predictor, replay_tracking_video
from feedwatch.video_inference import display_state, plan_windows
from test_video_mvp import make_video


CONFIG = json.loads(Path('config/simple_video.json').read_text(encoding='utf-8'))


class TrackingReplayTests(unittest.TestCase):
    def setUp(self):
        cv2.setNumThreads(1)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / 'source.mp4'
        make_video(self.source, frames=50, width=320, height=180)
        self.saved = self.root / 'saved'
        self.saved.mkdir()
        self.output = self.root / 'replayed'
        preset = MobileNet_V3_Small_Weights.IMAGENET1K_V1.transforms()
        self.metadata = {'width': 320, 'height': 180, 'frame_count': 50, 'fps': 25., 'duration_seconds': 2.}
        windows = plan_windows(self.metadata, 'clip')
        windows[0].update(status='ok', predicted_class='weak', predicted_id=1, legend_pt='Fraca',
                          scores={'none': .1, 'weak': .7, 'medium': .1, 'strong': .1})
        self.run = {'presentation_profile': 'simple', 'diagnostic': False, 'mode': 'clip',
            'input_signature': {'sha256': file_hash(self.source)}, 'metadata': self.metadata,
            'dynamic_config': deepcopy(CONFIG), 'windows': windows, 'status': 'completed',
            'food_model': {'extractor_identity': {'parameters': {'crop_xyxy': [.05, .1, .95, .95]},
                'weights': {'architecture': 'mobilenet_v3_small', 'resize_short_edge': preset.resize_size,
                    'center_crop': preset.crop_size, 'interpolation': str(preset.interpolation),
                    'antialias': preset.antialias}}},
            'counters': {'observations_with_speed': 99999}, 'reproduce_command': 'old analysis command',
            'total_processing_seconds': 99999, 'reload_verification': {'old': True}}
        self.rows = []
        for index in range(50):
            food = display_state(windows, 'clip', index / 25.)
            detection = {'xyxy': [60., 60., 100., 90.], 'score': .8, 'detector_class_id': 0,
                         'detector_class': 'Fish', 'track_id': 9999, 'trail_txy': [[0., 1., 1.]]}
            self.rows.append({'frame_index': index, 'timestamp_seconds': index / 25.,
                'food_status': food['status'], 'food_window_id': food['window']['window_id'],
                'food_class': food['window']['predicted_class'], 'detector_status': 'ok',
                'detected_count': 1, 'detections': [detection], 'motion_status': 'disabled',
                'image_motion': {'status': 'disabled'}, 'motion_mean_source_px_per_second': None,
                'motion_p90_source_px_per_second': None, 'moving_fraction': None, 'error': ''})
        self.rows[20].update(detector_status='unavailable', detected_count=None, detections=[])
        self.rows[30].update(detected_count=0, detections=[])
        self.config = deepcopy(CONFIG)
        self.config['tracker'] = {**CONFIG['tracker'], 'kind': 'occlusion-aware-v1', 'max_gap_seconds': 1.}
        self.write_saved()

    def write_saved(self):
        (self.saved / 'run.json').write_text(json.dumps(self.run), encoding='utf-8')
        (self.saved / 'frames.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in self.rows), encoding='utf-8')

    def replay(self):
        return replay_tracking_video(self.source, self.saved, self.output, self.config)

    def test_real_video_replay_preserves_predictions_boxes_counts_and_missing_observations(self):
        before = file_hash(self.source)
        report = self.replay()
        replayed = [json.loads(line) for line in (self.output / 'frames.jsonl').read_text().splitlines()]
        self.assertEqual(report['windows'], self.run['windows'])
        self.assertEqual(report['export_verification']['decoded_frames'], 50)
        self.assertEqual(file_hash(self.source), before)
        for prior, current in zip(self.rows, replayed):
            for key in ('timestamp_seconds', 'food_status', 'food_window_id', 'food_class',
                        'detected_count', 'detector_status', 'image_motion'):
                self.assertEqual(prior[key], current[key])
            self.assertEqual([{k: d[k] for k in ('xyxy', 'score', 'detector_class_id', 'detector_class')}
                              for d in prior['detections']],
                             [{k: d[k] for k in ('xyxy', 'score', 'detector_class_id', 'detector_class')}
                              for d in current['detections']])
        self.assertIsNone(replayed[20]['detected_count'])
        self.assertEqual(replayed[30]['detected_count'], 0)
        self.assertEqual(replayed[20]['suspended_track_ids'], [])
        self.assertEqual(replayed[21]['detections'][0]['trail_txy'], [])
        self.assertNotEqual(replayed[19]['detections'][0]['track_id'], replayed[21]['detections'][0]['track_id'])
        self.assertTrue(all(d['track_id'] != 9999 for row in replayed for d in row['detections']))
        self.assertLess(report['counters']['observations_with_speed'], 99999)
        self.assertEqual(report['counters']['detection_observations'], 48)
        self.assertNotIn('reload_verification', report)
        self.assertNotIn('reproduce_command', report)
        self.assertNotIn('total_processing_seconds', report)
        audit = report['tracking_replay']
        self.assertEqual(audit['before_detections_sha256'], audit['after_detections_sha256'])
        self.assertEqual(audit['before_windows_sha256'], audit['after_windows_sha256'])
        self.assertFalse(audit['detector_inference_run'])
        self.assertFalse(audit['feeding_inference_run'])

    def test_existing_output_is_never_overwritten(self):
        self.output.mkdir()
        sentinel = self.output / 'keep.txt'
        sentinel.write_text('keep')
        with self.assertRaisesRegex(ValueError, 'already exists'):
            self.replay()
        self.assertEqual(sentinel.read_text(), 'keep')

    def test_legacy_tracker_forgets_missing_frames_without_reusing_ids(self):
        self.config['tracker'] = {**CONFIG['tracker'], 'kind': 'conservative-v2', 'min_iou': .2,
                                 'max_innovation_diagonals': .3}
        self.replay()
        replayed = [json.loads(line) for line in (self.output / 'frames.jsonl').read_text().splitlines()]
        self.assertNotEqual(replayed[19]['detections'][0]['track_id'], replayed[21]['detections'][0]['track_id'])
        self.assertEqual(replayed[21]['detections'][0]['trail_txy'], [])

    def test_modified_cache_during_render_is_not_published(self):
        def change_cache(stage, done, total):
            if done == total:
                (self.saved / 'run.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'analysis changed'):
            replay_tracking_video(self.source, self.saved, self.output, self.config, progress=change_cache)
        self.assertFalse((self.output / 'annotated.mp4').exists())

    def test_original_hash_is_required_before_creating_output(self):
        self.run['input_signature']['sha256'] = '0' * 64
        self.write_saved()
        with self.assertRaisesRegex(ValueError, 'input hash'):
            self.replay()
        self.assertFalse(self.output.exists())

    def test_only_tracker_configuration_may_change_and_inputs_are_not_mutated(self):
        original = deepcopy(self.config)
        updated, _ = _configuration(self.config, self.run)
        self.assertEqual(self.config, original)
        self.assertFalse(updated['presentation']['show_speed'])
        for section, key in [('detector', 'conf'), ('motion', 'winsize'), ('presentation', 'card_seconds')]:
            config = deepcopy(self.config)
            config[section][key] = -1
            with self.subTest(section=section), self.assertRaisesRegex(ValueError, 'only the tracker'):
                _configuration(config, self.run)

    def test_invalid_cache_is_rejected_before_rendering(self):
        original = deepcopy(self.rows)
        mutations = [
            lambda: self.rows.pop(),
            lambda: self.rows[2].update(frame_index=9),
            lambda: self.rows[2].update(timestamp_seconds=float('nan')),
            lambda: self.rows[2].update(timestamp_seconds=99.),
            lambda: self.rows[2].update(timestamp_seconds=None),
            lambda: self.rows[2].update(detected_count=0),
            lambda: self.rows[20].update(detected_count=0),
            lambda: self.rows[2]['detections'][0].update(xyxy=[-1., 1., 2., 3.]),
            lambda: self.rows[2]['detections'][0].update(score=1.2),
            lambda: self.rows[2]['detections'][0].pop('detector_class'),
        ]
        for index, mutate in enumerate(mutations):
            with self.subTest(case=index):
                self.rows = deepcopy(original)
                mutate()
                self.write_saved()
                with self.assertRaises(ValueError):
                    self.replay()
                self.assertFalse(self.output.exists())

    def test_tracker_cannot_change_a_detector_box(self):
        class MutatingTracker:
            def update(self, detections, *args, **kwargs):
                detections[0]['xyxy'][0] += 1
                return detections
        with patch('feedwatch.tracking_replay.create_tracker', return_value=MutatingTracker()):
            with self.assertRaisesRegex(ValueError, 'Tracker changed'):
                self.replay()
        self.assertFalse((self.output / 'annotated.mp4').exists())

    def test_cached_food_display_must_agree_with_unchanged_windows(self):
        self.rows[0]['food_class'] = 'strong'
        self.write_saved()
        with self.assertRaisesRegex(ValueError, 'feeding display'):
            self.replay()
        self.assertFalse((self.output / 'annotated.mp4').exists())

    def test_geometry_is_reconstructed_from_a_preset_without_loading_a_network(self):
        with patch('torchvision.models.mobilenet_v3_small', side_effect=AssertionError('No network expected')):
            predictor = _geometry_predictor(self.run)
            self.assertEqual(predictor.extraction['crop_xyxy'], [.05, .1, .95, .95])
        self.run['food_model']['extractor_identity']['weights']['center_crop'] = [128]
        with self.assertRaisesRegex(ValueError, 'geometry'):
            _geometry_predictor(self.run)


if __name__ == '__main__':
    unittest.main()
