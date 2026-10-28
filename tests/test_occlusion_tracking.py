import copy
import unittest

import cv2
import numpy as np

from feedwatch.occlusion_tracking import (
    OcclusionAwareTracker, appearance_descriptor, appearance_distance,
)


def detection(x, y=40, width=40, height=20, score=.7):
    return {'xyxy': [float(x), float(y), float(x + width), float(y + height)],
            'score': score, 'detector_class_id': 0, 'detector_class': 'Fish'}


def scene(*fish):
    image = np.zeros((120, 300, 3), np.uint8)
    for box, color in fish:
        x1, y1, x2, y2 = np.rint(box['xyxy']).astype(int)
        cv2.rectangle(image, (x1, y1), (x2 - 1, y2 - 1), color, -1)
    return image


class OcclusionTrackingTests(unittest.TestCase):
    def test_distinct_fish_keep_ids_through_occlusion_and_reversed_detection_order(self):
        tracker = OcclusionAwareTracker({})
        first_ids = None
        for index in range(15):
            a = detection(20 + index * 10)
            b = detection(180 - index * 10)
            image = scene((a, (0, 220, 240)), (b, (240, 50, 0)))
            # At the complete overlap only the foreground fish is detected.
            inputs = [b] if index == 8 else [a, b] if index < 8 else [b, a]
            outputs = tracker.update(inputs, index / 10, index, image)
            self.assertEqual(len(outputs), len(inputs))
            for before, after in zip(inputs, outputs):
                self.assertEqual(before['xyxy'], after['xyxy'])
                self.assertEqual(before['score'], after['score'])
                self.assertFalse(after['position_predicted_only'])
            if index == 0:
                first_ids = [row['track_id'] for row in outputs]
            if index == 8:
                self.assertIn(first_ids[0], tracker.suspended_ids)
            if index >= 11:
                self.assertEqual([row['track_id'] for row in outputs], first_ids[::-1])

    def test_recovers_short_gap_without_inventing_observations_or_bridging_trail(self):
        tracker = OcclusionAwareTracker({})
        rows = []
        for index in range(8):
            box = detection(20 + index)
            rows.append(tracker.update([box], index / 30, index, scene((box, (0, 220, 240))))[0])
        old_id = rows[-1]['track_id']
        self.assertIsNotNone(rows[-1]['speed_source_px_per_second'])
        self.assertGreater(len(rows[-1]['trail_txy']), 2)
        for index in range(8, 14):
            self.assertEqual(tracker.update([], index / 30, index, scene()), [])
            self.assertIn(old_id, tracker.suspended_ids)
        box = detection(34)
        recovered = tracker.update([box], 14 / 30, 14, scene((box, (0, 220, 240))))[0]
        self.assertEqual(recovered['track_id'], old_id)
        self.assertTrue(recovered['recovered_after_gap'])
        self.assertEqual(recovered['consecutive_observations'], 1)
        self.assertEqual(recovered['trail_txy'], [])
        self.assertIsNone(recovered['speed_source_px_per_second'])
        self.assertEqual(recovered['center_xy'], [54., 50.])

    def test_expired_identity_is_not_recycled(self):
        tracker = OcclusionAwareTracker({})
        box = detection(20)
        first = tracker.update([box], 0., 0, scene((box, (0, 220, 240))))[0]
        self.assertEqual(tracker.update([], 1.01, 31, scene()), [])
        self.assertEqual(tracker.suspended_ids, [])
        returned = tracker.update([box], 1.1, 33, scene((box, (0, 220, 240))))[0]
        self.assertNotEqual(first['track_id'], returned['track_id'])
        self.assertFalse(returned['recovered_after_gap'])

    def test_indistinguishable_merged_detection_suspends_both_old_identities(self):
        tracker = OcclusionAwareTracker({})
        a, b = detection(20), detection(100)
        for index in range(3):
            initial = tracker.update([a, b], index / 10, index,
                                     scene((a, (0, 200, 230)), (b, (0, 200, 230))))
        ids = [row['track_id'] for row in initial]
        merged = detection(20, width=120)
        output = tracker.update([merged], .3, 3, scene((merged, (0, 200, 230))))[0]
        self.assertNotIn(output['track_id'], ids)
        self.assertEqual(output['tracking_status'], 'ambiguous')
        self.assertEqual(tracker.suspended_ids, ids)
        self.assertFalse(output['appearance_updated'])
        self.assertEqual(output['trail_txy'], [])
        self.assertIsNone(output['speed_source_px_per_second'])
        # An overlap-only temporary identity never gains clean-track priority.
        for index in range(4, 7):
            output = tracker.update([merged], index / 10, index,
                                    scene((merged, (0, 200, 230))))[0]
        self.assertEqual(tracker.tracks[output['track_id']]['clean_observations'], 0)
        returned = tracker.update([b, a], .7, 7,
                                  scene((a, (0, 200, 230)), (b, (0, 200, 230))))
        self.assertEqual([row['track_id'] for row in returned], ids[::-1])

    def test_overlap_does_not_update_appearance_or_show_motion_across_it(self):
        tracker = OcclusionAwareTracker({})
        a, b = detection(20), detection(120)
        for index in range(8):
            outputs = tracker.update([a, b], index / 30, index,
                                     scene((a, (0, 200, 230)), (b, (230, 50, 0))))
        original = {row['track_id']: tracker.tracks[row['track_id']]['appearance'].copy() for row in outputs}
        # A second box partially covers A's box but does not move A itself.
        b_overlap = detection(40)
        outputs = tracker.update([a, b_overlap], 8 / 30, 8,
                                 scene((a, (0, 200, 230)), (b_overlap, (230, 50, 0))))
        for row in outputs:
            self.assertTrue(row['occlusion_observed'])
            self.assertFalse(row['appearance_updated'])
            self.assertEqual(row['trail_txy'], [])
            self.assertIsNone(row['speed_source_px_per_second'])
            if row['track_id'] in original:
                np.testing.assert_array_equal(tracker.tracks[row['track_id']]['appearance'], original[row['track_id']])

    def test_reset_restarts_clip_but_error_clear_preserves_unique_counter(self):
        tracker = OcclusionAwareTracker({})
        box = detection(20)
        first = tracker.update([box], 0., 0)[0]
        tracker.clear_history()
        second = tracker.update([box], 0., 0)[0]
        self.assertGreater(second['track_id'], first['track_id'])
        tracker.reset()
        restarted = tracker.update([box], 0., 0)[0]
        self.assertEqual(restarted['track_id'], first['track_id'])

    def test_invalid_clock_boxes_and_image_fail_before_mutating_history(self):
        tracker = OcclusionAwareTracker({})
        box = detection(20)
        tracker.update([box], 0., 0)
        before = copy.deepcopy(tracker.tracks)
        for timestamp, frame in [(0., 1), (.1, 0), (float('nan'), 1), (-1., 1)]:
            with self.subTest(timestamp=timestamp, frame=frame), self.assertRaises(ValueError):
                tracker.update([box], timestamp, frame)
        for invalid in [[0, 0, 0, 10], [1, 2, float('nan'), 20], [1, 2, 3]]:
            with self.subTest(box=invalid), self.assertRaises(ValueError):
                tracker.update([{'xyxy': invalid}], .1, 1)
        with self.assertRaises(ValueError):
            tracker.update([box], .1, 1, np.zeros((10, 10), np.uint8))
        self.assertEqual(tracker.last_timestamp, 0.)
        self.assertEqual(tracker.next_id, 2)
        np.testing.assert_array_equal(tracker.tracks[1]['box'], before[1]['box'])

    def test_invalid_configuration_is_rejected(self):
        for key, value in [('max_gap_seconds', 0), ('prediction_uncertainty_diagonals_per_second', -1),
                           ('prediction_uncertainty_diagonals_per_second', float('nan')),
                           ('confirmation_frames', 0), ('min_speed_observations', 2.5),
                           ('speed_window_seconds', 0), ('appearance_weight', 1.01), ('max_area_ratio', .5)]:
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                OcclusionAwareTracker({key: value})

    def test_descriptor_distinguishes_colors_and_handles_clipped_tiny_boxes(self):
        box = detection(20)
        orange = appearance_descriptor(scene((box, (0, 200, 230))), box['xyxy'])
        blue = appearance_descriptor(scene((box, (230, 50, 0))), box['xyxy'])
        self.assertAlmostEqual(appearance_distance(orange, orange), 0.)
        self.assertGreater(appearance_distance(orange, blue), .7)
        self.assertIsNone(appearance_descriptor(scene(), [-20, -20, -1, -1]))
        self.assertIsNone(appearance_descriptor(scene(), [1, 1, 3, 3]))
        self.assertIsNone(appearance_descriptor(None, box['xyxy']))


if __name__ == '__main__':
    unittest.main()
