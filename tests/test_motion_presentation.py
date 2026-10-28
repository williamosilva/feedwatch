import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from feedwatch.image_motion import ImageMotion
from feedwatch.motion_presentation import NEUTRAL_BGR, export_motion_presentation, motion_heatmap


CONFIG = json.loads(Path("config/simple_video.json").read_text(encoding="utf-8"))["motion"]


class MotionPresentationTest(unittest.TestCase):
    def setUp(self):
        cv2.setNumThreads(1)

    def test_color_meaning_uses_one_fixed_scale_across_frames(self):
        first = np.zeros((8, 8, 2), dtype=np.float32)
        first[:, :, 0] = 100
        second = first.copy()
        second[0, 0, 0] = 400
        a = motion_heatmap(first, 400, (8, 8))
        b = motion_heatmap(second, 400, (8, 8))
        np.testing.assert_array_equal(a[4, 4], b[4, 4])
        expected = cv2.applyColorMap(np.full((1, 1), 63, dtype=np.uint8), cv2.COLORMAP_TURBO)[0, 0]
        np.testing.assert_array_equal(a[4, 4], expected)
        saturated = np.full((8, 8, 2), 500, dtype=np.float32)
        np.testing.assert_array_equal(motion_heatmap(saturated, 400, (8, 8))[0, 0],
                                      cv2.applyColorMap(np.full((1, 1), 255, np.uint8), cv2.COLORMAP_TURBO)[0, 0])

    def test_initialization_is_missing_and_static_zero_differs_from_missing(self):
        image = np.random.default_rng(4).integers(0, 256, (80, 120, 3), dtype=np.uint8)
        motion = ImageMotion(CONFIG)
        first, flow = motion.update(image, 0.)
        self.assertEqual(first["status"], "initializing")
        self.assertIsNone(first["mean_source_px_per_second"])
        neutral = motion_heatmap(flow, 400, (120, 80))
        np.testing.assert_array_equal(neutral[0, 0], NEUTRAL_BGR)
        static, flow = motion.update(image, .1)
        self.assertEqual(static["status"], "ok")
        self.assertLess(static["mean_source_px_per_second"], .1)
        self.assertFalse(np.array_equal(motion_heatmap(flow, 400, (120, 80))[40, 60], neutral[40, 60]))

    def test_apparent_displacement_uses_source_pixels_and_elapsed_seconds(self):
        rng = np.random.default_rng(6)
        base = rng.integers(0, 256, (160, 240, 3), dtype=np.uint8)
        shifted = cv2.warpAffine(base, np.float32([[1, 0, 3], [0, 1, 0]]), (240, 160),
                                 borderMode=cv2.BORDER_REFLECT)
        motion = ImageMotion({**CONFIG, "analysis_long_edge": 240})
        motion.update(base, 0.)
        metrics, flow = motion.update(shifted, .1)
        self.assertEqual(metrics["status"], "ok")
        self.assertAlmostEqual(float(np.median(flow[20:-20, 20:-20, 0])), 30., delta=2.)
        self.assertGreater(metrics["moving_fraction"], .7)

    def test_video_export_preserves_frames_clock_and_never_invents_initial_values(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "original.avi"
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 10., (96, 64))
            self.assertTrue(writer.isOpened())
            frame = np.random.default_rng(7).integers(0, 256, (64, 96, 3), dtype=np.uint8)
            for x in range(5):
                writer.write(np.roll(frame, x, axis=1))
            writer.release()
            output = Path(directory) / "presentation"
            result = export_motion_presentation(source, output, CONFIG)
            self.assertEqual(result["frame_count"], 5)
            self.assertEqual(result["duration_seconds"], .5)
            self.assertEqual([r["timestamp_seconds"] for r in result["frames"]], [0., .1, .2, .3, .4])
            self.assertEqual(result["frames"][0]["status"], "initializing")
            self.assertIsNone(result["frames"][0]["moving_fraction"])
            self.assertTrue(all(r["status"] == "ok" for r in result["frames"][1:]))
            self.assertEqual(json.loads((output / "motion.json").read_text(encoding="utf-8")), result)
            audit = json.loads((output / "motion-audit.json").read_text(encoding="utf-8"))
            self.assertEqual(audit["export_verification"]["decoded_frames"], 5)
            self.assertFalse(audit["feeding_or_detector_inference_run"])
            with self.assertRaisesRegex(ValueError, "already exists"):
                export_motion_presentation(source, output, CONFIG)

    def test_invalid_scale_or_field_cannot_be_drawn_as_real_motion(self):
        for maximum in (0, -1, float("nan")):
            with self.assertRaises(ValueError):
                motion_heatmap(None, maximum, (20, 20))
        with self.assertRaises(ValueError):
            motion_heatmap(np.full((3, 3, 2), np.nan), 400, (20, 20))


if __name__ == "__main__":
    unittest.main()
