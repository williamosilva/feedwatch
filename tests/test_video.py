import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from feedwatch.data import parse_relative_path
from feedwatch.video import inspect_video


class VideoTest(unittest.TestCase):
    def test_different_metadata_and_distributed_decode(self):
        with tempfile.TemporaryDirectory(prefix="feedwatch video space ") as directory:
            root = Path(directory).resolve()
            for label, size, fps, count in [("none", (64, 48), 6, 12), ("strong", (96, 64), 8, 24)]:
                relative = f"2022_6_19/AM_5/{label}/test.mp4"
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
                self.assertTrue(writer.isOpened(), "Codec mp4v indisponível para fixture sintética")
                try:
                    for index in range(count):
                        writer.write(np.full((size[1], size[0], 3), index * 5, dtype=np.uint8))
                finally:
                    writer.release()
                result = inspect_video(root, parse_relative_path(relative), 5)
                self.assertTrue(result["ok"], result["issues"])
                self.assertEqual((result["width"], result["height"]), size)
                self.assertEqual(result["frame_count"], count)
                self.assertAlmostEqual(result["fps"], fps)
                self.assertAlmostEqual(result["duration_seconds"], count / fps)
                self.assertEqual(result["frames_decoded"], 5)

    def test_corrupt_and_missing_video_are_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            relative = "2022_6_19/AM_5/none/bad.mp4"
            path = root / relative
            path.parent.mkdir(parents=True)
            for exists in [False, True]:
                if exists:
                    path.write_bytes(b"not a video")
                result = inspect_video(root, parse_relative_path(relative), 5)
                self.assertFalse(result["ok"])
                self.assertTrue(result["issues"])
                self.assertEqual(result["frames_decoded"], 0)


if __name__ == "__main__":
    unittest.main()
