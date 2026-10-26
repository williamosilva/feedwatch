import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pandas as pd
from torchvision.models import MobileNet_V3_Small_Weights

from feedwatch.data import CLASS_IDS, file_hash, parse_relative_path
from feedwatch.video_demo import select_validation
from feedwatch.video_export import export_video, verify_export
from feedwatch.video_inference import analyze_video, display_state, open_capture, plan_windows
from feedwatch.visual import sample_visual_indices


class FakePredictor:
    """Cabeça de teste determinística; não treina nem representa resultados do dataset."""
    extraction = {"crop_xyxy": [.05, .1, .95, .95]}
    transform = MobileNet_V3_Small_Weights.IMAGENET1K_V1.transforms()
    identification = {"test_fixture_only": True}

    def __init__(self):
        self.calls = 0

    def predict_frames(self, frames):
        assert frames.shape == (16, 3, 224, 224)
        self.calls += 1
        return {"predicted_id": 1, "predicted_class": "weak", "legend_pt": "Fraca",
                "scores": {"none": .1, "weak": .7, "medium": .1, "strong": .1}}

    def verify_frozen(self):
        pass


def make_video(path, frames=130, width=640, height=360):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25., (width, height))
    if not writer.isOpened():
        raise RuntimeError("Encoder de fixture indisponível")
    for index in range(frames):
        image = np.full((height, width, 3), (36, 65, 78), np.uint8)
        cv2.rectangle(image, (index % (width-50), height//2), (index % (width-50)+40, height//2+40), (150, 170, 210), -1)
        writer.write(image)
    writer.release()


class VideoMVPTest(unittest.TestCase):
    def setUp(self):
        import torch
        torch.set_num_threads(1)

    def test_temporal_plan_matches_original_and_does_not_pad_tail(self):
        windows = plan_windows({"fps": 25., "frame_count": 130, "duration_seconds": 5.2}, "sequential")
        original = sample_visual_indices(25., 50, 16, 2.)
        self.assertEqual(windows[0]["sampled_indices"], original)
        self.assertEqual(windows[1]["sampled_indices"], [50+i for i in original])
        self.assertEqual(windows[2]["status"], "incomplete")
        self.assertEqual(windows[2]["sampled_indices"], [])
        self.assertIsNone(windows[2]["predicted_class"])
        fractional = plan_windows({"fps": 30000/1001, "frame_count": 150, "duration_seconds": 150/(30000/1001)}, "sequential")
        self.assertEqual(fractional[0]["frame_stop_exclusive"], fractional[1]["frame_start"])
        self.assertTrue(all(len(w["sampled_indices"]) == 16 for w in fractional[:2]))

    def test_prediction_never_appears_before_window_end(self):
        windows = plan_windows({"fps": 25., "frame_count": 130, "duration_seconds": 5.2}, "sequential")
        for window in windows[:2]:
            window.update(status="ok", predicted_class="strong", legend_pt="Forte")
        self.assertEqual(display_state(windows, "sequential", 1.96)["status"], "analyzing")
        self.assertEqual(display_state(windows, "sequential", 2.)["window"]["window_id"], 0)
        self.assertEqual(display_state(windows, "sequential", 3.96)["window"]["window_id"], 0)
        tail = display_state(windows, "sequential", 4.)
        self.assertEqual(tail["window"]["window_id"], 1)
        self.assertTrue(tail["tail"])
        windows[1].update(status="failed", predicted_class=None)
        failed = display_state(windows, "sequential", 4.)
        self.assertEqual(failed["status"], "unavailable")
        self.assertEqual(failed["last_valid"]["window_id"], 0)

    def test_clip_contract_rejects_too_few_frames_or_long_clip(self):
        too_short = plan_windows({"fps": 25., "frame_count": 15, "duration_seconds": .6}, "clip")[0]
        too_long = plan_windows({"fps": 25., "frame_count": 51, "duration_seconds": 2.04}, "clip")[0]
        self.assertEqual(too_short["status"], "failed")
        self.assertEqual(too_long["status"], "failed")
        self.assertIsNone(too_short["scores"])

    def test_export_preserves_video_and_records_causal_display(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "source.mp4"
            make_video(source)
            before = file_hash(source)
            predictor = FakePredictor()
            report = export_video(source, root / "out", predictor)
            self.assertEqual(predictor.calls, 2)
            self.assertEqual(report["status"], "completed_with_unavailable_intervals")
            self.assertEqual(report["export_verification"]["decoded_frames"], 130)
            self.assertEqual(file_hash(source), before)
            self.assertEqual([(r["first_frame"], r["status"]) for r in report["display_events"]],
                             [(0, "analyzing"), (50, "prediction"), (100, "prediction")])
            self.assertEqual(report["display_events"][-1]["window_start_seconds"], 2.)
            self.assertTrue(report["display_events"][-1]["tail_incomplete"])
            with self.assertRaises(ValueError):
                verify_export(Path(report["output_video"]), {**report["metadata"], "frame_count": 131})
            with self.assertRaises(ValueError):
                export_video(source, root / "out", predictor)

    def test_read_failure_has_visual_diagnostic_and_never_none_prediction(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "bad.mp4"
            source.write_bytes(b"deliberately invalid")
            predictor = FakePredictor()
            report = export_video(source, root / "out", predictor, "clip")
            self.assertEqual(report["status"], "unavailable")
            self.assertIsNone(report["output_video"])
            self.assertEqual(predictor.calls, 0)
            self.assertTrue((root / "out" / "analysis_unavailable.png").exists())
            self.assertNotIn('"predicted_class": "none"', (root / "out" / "analysis.json").read_text(encoding="utf-8"))

    def test_offline_clip_displays_result_at_start_and_full_decode_is_checked(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "short.mp4"
            make_video(source, 50)
            predictor = FakePredictor()
            report = analyze_video(source, predictor, "clip")
            self.assertEqual(predictor.calls, 1)
            self.assertEqual(report["decoded_frames"], 50)
            self.assertEqual(display_state(report["windows"], "clip", 0)["status"], "prediction")

    def test_validation_selection_is_reproducible_and_does_not_need_predictions(self):
        rows = pd.DataFrame([parse_relative_path(f"2022_6_21/AM_5/{name}/{i}.mp4") for name in CLASS_IDS for i in range(10)])
        a, b = select_validation(rows, 42), select_validation(rows.iloc[::-1], 42)
        self.assertEqual(a.relative_path.tolist(), b.relative_path.tolist())
        self.assertEqual(a.class_name.value_counts().to_dict(), dict.fromkeys(CLASS_IDS, 3))
        wrong = rows.copy()
        wrong["relative_path"] = wrong.relative_path.str.replace("2022_6_21", "2022_6_22")
        wrong["date"] = "2022_6_22"
        with self.assertRaises(ValueError):
            select_validation(wrong, 42)

    def test_incompatible_timestamps_are_unavailable_without_inference(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "timestamps.mp4"
            make_video(path, 50)
            real = open_capture(path)
            class ShiftedCapture:
                def get(self, key):
                    value = real.get(key)
                    return value+500 if key == cv2.CAP_PROP_POS_MSEC else value
                def __getattr__(self, name):
                    return getattr(real, name)
            predictor = FakePredictor()
            with patch("feedwatch.video_inference.open_capture", return_value=ShiftedCapture()):
                report = analyze_video(path, predictor, "clip")
            self.assertEqual(predictor.calls, 0)
            self.assertEqual(report["windows"][0]["status"], "failed")
            self.assertIsNone(report["windows"][0]["predicted_class"])
            self.assertIsNone(report["windows"][0]["scores"])

    def test_cli_outputs_utf8_even_when_windows_pipe_defaults_to_cp1252(self):
        result = subprocess.run([sys.executable, "-m", "feedwatch", "annotate-video", "--help"],
                                 capture_output=True, env={**os.environ, "PYTHONIOENCODING": "cp1252"})
        self.assertEqual(result.returncode, 0)
        text = result.stdout.decode("utf-8")
        self.assertIn("análise offline", text)


if __name__ == "__main__":
    unittest.main()
