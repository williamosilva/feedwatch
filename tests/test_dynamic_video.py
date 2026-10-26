import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import torch

from feedwatch.cli import parser
from feedwatch.data import file_hash
from feedwatch.dynamic_overlay import DynamicOverlay
from feedwatch.dynamic_demo import write_page
from feedwatch.dynamic_tracking import ObservationTracker
from feedwatch.dynamic_video import export_dynamic
from feedwatch.fish_detector import validate_fish_classes
from feedwatch.image_motion import ImageMotion
from feedwatch.video_inference import display_state, plan_windows
from test_video_mvp import FakePredictor, make_video


CONFIG = json.loads(Path("config/dynamic.json").read_text(encoding="utf-8"))


def detection(x=100., y=100.):
    return {"xyxy":[x,y,x+40,y+20],"score":.7,"detector_class_id":0,"detector_class":"Fish"}


class Predictor(FakePredictor):
    device = torch.device("cpu")


class Detector:
    identification = {"technical_fixture":True}

    def detect(self, frame):
        return [detection()]


class DynamicVideoTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        cv2.setNumThreads(1)

    def test_only_verified_fish_class_is_accepted(self):
        validate_fish_classes({0:"Fish"},{"0":"Fish"})
        for names in ({0:"person"},{0:"fish",1:"person"},{0:"boat"}):
            with self.assertRaises(ValueError):
                validate_fish_classes(names,{"0":"Fish"})

    def test_tracks_follow_observations_speed_in_source_pixels_and_reset(self):
        tracker = ObservationTracker(CONFIG["tracker"])
        first = tracker.update([detection(),detection(500)],0.,0)
        self.assertIsNone(first[0]["speed_source_px_per_second"])
        # Reverse detector ordering: association, not output row, determines ID.
        second = tracker.update([detection(504),detection(104)],.04,1)
        self.assertEqual([obs["track_id"] for obs in second],[2,1])
        self.assertAlmostEqual(second[0]["speed_source_px_per_second"],100.)
        self.assertEqual(len(second[0]["trail_txy"]),2)
        self.assertEqual(tracker.update([],.08,2),[])  # no interpolated boxes
        recovered = tracker.update([detection(108)],.12,3)[0]
        self.assertEqual(recovered["track_id"],1)
        self.assertIsNone(recovered["speed_source_px_per_second"])
        self.assertEqual(len(recovered["trail_txy"]),1)  # no trail bridging a miss
        expired = tracker.update([detection(108)],.5,12)[0]
        self.assertGreater(expired["track_id"],2)
        tracker.reset()
        self.assertEqual(tracker.update([detection()],0.,0)[0]["track_id"],1)

    def test_bad_tracking_timestamp_is_failure_not_zero_speed(self):
        tracker = ObservationTracker(CONFIG["tracker"])
        tracker.update([detection()],0.,0)
        with self.assertRaises(ValueError):
            tracker.update([detection()],0.,1)

    def test_image_motion_is_measured_without_detecting_objects(self):
        cfg = {**CONFIG["motion"],"analysis_long_edge":160}
        rng = np.random.default_rng(42)
        image = rng.integers(0,256,(90,160,3),dtype=np.uint8)
        moving = cv2.warpAffine(image,np.float32([[1,0,2],[0,1,0]]),(160,90))
        motion = ImageMotion(cfg)
        initial,_ = motion.update(image,0.)
        self.assertEqual(initial["status"],"initializing")
        self.assertIsNone(initial["mean_source_px_per_second"])
        stats,flow = motion.update(moving,.04)
        # Translation 2 source pixels / .04 s, well inside 40-60 px/s.
        self.assertTrue(40 < np.median(flow[10:-10,10:-10,0]) < 60)
        self.assertTrue(stats["moving_fraction"]>.8)
        self.assertNotIn("fish",stats)
        with self.assertRaises(ValueError):
            motion.update(moving,.04)

    def test_visual_failure_remains_independent_of_food_prediction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/"technical_fixture.mp4"
            make_video(source,50)
            before = file_hash(source)
            detector = Detector()
            with patch.object(detector,"detect",side_effect=RuntimeError("deliberate failure")):
                report = export_dynamic(source,root/"out",Predictor(),CONFIG,detector,"clip")
            self.assertEqual(report["status"],"completed_with_unavailable_intervals")
            self.assertEqual(report["windows"][0]["predicted_class"],"weak")
            rows = [json.loads(line) for line in (root/"out"/"frames.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertTrue(all(row["detector_status"]=="unavailable" and row["detected_count"] is None for row in rows))
            self.assertTrue(all(row["detections"]==[] for row in rows))
            self.assertEqual(file_hash(source),before)
            self.assertEqual(report["export_verification"]["decoded_frames"],50)

    def test_sequential_dynamic_export_does_not_anticipate_food_or_pad_tail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/"technical_fixture.mp4"
            make_video(source,130)
            predictor = Predictor()
            report = export_dynamic(source,root/"out",predictor,CONFIG,None,"sequential")
            self.assertEqual(predictor.calls,2)
            self.assertEqual(report["windows"][-1]["status"],"incomplete")
            rows = [json.loads(line) for line in (root/"out"/"frames.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertTrue(all(r["food_class"] is None for r in rows[:50]))
            self.assertTrue(all(r["food_status"]=="analyzing" for r in rows[:50]))
            self.assertEqual(rows[50]["food_window_id"],0)
            self.assertEqual(rows[100]["food_window_id"],1)
            self.assertTrue(all(r["detected_count"] is None for r in rows))

    def test_render_does_not_mutate_frame_or_food_inputs(self):
        predictor = Predictor()
        renderer = DynamicOverlay(640,360,predictor,CONFIG)
        image = np.full((360,640,3),70,np.uint8)
        before = image.copy()
        windows = plan_windows({"fps":25.,"frame_count":50,"duration_seconds":2.},"sequential")
        state = display_state(windows,"sequential",0.)
        result = renderer.render(image,state,"sequential",[],"disabled",{"status":"initializing"},None,0.)
        np.testing.assert_array_equal(image,before)
        self.assertEqual(result.shape,image.shape)
        self.assertEqual(windows[0]["status"],"pending")

    def test_invalid_video_is_unavailable_and_never_none(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bad = root/"technical_invalid.mp4"
            bad.write_bytes(b"invalid technical fixture")
            report = export_dynamic(bad,root/"out",Predictor(),CONFIG,None)
            self.assertEqual(report["status"],"unavailable")
            self.assertIsNone(report["output_video"])
            self.assertEqual(report["rendered_frames"],0)

    def test_new_cli_keeps_original_commands_and_has_explicit_motion_fallback(self):
        args = parser().parse_args(["annotate-dynamic","--video","real.mp4","--output-dir","out","--motion-only"])
        self.assertTrue(args.motion_only)
        self.assertEqual(args.mode,"sequential")
        self.assertEqual(parser().parse_args(["annotate-video","--video","real.mp4","--output-dir","out"]).command,"annotate-video")

    def test_motion_only_gallery_does_not_claim_a_detector_was_used(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_page(root,[],False)
            text = (root/"index.html").read_text(encoding="utf-8")
            self.assertIn("Detector desligado explicitamente",text)
            self.assertNotIn("Fishial pré-treinado propõe caixas",text)


if __name__=="__main__":
    unittest.main()
