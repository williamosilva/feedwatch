import copy
import json
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pandas as pd

from feedwatch.data import CLASS_IDS, COLUMNS, file_hash, make_splits, parse_relative_path
from feedwatch.experiment import feature_matrix, load_train_val, run_experiment
from feedwatch.motion import (FEATURE_NAMES, aggregate_pairs, cached_extract, extract_video,
                              extraction_identity, identity_hash, pair_features, sample_indices)

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "motion.json"


def config():
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def synthetic_video(path, shift=0):
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (96, 64))
    if not writer.isOpened():
        raise RuntimeError("Fixture mp4v indisponível")
    try:
        for index in range(10):
            image = np.zeros((64, 96, 3), dtype=np.uint8)
            cv2.rectangle(image, (10 + index * shift, 20), (25 + index * shift, 35), (220, 220, 220), -1)
            writer.write(image)
    finally:
        writer.release()


class MotionTest(unittest.TestCase):
    def test_same_time_grid_at_different_fps(self):
        cfg = config()["extraction"]
        a, b = sample_indices(25, 50, cfg), sample_indices(30, 60, cfg)
        np.testing.assert_allclose(np.asarray(a) / 25, np.asarray(b) / 30)
        self.assertEqual(a, list(range(0, 50, 5)))
        with self.assertRaises(ValueError):
            sample_indices(2, 50, cfg)
        with self.assertRaises(ValueError):
            sample_indices(25, 2, cfg)

    def test_motion_signal_and_valid_stationary_example(self):
        cfg = config()["extraction"]
        first = np.zeros((64, 96), dtype=np.uint8)
        first[20:40, 20:40] = 255
        second = np.roll(first, 5, axis=1)
        static = aggregate_pairs([pair_features(first, first, .2, cfg)] * 2)
        moving = aggregate_pairs([pair_features(first, second, .2, cfg)] * 2)
        self.assertEqual(len(static), 42)
        self.assertEqual(static["diff_mean_temporal_mean"], 0)
        self.assertGreater(moving["diff_mean_temporal_mean"], static["diff_mean_temporal_mean"])
        self.assertGreater(moving["flow_mean_temporal_mean"], static["flow_mean_temporal_mean"])

    def test_corrupt_file_not_zero_features(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            relative = "2022_6_19/AM_5/none/same.mp4"
            path = root / relative
            path.parent.mkdir(parents=True)
            path.write_bytes(b"invalid video")
            result = extract_video(root, relative, config()["extraction"])
            self.assertFalse(result["ok"])
            self.assertIsNone(result["features"])
            self.assertTrue(result["error"])

    def test_failure_after_partial_decode_excludes_entire_file(self):
        cfg = config()["extraction"]
        class PartialCapture:
            def __init__(self):
                self.index = -1
            def isOpened(self): return True
            def getBackendName(self): return "mock"
            def get(self, key):
                return {cv2.CAP_PROP_FPS: 25, cv2.CAP_PROP_FRAME_COUNT: 50,
                        cv2.CAP_PROP_FRAME_WIDTH: 96, cv2.CAP_PROP_FRAME_HEIGHT: 64,
                        cv2.CAP_PROP_POS_FRAMES: self.index + 1,
                        cv2.CAP_PROP_POS_MSEC: self.index * 40}.get(key, 0)
            def grab(self):
                self.index += 1
                return self.index < 12
            def retrieve(self): return True, np.zeros((64, 96, 3), dtype=np.uint8)
            def release(self): pass
        with patch("feedwatch.motion.cv2.VideoCapture", return_value=PartialCapture()):
            result = extract_video(Path(tempfile.gettempdir()).resolve(), "2022_6_19/AM_5/none/same.mp4", cfg)
        self.assertFalse(result["ok"])
        self.assertIsNone(result["features"])
        self.assertEqual(result["details"]["frames_decoded"], 3)

    def test_cache_reuse_config_source_and_corruption(self):
        with tempfile.TemporaryDirectory(prefix="motion space ") as directory:
            root, cache = Path(directory).resolve() / "videos", Path(directory) / "cache"
            relative = "2022_6_19/AM_5/none/same.mp4"
            synthetic_video(root / relative, shift=1)
            cfg = config()["extraction"]
            identity = extraction_identity(cfg)
            key = identity_hash(identity)
            first = cached_extract(root, relative, cfg, cache, key)
            self.assertTrue(first["ok"], first["error"])
            with patch("feedwatch.motion.extract_video", side_effect=AssertionError("Cache deveria evitar decodificação")):
                second = cached_extract(root, relative, cfg, cache, key)
            self.assertTrue(second["cache_hit"])
            self.assertEqual(first["features"], second["features"])
            entry = next((cache / key).glob("*.json"))
            entry.write_text("not json", encoding="utf-8")
            third = cached_extract(root, relative, cfg, cache, key)
            self.assertFalse(third["cache_hit"])
            self.assertTrue(third["cache_warning"])
            cfg2 = copy.deepcopy(cfg)
            cfg2["sample_hz"] = 4.0
            other = cached_extract(root, relative, cfg2, cache, identity_hash(extraction_identity(cfg2)))
            self.assertFalse(other["cache_hit"])
            (root / relative).write_bytes(b"changed to corrupt")
            changed = cached_extract(root, relative, cfg, cache, key)
            self.assertFalse(changed["cache_hit"])
            self.assertFalse(changed["ok"])
            self.assertIsNone(changed["features"])

    def test_only_image_feature_allowlist(self):
        a = pd.DataFrame([{**{name: index for index, name in enumerate(FEATURE_NAMES)},
                           "date": "2022_6_19", "group": "AM_100", "class_id": 3, "relative_path": "x"}])
        b = a.assign(date="OTHER", group="OTHER", class_id=0, relative_path="OTHER")
        pd.testing.assert_frame_equal(feature_matrix(a), feature_matrix(b))
        self.assertEqual(feature_matrix(a).shape[1], 42)
        a.loc[0, FEATURE_NAMES[0]] = np.nan
        with self.assertRaises(ValueError):
            feature_matrix(a)


class ExperimentTest(unittest.TestCase):
    def test_end_to_end_excludes_failures_uses_cache_and_never_opens_test(self):
        with tempfile.TemporaryDirectory(prefix="experiment space ") as directory:
            base = Path(directory)
            root = (base / "videos").resolve()
            artifact = base / "artifacts"
            artifact.mkdir()
            records = [parse_relative_path(f"2022_6_{day}/AM_5/{label}/same.mp4")
                       for day in range(19, 23) for label in CLASS_IDS]
            manifest = pd.DataFrame(records, columns=COLUMNS)
            manifest.to_csv(artifact / "manifest.csv", index=False)
            assignment = {"train": ["2022_6_19", "2022_6_20"], "val": ["2022_6_21"], "test": ["2022_6_22"]}
            split_config = base / "splits.json"
            split_config.write_text(json.dumps(assignment), encoding="utf-8")
            make_splits(artifact / "manifest.csv", split_config, artifact / "splits")
            # Não existe test.csv nem vídeo de teste: o experimento precisa funcionar assim.
            (artifact / "splits" / "test.csv").unlink()
            for row in records:
                if row["date"] != "2022_6_22":
                    synthetic_video(root / row["relative_path"], row["class_id"])
            bad = root / "2022_6_19/AM_5/none/same.mp4"
            bad.write_bytes(b"corrupt")
            cfg = config()
            cfg["extraction"]["resize_width"] = 64
            cfg["extraction"]["resize_height"] = 48
            cfg["random_forest"].update(n_estimators=5, max_depth=3, min_samples_leaf=1)
            cfg_path = base / "motion.json"
            cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
            parts, provenance = load_train_val(artifact / "manifest.csv", artifact / "splits")
            self.assertEqual(len(parts["train"]), 8)
            self.assertFalse(provenance["test_csv_opened"])
            hashes = {p: file_hash(p) for p in [artifact / "manifest.csv", artifact / "splits" / "train.csv", artifact / "splits" / "val.csv"]}
            result = run_experiment(root, artifact / "manifest.csv", artifact / "splits", cfg_path,
                                    artifact / "cache", artifact / "run", 2, None)
            self.assertEqual(result["counts"]["train"]["used"], 7)
            self.assertEqual(result["counts"]["val"]["used"], 4)
            self.assertEqual(result["extraction"]["failed"], 1)
            self.assertFalse(result["test_evaluated"])
            features = pd.read_csv(artifact / "run" / "features.csv")
            self.assertNotIn("2022_6_22", features.date.tolist())
            self.assertNotIn("2022_6_19/AM_5/none/same.mp4", features.relative_path.tolist())
            with (artifact / "run" / "random_forest" / "model.pkl").open("rb") as handle:
                model = pickle.load(handle)
            stored_predictions = pd.read_csv(artifact / "run" / "random_forest" / "predictions.csv")
            np.testing.assert_array_equal(model["estimator"].predict(feature_matrix(features[features.split == "val"])), stored_predictions.predicted_id)
            for path, digest in hashes.items():
                self.assertEqual(file_hash(path), digest)
            second = run_experiment(root, artifact / "manifest.csv", artifact / "splits", cfg_path,
                                    artifact / "cache", artifact / "rerun", 2, None)
            self.assertEqual(second["extraction"]["cache_hits"], 11)
            self.assertEqual(result["models"]["random_forest"]["confusion_matrix"], second["models"]["random_forest"]["confusion_matrix"])


if __name__ == "__main__":
    unittest.main()
