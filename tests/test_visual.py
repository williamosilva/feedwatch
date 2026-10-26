import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights

from feedwatch.data import parse_relative_path
from feedwatch.evaluation import check_metadata
from feedwatch.linear_probe import load_probe, probabilities, train_probe, train_statistics
from feedwatch.visual import (cache_signature, decode_visual, encode_frames, frozen_model, load_embedding_cache,
                              preprocess_bgr, sample_visual_indices, save_embedding_cache, state_hash)
from feedwatch.visual_experiment import compare_predictions


class VisualTest(unittest.TestCase):
    def test_sampling_window_fps_and_short_duration(self):
        a = sample_visual_indices(25, 50, 16, 2)
        b = sample_visual_indices(30, 100, 16, 2)
        self.assertEqual((a[0], a[-1], len(set(a))), (0, 49, 16))
        self.assertEqual(b[-1], 59)
        self.assertLessEqual(max(sample_visual_indices(25, 25, 16, 2)), 24)
        self.assertTrue(all(i/25 < 2 for i in a))
        with self.assertRaises(ValueError):
            sample_visual_indices(25, 10, 16, 2)

    def test_bgr_to_rgb_and_official_normalization(self):
        cfg = {"crop_xyxy": [0, 0, 1, 1]}
        image = np.zeros((100, 300, 3), dtype=np.uint8)
        image[:, :, 2] = 255
        output = preprocess_bgr(image, cfg, MobileNet_V3_Small_Weights.IMAGENET1K_V1.transforms())
        self.assertEqual(tuple(output.shape), (3, 224, 224))
        np.testing.assert_allclose(output[:, 112, 112].numpy(), [(1-.485)/.229, - .456/.224, - .406/.225], rtol=1e-6)

    def test_extractor_frozen_batchnorm_unchanged_no_gradients(self):
        torch.set_num_threads(1)
        model = frozen_model(nn.Sequential(nn.Conv2d(3, 576, 1), nn.BatchNorm2d(576),
                                           nn.AdaptiveAvgPool2d(1), nn.Flatten()), torch.device("cpu"))
        before = state_hash(model)
        result = encode_frames(model, torch.rand(2, 3, 12, 12), torch.device("cpu"))
        self.assertEqual(result.shape, (2, 576))
        self.assertEqual(state_hash(model), before)
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in model.parameters()))
        model.train()
        with self.assertRaises(ValueError):
            encode_frames(model, torch.rand(2, 3, 12, 12), torch.device("cpu"))

    def test_per_frame_cache_and_source_identity(self):
        with tempfile.TemporaryDirectory(prefix="visual cache space ") as directory:
            root, cache = Path(directory).resolve() / "videos", Path(directory) / "cache"
            relative = "2022_6_19/AM_5/none/same.mp4"
            path = root / relative
            path.parent.mkdir(parents=True)
            path.write_bytes(b"fixture")
            signature = cache_signature(root, relative, "config1")
            embeddings = np.arange(16*576, dtype=np.float32).reshape(16, 576)
            save_embedding_cache(cache, signature, embeddings, {"times": list(range(16))}, .5)
            entry, warning = load_embedding_cache(cache, signature, 16)
            self.assertFalse(warning)
            np.testing.assert_array_equal(entry["embeddings"], embeddings)
            np.testing.assert_array_equal(entry["embeddings"].mean(0), embeddings.mean(0))
            self.assertIsNone(load_embedding_cache(cache, {**signature, "config_id": "other"}, 16)[0])
            self.assertIsNone(load_embedding_cache(cache, signature, 15)[0])
            path.write_bytes(b"changed")
            self.assertIsNone(load_embedding_cache(cache, cache_signature(root, relative, "config1"), 16)[0])
            other = "2022_6_19/AM_5/weak/same.mp4"
            other_path = root / other
            other_path.parent.mkdir()
            other_path.write_bytes(b"fixture")
            self.assertIsNone(load_embedding_cache(cache, cache_signature(root, other, "config1"), 16)[0])

    def test_partial_decoding_failure_never_becomes_zero_embeddings(self):
        cfg = {"crop_xyxy": [0, 0, 1, 1], "frames_per_video": 16, "window_seconds": 2}
        class PartialCapture:
            def __init__(self): self.index = -1
            def isOpened(self): return True
            def get(self, key):
                return {cv2.CAP_PROP_FPS: 25, cv2.CAP_PROP_FRAME_COUNT: 50,
                        cv2.CAP_PROP_FRAME_WIDTH: 96, cv2.CAP_PROP_FRAME_HEIGHT: 64,
                        cv2.CAP_PROP_POS_FRAMES: self.index+1,
                        cv2.CAP_PROP_POS_MSEC: self.index*40}.get(key, 0)
            def grab(self):
                self.index += 1
                return self.index < 12
            def retrieve(self): return True, np.zeros((64, 96, 3), dtype=np.uint8)
            def release(self): pass
        with patch("feedwatch.visual.cv2.VideoCapture", return_value=PartialCapture()):
            result = decode_visual(Path(tempfile.gettempdir()).resolve(), "2022_6_19/AM_5/none/x.mp4", cfg,
                                   MobileNet_V3_Small_Weights.IMAGENET1K_V1.transforms())
        self.assertFalse(result["ok"])
        self.assertIsNone(result["frames"])
        self.assertEqual(len(result["details"]["sampled_indices"]), 4)
        self.assertIn("Falha", result["error"])


class ProbeTest(unittest.TestCase):
    def test_train_only_statistics_balancing_and_checkpoint_selection(self):
        torch.set_num_threads(1)
        rng = np.random.default_rng(7)
        x = rng.normal(size=(24, 576)).astype(np.float32)
        y = np.tile(np.arange(4), 6)
        v = (rng.normal(size=(12, 576))+10).astype(np.float32)
        vy = np.tile(np.arange(4), 3)
        cfg = {"max_epochs": 3, "batch_size": 8, "learning_rate": .001, "weight_decay": .01,
               "balance": "none", "standardize_train_only": True}
        mean, _, balance = train_statistics(x, y, "inverse_frequency")
        np.testing.assert_array_equal(mean, x.mean(0))
        np.testing.assert_array_equal(balance, np.ones(4))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            result = train_probe(x, y, v, vy, cfg, 42, torch.device("cpu"), path, {})
            head, bundle = load_probe(path / "best_classifier.pt", torch.device("cpu"))
            history = pd.read_csv(path / "history.csv")
            self.assertEqual(result["selected_epoch"], int(history.val_macro_f1.argmax())+1)
            np.testing.assert_allclose(bundle["mean"], x.mean(0))
            self.assertGreater(abs(v.mean()-bundle["mean"].mean().item()), 5)
            predicted = probabilities(head, torch.from_numpy((v-bundle["mean"].numpy())/bundle["std"].numpy()))
            np.testing.assert_array_equal(predicted.argmax(1), result["val_probabilities"].argmax(1))
            self.assertEqual(result["trainable_parameters"], 2308)
            self.assertFalse(bundle["extractor_trained"])

    def test_common_subset_alignment_and_invalid_prediction_mapping(self):
        rows = [parse_relative_path(f"2022_6_21/AM_5/{label}/same.mp4") for label in ("none", "weak", "medium", "strong")]
        original = pd.DataFrame(rows)
        original["predicted_id"] = original.class_id
        original["predicted_class"] = original.class_name
        current = original.iloc[[3, 1]].copy()
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            for name in ("dummy", "random_forest"):
                (base / name).mkdir()
                original.to_csv(base / name / "predictions.csv", index=False)
            output = base / "out"
            output.mkdir()
            comparison = compare_predictions(current, base, output)
            self.assertEqual(comparison["common_examples"], 2)
            self.assertEqual(comparison["own_validation"]["random_forest"]["examples"], 4)
            self.assertEqual(comparison["delta_pp_vs"]["random_forest"]["accuracy"], 0)
            current.loc[current.index[0], "predicted_class"] = "none"
            with self.assertRaises(ValueError):
                compare_predictions(current, base, output)
        bad = pd.DataFrame(rows)
        bad.loc[0, "class_id"] = 3
        with self.assertRaises(ValueError):
            check_metadata(bad)


if __name__ == "__main__":
    unittest.main()
