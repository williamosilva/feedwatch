import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from feedwatch.data import CLASS_IDS, parse_relative_path
from feedwatch.motion import identity_hash
from feedwatch.temporal_data import validate_chronology
from feedwatch.temporal_heads import (EarlyStopping, TemporalHead, apply_permutations, evaluate_head,
                                     load_temporal_checkpoint, train_head, validation_permutations)
from feedwatch.temporal_reporting import mean_sd
from feedwatch.visual import load_embedding_cache

HEADS = {"embedding_dim": 576, "gru_hidden_size": 32, "mlp_hidden_size": 101, "dropout": .3}


class TemporalInputsTest(unittest.TestCase):
    def test_chronology_rejects_reordered_repeated_or_missing_frames(self):
        config = {"frames_per_video": 16, "window_seconds": 2}
        indices = np.rint(np.linspace(0, 49, 16)).astype(int).tolist()
        details = {"fps": 25, "frame_count": 50, "sampled_indices": indices,
                   "requested_indices": indices, "sampled_times_seconds": (np.asarray(indices)/25).tolist()}
        validate_chronology(details, config)
        for changed in [{**details, "sampled_indices": indices[::-1]},
                        {**details, "sampled_times_seconds": details["sampled_times_seconds"][::-1]},
                        {**details, "sampled_times_seconds": [0]*16},
                        {**details, "sampled_indices": indices[:-1]}]:
            with self.assertRaises(ValueError):
                validate_chronology(changed, config)

    def test_mean_only_cache_cannot_be_used_as_sequence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            signature = {"config_id": "id", "relative_path": "2022_6_19/AM_5/none/x.mp4"}
            (root / "id").mkdir()
            with (root / "id" / f"{identity_hash(signature)}.npz").open("wb") as handle:
                np.savez(handle, embeddings=np.zeros(576, dtype=np.float32),
                         metadata=np.asarray(json.dumps({"signature": signature, "details": {}, "seconds": 0})))
            entry, warning = load_embedding_cache(root, signature, 16)
            self.assertIsNone(entry)
            self.assertTrue(warning)
        with self.assertRaises(ValueError):
            TemporalHead("gru", HEADS)(torch.zeros(2, 576))

    def test_permutations_are_fixed_per_path_and_preserve_frames(self):
        paths = ["a", "b", "c"]
        orders = validation_permutations(paths, 123)
        np.testing.assert_array_equal(orders, validation_permutations(paths, 123))
        np.testing.assert_array_equal(orders[::-1], validation_permutations(paths[::-1], 123))
        self.assertTrue(np.all(np.sort(orders, axis=1) == np.arange(16)))
        x = torch.arange(3*16*2).reshape(3, 16, 2)
        shuffled = apply_permutations(x, orders)
        self.assertTrue(torch.equal(shuffled[0], x[0, orders[0]]))
        with self.assertRaises(ValueError):
            apply_permutations(x, np.zeros((3, 16), dtype=np.int64))


class TemporalHeadsTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_parameter_matching_and_single_direction_layer(self):
        mlp, gru = TemporalHead("mlp", HEADS), TemporalHead("gru", HEADS)
        self.assertEqual(sum(p.numel() for p in mlp.parameters()), 58685)
        self.assertEqual(sum(p.numel() for p in gru.parameters()), 58692)
        self.assertEqual(gru.gru.num_layers, 1)
        self.assertFalse(gru.gru.bidirectional)
        self.assertEqual(gru.gru.dropout, 0)

    def test_mlp_is_order_invariant_gru_can_use_order(self):
        torch.manual_seed(42)
        x = torch.randn(4, 16, 576)
        orders = validation_permutations([str(i) for i in range(4)], 123)
        shuffled = apply_permutations(x, orders)
        mlp, gru = TemporalHead("mlp", HEADS).eval(), TemporalHead("gru", HEADS).eval()
        with torch.inference_mode():
            torch.testing.assert_close(mlp(x), mlp(shuffled), atol=1e-6, rtol=1e-5)
            self.assertGreater(float((gru(x)-gru(shuffled)).abs().max()), 1e-4)

    def test_early_stopping_does_not_reset_on_ties(self):
        stop = EarlyStopping(3)
        observed = [stop.step(value) for value in [.1, .2, .2, .19, .2]]
        self.assertEqual(observed, [(True, False), (True, False), (False, False), (False, False), (False, True)])
        self.assertEqual(stop.best, .2)
        self.assertAlmostEqual(mean_sd([1, 2, 3])["std"], 1)
        self.assertIsNone(mean_sd([1])["std"])

    def test_checkpoint_replay_and_diagnostic_does_not_refit(self):
        rng = np.random.default_rng(42)
        x = torch.from_numpy(rng.normal(size=(16, 16, 576)).astype(np.float32))
        v = torch.from_numpy(rng.normal(size=(8, 16, 576)).astype(np.float32))
        y, vy = np.tile(np.arange(4), 4), np.tile(np.arange(4), 2)
        labels = list(CLASS_IDS)
        train_rows = pd.DataFrame([parse_relative_path(f"2022_6_19/AM_5/{labels[label]}/{i}.mp4") for i, label in enumerate(y)])
        val_rows = pd.DataFrame([parse_relative_path(f"2022_6_21/AM_5/{labels[label]}/{i}.mp4") for i, label in enumerate(vy)])
        cfg = {"balance": "none", "max_epochs": 3, "batch_size": 8, "evaluation_batch_size": 8,
               "learning_rate": .0003, "weight_decay": .01, "max_grad_norm": 1., "patience": 2, "min_delta": 0.}
        orders = validation_permutations(val_rows.relative_path.tolist(), 123)
        with tempfile.TemporaryDirectory() as directory, patch("feedwatch.temporal_heads.plot_history"):
            output = Path(directory)
            result = train_head("gru", 42, HEADS, cfg, x, y, v, vy, train_rows, val_rows,
                                np.zeros(576, dtype=np.float32), np.ones(576, dtype=np.float32), output,
                                {"permutation_sha256": "test"}, orders)
            restored, bundle = load_temporal_checkpoint(output / "best_head.pt", torch.device("cpu"))
            pred, _, _ = evaluate_head(restored, v, vy, 8)
            saved = pd.read_csv(output / "val" / "predictions.csv")
            np.testing.assert_array_equal(pred.argmax(1), saved.predicted_id)
            history = pd.read_csv(output / "history.csv")
            self.assertEqual(bundle["selected_epoch"], int(history.val_macro_f1.argmax())+1)
            diagnostic = json.loads((output / "val_shuffled" / "diagnostic.json").read_text())
            self.assertFalse(diagnostic["used_for_selection"])
            self.assertFalse(diagnostic["retrained"])
            self.assertTrue(result["checkpoint_reproduces_predictions"])
            self.assertFalse(bundle["extractor_trained"])


if __name__ == "__main__":
    unittest.main()
