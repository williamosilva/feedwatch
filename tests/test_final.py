import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch import nn

from feedwatch.data import CLASS_IDS, file_hash, parse_relative_path, write_json
from feedwatch.final_evaluation import infer_checkpoint, verify_predictions
from feedwatch.final_inputs import motion_item, validate_motion_details
from feedwatch.final_protocol import freeze_protocol, verify_protocol
from feedwatch.final_reporting import focus, report_final, summarize_values
from feedwatch.evaluation import metrics
from feedwatch.motion import identity_hash


class FinalProtocolTest(unittest.TestCase):
    def test_registration_does_not_require_test_data_and_detects_mutation(self):
        base = json.loads(Path("config/final_test.json").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            base.update(data_dir=str(root / "ABSENT"), manifest=str(root / "ABSENT.csv"), splits_dir=str(root / "ABSENT_SPLITS"))
            for spec in base["models"]:
                for key in ("checkpoint", "validation_predictions"):
                    path = root / f"{spec['id']}_{key}"
                    path.write_text("fixed previous artifact", encoding="utf-8")
                    spec[key] = str(path)
            weights = root / "weights.pth"
            weights.write_bytes(b"fixed weights")
            motion, visual = root / "motion.json", root / "visual.json"
            write_json(motion, {"extraction_identity": {"parameters": {}}})
            write_json(visual, {"extraction_identity": {"weights": {"weights_file": str(weights)}}})
            base.update(motion_config=str(motion), visual_config=str(visual))
            source, output = root / "input.json", root / "protocol"
            write_json(source, base)
            freeze_protocol(source, output)
            frozen = verify_protocol(output)
            self.assertEqual(frozen["operational_seed"], 42)
            self.assertFalse((root / "ABSENT").exists())
            with self.assertRaises(ValueError):
                freeze_protocol(source, output)
            Path(base["models"][0]["checkpoint"]).write_bytes(b"changed")
            with self.assertRaises(ValueError):
                verify_protocol(output)


class FinalInferenceTest(unittest.TestCase):
    def test_prediction_replay_rejects_wrong_order_or_class_id(self):
        rows = pd.DataFrame([parse_relative_path(f"2022_6_22/AM_5/{name}/{i}.mp4") for i, name in enumerate(CLASS_IDS)])
        probabilities = np.eye(4)
        stored = rows.assign(predicted_id=np.arange(4), predicted_class=list(CLASS_IDS), correct=True)
        for i, name in enumerate(CLASS_IDS):
            stored[f"prob_{name}"] = probabilities[:, i]
        self.assertTrue(verify_predictions(stored, rows, probabilities)["ids_exact"])
        with self.assertRaises(ValueError):
            verify_predictions(stored.iloc[::-1].reset_index(drop=True), rows, probabilities)
        stored.loc[0, "predicted_class"] = "medium"
        with self.assertRaises(ValueError):
            verify_predictions(stored, rows, probabilities)

    def test_linear_uses_checkpoint_statistics_without_fitting(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path, splits = root / "head.pt", root / "splits"
            splits.mkdir()
            path.write_bytes(b"fixed")
            (splits / "train.csv").write_bytes(b"fixed_train")
            identity = {"weights": {"weights_sha256": "frozen"}}
            head = nn.Linear(576, 4)
            with torch.no_grad():
                head.weight.zero_(); head.bias.zero_()
                head.weight[:, :4] = torch.eye(4)
            mean, std = torch.full((576,), 2.), torch.full((576,), 3.)
            bundle = {"seed": 42, "extractor_trained": False, "mean": mean, "std": std,
                      "selected_epoch": 7, "selection": "validation", "validation_macro_f1": .2,
                      "training_config": {}, "class_mapping": CLASS_IDS,
                      "provenance": {"config_id": identity_hash(identity), "used_train_sha256": file_hash(splits / "train.csv")}}
            config = {"locked_input_sha256": {str(path): file_hash(path)}, "splits_dir": str(splits),
                      "extraction_identities": {"visual": identity}, "evaluation_batch_size": 2}
            x = np.ones((3, 16, 576), dtype=np.float32)*8
            x[:, :, 1] = 14
            with patch("feedwatch.final_evaluation.load_probe", return_value=(head, bundle)):
                result, info = infer_checkpoint({"checkpoint": str(path), "family": "linear", "seed": 42}, config,
                                                 pd.DataFrame(), x, torch.device("cpu"))
            expected = torch.tensor([2., 4., 2., 2.]).softmax(0).numpy()
            np.testing.assert_allclose(result, np.tile(expected, (3, 1)), rtol=1e-6)
            self.assertFalse(info["trained_in_this_stage"])
            np.testing.assert_array_equal(bundle["mean"].numpy(), np.full(576, 2))

    def test_decode_failure_is_recorded_without_zero_features(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            row = parse_relative_path("2022_6_22/AM_5/medium/missing.mp4")
            result = motion_item(row, root, {"parameters": {}}, [root / "old", root / "new"])
            self.assertFalse(result["ok"])
            self.assertIsNone(result["features"])
            self.assertTrue(result["error"])

    def test_cached_motion_chronology_must_match_original_grid(self):
        parameters = {"sample_hz": 5., "window_seconds": 2.}
        indices = list(range(0, 50, 5))
        details = {"fps": 25., "frame_count": 50, "sampled_indices": indices,
                   "requested_indices": indices, "sampled_times_seconds": (np.asarray(indices)/25).tolist()}
        validate_motion_details(details, parameters)
        with self.assertRaises(ValueError):
            validate_motion_details({**details, "sampled_indices": indices[::-1]}, parameters)

    def test_seed_aggregation_is_sample_sd_not_ensemble(self):
        y = np.tile(np.arange(4), 2)
        values = [metrics(y, p) for p in (y, np.zeros(8, dtype=int), np.ones(8, dtype=int))]
        with tempfile.TemporaryDirectory() as folder:
            result = summarize_values(values, Path(folder))
            scores = [v["macro_f1"] for v in values]
            self.assertAlmostEqual(result["macro_f1"]["mean"], float(np.mean(scores)))
            self.assertAlmostEqual(result["macro_f1"]["std"], float(np.std(scores, ddof=1)))
            self.assertTrue((Path(folder) / "confusion_std.csv").exists())
        self.assertEqual(focus(values[0])["medium_recall"], 1)

    def test_comparison_uses_intersection_and_reports_missing_video(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            splits = root / "splits"
            splits.mkdir()
            labels = list(CLASS_IDS)*2
            requested = pd.DataFrame([parse_relative_path(f"2022_6_22/AM_5/{name}/{i}.mp4")
                                      for i, name in enumerate(labels)])
            validation = pd.DataFrame([parse_relative_path(f"2022_6_21/AM_5/{name}/{i}.mp4")
                                       for i, name in enumerate(labels)])
            validation.to_csv(splits / "val.csv", index=False)

            def predictions(rows):
                result = rows.assign(predicted_id=rows.class_id, predicted_class=rows.class_name, correct=True)
                for i, name in enumerate(CLASS_IDS):
                    result[f"prob_{name}"] = result.class_id.eq(i).astype(float)
                return result

            val_path = root / "validation.csv"
            predictions(validation).to_csv(val_path, index=False)
            specs = json.loads(Path("config/final_test.json").read_text(encoding="utf-8"))["models"]
            runs = []
            for original in specs:
                spec = {**original, "validation_predictions": str(val_path)}
                rows = requested.iloc[1:] if spec["family"] == "random_forest" else requested
                directory = root / "models" / spec["id"]
                directory.mkdir(parents=True)
                predictions(rows).to_csv(directory / "predictions.csv", index=False)
                runs.append({"spec": spec, "identification": {"used": len(rows), "failed": 8-len(rows),
                                                               "coverage_fraction": len(rows)/8}})
            result = report_final(root, {"splits_dir": str(splits)}, requested, runs)
            self.assertEqual(result["common_test"], 7)
            self.assertTrue(all(value["test_common"]["examples"] == 7 for value in result["models"].values()))
            self.assertEqual(next(value for value in result["coverage"] if value["model"] == "random_forest")["failed"], 1)
            self.assertFalse(result["ensemble"])
            self.assertEqual(result["operational_reference"], "gru_seed_42")


if __name__ == "__main__":
    unittest.main()
