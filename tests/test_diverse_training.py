import io
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from feedwatch.data import CLASS_IDS, parse_relative_path
from feedwatch.diverse_training import (day_metrics, session_bootstrap_delta, session_weights,
                                       validate_development_days, validation_gate)
from feedwatch.evaluation import metrics
from feedwatch.experiment_integrity import seal_inputs, verify_checkpoint_run
from feedwatch.research_acquisition import RemoteZip, verified_member


def rows(day="2022_6_17", group="AM_5", repeat=1):
    return pd.DataFrame([parse_relative_path(f"{day}/{group}/{label}/{n}.mp4")
                         for label in CLASS_IDS for n in range(repeat)])


class DiverseTrainingTest(unittest.TestCase):
    def test_session_sampling_balances_observed_strata_without_validation(self):
        frame = pd.concat([rows(repeat=1), rows(group="PM_5", repeat=4)], ignore_index=True).assign(split="train")
        weights = session_weights(frame)
        self.assertAlmostEqual(weights[:4].sum(), .5)
        self.assertAlmostEqual(weights[4:].sum(), .5)
        np.testing.assert_allclose(weights[:4], weights[4:8]*4)
        with self.assertRaises(ValueError): session_weights(frame.assign(split="val"))

    def test_day_assignment_rejects_overlap_and_unexpected_heldout(self):
        frame = pd.concat([rows("2022_6_16").assign(split="train"), rows().assign(split="val")], ignore_index=True)
        validate_development_days(frame, {"train": ["2022_6_16"], "val": ["2022_6_17"]})
        for assignment in ({"train": ["2022_6_16"], "val": ["2022_6_16"]},
                           {"train": ["2022_6_16"], "val": ["2022_6_18"]}):
            with self.assertRaises(ValueError): validate_development_days(frame, assignment)

    def test_day_average_does_not_allow_large_day_to_hide_small_day_failure(self):
        frame = pd.concat([rows("2022_6_17"), rows("2022_6_21", repeat=20)], ignore_index=True)
        predicted = frame.class_id.to_numpy().copy()
        predicted[:4] = (predicted[:4]+1) % 4
        result = day_metrics(frame, np.eye(4)[predicted])
        self.assertEqual(result["2022_6_17"]["macro_f1"], 0)
        self.assertEqual(result["2022_6_21"]["macro_f1"], 1)
        old = {day: metrics(np.arange(4), np.arange(4)) for day in result}
        gate = validation_gate([result], [old], {"mean_day_macro_f1_gain": .03,
            "mean_day_medium_recall_gain": .1, "max_other_mean_day_recall_drop": .05,
            "max_existing_day_macro_f1_drop": .01})
        self.assertAlmostEqual(gate["candidate"]["macro_f1"], .5)
        self.assertFalse(gate["eligible_for_heldout"])

    def test_session_bootstrap_is_paired_and_reproducible(self):
        frame = pd.concat([rows(group="AM_5"), rows(group="PM_5")], ignore_index=True)
        actual = frame.class_id.to_numpy()
        baseline, candidate = (actual+1) % 4, actual.copy()
        result = session_bootstrap_delta(frame, baseline, candidate, 42, 100)
        self.assertEqual(result, session_bootstrap_delta(frame, baseline, candidate, 42, 100))
        self.assertEqual(result["clusters"], 2)
        self.assertEqual(result["lower_95"], 1)
        self.assertEqual(session_bootstrap_delta(frame, candidate, baseline, 42, 100)["upper_95"], -1)


class AcquisitionTest(unittest.TestCase):
    def test_exact_member_transfer_checks_directory_name_size_and_crc(self):
        content = b"real research data"*100
        memory = io.BytesIO()
        with zipfile.ZipFile(memory, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("2022_6_17/AM_5/none/1.mp4", content)
        payload = memory.getvalue()
        with zipfile.ZipFile(io.BytesIO(payload)) as archive: member = archive.infolist()[0]
        def read(remote, size=-1):
            start = remote.position
            value = payload[start:start+size]
            remote.position += len(value)
            remote.transferred += len(value)
            return value
        with tempfile.TemporaryDirectory() as directory, patch.object(RemoteZip, "read", read):
            target = Path(directory)/"video.mp4"
            result = verified_member("https://example.org/archive", len(payload), member, target)
            self.assertTrue(result["crc_verified"])
            self.assertEqual(target.read_bytes(), content)
            self.assertLess(result["transferred_bytes"], len(payload))
            target.write_bytes(b"corrupted")
            with self.assertRaises(ValueError): verified_member("https://example.org/archive", len(payload), member, target)

    def test_range_reader_refuses_full_archive_and_out_of_bounds_seek(self):
        remote = RemoteZip("https://example.org/archive", 10**9)
        with self.assertRaises(ValueError): remote.read()
        with self.assertRaises(ValueError): remote.seek(-1)
        with self.assertRaises(ValueError): remote.seek(10**9+1)
        with self.assertRaises(ValueError): RemoteZip("http://example.org/archive", 100)


class ResumeIntegrityTest(unittest.TestCase):
    def test_input_seal_rejects_changed_checkpoint_validation_selection_or_architecture(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"seal.json"
            inputs = {"candidate_sha256": "a", "validation_sha256": "b", "selection_sha256": "c",
                      "heads": {"hidden": 32}, "baseline_sha256": {"42": "d"}}
            original = seal_inputs(path, inputs)
            self.assertEqual(seal_inputs(path, inputs), original)
            for key in inputs:
                changed = {**inputs, key: "changed"}
                with self.assertRaises(ValueError): seal_inputs(path, changed)
            self.assertEqual(seal_inputs(path, inputs), original)

    def test_seed_resume_rejects_wrong_provenance_hash_and_configuration(self):
        import hashlib
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory)/"checkpoint.pt"
            checkpoint.write_bytes(b"immutable checkpoint")
            sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            inputs, heads, config = {"train_sha256": "a", "val_sha256": "b"}, {"hidden": 32}, {"seed": [42]}
            result = {"seed": 42, "training_inputs": inputs, "checkpoint_sha256": sha, "selected_epoch": 3}
            bundle = {"seed": 42, "kind": "gru", "head_config": heads, "training_config": config,
                      "provenance": inputs, "selected_epoch": 3}
            verify_checkpoint_run(checkpoint, result, bundle, inputs, 42, heads, config)
            for key, value in (("training_inputs", {"train_sha256": "changed"}), ("checkpoint_sha256", "bad"), ("seed", 43)):
                with self.assertRaises(ValueError): verify_checkpoint_run(checkpoint, {**result, key: value}, bundle, inputs, 42, heads, config)
            for key, value in (("head_config", {"hidden": 16}), ("training_config", {}), ("provenance", {}), ("selected_epoch", 4)):
                with self.assertRaises(ValueError): verify_checkpoint_run(checkpoint, result, {**bundle, key: value}, inputs, 42, heads, config)


if __name__ == "__main__": unittest.main()
