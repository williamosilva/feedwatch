import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from feedwatch.data import (CLASS_IDS, COLUMNS, index_dataset, make_splits, parse_relative_path,
                            protect_originals, read_manifest, stratified_sample, validate_splits)


def fixture_manifest():
    return pd.DataFrame([parse_relative_path(f"2022_6_{day}/{group}/{label}/same.mp4")
                         for day in range(19, 23) for group in ["AM_5", "PM_100"]
                         for label in CLASS_IDS], columns=COLUMNS)


class PathsTest(unittest.TestCase):
    def test_labels_and_opaque_group(self):
        for label, class_id in CLASS_IDS.items():
            row = parse_relative_path(f"2022_6_19/AM_999/{label}/19_video_1.mp4")
            self.assertEqual(row["class_id"], class_id)
            self.assertEqual(row["group"], "AM_999")

    def test_invalid_paths_and_unknown_class(self):
        for path in ["2022_6_19/AM_5/hungry/x.mp4", "2022_2_30/AM_5/none/x.mp4",
                     "2022_6_19/bad/none/x.mp4", "2022_6_19/AM_5/none/nested/x.mp4",
                     "../AM_5/none/x.mp4", "2022_6_19/AM_5/none/x.txt",
                     "/2022_6_19/AM_5/none/x.mp4", "2022_6_19\\AM_5\\none\\x.mp4"]:
            with self.subTest(path=path), self.assertRaises(ValueError):
                parse_relative_path(path)

    def test_repeated_basenames_and_issue_log(self):
        with tempfile.TemporaryDirectory(prefix="feedwatch space ") as directory:
            base = Path(directory)
            root, output = base / "original videos", base / "out"
            paths = ["2022_6_19/AM_5/none/same.mp4", "2022_6_19/AM_5/weak/same.mp4",
                     "2022_6_20/PM_5/none/same.mp4", "2022_6_19/AM_5/unknown/bad.mp4", "stray.mp4"]
            for relative in paths:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"original untouched")
            manifest, issues = index_dataset(root, output)
            self.assertEqual(len(manifest), 3)
            self.assertEqual(manifest.relative_path.nunique(), 3)
            self.assertEqual(len(issues), 2)
            self.assertEqual(len(read_manifest(output / "manifest.csv")), 3)
            self.assertEqual(len(json.loads((output / "index_report.json").read_text(encoding="utf-8"))["issues"]), 2)
            for relative in paths:
                self.assertEqual((root / relative).read_bytes(), b"original untouched")

    def test_output_cannot_be_inside_originals(self):
        root = Path(tempfile.gettempdir()).resolve() / "originals"
        with self.assertRaises(ValueError):
            protect_originals(root, root / "generated")

    def test_manifest_detects_label_tampering_and_duplicate_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.csv"
            frame = fixture_manifest()
            frame.loc[0, "class_id"] = 3
            frame.to_csv(path, index=False)
            with self.assertRaises(ValueError):
                read_manifest(path)
            pd.concat([fixture_manifest(), fixture_manifest().iloc[:1]]).to_csv(path, index=False)
            with self.assertRaises(ValueError):
                read_manifest(path)


class SplitsTest(unittest.TestCase):
    def setUp(self):
        self.manifest = fixture_manifest()
        self.parts = {"train": self.manifest[self.manifest.date.isin(["2022_6_19", "2022_6_20"])].copy(),
                      "val": self.manifest[self.manifest.date == "2022_6_21"].copy(),
                      "test": self.manifest[self.manifest.date == "2022_6_22"].copy()}

    def test_whole_days_no_leakage_and_all_classes(self):
        result = validate_splits(self.manifest, self.parts)
        self.assertTrue(result["valid"], result["errors"])
        self.assertEqual([result["composition"][n]["videos"] for n in ["train", "val", "test"]], [16, 8, 8])

    def test_rejects_overlap_of_paths(self):
        self.parts["test"] = pd.concat([self.parts["test"], self.parts["train"].iloc[:1]])
        result = validate_splits(self.manifest, self.parts)
        self.assertTrue(any("sobreposição de caminhos" in error for error in result["errors"]))

    def test_rejects_day_leakage_even_with_different_paths(self):
        moved = self.parts["train"].iloc[:1]
        self.parts["train"] = self.parts["train"].iloc[1:]
        self.parts["test"] = pd.concat([self.parts["test"], moved])
        result = validate_splits(self.manifest, self.parts)
        self.assertTrue(any("sobreposição de datas" in error for error in result["errors"]))

    def test_rejects_missing_class_and_missing_path(self):
        self.parts["test"] = self.parts["test"][self.parts["test"].class_name != "strong"]
        result = validate_splits(self.manifest, self.parts)
        self.assertFalse(result["valid"])
        self.assertTrue(any("classes ausentes" in error for error in result["errors"]))
        self.assertTrue(any("exatamente os caminhos" in error for error in result["errors"]))

    def test_saved_splits_revalidate(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            self.manifest.to_csv(base / "manifest.csv", index=False)
            config = {name: sorted(frame.date.unique()) for name, frame in self.parts.items()}
            (base / "config.json").write_text(json.dumps(config), encoding="utf-8")
            result = make_splits(base / "manifest.csv", base / "config.json", base / "splits")
            self.assertTrue(result["valid"])
            saved = {name: read_manifest(base / "splits" / f"{name}.csv") for name in self.parts}
            self.assertTrue(validate_splits(self.manifest, saved)["valid"])
            config["val"].append("2022_6_19")
            (base / "config.json").write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaises(ValueError):
                make_splits(base / "manifest.csv", base / "config.json", base / "invalid")


class SamplingTest(unittest.TestCase):
    def test_reproducible_independent_of_input_order_and_covers_strata(self):
        first = fixture_manifest()
        second = first.copy()
        second["relative_path"] = second.relative_path.str.replace("same.mp4", "other.mp4", regex=False)
        frame = pd.concat([first, second], ignore_index=True)
        a = stratified_sample(frame, 32, 42)
        b = stratified_sample(frame.sample(frac=1, random_state=9), 32, 42)
        self.assertEqual(a.relative_path.tolist(), b.relative_path.tolist())
        self.assertEqual(a.groupby(["date", "group", "class_name"]).ngroups, 32)
        self.assertEqual(len(stratified_sample(frame, 100, 42)), 64)
        self.assertEqual(a.relative_path.nunique(), 32)


if __name__ == "__main__":
    unittest.main()
