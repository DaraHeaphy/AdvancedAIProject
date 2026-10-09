"""Independent metric checks and evidence corruption/portability checks."""

import argparse
import contextlib
import hashlib
import io
import json
import math
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from dataset import LABELS, MatchHistoryDataset, load_datasets
from evidence import (bundle_run, dependency_versions, metrics_from_predictions,
                      portable_path, read_json, verify_run, write_json)
from train import train


class MetricEvidenceTests(unittest.TestCase):
    def rows(self):
        return [{"split": "val", "index": index, "fixture_id": str(index),
                 "label": label, "label_name": LABELS[label],
                 "probabilities": dict(zip(LABELS, probabilities))}
                for index, (label, probabilities) in enumerate(
                    [(0, [0.7, 0.2, 0.1]), (2, [0.2, 0.5, 0.3])])]

    def test_metrics_match_hand_calculated_values(self):
        metrics = metrics_from_predictions(self.rows())
        self.assertAlmostEqual(metrics["log_loss"], (-math.log(0.7) - math.log(0.3)) / 2)
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertAlmostEqual(metrics["brier_score"], 0.46)
        self.assertEqual(metrics["confusion_matrix"], [[1, 0, 0], [0, 0, 0], [0, 1, 0]])

    def test_invalid_probabilities_labels_and_duplicate_fixtures_are_rejected(self):
        bad_rows = []
        for values in ([0.7, 0.2, 0.2], [float("nan"), 0.2, 0.1], [-0.1, 0.2, 0.9], [0., 0.5, 0.5]):
            rows = self.rows()
            rows[0]["probabilities"] = dict(zip(LABELS, values))
            bad_rows.append(rows)
        rows = self.rows()
        rows[1]["fixture_id"] = rows[0]["fixture_id"]
        bad_rows.append(rows)
        rows = self.rows()
        rows[0]["label"] = -1
        bad_rows.append(rows)
        bad_rows.append([])
        for rows in bad_rows:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                metrics_from_predictions(rows)

    def test_dependency_closure_includes_transitive_packages(self):
        versions = dependency_versions()
        for name in ("torch", "packaging", "markupsafe", "mpmath"):
            self.assertIn(name, versions)
        self.assertNotIn("pytest", versions)

    def test_portable_paths_cannot_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            for path in ("../secret.txt", "/secret.txt", "C:/secret.txt", "..\\secret.txt"):
                with self.subTest(path=path), self.assertRaises(ValueError):
                    portable_path(Path(directory), path)


class RunEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        datasets = load_datasets()
        # Deliberately provide no test dataset to the training/verification APIs.
        cls.small = {name: MatchHistoryDataset(datasets[name].metadata[:size], datasets[name].report)
                     for name, size in (("train", 35), ("val", 5))}
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        args = argparse.Namespace(epochs=2, seed=42, positional_encoding="none",
                                  device="cpu", threads=1, runs_dir=Path(cls.temp.name))
        with patch("train.load_datasets", return_value=cls.small), contextlib.redirect_stdout(io.StringIO()):
            cls.run_dir = train(args)

    def cloned_run(self, directory):
        import shutil
        target = Path(directory) / "run"
        shutil.copytree(self.run_dir, target)
        return target

    def test_all_predictions_and_automatic_verification_are_saved(self):
        self.assertEqual(len(read_json(self.run_dir / "validation_predictions.json")), 5)
        result = read_json(self.run_dir / "verification.json")
        self.assertTrue(result["passed"])
        self.assertFalse(result["test_evaluated"])
        self.assertTrue(result["checks"]["independent_metrics_match_training"])
        self.assertIn("git", read_json(self.run_dir / "provenance.json"))

    def test_corrupted_predictions_are_rejected_and_failure_is_saved(self):
        with tempfile.TemporaryDirectory() as directory:
            target = self.cloned_run(directory)
            rows = read_json(target / "validation_predictions.json")
            rows[0]["probabilities"] = {name: 1 / 3 for name in LABELS}
            write_json(target / "validation_predictions.json", rows)
            with self.assertRaisesRegex(ValueError, "Run verification failed"):
                verify_run(target, datasets=self.small)
            result = read_json(target / "verification.json")
            self.assertFalse(result["passed"])
            self.assertFalse(result["checks"]["validation_predictions_reproduced"])

    def test_corrupted_source_and_checkpoint_are_rejected(self):
        for filename in ("source/model.py", "best.pt"):
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as directory:
                target = self.cloned_run(directory)
                (target / filename).write_text("corrupted", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "Run verification failed"):
                    verify_run(target, datasets=self.small)
                self.assertFalse(read_json(target / "verification.json")["passed"])

    def test_bundle_contains_reproducible_allowlisted_files_and_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "evidence.zip"
            with patch("evidence.load_datasets", return_value=self.small):
                bundle_run(self.run_dir, output)
                with self.assertRaises(FileExistsError):
                    bundle_run(self.run_dir, output)
            with zipfile.ZipFile(output) as archive:
                manifest = json.loads(archive.read("bundle_manifest.json"))
                for name, expected in manifest["files_sha256"].items():
                    self.assertEqual(hashlib.sha256(archive.read(name)).hexdigest(), expected)
                self.assertEqual(set(archive.namelist()), set(manifest["files_sha256"]) | {"bundle_manifest.json"})
                self.assertIn("dataset/matches.csv", archive.namelist())
                self.assertIn(f"runs/{self.run_dir.name}/best.pt", archive.namelist())
                self.assertIn("tests/test_evidence.py", archive.namelist())
                self.assertFalse(any(".git/" in name or ".venv/" in name for name in archive.namelist()))


if __name__ == "__main__":
    unittest.main()
