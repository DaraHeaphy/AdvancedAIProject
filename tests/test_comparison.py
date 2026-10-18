"""Check paired statistics and reject invalid or confounded experiments."""

import copy
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from comparison import (CONDITIONS, PROTOCOL, SEEDS, check_grid, collect_runs,
                        controlled_config, distribution, paired_results)
from dataset import LABELS
from evidence import write_json


class ComparisonTests(unittest.TestCase):
    def rows(self):
        return [{"condition": condition, "seed": seed,
                 "validation_log_loss": [1.0, 1.1, 1.2][index] + (0.02 if condition == "sinusoidal" else 0),
                 "validation_accuracy": [0.4, 0.5, 0.6][index] - (0.01 if condition == "sinusoidal" else 0)}
                for condition in CONDITIONS for index, seed in enumerate(SEEDS)]

    def test_sample_variation_and_insufficient_results(self):
        result = distribution([1.0, 1.1, 1.2])
        self.assertAlmostEqual(result["mean"], 1.1)
        self.assertAlmostEqual(result["sample_sd"], 0.1)
        for values in ([1, 2], [1, 2, math.nan], [1, 2, math.inf]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                distribution(values)

    def test_paired_statistics_use_all_seeds_and_correct_sign(self):
        pairs, summary = paired_results(list(reversed(self.rows())))
        self.assertEqual([pair["seed"] for pair in pairs], list(SEEDS))
        for pair in pairs:
            self.assertAlmostEqual(pair["log_loss_difference"], 0.02)
            self.assertAlmostEqual(pair["accuracy_difference_pp"], -1.0)
        self.assertAlmostEqual(summary["none"]["log_loss"]["mean"], 1.1)
        self.assertAlmostEqual(summary["sinusoidal"]["log_loss"]["mean"], 1.12)

    def test_missing_duplicate_and_unplanned_seeds_are_rejected(self):
        rows = self.rows()
        wrong_seed = copy.deepcopy(rows)
        wrong_seed[0]["seed"] = 99
        for invalid in (rows[:-1], rows + [rows[0]], wrong_seed):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                check_grid(invalid)

    def test_only_seed_condition_and_role_are_ignored_in_fixed_config(self):
        config = {"seed": 42, "condition_role": "reference", "epochs": 10,
                  "model": {"positional_encoding": "none", "d_model": 32}}
        other = copy.deepcopy(config)
        other.update(seed=43, condition_role="comparison")
        other["model"]["positional_encoding"] = "sinusoidal"
        self.assertEqual(controlled_config(config), controlled_config(other))
        other["model"]["d_model"] = 64
        self.assertNotEqual(controlled_config(config), controlled_config(other))

    def test_collection_rejects_confounds_and_uses_independent_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = {"protocol": PROTOCOL, "status": "complete", "runs": []}
            for condition in CONDITIONS:
                for seed in SEEDS:
                    run = root / f"{condition}_{seed}"
                    run.mkdir()
                    config = {"seed": seed, "condition_role": condition, "epochs": 10,
                              "device": "cpu", "threads": 1,
                              "model": {"positional_encoding": condition, "d_model": 32}}
                    predictions = [{"split": "val", "index": 0, "fixture_id": "example", "label": 0,
                                    "label_name": LABELS[0], "probabilities": dict(zip(LABELS, [0.6, 0.2, 0.2]))}]
                    for name, value in {
                        "config": config, "summary": {"parameter_count": 8867, "test_evaluated": False,
                        "best_epoch": 1, "training_seconds": 1}, "validation_predictions": predictions,
                        "environment": {"python": "same"}, "data_report": {"source_sha256": "same"},
                        "provenance": {"source_files_sha256": {"model.py": "same"}},
                        "baseline": {"validation_log_loss": 1.1}, "metrics": [],
                    }.items():
                        write_json(run / f"{name}.json", value)
                    manifest["runs"].append({"condition": condition, "seed": seed, "run_dir": run.name})
            with patch("comparison.safe_run", side_effect=lambda name: root / name), \
                    patch("comparison.verify_run", return_value={"passed": True}):
                rows, _, _ = collect_runs(manifest)
                self.assertEqual(len(rows), 6)
                self.assertAlmostEqual(rows[0]["validation_log_loss"], -math.log(0.6))
                self.assertEqual(rows[0]["validation_accuracy"], 1.0)
                target = root / "sinusoidal_44"
                mutations = {
                    "environment": {"python": "different"},
                    "provenance": {"source_files_sha256": {"model.py": "different"}},
                    "data_report": {"source_sha256": "different"},
                    "validation_predictions": [{**predictions[0], "fixture_id": "different"}],
                }
                for name, value in mutations.items():
                    path = target / f"{name}.json"
                    original = path.read_bytes()
                    write_json(path, value)
                    with self.subTest(name=name), self.assertRaisesRegex(ValueError, "Runs differ"):
                        collect_runs(manifest)
                    path.write_bytes(original)


if __name__ == "__main__":
    unittest.main()
