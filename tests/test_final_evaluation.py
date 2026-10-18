"""Final evaluation controls, using synthetic probabilities; no real test predictions."""
import copy
import math
import unittest
from unittest.mock import patch

from evidence import metrics_from_predictions
from final_evaluation import aggregate, validate_frozen


class FinalEvaluationTests(unittest.TestCase):
    def test_test_metrics_require_explicit_split_and_correct_log_loss(self):
        rows = [{"split": "test", "index": 0, "fixture_id": "synthetic", "label": 0,
                 "label_name": "home_win", "probabilities": {"home_win": .6, "draw": .3, "away_win": .1}}]
        with self.assertRaises(ValueError):
            metrics_from_predictions(rows)
        actual = metrics_from_predictions(rows, split="test")
        self.assertEqual(actual["split"], "test")
        self.assertAlmostEqual(actual["log_loss"], -math.log(.6))
        self.assertAlmostEqual(actual["brier_score"], .26)

    def test_all_seed_pairs_and_difference_sign(self):
        rows = [{"condition": condition, "seed": seed, "log_loss": 1 + i*.1 + offset,
                 "accuracy": .5, "brier_score": .6}
                for condition, offset in (("none", 0), ("sinusoidal", .02))
                for i, seed in enumerate((42, 43, 44))]
        actual = aggregate(rows)
        self.assertAlmostEqual(actual["paired_log_loss"]["mean"], .02)
        self.assertAlmostEqual(actual["conditions"]["none"]["log_loss"]["sample_sd"], .1)
        with self.assertRaises(ValueError):
            aggregate(rows[:-1])

    def test_frozen_sources_and_checkpoints_cannot_change(self):
        protocol = {"runs": [{"condition": c, "seed": s, "run_dir": f"runs/{c}{s}",
                              "checkpoint_sha256": "expected"}
                             for c in ("none", "sinusoidal") for s in (42, 43, 44)],
                    "data_sha256": "expected", "study_manifest": "manifest.json",
                    "study_manifest_sha256": "expected", "source_sha256": {"model.py": "expected"}}
        with patch("final_evaluation.sha256", return_value="expected"):
            validate_frozen(protocol)
        for key in ("data_sha256", "study_manifest_sha256"):
            changed = copy.deepcopy(protocol)
            changed[key] = "changed"
            with patch("final_evaluation.sha256", return_value="expected"), self.assertRaises(ValueError):
                validate_frozen(changed)
        changed = copy.deepcopy(protocol)
        changed["runs"][0]["checkpoint_sha256"] = "changed"
        with patch("final_evaluation.sha256", return_value="expected"), self.assertRaises(ValueError):
            validate_frozen(changed)


if __name__ == "__main__":
    unittest.main()
