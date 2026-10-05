"""Focused architecture and pipeline checks; no held-out test evaluation."""

import argparse
import contextlib
import io
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from dataset import MatchHistoryDataset, load_datasets
from model import MatchTransformer
from train import (constant_baseline, evaluate, load_checkpoint, save_checkpoint,
                   set_seed, train, validation_prediction)


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        set_seed(42)
        self.home = torch.randn(4, 10, 3)
        self.away = torch.randn(4, 10, 3)
        self.labels = torch.tensor([0, 1, 2, 0])

    def test_shape_finite_logits_gradients_and_probabilities(self):
        for setting in ("none", "sinusoidal"):
            with self.subTest(setting=setting):
                model = MatchTransformer(positional_encoding=setting)
                self.assertIsInstance(model.encoder, nn.TransformerEncoderLayer)
                self.assertEqual(model.projection.in_features, 3)
                self.assertEqual(model.projection.out_features, 32)
                self.assertEqual(model.encoder.self_attn.num_heads, 2)
                self.assertEqual(model.encoder.linear1.out_features, 64)
                self.assertEqual(model.encoder.dropout.p, 0.1)
                self.assertEqual(model.classifier.in_features, 64)
                logits = model(self.home, self.away)
                self.assertEqual(tuple(logits.shape), (4, 3))
                self.assertTrue(torch.isfinite(logits).all())
                F.cross_entropy(logits, self.labels).backward()
                for name, parameter in model.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                    self.assertGreater(parameter.grad.abs().sum().item(), 0, name)
                probabilities = logits.detach().softmax(dim=1)
                torch.testing.assert_close(probabilities.sum(dim=1), torch.ones(4))
                self.assertTrue(((probabilities >= 0) & (probabilities <= 1)).all())

    def test_identical_parameters_and_fixed_position_buffer(self):
        set_seed(7)
        none = MatchTransformer("none")
        set_seed(7)
        sinusoidal = MatchTransformer("sinusoidal")
        counts = [sum(p.numel() for p in model.parameters() if p.requires_grad)
                  for model in (none, sinusoidal)]
        self.assertEqual(counts[0], counts[1])
        self.assertEqual(counts[0], 8867)
        for left, right in zip(none.parameters(), sinusoidal.parameters()):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        self.assertIn("positions", dict(sinusoidal.named_buffers()))
        self.assertNotIn("positions", dict(sinusoidal.named_parameters()))
        self.assertFalse(sinusoidal.positions.requires_grad)
        torch.testing.assert_close(sinusoidal.positions[0, 0, 0::2], torch.zeros(16))
        torch.testing.assert_close(sinusoidal.positions[0, 0, 1::2], torch.ones(16))
        torch.testing.assert_close(sinusoidal.positions[0, 1, 0], torch.tensor(math.sin(1)))

    def test_checkpoint_reload_preserves_eval_predictions(self):
        for setting in ("none", "sinusoidal"):
            with self.subTest(setting=setting), tempfile.TemporaryDirectory() as directory:
                model = MatchTransformer(setting)
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                F.cross_entropy(model(self.home, self.away), self.labels).backward()
                optimizer.step()
                model.eval()
                with torch.no_grad():
                    expected = model(self.home, self.away).softmax(dim=1)
                checkpoint_path = Path(directory) / "best.pt"
                save_checkpoint(checkpoint_path, model, 1, 1.0, {"seed": 42})
                reloaded, checkpoint = load_checkpoint(checkpoint_path)
                self.assertEqual(reloaded.config, model.config)
                self.assertEqual(checkpoint["epoch"], 1)
                self.assertFalse(reloaded.training)
                with torch.no_grad():
                    actual = reloaded(self.home, self.away).softmax(dim=1)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_position_free_predictions_ignore_independent_permutations(self):
        model = MatchTransformer("none")
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        F.cross_entropy(model(self.home, self.away), self.labels).backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            expected = model(self.home, self.away).softmax(dim=1)
            for _ in range(5):
                home = torch.stack([history[torch.randperm(10)] for history in self.home])
                away = torch.stack([history[torch.randperm(10)] for history in self.away])
                actual = model(home, away).softmax(dim=1)
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_validation_weights_final_batch_and_disables_gradients(self):
        class FixedLogits(nn.Module):
            def __init__(self):
                super().__init__()
                self.gradient_flags = []

            def forward(self, home, away):
                self.gradient_flags.append(torch.is_grad_enabled())
                return home

        logits = torch.tensor([[4., 0., 0.], [0., 4., 0.], [0., 0., 4.],
                               [4., 0., 0.], [0., 0., 4.]])
        labels = torch.tensor([0, 1, 2, 0, 0])
        loader = DataLoader(TensorDataset(logits, logits, labels), batch_size=4)
        model = FixedLogits()
        actual = evaluate(model, loader, "cpu")
        self.assertAlmostEqual(actual["log_loss"], F.cross_entropy(logits, labels).item(), places=6)
        self.assertAlmostEqual(actual["accuracy"], 0.8)
        self.assertEqual(model.gradient_flags, [False, False])
        self.assertFalse(model.training)

    def test_baseline_uses_training_frequencies_only(self):
        actual = constant_baseline(torch.tensor([0, 0, 0, 1, 2]), torch.tensor([2, 2, 1]))
        self.assertEqual(actual["training_class_counts"], {"home_win": 3, "draw": 1, "away_win": 1})
        self.assertEqual(actual["probabilities"], {"home_win": 0.6, "draw": 0.2, "away_win": 0.2})
        self.assertAlmostEqual(actual["validation_log_loss"], -math.log(0.2))
        self.assertEqual(actual["validation_accuracy"], 0)

    def test_small_pipeline_saves_reconstructible_distinct_seeded_runs(self):
        datasets = load_datasets()
        # No test entry: any attempt by the training pipeline to use it fails.
        small = {name: MatchHistoryDataset(datasets[name].metadata[:size], datasets[name].report)
                 for name, size in (("train", 35), ("val", 5))}
        with tempfile.TemporaryDirectory() as directory:
            args = argparse.Namespace(epochs=2, seed=42, positional_encoding="none",
                                      device="cpu", threads=1, runs_dir=Path(directory))
            with patch("train.load_datasets", return_value=small), contextlib.redirect_stdout(io.StringIO()):
                first = train(args)
                second = train(args)
            self.assertNotEqual(first, second)
            metrics = json.loads((first / "metrics.json").read_text())
            self.assertEqual(metrics, json.loads((second / "metrics.json").read_text()))
            summary = json.loads((first / "summary.json").read_text())
            best = min(metrics, key=lambda row: row["validation_log_loss"])
            self.assertEqual(summary["best_epoch"], best["epoch"])
            self.assertEqual(summary["best_validation_log_loss"], best["validation_log_loss"])
            self.assertFalse(summary["test_evaluated"])
            for name in ("config", "environment", "data_report", "baseline", "validation_prediction"):
                self.assertIsInstance(json.loads((first / f"{name}.json").read_text()), dict)
            model, checkpoint = load_checkpoint(first / "best.pt")
            config = json.loads((first / "config.json").read_text())
            self.assertEqual(model.config, config["model"])
            self.assertEqual(checkpoint["run_config"], config)
            prediction = validation_prediction(model, small["val"], 0, "cpu")
            saved = json.loads((first / "validation_prediction.json").read_text())
            self.assertEqual(prediction, saved)


if __name__ == "__main__":
    unittest.main()
