"""Train, check small-batch learning, or predict a validation fixture only."""

import argparse
import json
import math
import os
import platform
import random
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from dataset import DEFAULT_CSV, LABELS, load_datasets
from model import MatchTransformer
from evidence import (dependency_versions, metrics_from_predictions, record_provenance,
                      validation_predictions, verify_run)


ROOT = Path(__file__).resolve().parent


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def prepare_device(name, threads):
    if threads < 1:
        raise ValueError("threads must be positive")
    torch.set_num_threads(threads)
    # Required for deterministic CUDA matrix multiplication; set before use.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable; use --device cpu")
    return device


def environment_details(device):
    versions = dependency_versions()
    return {
        "python": platform.python_version(), "dependencies": versions,
        "torch_build": str(torch.__version__), "platform": platform.platform(),
        "machine": platform.machine(), "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(), "torch_threads": torch.get_num_threads(),
        "device": str(device), "cuda_build": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }


def evaluate(model, loader, device):
    """Mean multiclass log loss (natural log) and accuracy over all examples."""
    model.eval()
    total_loss, correct, count = 0.0, 0, 0
    with torch.no_grad():
        for home, away, labels in loader:
            home, away, labels = home.to(device), away.to(device), labels.to(device)
            logits = model(home, away)
            total_loss += F.cross_entropy(logits, labels, reduction="sum").item()
            correct += (logits.argmax(dim=1) == labels).sum().item()
            count += labels.numel()
    if not count:
        raise ValueError("Validation dataset is empty")
    return {"log_loss": total_loss / count, "accuracy": correct / count}


def constant_baseline(train_labels, val_labels):
    counts = torch.bincount(train_labels, minlength=3)
    probabilities = counts.double() / counts.sum()
    if (probabilities == 0).any():
        raise ValueError("Training labels must contain all three classes for this baseline")
    return {
        "training_class_counts": dict(zip(LABELS, counts.tolist())),
        "probabilities": dict(zip(LABELS, probabilities.tolist())),
        "validation_log_loss": -probabilities[val_labels].log().mean().item(),
        "validation_accuracy": (val_labels == probabilities.argmax()).double().mean().item(),
    }


def save_checkpoint(path, model, epoch, validation_log_loss, run_config):
    torch.save({
        "model_config": model.config, "model_state_dict": model.state_dict(),
        "epoch": epoch, "validation_log_loss": validation_log_loss,
        "run_config": run_config,
    }, path)


def load_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = MatchTransformer(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, checkpoint


def validation_prediction(model, val, index, device):
    if not 0 <= index < len(val):
        raise ValueError(f"Validation index must be between 0 and {len(val) - 1}")
    home, away, label = val[index]
    model.eval()
    with torch.no_grad():
        logits = model(home.unsqueeze(0).to(device), away.unsqueeze(0).to(device))
        probabilities = logits.softmax(dim=1)[0].cpu()
    target = val.metadata[index].target
    return {
        "split": "val", "index": index, "fixture_id": target.fixture_id,
        "home": target.home, "away": target.away, "kickoff": target.kickoff.isoformat(),
        "label": label.item(), "label_name": LABELS[label.item()],
        "probabilities": dict(zip(LABELS, probabilities.tolist())),
    }


def train(args):
    if args.epochs < 1:
        raise ValueError("epochs must be positive")
    device = prepare_device(args.device, args.threads)
    set_seed(args.seed)
    datasets = load_datasets()
    train_data, val_data = datasets["train"], datasets["val"]
    # A separate seeded generator makes batch order independent of model RNG use.
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, generator=generator)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
    model = MatchTransformer(positional_encoding=args.positional_encoding).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    started_at = datetime.now(timezone.utc)
    name = f"{started_at:%Y%m%dT%H%M%SZ}_{args.positional_encoding}_seed{args.seed}_{uuid.uuid4().hex[:8]}"
    run_dir = args.runs_dir / name
    run_dir.mkdir(parents=True, exist_ok=False)
    config = {
        "condition_role": "reference" if args.positional_encoding == "none" else "comparison",
        "seed": args.seed, "epochs": args.epochs, "batch_size": 32,
        "optimizer": "Adam", "learning_rate": 0.001,
        "loss": "cross_entropy_on_logits", "checkpoint_selection": "minimum_validation_log_loss",
        "shuffle_train": True, "shuffle_validation": False, "num_workers": 0,
        "device": str(device), "threads": args.threads, "model": model.config,
    }
    record_provenance(run_dir, [
        "python", "train.py", "train", "--positional-encoding", args.positional_encoding,
        "--seed", str(args.seed), "--epochs", str(args.epochs),
        "--device", str(device), "--threads", str(args.threads),
    ])
    write_json(run_dir / "config.json", config)
    write_json(run_dir / "environment.json", environment_details(device))
    # Use a portable source path, leaving Dara's committed preparation report intact.
    report = dict(train_data.report)
    report["source"] = DEFAULT_CSV.relative_to(ROOT).as_posix()
    write_json(run_dir / "data_report.json", report)
    baseline = constant_baseline(train_data.labels, val_data.labels)
    write_json(run_dir / "baseline.json", baseline)
    print(f"Run directory: {run_dir}", flush=True)
    print(f"Constant baseline: validation log loss={baseline['validation_log_loss']:.6f}, "
          f"accuracy={baseline['validation_accuracy']:.6f}", flush=True)

    metrics, best_epoch, best_loss = [], None, math.inf
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, count = 0.0, 0
        for home, away, labels in train_loader:
            home, away, labels = home.to(device), away.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(home, away), labels)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.numel()
            count += labels.numel()
        validation = evaluate(model, val_loader, device)
        row = {"epoch": epoch, "training_loss": total_loss / count,
               "validation_log_loss": validation["log_loss"],
               "validation_accuracy": validation["accuracy"]}
        metrics.append(row)
        write_json(run_dir / "metrics.json", metrics)
        if validation["log_loss"] < best_loss:
            best_loss, best_epoch = validation["log_loss"], epoch
            save_checkpoint(run_dir / "best.pt", model, epoch, best_loss, config)
        print(f"Epoch {epoch:02}: train={row['training_loss']:.6f}, "
              f"val log loss={validation['log_loss']:.6f}, "
              f"val accuracy={validation['accuracy']:.6f}", flush=True)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    training_seconds = time.perf_counter() - start
    best_model, _ = load_checkpoint(run_dir / "best.pt", device)
    reloaded = evaluate(best_model, val_loader, device)
    if not math.isclose(reloaded["log_loss"], best_loss, rel_tol=1e-6, abs_tol=1e-7):
        raise ValueError("Reloaded checkpoint does not reproduce best validation log loss")
    summary = {
        "started_at_utc": started_at.isoformat(), "training_seconds": training_seconds,
        "parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "best_epoch": best_epoch, "best_validation_log_loss": best_loss,
        "best_validation_accuracy": metrics[best_epoch - 1]["validation_accuracy"],
        "checkpoint": "best.pt", "checkpoint_reload_verified": True,
        "reloaded_validation_metrics": reloaded, "test_evaluated": False,
        "training_examples": len(train_data), "validation_examples": len(val_data),
    }
    write_json(run_dir / "summary.json", summary)
    prediction = validation_prediction(best_model, val_data, 0, device)
    write_json(run_dir / "validation_prediction.json", prediction)
    predictions = validation_predictions(best_model, val_data, device)
    write_json(run_dir / "validation_predictions.json", predictions)
    write_json(run_dir / "validation_metrics.json", metrics_from_predictions(predictions))
    verify_run(run_dir, device, datasets=datasets)
    print(json.dumps(summary, indent=2), flush=True)
    print(json.dumps(prediction, indent=2), flush=True)
    return run_dir


def predict(args):
    device = prepare_device(args.device, args.threads)
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    val = load_datasets()["val"]
    data_report = args.checkpoint.parent / "data_report.json"
    if data_report.exists():
        saved_report = json.loads(data_report.read_text(encoding="utf-8"))
        if saved_report["source_sha256"] != val.report["source_sha256"]:
            raise ValueError("Data source differs from the checkpoint run")
    result = validation_prediction(model, val, args.index, device)
    result["checkpoint_epoch"] = checkpoint["epoch"]
    print(json.dumps(result, indent=2))


def overfit(args):
    """Diagnostic only: repeatedly optimise eight training examples, no validation."""
    if args.steps < 1:
        raise ValueError("steps must be positive")
    device = prepare_device(args.device, args.threads)
    set_seed(args.seed)
    data = load_datasets()["train"]
    home, away, labels = (value.to(device) for value in
                          (data.home_histories[:8], data.away_histories[:8], data.labels[:8]))
    model = MatchTransformer(positional_encoding=args.positional_encoding).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    def batch_loss():
        model.eval()
        with torch.no_grad():
            return F.cross_entropy(model(home, away), labels).item()

    initial_loss = batch_loss()
    start = time.perf_counter()
    for _ in range(args.steps):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(home, away), labels)
        loss.backward()
        optimizer.step()
    final_loss = batch_loss()
    result = {
        "diagnostic": "overfit_first_eight_training_examples", "seed": args.seed,
        "model": model.config, "steps": args.steps, "learning_rate": 0.001,
        "initial_eval_loss": initial_loss, "final_eval_loss": final_loss,
        "loss_reduction_fraction": 1 - final_loss / initial_loss,
        "required_reduction_fraction": 0.5, "passed": final_loss < initial_loss * 0.5,
        "seconds": time.perf_counter() - start, "device": str(device),
        "fixture_ids": [m.target.fixture_id for m in data.metadata[:8]],
        "test_evaluated": False,
    }
    print(json.dumps(result, indent=2))
    if args.output:
        # Diagnostics must not overwrite a previous record either.
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if not result["passed"]:
        raise RuntimeError("Small-batch loss did not decrease by at least 50%")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train_parser = commands.add_parser("train", help="Train using train and validation only")
    train_parser.add_argument("--epochs", type=int, default=10)
    train_parser.add_argument("--runs-dir", type=Path, default=ROOT / "runs")
    overfit_parser = commands.add_parser("overfit", help="Separate small-batch learning diagnostic")
    overfit_parser.add_argument("--steps", type=int, default=300)
    overfit_parser.add_argument("--output", type=Path)
    for command in (train_parser, overfit_parser):
        command.add_argument("--seed", type=int, default=42)
        command.add_argument("--positional-encoding", choices=("none", "sinusoidal"), default="none")
    predict_parser = commands.add_parser("predict", help="Reload a checkpoint for a validation fixture")
    predict_parser.add_argument("--checkpoint", type=Path, required=True)
    predict_parser.add_argument("--index", type=int, default=0)
    for command in (train_parser, overfit_parser, predict_parser):
        command.add_argument("--device", default="cpu")
        command.add_argument("--threads", type=int, default=1,
                             help="CPU worker threads; one is efficient for this tiny model")
    args = parser.parse_args()
    {"train": train, "predict": predict, "overfit": overfit}[args.command](args)


if __name__ == "__main__":
    main()
