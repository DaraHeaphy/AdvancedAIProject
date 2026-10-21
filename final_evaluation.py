"""Freeze, evaluate and verify the revised experiment without choosing models on test."""

import argparse
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from comparison import collect_runs, check_grid, safe_run, write_csv
from dataset import DEFAULT_CSV, LABELS, load_datasets
from evidence import metrics_from_predictions, read_json, sha256, write_json
from train import load_checkpoint, prepare_device, evaluate

ROOT = Path(__file__).resolve().parent


def now():
    return datetime.now(timezone.utc).isoformat()


def predict_rows(model, data):
    rows = []
    model.eval()
    with torch.no_grad():
        for home, away, labels in DataLoader(data, batch_size=32, shuffle=False):
            probabilities = model(home, away).double().softmax(dim=1).tolist()
            for probability, label in zip(probabilities, labels.tolist()):
                index = len(rows)
                target = data.metadata[index].target
                rows.append({"split": "test", "index": index, "fixture_id": target.fixture_id,
                             "home": target.home, "away": target.away,
                             "kickoff": target.kickoff.isoformat(), "label": label,
                             "label_name": LABELS[label],
                             "probabilities": dict(zip(LABELS, probability))})
    return rows


def aggregate(rows):
    check_grid(rows)
    conditions = {}
    for condition in ("none", "sinusoidal"):
        selected = [row for row in rows if row["condition"] == condition]
        conditions[condition] = {
            key: {"mean": statistics.mean(row[key] for row in selected),
                  "sample_sd": statistics.stdev(row[key] for row in selected)}
            for key in ("log_loss", "accuracy", "brier_score")}
    lookup = {(row["condition"], row["seed"]): row for row in rows}
    pairs = [{"seed": seed,
              "log_loss_difference": lookup[("sinusoidal", seed)]["log_loss"] - lookup[("none", seed)]["log_loss"],
              "accuracy_difference_pp": 100 * (lookup[("sinusoidal", seed)]["accuracy"] - lookup[("none", seed)]["accuracy"])}
             for seed in (42, 43, 44)]
    return {"conditions": conditions, "paired_differences": pairs,
            "paired_log_loss": {"mean": statistics.mean(p["log_loss_difference"] for p in pairs),
                                "sample_sd": statistics.stdev(p["log_loss_difference"] for p in pairs)}}


def baseline_rows(train, test):
    counts = torch.bincount(train.labels, minlength=3).double()
    probabilities = (counts / counts.sum()).tolist()
    rows = []
    for index, metadata in enumerate(test.metadata):
        target = metadata.target
        rows.append({"split": "test", "index": index, "fixture_id": target.fixture_id,
                     "label": target.label, "label_name": LABELS[target.label],
                     "probabilities": dict(zip(LABELS, probabilities))})
    return rows


def freeze(study, output):
    if output.exists() and any(output.iterdir()):
        raise ValueError("Freeze requires a new empty output directory; existing evidence is never replaced")
    manifest = read_json(study / "experiment.json")
    # Validation verification only. No test prediction or test metric occurs here.
    collect_runs(manifest)
    output.mkdir(parents=True, exist_ok=True)
    sources = ["dataset.py", "model.py", "train.py", "evidence.py", "comparison.py", "final_evaluation.py"]
    protocol = {
        "frozen_before_test_evaluation_at_utc": now(),
        "research_question": manifest["protocol"]["question"],
        "study_manifest": (study / "experiment.json").relative_to(ROOT).as_posix(),
        "study_manifest_sha256": sha256(study / "experiment.json"),
        "source_sha256": {name: sha256(ROOT / name) for name in sources},
        "data_sha256": sha256(DEFAULT_CSV),
        "primary_metric": "test multiclass log loss, natural logarithm; lower is better",
        "secondary_metrics": ["accuracy", "multiclass Brier score", "confusion matrix and class recall"],
        "checkpoint_selection": "minimum validation log loss within each fixed ten-epoch run",
        "comparison": "sinusoidal minus none, paired by training seed; retain all three pairs",
        "uncertainty": "sample SD across training seeds; no significance claim or independent-fixture assumption",
        "test_policy": "available 2021-2022 season; frozen models; earlier completed test fixtures may enter later histories",
        "no_test_feedback": "No architecture, training, checkpoint, seed, threshold or ensemble selection after evaluation",
        "evaluation_status": "Revised analysis of a previously inspected test season; not a new untouched holdout",
        "baseline": "constant class probabilities fitted to training labels only",
        "runs": [{**entry, "checkpoint_sha256": sha256(safe_run(entry["run_dir"]) / "best.pt")}
                 for entry in manifest["runs"]],
    }
    write_json(output / "protocol.json", protocol)
    print(f"Frozen before test evaluation: {output / 'protocol.json'}")


def validate_frozen(protocol):
    check_grid(protocol["runs"])
    if sha256(DEFAULT_CSV) != protocol["data_sha256"]:
        raise ValueError("Frozen data changed")
    if sha256(ROOT / protocol["study_manifest"]) != protocol["study_manifest_sha256"]:
        raise ValueError("Frozen study manifest changed")
    for name, digest in protocol["source_sha256"].items():
        if sha256(ROOT / name) != digest:
            raise ValueError(f"Frozen source changed: {name}")
    for entry in protocol["runs"]:
        if sha256(safe_run(entry["run_dir"]) / "best.pt") != entry["checkpoint_sha256"]:
            raise ValueError("Frozen checkpoint changed")


def class_rows(entry, metrics):
    confusion = metrics["confusion_matrix"]
    return [{"condition": entry["condition"], "seed": entry["seed"], "class": label,
             "actual_count": sum(confusion[i]),
             "predicted_count": sum(row[i] for row in confusion),
             "correct_count": confusion[i][i],
             "recall": confusion[i][i] / sum(confusion[i]) if sum(confusion[i]) else None}
            for i, label in enumerate(LABELS)]


def evaluate_frozen(output):
    protocol = read_json(output / "protocol.json")
    validate_frozen(protocol)
    if any(path.name != "protocol.json" for path in output.iterdir()):
        raise ValueError("Evaluation output already contains evidence; use verify instead of replacing results")
    prepare_device("cpu", 1)
    started = now()
    write_json(output / "evaluation_started.json", {"started_at_utc": started,
               "protocol_sha256": sha256(output / "protocol.json")})
    data = load_datasets()
    rows, classes = [], []
    for entry in protocol["runs"]:
        model, checkpoint = load_checkpoint(safe_run(entry["run_dir"]) / "best.pt")
        predictions = predict_rows(model, data["test"])
        metrics = metrics_from_predictions(predictions, split="test")
        independent_torch = evaluate(model, DataLoader(data["test"], batch_size=32), "cpu")
        for key in ("log_loss", "accuracy"):
            if not math.isclose(metrics[key], independent_torch[key], rel_tol=1e-6, abs_tol=1e-7):
                raise ValueError("Independent test metrics disagree")
        stem = f"{entry['condition']}_seed{entry['seed']}"
        write_json(output / f"{stem}_predictions.json", predictions)
        write_json(output / f"{stem}_metrics.json", metrics)
        rows.append({"condition": entry["condition"], "seed": entry["seed"],
                     "selected_epoch": checkpoint["epoch"], "examples": len(predictions),
                     **{key: metrics[key] for key in ("log_loss", "accuracy", "brier_score")}})
        classes.extend(class_rows(entry, metrics))
    baseline = metrics_from_predictions(baseline_rows(data["train"], data["test"]), split="test")
    result = {"evaluation_started_at_utc": started, "completed_at_utc": now(), "split": "test",
              "protocol_sha256": sha256(output / "protocol.json"), "examples": len(data["test"]),
              "runs": rows, **aggregate(rows), "training_frequency_baseline": baseline,
              "data_report": data["test"].report}
    write_json(output / "results.json", result)
    write_csv(output / "individual_runs.csv", rows)
    write_csv(output / "paired_differences.csv", result["paired_differences"])
    write_csv(output / "class_errors.csv", classes)
    plots(output, result)
    verify(output)
    print(f"Final held-out evaluation complete: {output / 'results.json'}")


def plots(output, result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2), layout="constrained")
    for seed, color in zip((42, 43, 44), ("#246c91", "#cf7234", "#52844c")):
        values = [next(r["log_loss"] for r in result["runs"] if r["seed"] == seed and r["condition"] == c)
                  for c in ("none", "sinusoidal")]
        axes[0].plot([0, 1], values, "o-", color=color, label=f"Seed {seed}")
    axes[0].axhline(result["training_frequency_baseline"]["log_loss"], color="gray", linestyle="--", label="Constant baseline")
    axes[0].set(xticks=[0, 1], xticklabels=["No position", "Sinusoidal"], ylabel="Test log loss (lower is better)", title="All frozen checkpoints")
    axes[0].legend(fontsize=9)
    pairs = result["paired_differences"]
    axes[1].bar([str(p["seed"]) for p in pairs], [p["log_loss_difference"] for p in pairs],
                color=["#cf7234" if p["log_loss_difference"] >= 0 else "#246c91" for p in pairs])
    axes[1].axhline(0, color="gray", linewidth=1)
    axes[1].set(xlabel="Training seed", ylabel="Sinusoidal minus no position", title="Negative favours sinusoidal")
    for suffix in ("png", "svg"):
        fig.savefig(output / f"test_comparison.{suffix}", dpi=180)
    plt.close(fig)


def verify(output):
    protocol = read_json(output / "protocol.json")
    validate_frozen(protocol)
    results = read_json(output / "results.json")
    if results["protocol_sha256"] != sha256(output / "protocol.json"):
        raise ValueError("Protocol differs from evaluated protocol")
    if not protocol["frozen_before_test_evaluation_at_utc"] <= results["evaluation_started_at_utc"]:
        raise ValueError("Protocol was not frozen before evaluation")
    prepare_device("cpu", 1)
    data = load_datasets()
    recalculated = []
    for entry in protocol["runs"]:
        stem = f"{entry['condition']}_seed{entry['seed']}"
        saved = read_json(output / f"{stem}_predictions.json")
        model, _ = load_checkpoint(safe_run(entry["run_dir"]) / "best.pt")
        fresh = predict_rows(model, data["test"])
        if len(saved) != len(fresh):
            raise ValueError("Test prediction count changed")
        for left, right in zip(saved, fresh):
            if {k: v for k, v in left.items() if k != "probabilities"} != {k: v for k, v in right.items() if k != "probabilities"}:
                raise ValueError("Test fixture or label changed")
            if any(not math.isclose(left["probabilities"][label], right["probabilities"][label], rel_tol=1e-6, abs_tol=1e-7) for label in LABELS):
                raise ValueError("Test predictions did not reproduce")
        metrics = metrics_from_predictions(saved, split="test")
        if metrics != read_json(output / f"{stem}_metrics.json"):
            raise ValueError("Saved test metrics disagree")
        row = next(row for row in results["runs"] if (row["condition"], row["seed"]) == (entry["condition"], entry["seed"]))
        if any(row[key] != metrics[key] for key in ("log_loss", "accuracy", "brier_score")):
            raise ValueError("Result table differs from probabilities")
        recalculated.append(row)
    if any(results[key] != value for key, value in aggregate(recalculated).items()):
        raise ValueError("Paired summary differs from individual results")
    if results["training_frequency_baseline"] != metrics_from_predictions(baseline_rows(data["train"], data["test"]), split="test"):
        raise ValueError("Training-only baseline differs")
    verification = {"verified_at_utc": now(), "passed": True, "test_evaluated": True,
                    "checks": ["frozen source/data/checkpoints", "protocol preceded evaluation",
                               "all six test predictions reproduced", "independent probability metrics",
                               "paired summaries", "training-only baseline"]}
    write_json(output / "verification.json", verification)
    print("Verified all six frozen test evaluations")
    return verification


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "evaluate", "verify"))
    parser.add_argument("--study", type=Path, default=Path("research/revised_validation"))
    parser.add_argument("--output", type=Path, default=Path("research/revised_evaluation"))
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / "research") or output == ROOT / "research":
        raise ValueError("Final evaluation must be within a study directory in research/")
    if args.command == "freeze":
        freeze(args.study.resolve(), output)
    elif args.command == "evaluate":
        evaluate_frozen(output)
    else:
        verify(output)


if __name__ == "__main__":
    main()
