"""Run and report the fixed paired comparison using validation evidence only."""

import argparse
import csv
import importlib.metadata
import json
import math
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dataset import LABELS
from evidence import metrics_from_predictions, read_json, sha256, verify_run, write_json
from train import prepare_device

ROOT = Path(__file__).resolve().parent
CONDITIONS = ("none", "sinusoidal")
SEEDS = (42, 43, 44)
PROTOCOL = {
    "question": "Does fixed sinusoidal positional encoding improve next-fixture prediction compared with no positional encoding?",
    "conditions": list(CONDITIONS), "seeds": list(SEEDS),
    "epochs": 10, "device": "cpu", "threads": 1,
    "primary_metric": "validation multiclass log loss, natural logarithm",
    "secondary_metric": "validation accuracy",
    "checkpoint_selection": "minimum validation log loss within each run",
    "paired_difference": "sinusoidal minus none; negative favours sinusoidal",
    "variation": "sample standard deviation across training seeds (ddof=1)",
    "test_evaluated": False,
    "history_policy": "Reset team history after a missing Premier League season",
    "evaluation_status": "Revised analysis; the final test season was previously inspected",
}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def distribution(values):
    if len(values) < 3 or any(not math.isfinite(v) for v in values):
        raise ValueError("At least three finite independent-seed results are required")
    return {"mean": statistics.mean(values), "sample_sd": statistics.stdev(values),
            "min": min(values), "max": max(values)}


def check_grid(entries):
    pairs = [(entry["condition"], entry["seed"]) for entry in entries]
    expected = {(condition, seed) for condition in CONDITIONS for seed in SEEDS}
    if len(pairs) != len(set(pairs)) or set(pairs) != expected:
        raise ValueError("Expected exactly one run for each of two conditions and three seeds")


def controlled_config(config):
    fixed = {key: value for key, value in config.items()
             if key not in ("seed", "condition_role", "model")}
    fixed["model"] = {key: value for key, value in config["model"].items()
                      if key != "positional_encoding"}
    return fixed


def paired_results(rows):
    check_grid(rows)
    lookup = {(row["condition"], row["seed"]): row for row in rows}
    pairs = [{"seed": seed,
              "log_loss_difference": lookup[("sinusoidal", seed)]["validation_log_loss"]
              - lookup[("none", seed)]["validation_log_loss"],
              "accuracy_difference_pp": 100 * (lookup[("sinusoidal", seed)]["validation_accuracy"]
              - lookup[("none", seed)]["validation_accuracy"])} for seed in SEEDS]
    summaries = {condition: {
        "log_loss": distribution([row["validation_log_loss"] for row in rows if row["condition"] == condition]),
        "accuracy": distribution([row["validation_accuracy"] for row in rows if row["condition"] == condition]),
    } for condition in CONDITIONS}
    return pairs, summaries


def safe_run(relative):
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(ROOT / "runs") or path == ROOT / "runs":
        raise ValueError("Run path must be a directory within the project's runs folder")
    return path


def run_study(output):
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "experiment.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest["protocol"] != PROTOCOL:
            raise ValueError("Existing study protocol differs; use a new output folder")
    else:
        if any(output.iterdir()):
            raise ValueError("A new study needs an empty output folder")
        manifest = {"created_before_training_at_utc": utc_now(), "protocol": PROTOCOL,
                    "status": "incomplete", "runs": [],
                    "script_sha256": sha256(Path(__file__)), "wall_seconds": 0.0}
        write_json(manifest_path, manifest)
    if manifest["script_sha256"] != sha256(Path(__file__)):
        raise ValueError("Study runner changed since training began; use a new output folder")
    prepare_device("cpu", 1)
    started = time.perf_counter()
    for seed in SEEDS:
        for condition in CONDITIONS:
            existing = [entry for entry in manifest["runs"]
                        if (entry["condition"], entry["seed"]) == (condition, seed)]
            if existing:
                if len(existing) != 1:
                    raise ValueError("Duplicate study entries")
                verify_run(safe_run(existing[0]["run_dir"]), "cpu")
                continue
            command = [sys.executable, str(ROOT / "train.py"), "train",
                       "--positional-encoding", condition, "--seed", str(seed),
                       "--epochs", "10", "--device", "cpu", "--threads", "1"]
            before = set((ROOT / "runs").glob("*"))
            print(f"Training {condition}, seed {seed}", flush=True)
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            (output / f"training_{condition}_seed{seed}.log").write_text(
                result.stdout + result.stderr, encoding="utf-8")
            if result.returncode:
                raise RuntimeError(f"Training failed; inspect training_{condition}_seed{seed}.log")
            fresh = [path for path in set((ROOT / "runs").glob("*")) - before
                     if path.is_dir() and (path / "config.json").exists()
                     and read_json(path / "config.json")["seed"] == seed
                     and read_json(path / "config.json")["model"]["positional_encoding"] == condition]
            if len(fresh) != 1:
                raise ValueError("Cannot identify exactly one fresh training run")
            run = fresh[0]
            verify_run(run, "cpu")
            manifest["runs"].append({"condition": condition, "seed": seed,
                                     "run_dir": run.relative_to(ROOT).as_posix()})
            write_json(manifest_path, manifest)
            saved = read_json(run / "summary.json")
            print(f"  {run.name}: epoch {saved['best_epoch']}, log loss {saved['best_validation_log_loss']:.6f}", flush=True)
    manifest["wall_seconds"] += time.perf_counter() - started
    manifest["status"] = "complete"
    write_json(manifest_path, manifest)
    report_study(output)


def collect_runs(manifest):
    if manifest["protocol"] != PROTOCOL or manifest["status"] != "complete":
        raise ValueError("Study must be complete and use the fixed protocol")
    check_grid(manifest["runs"])
    prepare_device("cpu", 1)
    rows, evidence = [], {}
    reference = None
    for entry in sorted(manifest["runs"], key=lambda item: (item["condition"], item["seed"])):
        run = safe_run(entry["run_dir"])
        verification = verify_run(run, "cpu")
        config, summary = read_json(run / "config.json"), read_json(run / "summary.json")
        if (config["model"]["positional_encoding"], config["seed"]) != (entry["condition"], entry["seed"]):
            raise ValueError("Manifest condition or seed differs from saved configuration")
        if any(config[key] != PROTOCOL[key] for key in ("epochs", "device", "threads")):
            raise ValueError("Training settings differ from the protocol")
        if summary["parameter_count"] != 8867 or summary["test_evaluated"]:
            raise ValueError("Unexpected model size or test evaluation")
        predictions = read_json(run / "validation_predictions.json")
        metrics = metrics_from_predictions(predictions)
        environment = read_json(run / "environment.json")
        signature = {"config": controlled_config(config),
                     "data": read_json(run / "data_report.json"),
                     "source_hashes": read_json(run / "provenance.json")["source_files_sha256"],
                     "environment": environment,
                     "fixtures": [{k: v for k, v in p.items() if k != "probabilities"} for p in predictions],
                     "baseline": read_json(run / "baseline.json")}
        if reference is None:
            reference = signature
        elif signature != reference:
            raise ValueError("Runs differ in controlled settings, source/data, environment, fixtures or baseline")
        rows.append({"condition": entry["condition"], "seed": entry["seed"],
                     "selected_epoch": summary["best_epoch"],
                     "validation_log_loss": metrics["log_loss"],
                     "validation_accuracy": metrics["accuracy"],
                     "training_seconds": summary["training_seconds"],
                     "parameter_count": summary["parameter_count"], "verified": verification["passed"],
                     "run_dir": entry["run_dir"]})
        evidence[(entry["condition"], entry["seed"])] = {
            "predictions": predictions, "metrics": metrics,
            "epochs": read_json(run / "metrics.json")}
    return rows, evidence, reference


def class_errors(evidence):
    rows = []
    for (condition, seed), run in sorted(evidence.items()):
        confusion = run["metrics"]["confusion_matrix"]
        for index, label in enumerate(LABELS):
            examples = [p for p in run["predictions"] if p["label"] == index]
            rows.append({"condition": condition, "seed": seed, "class": label,
                         "actual_count": len(examples),
                         "predicted_count": sum(row[index] for row in confusion),
                         "correct_count": confusion[index][index],
                         "recall": confusion[index][index] / len(examples),
                         "mean_true_class_probability": statistics.mean(p["probabilities"][label] for p in examples),
                         "class_log_loss": statistics.mean(-math.log(p["probabilities"][label]) for p in examples)})
    return rows


def failure_cases(evidence):
    # Rank actual fixtures by average individual-model loss, not an ensemble prediction.
    cases = []
    for index, first in enumerate(evidence[("none", SEEDS[0])]["predictions"]):
        case = {key: first[key] for key in ("fixture_id", "home", "away", "kickoff", "label_name")}
        for condition in CONDITIONS:
            examples = [evidence[(condition, seed)]["predictions"][index] for seed in SEEDS]
            case[f"{condition}_mean_log_loss"] = statistics.mean(
                -math.log(p["probabilities"][p["label_name"]]) for p in examples)
            case[f"{condition}_mean_true_probability"] = statistics.mean(
                p["probabilities"][p["label_name"]] for p in examples)
            case[f"{condition}_wrong_seeds"] = sum(
                max(p["probabilities"], key=p["probabilities"].__getitem__) != p["label_name"] for p in examples)
        case["ranking_mean_log_loss"] = statistics.mean(case[f"{c}_mean_log_loss"] for c in CONDITIONS)
        cases.append(case)
    return sorted(cases, key=lambda case: case["ranking_mean_log_loss"], reverse=True)[:10]


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def figures(output, rows, pairs, evidence, baseline):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    colours = {"none": "#2563a6", "sinusoidal": "#c55d18"}
    lookup = {(r["condition"], r["seed"]): r for r in rows}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), layout="constrained")
    for seed in SEEDS:
        ys = [lookup[(c, seed)]["validation_log_loss"] for c in CONDITIONS]
        axes[0].plot([0, 1], ys, marker="o", label=f"Seed {seed}")
        for x, y in enumerate(ys):
            axes[0].annotate(f"{seed}", (x, y), xytext=(7, 0), textcoords="offset points", fontsize=8)
    axes[0].axhline(baseline["validation_log_loss"], color="#666666", linestyle="--", label="Constant baseline")
    axes[0].set(xticks=[0, 1], xticklabels=["No position", "Sinusoidal"],
                ylabel="Validation log loss (lower is better)", title="Selected checkpoint, each seed", xlim=(-0.2, 1.3))
    axes[0].legend(fontsize=8)
    differences = [p["log_loss_difference"] for p in pairs]
    axes[1].bar([str(s) for s in SEEDS], differences,
                color=["#2563a6" if d < 0 else "#c55d18" for d in differences], width=0.55)
    axes[1].axhline(0, color="#555555", linewidth=1)
    axes[1].set(xlabel="Training seed", ylabel="Sinusoidal minus no-position log loss",
                title="Paired differences; negative favours sinusoidal")
    for index, difference in enumerate(differences):
        axes[1].annotate(f"{difference:+.4f}", (index, difference), xytext=(0, 5 if difference >= 0 else -14),
                         textcoords="offset points", ha="center", fontsize=9)
    axes[1].margins(y=0.3)
    fig.suptitle("Positional encoding comparison on the corrected validation split")
    for suffix in ("png", "svg"):
        fig.savefig(output / f"paired_log_loss.{suffix}", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True, layout="constrained")
    for axis, condition in zip(axes, CONDITIONS):
        for seed in SEEDS:
            epochs = evidence[(condition, seed)]["epochs"]
            axis.plot([r["epoch"] for r in epochs], [r["validation_log_loss"] for r in epochs], label=f"Seed {seed}")
            selected = lookup[(condition, seed)]
            axis.scatter(selected["selected_epoch"], selected["validation_log_loss"], s=50, edgecolors="black", zorder=3)
        axis.axhline(baseline["validation_log_loss"], color="#666666", linestyle="--", label="Constant baseline")
        axis.set(title="No positional encoding" if condition == "none" else "Sinusoidal encoding",
                 xlabel="Epoch", xticks=range(1, 11))
        axis.legend(fontsize=8)
    axes[0].set_ylabel("Validation log loss (lower is better)")
    fig.suptitle("Validation learning curves; outlined points mark selected checkpoints")
    for suffix in ("png", "svg"):
        fig.savefig(output / f"learning_curves.{suffix}", dpi=180)
    plt.close(fig)


def interpretation(pairs):
    differences = [p["log_loss_difference"] for p in pairs]
    wins = sum(d < 0 for d in differences)
    if wins == 0:
        return ("Sinusoidal encoding did not improve validation log loss for any of the three seeds. "
                "Under this fixed setup, the results favour the no-position reference. "
                "This does not establish that positional information is unhelpful in general.")
    if wins == 3:
        return ("Sinusoidal encoding achieved lower validation log loss for all three seeds. "
                "This is consistent evidence of an advantage in this fixed setup, but three seeds and one "
                "validation season provide limited evidence about stability and no claim of statistical significance.")
    return (f"Sinusoidal encoding improved validation log loss for {wins} of three seeds. "
            "The direction varies across seeds, so there is no consistent improvement in this comparison. "
            "A small mean difference should be interpreted alongside the paired variation.")


def report_text(rows, pairs, summaries, errors, cases, result):
    text = ["# Revised validation comparison", "", PROTOCOL["question"], "",
            "Team histories reset after a missing Premier League season. Architecture, optimiser, "
            "seeds and ten-epoch budget are unchanged. The test season was previously inspected; "
            "subsequent test results are a revised analysis, not a new untouched holdout.", "",
            "| Condition | Seed | Selected epoch | Validation log loss | Accuracy |",
            "| --- | ---: | ---: | ---: | ---: |"]
    for row in rows:
        text.append(f"| {row['condition']} | {row['seed']} | {row['selected_epoch']} | "
                    f"{row['validation_log_loss']:.6f} | {100*row['validation_accuracy']:.2f}% |")
    text += ["", interpretation(pairs), "",
             f"Mean paired log-loss difference: {result['paired_log_loss']['mean']:+.6f}; "
             f"sample SD: {result['paired_log_loss']['sample_sd']:.6f}. "
             "Negative favours sinusoidal encoding.", "",
             "The same validation fixtures select checkpoints and provide these scores. "
             "Three seeds measure training variation on one season, not uncertainty across seasons. "
             "Fixtures share teams and overlapping histories. The incomplete final season and "
             "exclusion of fixtures with insufficient history limit generalisation.", "",
             f"Six training runs; total training/validation-loop time "
             f"{result['total_training_seconds']:.2f} seconds. One CPU thread per run. "
             "Both conditions have 8,867 trainable parameters.", "",
             "All six checkpoints and exported predictions were verified. Class errors and "
             "selected validation failure cases accompany this report. They do not establish "
             "the football-specific causes of errors.", "",
             "Codex assisted with implementation, analysis and writing. Regression tests, "
             "data audits, checkpoint reloads and independent probability metrics check its output. "
             "Individual contributions and lecturer gate decisions require the team's records.", ""]
    return "\n".join(text)


def report_study(output):
    manifest = read_json(output / "experiment.json")
    rows, evidence, reference = collect_runs(manifest)
    pairs, summaries = paired_results(rows)
    errors, cases = class_errors(evidence), failure_cases(evidence)
    result = {"generated_at_utc": utc_now(), "split": "val", "test_evaluated": False,
              "conditions": summaries, "paired_differences": pairs,
              "paired_log_loss": distribution([p["log_loss_difference"] for p in pairs]),
              "baseline": reference["baseline"], "training_environment": reference["environment"],
              "total_training_seconds": sum(r["training_seconds"] for r in rows),
              "runner_wall_seconds": manifest["wall_seconds"],
              "interpretation": interpretation(pairs)}
    write_csv(output / "individual_runs.csv", rows)
    write_csv(output / "paired_differences.csv", pairs)
    write_csv(output / "class_errors.csv", errors)
    write_csv(output / "failure_cases.csv", cases)
    write_json(output / "summary.json", result)
    figures(output, rows, pairs, evidence, reference["baseline"])
    (output / "REPORT.md").write_text(report_text(rows, pairs, summaries, errors, cases, result), encoding="utf-8")
    checks = {"verified_at_utc": utc_now(), "passed": True, "test_evaluated": False,
              "checks": {"complete_unique_seed_grid": True, "checkpoint_reload_and_artifact_verification": True,
                         "matched_source_data_environment_and_settings": True,
                         "identical_validation_fixtures": True, "independent_metrics": True},
              "runs": manifest["runs"], "analysis_source_sha256": sha256(Path(__file__)),
              "analysis_dependencies": {name: importlib.metadata.version(name) for name in (
                  "matplotlib", "numpy", "contourpy", "cycler", "fonttools", "kiwisolver",
                  "packaging", "pillow", "pyparsing", "python-dateutil", "six")}}
    write_json(output / "checks.json", checks)
    print(f"Verified six-run validation report: {output / 'REPORT.md'}", flush=True)
    print(json.dumps({"conditions": summaries, "paired_log_loss": result["paired_log_loss"]}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "report"))
    parser.add_argument("--output", type=Path, default=ROOT / "research" / "revised_validation")
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / "research") or output == ROOT / "research":
        raise ValueError("Output must be a study directory within research/")
    {"run": run_study, "report": report_study}[args.command](output)


if __name__ == "__main__":
    main()
