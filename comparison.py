"""Run and report the fixed Week 6 comparison, using validation evidence only."""

import argparse
import csv
import importlib.metadata
import json
import math
import statistics
import subprocess
import sys
import time
import uuid
import zipfile
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
    fig.suptitle("Week 6: positional encoding comparison on 380 validation fixtures")
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
    baseline, env = result["baseline"], result["training_environment"]
    text = ["# Week 6 — results checkpoint", "", "Team: Trivial  ",
            "Dara Heaphy — 23369914  ", "Tiernan Scully — 23365528", "",
            "## Question and fixed experiment", "", PROTOCOL["question"], "",
            "We compared no positional encoding with fixed sinusoidal encoding using seeds 42, 43 and 44 for both conditions. "
            "All six runs were trained fresh for this checkpoint. The seed set and protocol were saved before training. "
            "Each used ten epochs, Adam at 0.001, batch size 32, dropout 0.1, ten-match histories, CPU and one PyTorch thread. "
            "The shared one-layer transformer has 8,867 trainable parameters in both conditions. "
            "Only positional encoding changes within each seed pair.", "",
            "The chronological splits contain 6,912 training, 380 validation and 357 reserved test examples. "
            "The lowest validation log loss selects each run's checkpoint. Results below are validation results used for "
            "development and checkpoint selection; final held-out evaluation has not been performed. "
            "There is no ensemble or best-seed selection.", "",
            "## Individual results", "", "Log loss uses natural logarithms; lower is better. Accuracy is secondary. "
            "Table log loss is independently recalculated from exported probabilities; tiny rounding differences from "
            "PyTorch cross-entropy are covered by the run verifier's tolerance.", "",
            "| Condition | Seed | Selected epoch | Validation log loss | Accuracy | Training seconds |",
            "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        text.append(f"| {row['condition']} | {row['seed']} | {row['selected_epoch']} | {row['validation_log_loss']:.6f} | {100*row['validation_accuracy']:.2f}% | {row['training_seconds']:.2f} |")
    text += [f"| Training-frequency constant baseline | — | — | {baseline['validation_log_loss']:.6f} | {100*baseline['validation_accuracy']:.2f}% | — |", "",
             "The constant predictor is fitted only to training class frequencies. It is a sanity baseline; "
             "the no-position transformer is the reference for the positional-information question.", "",
             "## Variation and paired comparison", "",
             "| Condition | Mean log loss | Sample SD | Range | Mean accuracy | Accuracy SD (percentage points) |",
             "| --- | ---: | ---: | --- | ---: | ---: |"]
    for condition in CONDITIONS:
        loss, accuracy = summaries[condition]["log_loss"], summaries[condition]["accuracy"]
        text.append(f"| {condition} | {loss['mean']:.6f} | {loss['sample_sd']:.6f} | {loss['min']:.6f}–{loss['max']:.6f} | {100*accuracy['mean']:.2f}% | {100*accuracy['sample_sd']:.2f} |")
    text += ["", "Sample SD uses ddof=1 across three training seeds. It measures training variation on the same "
             "fixtures, not uncertainty across future seasons.", "",
             "| Seed | Log-loss difference: sinusoidal − none | Accuracy difference (percentage points) |",
             "| --- | ---: | ---: |"]
    for pair in pairs:
        text.append(f"| {pair['seed']} | {pair['log_loss_difference']:+.6f} | {pair['accuracy_difference_pp']:+.2f} |")
    delta = result["paired_log_loss"]
    text += ["", f"Mean paired log-loss difference: **{delta['mean']:+.6f}**, sample SD **{delta['sample_sd']:.6f}**. "
             "Negative favours sinusoidal encoding. Three runs do not establish statistical significance.", "",
             "![Every run and its paired difference](paired_log_loss.png)", "",
             "## Current interpretation", "", interpretation(pairs), "",
             "Both conditions beat the constant baseline in every run." if all(r["validation_log_loss"] < baseline["validation_log_loss"] for r in rows)
             else "Performance against the constant baseline varies across runs; inspect the full table.", "",
             "The experiment tests whether this particular sinusoidal representation helps this small model. "
             "It does not directly test whether football recency is useful in every model. "
             "The no-position model ignores history order in evaluation; adding position allows order to matter "
             "but does not guarantee the model learns a useful recency rule.", "",
             "## Anomalies and failure cases", "",
             "![Validation learning curves](learning_curves.png)", "",
             "Selected epochs: " + "; ".join(f"{r['condition']} seed {r['seed']}: {r['selected_epoch']}" for r in rows) + ". "
             "The curves show whether later training improves validation loss. An early best checkpoint is an observation; "
             "it alone does not prove a particular cause such as overfitting.", "",
             "### Draw predictions", "",
             "| Condition | Seed | Actual draws | Predicted draws | Correct draws | Draw recall | Mean probability assigned to actual draws |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in errors:
        if row["class"] == "draw":
            text.append(f"| {row['condition']} | {row['seed']} | {row['actual_count']} | {row['predicted_count']} | {row['correct_count']} | {100*row['recall']:.2f}% | {row['mean_true_class_probability']:.4f} |")
    text += ["", "Draw predictions above mean the highest-probability class, not zero probability of a draw. "
             "Accuracy can obscure poor class recall; log loss still penalises the probability assigned to actual draws. "
             "Full class counts, recalls and class log losses are in `class_errors.csv`; each run retains its confusion matrix.", "",
             "### Examples with the largest average log loss", "",
             "These are validation fixtures ranked by mean individual-model log loss across all six selected checkpoints. "
             "They are selected examples, not an unbiased estimate of overall performance. Condition-level values are "
             "averages across its three seeds, not ensemble metrics.", "",
             "| Fixture | Actual outcome | none: mean true-outcome probability | sinusoidal: mean true-outcome probability | Wrong seeds (none / sinusoidal) |",
             "| --- | --- | ---: | ---: | --- |"]
    for case in cases[:5]:
        text.append(f"| {case['home']} vs {case['away']} ({case['fixture_id']}) | {case['label_name']} | {case['none_mean_true_probability']:.4f} | {case['sinusoidal_mean_true_probability']:.4f} | {case['none_wrong_seeds']} / {case['sinusoidal_wrong_seeds']} |")
    text += ["", "The ten highest-loss cases, dates and condition-level losses are in `failure_cases.csv`. "
             "Inspect the underlying histories before attributing an error to a football-specific cause.", "",
             "## Limitations and final evaluation", "",
             "- Only three seeds and one validation season are available. Fixtures share teams and overlapping histories, "
             "so they are not independent observations.",
             "- Each checkpoint is selected on this validation season; the reported validation results can be optimistic.",
             "- Inputs contain goals scored, goals conceded and home status, but no team or opponent identity, injuries, "
             "squad changes or time gaps. Ten positions need not represent equal intervals of time.",
             "- The small model, raw unscaled features and ten-epoch budget limit the conclusion to this setup.",
             "- The final season has 367 unique completed fixtures and only 357 usable targets, not a full season.",
             "- Future performance and betting value are not established by this experiment.", "",
             "For Week 7, freeze the experiment and agree a final evaluation protocol before opening the test results. "
             "Retain all seeds; do not choose the best seed. The existing training and reporting commands produce no test "
             "predictions or metrics. The checkpoint's final held-out evaluation remains pending.", "",
             "## Compute and verification", "",
             f"Six ten-epoch CPU runs, 8,867 parameters each. Total training/validation-loop time: "
             f"{result['total_training_seconds']:.2f} seconds. Runner wall time, including process startup and "
             f"verification before reporting: {result['runner_wall_seconds']:.2f} seconds. "
             "Training timings exclude preprocessing and evidence export. No GPU or paid compute was used.", "",
             f"Recorded processor: `{env['processor']}`; {env['logical_cpu_count']} logical CPUs, one PyTorch thread. "
             f"Python {env['python']}, PyTorch {env['torch_build']}, {env['platform']}.", "",
             f"An earlier packaging audit retained {result['packaging_audit']['runs']} additional full training runs "
             f"({result['packaging_audit']['training_seconds']:.2f} training-loop seconds). These used the same seeds "
             "and were excluded because the original source snapshot omitted the new comparison script imported by tests. "
             "After correcting the packager, all six main runs were repeated. This was a portability correction, "
             "not selection based on results. See `research/week6_packaging_audit/NOTE.md`.", "",
             "All six checkpoints were reloaded and verified against their saved predictions. Independent Python metric "
             "calculations matched the training metrics within the recorded tolerances. The comparison also checked "
             "identical controlled settings, parameter counts, source/data hashes, environments, fixture order and baseline. "
             "See `checks.json` and each run's `verification.json`. Existing Week 5 repeats of seed 42 are not counted "
             "as independent runs in this study.", "",
             "## Reproduction and evidence", "",
             "In PowerShell, from `C:\\Users\\User\\repos\\AdvancedAIProject`:", "", "```powershell",
             "python -m pip install -r requirements-analysis.txt", "python -m unittest discover -s tests -v",
             "python comparison.py report --output research/week6", "```", "",
             "To train a fresh complete comparison without replacing this evidence:", "", "```powershell",
             "python comparison.py run --output research/week6_reproduction", "```", "",
             "A resumed `run` uses only its manifest's completed, verified runs. Failed or unlisted runs are not "
             "silently substituted. A report rejects incomplete or mismatched conditions. "
             "`python comparison.py bundle --output research/week6` creates an allowlisted ZIP including all six "
             "checkpoints, source snapshots, input CSV, reports, figures and SHA-256 manifest. Checkpoints and source "
             "snapshots remain ignored by Git, so retain the bundle or regenerate after cloning.", "",
             "Run directories:", ""]
    text += [f"- `{r['condition']}`, seed {r['seed']}: `{r['run_dir']}`" for r in rows]
    text += ["", "## External tools, provenance and team responsibility", "",
             "Codex helped implement the comparison runner, aggregate the saved evidence, generate figures and draft "
             "this checkpoint. Its output was checked using unit tests, checkpoint reloads, independent metric "
             "calculations, controlled-run checks and visual inspection of the figures. "
             "The earlier report cited a run folder absent from this local repository; this report uses the actual "
             "manifest paths. Both team members must review the interpretation and be able to explain the code.", "",
             "Dataset source and licence are described in `dataset/README.md`. Its CC0 statement is carried over "
             "from the proposal and has not been independently confirmed by this checkpoint work. Confirm the "
             "source terms before final submission. No pretrained model is used.", "",
             "The final contribution statement needs confirmation from Dara and Tiernan; generated work is not evidence "
             "of either person's individual contribution.", ""]
    return "\n".join(text)


def report_study(output):
    manifest = read_json(output / "experiment.json")
    rows, evidence, reference = collect_runs(manifest)
    pairs, summaries = paired_results(rows)
    errors, cases = class_errors(evidence), failure_cases(evidence)
    audit_manifest = ROOT / "research" / "week6_packaging_audit" / "experiment.json"
    audit_runs = read_json(audit_manifest)["runs"] if audit_manifest.exists() else []
    result = {"generated_at_utc": utc_now(), "split": "val", "test_evaluated": False,
              "conditions": summaries, "paired_differences": pairs,
              "paired_log_loss": distribution([p["log_loss_difference"] for p in pairs]),
              "baseline": reference["baseline"], "training_environment": reference["environment"],
              "total_training_seconds": sum(r["training_seconds"] for r in rows),
              "runner_wall_seconds": manifest["wall_seconds"],
              "packaging_audit": {"runs": len(audit_runs), "training_seconds": sum(
                  read_json(safe_run(entry["run_dir"]) / "summary.json")["training_seconds"] for entry in audit_runs)},
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
    print(f"Verified six-run Week 6 report: {output / 'REPORT.md'}", flush=True)
    print(json.dumps({"conditions": summaries, "paired_log_loss": result["paired_log_loss"]}, indent=2))


def bundle_study(output):
    report_study(output)
    manifest = read_json(output / "experiment.json")
    members = {}
    source_names = ["comparison.py", "requirements-analysis.txt"]
    source_names += [p.relative_to(ROOT).as_posix() for p in (ROOT / "tests").glob("test_*.py")]
    first = safe_run(manifest["runs"][0]["run_dir"])
    source_names += list(read_json(first / "provenance.json")["source_files_sha256"])
    source_names += ["dataset/matches.csv"]
    for name in set(source_names):
        members[name] = ROOT / name
    allowed = {"config.json", "environment.json", "data_report.json", "baseline.json", "metrics.json",
               "summary.json", "validation_prediction.json", "validation_predictions.json", "validation_metrics.json",
               "provenance.json", "requirements-lock.txt", "verification.json", "best.pt"}
    for entry in manifest["runs"]:
        run = safe_run(entry["run_dir"])
        for name in allowed:
            members[f"{entry['run_dir']}/{name}"] = run / name
        for name in read_json(run / "provenance.json")["source_files_sha256"]:
            members[f"{entry['run_dir']}/source/{name}"] = run / "source" / name
    study_files = {"experiment.json", "individual_runs.csv", "paired_differences.csv", "class_errors.csv",
                   "failure_cases.csv", "summary.json", "checks.json", "REPORT.md",
                   "paired_log_loss.png", "paired_log_loss.svg", "learning_curves.png", "learning_curves.svg"}
    study_files.update(f"training_{c}_seed{s}.log" for c in CONDITIONS for s in SEEDS)
    for name in study_files:
        members[f"research/week6/{name}"] = output / name
    audit = ROOT / "research" / "week6_packaging_audit"
    if (audit / "experiment.json").exists():
        for name in ("experiment.json", "NOTE.md"):
            members[f"research/week6_packaging_audit/{name}"] = audit / name
        # Audit metadata accounts for extra compute; main runs retain full evidence.
        for entry in read_json(audit / "experiment.json")["runs"]:
            members[f"{entry['run_dir']}/summary.json"] = safe_run(entry["run_dir"]) / "summary.json"
    destination = ROOT / "evidence_bundles" / f"week6_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{uuid.uuid4().hex[:8]}.zip"
    destination.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        hashes = {}
        for name, path in sorted(members.items()):
            archive.write(path, name)
            hashes[name] = sha256(path)
        instructions = ("# Week 6 comparison evidence\n\nExtract the ZIP and enter its root in PowerShell.\n\n"
                        "```powershell\npython -m pip install -r requirements-lock.txt\n"
                        "python -m pip install -r requirements-analysis.txt\n"
                        "python -m unittest discover -s tests -v\n"
                        "python comparison.py report --output research/week6\n```\n\n"
                        "Expected: tests pass and the six-run report verifies. No test metrics are produced.\n"
                        "The original Python/platform/build are recorded per run. Other builds may differ numerically.\n")
        archive.writestr("WEEK6_README.md", instructions)
        import hashlib
        hashes["WEEK6_README.md"] = hashlib.sha256(instructions.encode()).hexdigest()
        archive.writestr("bundle_manifest.json", json.dumps({"created_at_utc": utc_now(),
                         "test_evaluated": False, "files_sha256": hashes}, indent=2) + "\n")
    print(f"Week 6 evidence bundle: {destination}")
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "report", "bundle"))
    parser.add_argument("--output", type=Path, default=ROOT / "research" / "week6")
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / "research") or output == ROOT / "research":
        raise ValueError("Output must be a study directory within research/")
    {"run": run_study, "report": report_study, "bundle": bundle_study}[args.command](output)


if __name__ == "__main__":
    main()
