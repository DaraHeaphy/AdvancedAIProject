"""Verify validation-only runs and package a portable pipeline demonstration."""

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import pickle
import shutil
import subprocess
import sys
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import torch
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from torch.utils.data import DataLoader

from dataset import DEFAULT_CSV, LABELS, load_datasets


ROOT = Path(__file__).resolve().parent
RTOL = 1e-6
ATOL = 1e-7
RUN_FILES = (
    "config.json", "environment.json", "data_report.json", "baseline.json",
    "metrics.json", "summary.json", "validation_prediction.json",
    "validation_predictions.json", "validation_metrics.json", "provenance.json",
    "requirements-lock.txt", "verification.json",
)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dependency_versions():
    """Pin the installed dependency closure, excluding inactive optional extras."""
    versions, processed = {}, {}
    pending = [("torch", set()), ("packaging", set())]
    while pending:
        requested_name, extras = pending.pop()
        name = canonicalize_name(requested_name)
        if name in processed and extras.issubset(processed[name]):
            continue
        extras = extras | processed.get(name, set())
        processed[name] = extras
        distribution = importlib.metadata.distribution(name)
        versions[name] = distribution.version
        for text in distribution.requires or []:
            requirement = Requirement(text)
            if requirement.marker is None or any(
                    requirement.marker.evaluate({"extra": extra}) for extra in {"", *extras}):
                pending.append((requirement.name, set(requirement.extras)))
    return dict(sorted(versions.items()))


def lock_text():
    versions = dependency_versions()
    return (
        f"# Recorded on Python {platform.python_version()}, {platform.system()} {platform.machine()}.\n"
        "# Package versions are pinned; the exact torch build is in environment.json.\n"
        + "".join(f"{name}=={version}\n" for name, version in versions.items())
    )


def portable_path(directory, relative):
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts or "\\" in relative or ":" in relative:
        raise ValueError(f"Unsafe evidence path: {relative}")
    resolved = (directory / relative).resolve()
    if not resolved.is_relative_to(directory.resolve()):
        raise ValueError(f"Evidence path escapes its directory: {relative}")
    return resolved


def record_provenance(run_dir, command):
    files = ["dataset.py", "model.py", "train.py", "evidence.py",
             "requirements.txt", "requirements-lock.txt", "README.md", "dataset/README.md"]
    files += [path.relative_to(ROOT).as_posix() for path in sorted((ROOT / "tests").glob("test_*.py"))]
    hashes = {}
    for relative in files:
        source = portable_path(ROOT, relative)
        destination = portable_path(run_dir / "source", relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        hashes[relative] = sha256(destination)
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                capture_output=True, text=True, check=True).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                                capture_output=True, text=True, check=True).stdout
        git = {"commit": commit, "dirty": bool(status.strip())}
    except (OSError, subprocess.CalledProcessError):
        git = {"commit": None, "dirty": None, "reason": "Git metadata unavailable"}
    write_json(run_dir / "provenance.json", {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_command": command, "git": git,
        "source_files_sha256": hashes, "source_snapshot": "source",
        "note": "Source hashes identify the executed files even when Git is dirty or absent.",
    })
    (run_dir / "requirements-lock.txt").write_text(lock_text(), encoding="utf-8")


def validation_predictions(model, val, device, batch_size=32):
    """Export every validation fixture in order, using double-precision softmax."""
    rows = []
    model.eval()
    with torch.no_grad():
        for home, away, labels in DataLoader(val, batch_size=batch_size, shuffle=False):
            probabilities = model(home.to(device), away.to(device)).double().softmax(dim=1).cpu()
            for probability, label in zip(probabilities.tolist(), labels.tolist()):
                index = len(rows)
                target = val.metadata[index].target
                rows.append({
                    "split": "val", "index": index, "fixture_id": target.fixture_id,
                    "home": target.home, "away": target.away,
                    "kickoff": target.kickoff.isoformat(), "label": label,
                    "label_name": LABELS[label],
                    "probabilities": dict(zip(LABELS, probability)),
                })
    return rows


def metrics_from_predictions(rows):
    """Calculate metrics with Python maths only, independently of torch losses."""
    if not rows:
        raise ValueError("Validation predictions are empty")
    losses, brier = [], []
    confusion = [[0] * 3 for _ in range(3)]
    identifiers = set()
    for index, row in enumerate(rows):
        if row["split"] != "val" or row["index"] != index:
            raise ValueError("Predictions must be ordered validation fixtures")
        if row["fixture_id"] in identifiers:
            raise ValueError("Duplicate prediction fixture")
        identifiers.add(row["fixture_id"])
        label = row["label"]
        if type(label) is not int or not 0 <= label < 3 or row["label_name"] != LABELS[label]:
            raise ValueError("Invalid prediction label")
        probabilities = [row["probabilities"][name] for name in LABELS]
        if set(row["probabilities"]) != set(LABELS):
            raise ValueError("Invalid probability classes")
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in probabilities):
            raise ValueError("Invalid probability value")
        if not math.isclose(math.fsum(probabilities), 1.0, abs_tol=1e-9, rel_tol=0):
            raise ValueError("Probabilities do not sum to one")
        if probabilities[label] <= 0:
            raise ValueError("True-label probability must be positive for finite log loss")
        predicted = max(range(3), key=probabilities.__getitem__)
        confusion[label][predicted] += 1
        losses.append(-math.log(probabilities[label]))
        brier.append(math.fsum((value - int(i == label)) ** 2 for i, value in enumerate(probabilities)))
    return {
        "split": "val", "examples": len(rows), "log_loss": math.fsum(losses) / len(rows),
        "accuracy": sum(confusion[i][i] for i in range(3)) / len(rows),
        "brier_score": math.fsum(brier) / len(rows), "class_order": LABELS,
        "confusion_matrix": confusion, "confusion_matrix_axes": "rows=true, columns=predicted",
        "brier_definition": "mean sum of squared errors across the three classes",
        "calculation": "Python math from exported probabilities; natural logarithm",
    }


def verify_run(run_dir, device="cpu", datasets=None, command=None):
    """Check saved artifacts without training or evaluating held-out examples."""
    from train import constant_baseline, evaluate, load_checkpoint, validation_prediction

    checks, details = {}, {}

    def check(name, condition):
        checks[name] = bool(condition)

    def close(left, right):
        return math.isclose(left, right, rel_tol=RTOL, abs_tol=ATOL)

    try:
        config = read_json(run_dir / "config.json")
        summary = read_json(run_dir / "summary.json")
        epochs = read_json(run_dir / "metrics.json")
        report = read_json(run_dir / "data_report.json")
        provenance = read_json(run_dir / "provenance.json")
        hashes = provenance["source_files_sha256"]
        check("source_snapshot_hashes", bool(hashes) and all(
            sha256(portable_path(run_dir / "source", name)) == digest for name, digest in hashes.items()))
        check("executed_source_matches_snapshot", all(
            sha256(portable_path(ROOT, name)) == digest for name, digest in hashes.items()))
        check("source_csv_hash", sha256(DEFAULT_CSV) == report["source_sha256"])
        check("all_epochs_recorded", [row["epoch"] for row in epochs] == list(range(1, config["epochs"] + 1)))
        check("finite_epoch_metrics", all(math.isfinite(row[key]) for row in epochs for key in
                                         ("training_loss", "validation_log_loss", "validation_accuracy")))
        best = min(epochs, key=lambda row: row["validation_log_loss"])
        model, checkpoint = load_checkpoint(run_dir / "best.pt", device)
        check("checkpoint_configuration", checkpoint["run_config"] == config and model.config == config["model"])
        check("checkpoint_selection", best["epoch"] == summary["best_epoch"] == checkpoint["epoch"]
              and close(best["validation_log_loss"], summary["best_validation_log_loss"])
              and close(best["validation_log_loss"], checkpoint["validation_log_loss"])
              and close(best["validation_accuracy"], summary["best_validation_accuracy"]))
        data = datasets if datasets is not None else load_datasets()
        val = data["val"]
        check("example_counts", len(val) == summary["validation_examples"]
              and len(data["train"]) == summary["training_examples"])
        actual = evaluate(model, DataLoader(val, batch_size=config["batch_size"], shuffle=False), device)
        check("checkpoint_metrics_reproduced", all(
            close(actual[key], summary["reloaded_validation_metrics"][key])
            and close(actual[key], best["validation_" + key]) for key in ("log_loss", "accuracy")))
        check("single_prediction_reproduced", validation_prediction(model, val, 0, device)
              == read_json(run_dir / "validation_prediction.json"))
        rows = read_json(run_dir / "validation_predictions.json")
        regenerated = validation_predictions(model, val, device, config["batch_size"])
        check("all_validation_fixtures_exported", len(rows) == len(val))
        check("validation_predictions_reproduced", len(rows) == len(regenerated) and all(
            {key: value for key, value in row.items() if key != "probabilities"}
            == {key: value for key, value in fresh.items() if key != "probabilities"}
            and all(close(row["probabilities"][label], fresh["probabilities"][label]) for label in LABELS)
            for row, fresh in zip(rows, regenerated)))
        independent = metrics_from_predictions(rows)
        check("exported_metrics_recalculated", independent == read_json(run_dir / "validation_metrics.json"))
        check("independent_metrics_match_training", all(close(independent[key], actual[key])
                                                       for key in ("log_loss", "accuracy")))
        baseline = constant_baseline(data["train"].labels, val.labels)
        check("training_only_baseline_reproduced", baseline == read_json(run_dir / "baseline.json"))
        counts = Counter(data["train"].labels.tolist())
        baseline_loss = math.fsum(-math.log(counts[row["label"]] / len(data["train"])) for row in rows) / len(rows)
        check("baseline_log_loss_independently_recalculated", close(baseline_loss, baseline["validation_log_loss"]))
        check("test_not_evaluated", summary["test_evaluated"] is False)
        environment = read_json(run_dir / "environment.json")
        expected_lock = {line.split("==", 1)[0]: line.split("==", 1)[1]
                         for line in (run_dir / "requirements-lock.txt").read_text().splitlines()
                         if line and not line.startswith("#")}
        check("pinned_environment_matches_record", expected_lock == environment["dependencies"])
        details["verification_environment"] = {
            "python": platform.python_version(), "torch_build": str(torch.__version__),
            "dependencies": dependency_versions(), "device": str(device),
        }
        details["recorded_environment_matches_current"] = (
            environment["python"] == platform.python_version()
            and environment["torch_build"] == str(torch.__version__)
            and expected_lock == details["verification_environment"]["dependencies"])
        if (run_dir / "unit_tests.json").exists():
            tests = read_json(run_dir / "unit_tests.json")
            check("unit_tests_passed", tests["returncode"] == 0 and tests["tests_run"] > 0)
            details["unit_tests_run"] = tests["tests_run"]
        details.update({"validation_examples": len(rows), "independent_validation_metrics": independent,
                        "checkpoint_sha256": sha256(run_dir / "best.pt")})
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, IndexError,
            pickle.UnpicklingError, EOFError,
            importlib.metadata.PackageNotFoundError) as error:
        checks["artifacts_readable_and_valid"] = False
        details["error"] = f"{type(error).__name__}: {error}"
    result = {
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": command or ["python", "evidence.py", "verify", "--run-dir", f"runs/{run_dir.name}"],
        "passed": bool(checks) and all(checks.values()), "checks": checks,
        "tolerances": {"relative": RTOL, "absolute": ATOL},
        "test_evaluated": False, **details,
    }
    write_json(run_dir / "verification.json", result)
    if not result["passed"]:
        failed = [name for name, passed in checks.items() if not passed]
        raise ValueError(f"Run verification failed: {', '.join(failed)}; {details.get('error', '')}")
    return result


def bundle_run(run_dir, output, device="cpu"):
    """Create an allowlisted ZIP with checkpoint, input CSV and source snapshot."""
    verify_run(run_dir, device)
    provenance = read_json(run_dir / "provenance.json")
    members = {name: portable_path(run_dir / "source", name)
               for name in provenance["source_files_sha256"]}
    members["dataset/matches.csv"] = DEFAULT_CSV
    for name in RUN_FILES:
        members[f"runs/{run_dir.name}/{name}"] = run_dir / name
    if (run_dir / "unit_tests.json").exists():
        members[f"runs/{run_dir.name}/unit_tests.json"] = run_dir / "unit_tests.json"
    members[f"runs/{run_dir.name}/best.pt"] = run_dir / "best.pt"
    # Retain a snapshot for verification within the extracted project as well.
    for name in provenance["source_files_sha256"]:
        members[f"runs/{run_dir.name}/source/{name}"] = portable_path(run_dir / "source", name)
    instructions = (
        "# Portable pipeline evidence\n\n"
        "Extract the ZIP. In PowerShell, enter the extracted folder containing train.py.\n"
        f"Use Python {read_json(run_dir / 'environment.json')['python']} and the recorded platform for closest reproduction.\n\n"
        "```powershell\npython -m venv .venv\n"
        f".\\.venv\\Scripts\\python.exe -m pip install -r runs/{run_dir.name}/requirements-lock.txt\n"
        ".\\.venv\\Scripts\\python.exe -m unittest discover -s tests -v\n"
        f".\\.venv\\Scripts\\python.exe evidence.py verify --run-dir runs/{run_dir.name}\n"
        f".\\.venv\\Scripts\\python.exe train.py predict --checkpoint runs/{run_dir.name}/best.pt --index 0\n"
        "```\n\nExpected: tests pass and verification reports passed=true.\n"
        "The lock pins installed package versions; environment.json also records the torch build.\n"
        "Other platforms/builds can differ numerically; verification records the current environment.\n"
        "bundle_manifest.json lists SHA-256 hashes for all other ZIP members.\n"
        "The reference condition is none. This evidence uses validation only.\n"
    ).encode("utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        hashes = {}
        for name, path in sorted(members.items()):
            data = path.read_bytes()
            archive.writestr(name, data)
            hashes[name] = hashlib.sha256(data).hexdigest()
        archive.writestr("EVIDENCE_README.md", instructions)
        hashes["EVIDENCE_README.md"] = hashlib.sha256(instructions).hexdigest()
        archive.writestr("bundle_manifest.json", json.dumps({
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "run": run_dir.name, "reference_condition": "none",
            "run_condition": read_json(run_dir / "config.json")["model"]["positional_encoding"],
            "test_evaluated": False, "files_sha256": hashes,
        }, indent=2) + "\n")
    return output


def demo(args):
    """One command: tests, full training, verification and a portable ZIP."""
    import re
    from train import train

    command = ["python", "-m", "unittest", "discover", "-s", "tests", "-v"]
    tests = subprocess.run([sys.executable, *command[1:]], cwd=ROOT, capture_output=True, text=True)
    log = tests.stdout + tests.stderr
    print(log, flush=True)
    if tests.returncode:
        raise RuntimeError("Unit tests failed; demonstration training was not started")
    count = re.search(r"Ran (\d+) tests?", log)
    run_dir = train(args)
    write_json(run_dir / "unit_tests.json", {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(), "command": command,
        "returncode": tests.returncode, "tests_run": int(count[1]) if count else 0, "output": log,
    })
    output = bundle_run(run_dir, ROOT / "evidence_bundles" / f"{run_dir.name}.zip", args.device)
    print(f"Verified run: {run_dir}\nEvidence bundle: {output}", flush=True)
    return run_dir, output


def main():
    from train import prepare_device

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    demonstration = commands.add_parser("demo", help="Run tests, train, verify and bundle")
    demonstration.add_argument("--epochs", type=int, default=10)
    demonstration.add_argument("--seed", type=int, default=42)
    demonstration.add_argument("--positional-encoding", choices=("none", "sinusoidal"), default="none")
    demonstration.add_argument("--runs-dir", type=Path, default=ROOT / "runs")
    verification = commands.add_parser("verify", help="Check an existing evidence-enabled run")
    bundling = commands.add_parser("bundle", help="Verify and package an existing run")
    for command in (verification, bundling):
        command.add_argument("--run-dir", type=Path, required=True)
    bundling.add_argument("--output", type=Path, required=True)
    for command in (demonstration, verification, bundling):
        command.add_argument("--device", default="cpu")
        command.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    device = prepare_device(args.device, args.threads)
    if args.command == "demo":
        demo(args)
    elif args.command == "verify":
        print(json.dumps(verify_run(args.run_dir, device), indent=2))
    else:
        print(f"Evidence bundle: {bundle_run(args.run_dir, args.output, device)}")


if __name__ == "__main__":
    main()
