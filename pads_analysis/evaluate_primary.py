"""Compare held-out adaptive and optimized fixed protocols on the full cohort.

Required inputs are genuine out-of-fold predictions and a separately frozen
cohort/split manifest. This program does not generate or fit predictions.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from acquisition_rules import full_cohort_output_summary

METHODS = ("adaptive", "fixed_optimized")


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def align_predictions(cohort, predictions):
    """Require the same independently specified people, labels and folds."""
    expected = {}
    for row in cohort:
        subject = str(row["subject_id"])
        if subject in expected:
            raise ValueError("Duplicate participant in frozen cohort manifest")
        expected[subject] = (int(row["y"]), int(row["outer_fold"]))
    if not expected or set(y for y, _ in expected.values()) != {0, 1}:
        raise ValueError("The frozen cohort must contain both diagnostic classes")
    grouped = {method: {} for method in METHODS}
    for row in predictions:
        if row["method"] not in METHODS:
            continue
        subject, method = str(row["subject_id"]), row["method"]
        if subject in grouped[method]:
            raise ValueError("Multiple predictions per participant and method; analyse each repeat separately")
        if subject not in expected or (int(row["y"]), int(row["outer_fold"])) != expected[subject]:
            raise ValueError("Prediction label, fold or participant differs from frozen manifest")
        grouped[method][subject] = row
    if any(set(grouped[m]) != set(expected) for m in METHODS):
        raise ValueError("A method omits or adds participants relative to the frozen cohort")
    ids = sorted(expected)
    y = np.array([expected[s][0] for s in ids])
    arrays = {}
    bases = set()
    for method in METHODS:
        rows = [grouped[method][subject] for subject in ids]
        binary = np.array([int(row["binary_prediction"]) for row in rows])
        actual = np.array([int(row["actual_output"]) for row in rows])
        seconds = np.array([float(row["movement_seconds"]) for row in rows])
        complete = np.array([float(row["full_movement_seconds"]) for row in rows])
        bases.update(row["cost_basis"] for row in rows)
        if not np.isfinite(seconds).all() or not np.isfinite(complete).all() or (seconds <= 0).any() or (complete <= 0).any():
            raise ValueError("Every evaluated participant must have valid positive movement times")
        if (seconds > .5 * complete + 1e-9).any():
            raise ValueError("An abbreviated protocol exceeds a participant's 50% movement budget")
        full_cohort_output_summary(y, binary, actual)  # Check output encodings and consistency.
        arrays[method] = {"binary": binary, "actual": actual, "seconds": seconds, "full_seconds": complete}
    if len(bases) != 1 or not bases.issubset({"protocol_defined", "timestamp_estimated"}):
        raise ValueError("All compared times must use one explicitly named accounting basis")
    if not np.allclose(arrays[METHODS[0]]["full_seconds"], arrays[METHODS[1]]["full_seconds"], rtol=0, atol=1e-9):
        raise ValueError("Methods use different full-protocol reference durations")
    return ids, y, arrays, next(iter(bases))


def compare(cohort, predictions, bootstrap_draws=5000, seed=20260913):
    ids, y, arrays, cost_basis = align_predictions(cohort, predictions)
    if bootstrap_draws < 100:
        raise ValueError("Use at least 100 resamples; the study default is 5000")
    rng = np.random.default_rng(seed)
    indices = {label: np.flatnonzero(y == label) for label in [0, 1]}
    draws = {label: index[rng.integers(0, len(index), size=(bootstrap_draws, len(index)))]
             for label, index in indices.items()}
    estimates, bootstrap = {}, {}
    for method in METHODS:
        data = arrays[method]
        correct = data["binary"] == y
        sensitivity = correct[draws[1]].mean(axis=1)
        specificity = correct[draws[0]].mean(axis=1)
        clear = data["actual"] != -1
        coverage = sum(clear[draws[label]].mean(axis=1) * len(indices[label]) / len(y) for label in [0, 1])
        bootstrap[method] = {"balanced_accuracy": .5 * (sensitivity + specificity),
                             "sensitivity": sensitivity, "specificity": specificity, "coverage": coverage}
        estimate = full_cohort_output_summary(y, data["binary"], data["actual"])
        estimate["confidence_intervals_95"] = {
            key: np.quantile(value, [.025, .975]).tolist() for key, value in bootstrap[method].items()}
        estimate["movement_seconds"] = {"mean": float(data["seconds"].mean()),
                                        "median": float(np.median(data["seconds"])),
                                        "max_fraction_of_full": float(np.max(data["seconds"] / data["full_seconds"]))}
        estimate["coverage_target"] = .8
        estimate["observed_coverage_minus_target"] = estimate["coverage"] - .8
        estimates[method] = estimate
    paired = {}
    point_keys = {"balanced_accuracy": "underlying_binary_balanced_accuracy",
                  "sensitivity": "underlying_binary_sensitivity", "specificity": "underlying_binary_specificity",
                  "coverage": "coverage"}
    for key, point_key in point_keys.items():
        difference = bootstrap["adaptive"][key] - bootstrap["fixed_optimized"][key]
        paired[key] = {"adaptive_minus_fixed": estimates["adaptive"][point_key] - estimates["fixed_optimized"][point_key],
                       "confidence_interval_95": np.quantile(difference, [.025, .975]).tolist()}
    return {"participants": len(ids), "class_counts": {"PD": int((y == 1).sum()), "differential_diagnosis": int((y == 0).sum())},
            "primary_contrast": "Adaptive minus optimized fixed, full-cohort underlying binary balanced accuracy",
            "cost_basis": cost_basis, "bootstrap_draws": bootstrap_draws, "bootstrap_seed": seed,
            "interval_boundary": "Class-stratified paired participant resampling conditional on the supplied held-out predictions; excludes full model-retraining uncertainty.",
            "methods": estimates, "paired_differences": paired,
            "interpretation": "Report the effect and interval together with class-specific errors, indeterminate outputs and actual cost. An interval favoring one method is conditional internal-validation evidence, not prospective clinical validation.",
            "input_boundary": "Matching manifests and budgets do not by themselves prove that upstream model fitting was free from leakage; retain fitted-split and acquisition logs."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("primary_comparison.json"))
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    args = parser.parse_args()
    results = compare(read_csv(args.cohort), read_csv(args.predictions), args.bootstrap_draws)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False))
    print(f"Saved paired full-cohort evaluation to {args.output}")
