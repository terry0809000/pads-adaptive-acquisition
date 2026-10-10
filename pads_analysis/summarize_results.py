"""Validate paired outputs and summarize the prespecified analyses."""

import csv
import json
import hashlib
from collections import Counter
from pathlib import Path

import numpy as np
import joblib
from sklearn.metrics import roc_auc_score

from acquisition_rules import full_cohort_output_summary
from evaluate_primary import compare
from run_analysis import TASKS, VALID, nominal_units, task_indices, csv_write, json_write

BASE = Path(__file__).resolve().parent
FEATURES = BASE.parent / "pads_features_received/PADS_features_return"
CORE = ["adaptive", "fixed_optimized", "fixed_brier", "fixed_sequence_stop", "rest_only", "clinical_only", "full_protocol"]


def read(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def gather(root):
    rows = []
    checks = []
    for fold in range(5):
        p = root / f"fold_{fold}"
        complete = json.loads((p / "COMPLETE.json").read_text())
        assert complete["n_test"] == 78 and complete["n_train"] == 312
        ledger = json.loads((p / "training_ledger.json").read_text())
        outer = set(ledger["outer_test_ids"])
        final = set(ledger["final_fit"]["training_ids"])
        assert len(outer) == 78 and len(final) == 312 and not outer & final
        for fit in [ledger["final_fit"]] + [m["fit"] for m in ledger["middle_fits"]]:
            fit_ids = set(fit["training_ids"])
            assert not fit_ids & outer
            inner_held = []
            for inner in fit["state_crossfit"]:
                t, h = set(inner["train_ids"]), set(inner["held_out_ids"])
                assert not t & h and t | h == fit_ids
                inner_held.extend(h)
            assert len(inner_held) == len(set(inner_held)) == len(fit_ids)
        middle_held = []
        for middle in ledger["middle_fits"]:
            t, h = set(middle["fit"]["training_ids"]), set(middle["threshold_holdout_ids"])
            assert not t & h and t | h == final
            middle_held.extend(h)
        assert len(middle_held) == len(set(middle_held)) == 312
        rows.extend(read(p / "held_out_predictions.csv"))
        checks.append({"fold": fold, "outer_and_nested_split_disjointness_passed": True,
                       "clinical_fits": 4 * 4 * 259, "elapsed_seconds": complete["elapsed_seconds"]})
    manifest = {r["subject_id"]: r for r in read(FEATURES / "primary_outer_folds.csv")}
    grouped = {}
    for row in rows:
        sid, method = row["subject_id"], row["method"]
        assert row["y"] == manifest[sid]["y"] and row["outer_fold"] == manifest[sid]["outer_fold"]
        group = grouped.setdefault(method, {})
        assert sid not in group
        mask = int(row["task_mask"])
        if method not in {"clinical_only", "full_protocol"}:
            budget = int(method.rsplit("_", 1)[1]) if "_budget_" in method else 7
            assert mask in VALID and nominal_units(mask) <= budget
        assert abs(float(row["movement_seconds"]) - nominal_units(mask) * 10.24) < 1e-9
        assert row["task_path"] == "|".join(TASKS[j] for j in task_indices(mask))
        group[sid] = row
    assert all(set(g) == set(manifest) for g in grouped.values())
    return rows, {m: [g[s] for s in sorted(g)] for m, g in grouped.items()}, checks


def arrays(rows):
    return {"y": np.array([int(r["y"]) for r in rows]),
            "binary": np.array([int(r["binary_prediction"]) for r in rows]),
            "actual": np.array([int(r["actual_output"]) for r in rows]),
            "score": np.array([float(r["score"]) for r in rows]),
            "seconds": np.array([float(r["movement_seconds"]) for r in rows]),
            "timestamp": np.array([float(r["timestamp_seconds"]) for r in rows]),
            "full_timestamp": np.array([float(r["full_timestamp_seconds"]) for r in rows]),
            "mask": np.array([int(r["task_mask"]) for r in rows])}


def resampling(y):
    rng = np.random.default_rng(20260913)
    return {label: (idx := np.flatnonzero(y == label))[rng.integers(0, len(idx), size=(5000, len(idx)))] for label in [0, 1]}


def summarize(grouped):
    y = arrays(next(iter(grouped.values())))["y"]
    draws = resampling(y)
    results, bootstrap = {}, {}
    for method, rows in grouped.items():
        a = arrays(rows)
        assert np.array_equal(y, a["y"])
        r = full_cohort_output_summary(y, a["binary"], a["actual"])
        correct, clear = a["binary"] == y, a["actual"] != -1
        bs = {"sensitivity": correct[draws[1]].mean(1), "specificity": correct[draws[0]].mean(1)}
        bs["balanced_accuracy"] = .5 * (bs["sensitivity"] + bs["specificity"])
        for name, value in [("coverage", clear), ("mean_seconds", a["seconds"]), ("mean_timestamp", a["timestamp"])]:
            bs[name] = sum(value[draws[k]].mean(1) * (y == k).mean() for k in [0, 1])
        r.update(mean_seconds=float(a["seconds"].mean()), mean_timestamp=float(a["timestamp"].mean()),
                 mean_nominal_fraction=float(a["seconds"].mean() / 143.36),
                 mean_tasks=float(np.mean([len(task_indices(int(m))) for m in a["mask"]])),
                 unique_task_paths=int(len(np.unique(a["mask"]))),
                 timestamp_over_half_count=int((a["timestamp"] > .5 * a["full_timestamp"] + 1e-9).sum()),
                 maximum_timestamp_fraction=float((a["timestamp"] / a["full_timestamp"]).max()),
                 maximum_timestamp_excess_seconds=float(np.maximum(a["timestamp"] - .5 * a["full_timestamp"], 0).max()),
                 auroc=float(roc_auc_score(y, a["score"])),
                 clear_case_accuracy=float(correct[clear].mean()) if clear.any() else None,
                 confidence_intervals_95={name: np.quantile(value, [.025, .975]).tolist() for name, value in bs.items()})
        results[method], bootstrap[method] = r, bs
    return results, bootstrap


def paired(results, bootstrap, first, second):
    names = {"balanced_accuracy": "underlying_binary_balanced_accuracy", "sensitivity": "underlying_binary_sensitivity",
             "specificity": "underlying_binary_specificity", "coverage": "coverage", "mean_seconds": "mean_seconds"}
    return {name: {"difference": results[first][key] - results[second][key],
                   "conditional_interval_95": np.quantile(bootstrap[first][name] - bootstrap[second][name], [.025, .975]).tolist()}
            for name, key in names.items()}


def main():
    destination = BASE / "summary"
    destination.mkdir(exist_ok=True)
    all_results, compact_rows, gap_rows, fold_rows, selected_rows = {}, [], [], [], []
    for analysis in ["primary", "without_nms"]:
        rows, grouped, checks = gather(BASE / f"results_{analysis}")
        csv_write(destination / f"held_out_predictions_{analysis}.csv", rows)
        result, bootstrap = summarize(grouped)
        comparison = compare(read(FEATURES / "primary_outer_folds.csv"), rows)
        assert abs(comparison["paired_differences"]["balanced_accuracy"]["adaptive_minus_fixed"] -
                   (result["adaptive"]["underlying_binary_balanced_accuracy"] - result["fixed_optimized"]["underlying_binary_balanced_accuracy"])) < 1e-12
        entry = {"methods": result, "primary_comparison": comparison,
                 "paired_secondary_contrasts": {f"{a}_minus_{b}": paired(result, bootstrap, a, b)
                                                for a, b in [("adaptive", "fixed_sequence_stop"), ("fixed_sequence_stop", "fixed_optimized"),
                                                             ("adaptive", "fixed_brier"), ("adaptive", "full_protocol")]},
                 "validation_checks": checks}
        entry["extended_primary_contrast"] = paired(result, bootstrap, "adaptive", "fixed_optimized")
        entry["conservative_budget_contrast"] = paired(result, bootstrap, "adaptive_budget_6", "fixed_budget_6")
        clean = {m: [r for r in g if r["has_gap_record"] == "False"] for m, g in grouped.items()}
        clean_result, clean_bootstrap = summarize(clean)
        entry["gap_scoring_contrast"] = paired(clean_result, clean_bootstrap, "adaptive", "fixed_optimized")
        all_results[analysis] = entry
        json_write(destination / f"analysis_{analysis}.json", entry)
        for method, r in result.items():
            compact_rows.append({"analysis": analysis, "method": method, "n": r["n"],
                                 "balanced_accuracy": r["underlying_binary_balanced_accuracy"],
                                 "ba_low": r["confidence_intervals_95"]["balanced_accuracy"][0],
                                 "ba_high": r["confidence_intervals_95"]["balanced_accuracy"][1],
                                 "sensitivity": r["underlying_binary_sensitivity"], "specificity": r["underlying_binary_specificity"],
                                 "coverage": r["coverage"], "mean_nominal_seconds": r["mean_seconds"],
                                 "mean_timestamp_seconds": r["mean_timestamp"], "timestamp_over_half_count": r["timestamp_over_half_count"],
                                 "mean_tasks": r["mean_tasks"]})
        for method in CORE:
            filtered = [r for r in grouped[method] if r["has_gap_record"] == "False"]
            a = arrays(filtered)
            sr = full_cohort_output_summary(a["y"], a["binary"], a["actual"])
            gap_rows.append({"analysis": analysis, "method": method, "n": len(filtered),
                             "balanced_accuracy": sr["underlying_binary_balanced_accuracy"], "coverage": sr["coverage"]})
            for fold in range(5):
                g = [r for r in grouped[method] if int(r["outer_fold"]) == fold]
                a = arrays(g)
                sr = full_cohort_output_summary(a["y"], a["binary"], a["actual"])
                fold_rows.append({"analysis": analysis, "method": method, "fold": fold,
                                  "balanced_accuracy": sr["underlying_binary_balanced_accuracy"],
                                  "coverage": sr["coverage"], "mean_nominal_seconds": float(a["seconds"].mean())})
        for fold in range(5):
            protocols = json.loads((BASE / f"results_{analysis}/fold_{fold}/fixed_protocols.json").read_text())
            for budget, spec in protocols["by_budget"].items():
                selected_rows.append({"analysis": analysis, "fold": fold, "budget_units": int(budget),
                                      "task_path": "|".join(spec["tasks"]), "task_mask": spec["mask"],
                                      "inner_ba_at_half": spec["inner_ba_at_half"]})
    csv_write(destination / "method_metrics.csv", compact_rows)
    csv_write(destination / "gap_scoring_sensitivity.csv", gap_rows)
    csv_write(destination / "fold_metrics.csv", fold_rows)
    csv_write(destination / "selected_fixed_protocols.csv", selected_rows)
    metadata = {r["subject_id"]: r for r in read(FEATURES / "participant_metadata.csv")}
    _, primary, _ = gather(BASE / "results_primary")
    subgroup_rows = []
    for name, predicate in [("age_under65", lambda r: float(r["age"]) < 65), ("age_65plus", lambda r: float(r["age"]) >= 65),
                            ("female", lambda r: r["gender_male"] == "0"), ("male", lambda r: r["gender_male"] == "1")]:
        for method in CORE:
            g = [r for r in primary[method] if predicate(metadata[r["subject_id"]])]
            a = arrays(g)
            sr = full_cohort_output_summary(a["y"], a["binary"], a["actual"])
            subgroup_rows.append({"group": name, "method": method, "n": len(g), "pd": int(a["y"].sum()),
                                  "dd": int((a["y"] == 0).sum()), "balanced_accuracy": sr["underlying_binary_balanced_accuracy"],
                                  "sensitivity": sr["underlying_binary_sensitivity"], "specificity": sr["underlying_binary_specificity"],
                                  "coverage": sr["coverage"]})
    csv_write(destination / "subgroup_metrics.csv", subgroup_rows)
    diagnoses = sorted(set(metadata[r["subject_id"]]["condition"] for r in primary["adaptive"] if r["y"] == "0"))
    subtype_rows = []
    for diagnosis in diagnoses:
        for method in CORE:
            g = [r for r in primary[method] if metadata[r["subject_id"]]["condition"] == diagnosis]
            a = arrays(g)
            subtype_rows.append({"diagnosis": diagnosis, "method": method, "n": len(g),
                                 "specificity": float((a["binary"] == 0).mean()), "coverage": float((a["actual"] != -1).mean())})
    csv_write(destination / "dd_subtype_metrics.csv", subtype_rows)
    task_rows = []
    for method in CORE:
        masks = arrays(primary[method])["mask"]
        for j, task in enumerate(TASKS):
            task_rows.append({"method": method, "task": task, "n_acquired": int(((masks & (1 << j)) != 0).sum()),
                              "fraction_acquired": float(((masks & (1 << j)) != 0).mean())})
    csv_write(destination / "task_acquisition_frequencies.csv", task_rows)
    pca_rows = []
    for fold in range(5):
        system = joblib.load(BASE / f"results_primary/fold_{fold}/fitted_system.joblib")
        for task, pca in zip(TASKS, system.representation.pcas):
            pca_rows.append({"fold": fold, "task": task,
                             "variance_retained": float(pca.explained_variance_ratio_.sum())})
    csv_write(destination / "pca_variance_retained.csv", pca_rows)
    json_write(destination / "ALL_RESULTS.json", all_results)
    print(json.dumps({"primary_contrast": all_results["primary"]["primary_comparison"]["paired_differences"],
                      "primary_methods": {k: {x: all_results["primary"]["methods"][k][x] for x in
                                              ["underlying_binary_balanced_accuracy", "coverage", "mean_seconds", "mean_timestamp", "timestamp_over_half_count", "unique_task_paths", "actual_outputs_by_reference_group"]} for k in CORE},
                      "without_nms_primary_contrast": all_results["without_nms"]["primary_comparison"]["paired_differences"],
                      "secondary_stop_contrast": all_results["primary"]["paired_secondary_contrasts"]["fixed_sequence_stop_minus_fixed_optimized"]}, indent=2))


if __name__ == "__main__":
    main()
