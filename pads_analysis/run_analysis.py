"""Nested participant-level evaluation of adaptive PADS task acquisition."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
from scipy.special import expit
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from acquisition_rules import select_binary_threshold, fit_indeterminate_threshold, predict_with_indeterminate

SEED = 20260913
TASKS = ["Relaxed", "RelaxedTask", "StretchHold", "LiftHold", "HoldWeight",
         "PointFinger", "DrinkGlas", "CrossArms", "TouchIndex", "TouchNose", "Entrainment"]
UNITS = np.array([2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2])
FULL = (1 << 11) - 1
FEATURE_HASH = "55d40eec48a3a7cfa5d891bd1cf9a9f56f187260f5487c2b7a8e037c013791c3"
FOLD_HASH = "972ff7af4c4d4946f1166d43fcf18e5e6528915597870d741b608f9cf061377a"


def task_indices(mask):
    return [j for j in range(11) if mask & (1 << j)]


def nominal_units(mask):
    return int(UNITS[task_indices(mask)].sum())


def tie_order(mask):
    tasks = task_indices(mask)
    return nominal_units(mask), len(tasks), tuple(tasks)


def admissible_masks(budget=7):
    masks = []
    for mask in range(1, FULL + 1, 2):
        if nominal_units(mask) > budget:
            continue
        if mask & (1 << 3) and not mask & (1 << 2):
            continue
        if mask & (1 << 10) and not mask & (1 << 9):
            continue
        masks.append(mask)
    return sorted(masks, key=tie_order)


VALID = admissible_masks()
STATES = [0] + VALID + [FULL]
STATE_INDEX = {mask: i for i, mask in enumerate(STATES)}


def json_write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def csv_write(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class Representation:
    def fit(self, clinical, movement):
        self.clinical_scaler = StandardScaler().fit(clinical)
        self.task_scalers, self.pcas = [], []
        for j in range(11):
            scaler = StandardScaler().fit(movement[:, j])
            pca = PCA(n_components=8, whiten=True, svd_solver="full").fit(scaler.transform(movement[:, j]))
            if not np.isfinite(pca.explained_variance_).all() or (pca.explained_variance_ <= 1e-15).any():
                raise ValueError("Task PCA cannot support eight nondegenerate components")
            self.task_scalers.append(scaler)
            self.pcas.append(pca)
        self.baseline_width = clinical.shape[1]
        self.columns = {mask: np.asarray(list(range(self.baseline_width)) +
                       [self.baseline_width + j * 8 + k for j in task_indices(mask) for k in range(8)])
                       for mask in STATES}
        self.distance_scale = np.r_[np.full(self.baseline_width, 1 / np.sqrt(self.baseline_width)),
                                   np.full(88, 1 / np.sqrt(8))]
        return self

    def transform(self, clinical, movement):
        return np.column_stack([self.clinical_scaler.transform(clinical)] +
                               [p.transform(s.transform(movement[:, j]))
                                for j, (s, p) in enumerate(zip(self.task_scalers, self.pcas))])


def fit_bank(z, y, columns):
    weights = np.zeros((len(STATES), z.shape[1]))
    intercepts = np.zeros(len(STATES))
    max_iterations = 0
    for i, mask in enumerate(STATES):
        cols = columns[mask]
        model = LogisticRegression(C=1.0, solver="liblinear", l1_ratio=0, class_weight="balanced",
                                   intercept_scaling=100, tol=1e-6, max_iter=2000, random_state=SEED)
        model.fit(z[:, cols], y)
        weights[i, cols] = model.coef_[0]
        intercepts[i] = model.intercept_[0]
        max_iterations = max(max_iterations, int(model.n_iter_.max()))
    return weights, intercepts, max_iterations


class AcquisitionSystem:
    def fit(self, clinical, movement, y, ids, seed=SEED):
        self.ids = np.asarray(ids).copy()
        self.y = np.asarray(y).copy()
        self.oof = np.full((len(y), len(STATES)), np.nan)
        self.ledger = {"training_ids": self.ids.tolist(), "state_crossfit": [], "maximum_solver_iterations": 0}
        for fold, (train, held) in enumerate(StratifiedKFold(3, shuffle=True, random_state=seed).split(clinical, y)):
            assert not set(train) & set(held)
            representation = Representation().fit(clinical[train], movement[train])
            ztrain = representation.transform(clinical[train], movement[train])
            zheld = representation.transform(clinical[held], movement[held])
            weights, intercepts, iterations = fit_bank(ztrain, y[train], representation.columns)
            self.oof[held] = expit(zheld @ weights.T + intercepts)
            self.ledger["maximum_solver_iterations"] = max(self.ledger["maximum_solver_iterations"], iterations)
            self.ledger["state_crossfit"].append({"fold": fold, "train_ids": self.ids[train].tolist(),
                                                "held_out_ids": self.ids[held].tolist()})
        assert np.isfinite(self.oof).all()
        self.loss = (self.oof - y[:, None]) ** 2
        self.class_weights = np.where(y == 1, len(y) / (2 * (y == 1).sum()), len(y) / (2 * (y == 0).sum()))
        self.state_ba = .5 * ((self.oof[y == 1] >= .5).mean(axis=0) + (self.oof[y == 0] < .5).mean(axis=0))
        self.state_brier = np.average(self.loss, axis=0, weights=self.class_weights)
        self.fixed = {b: min(admissible_masks(b), key=lambda m: (-self.state_ba[STATE_INDEX[m]], tie_order(m)))
                      for b in range(2, 8)}
        self.fixed_brier = min(VALID, key=lambda m: (self.state_brier[STATE_INDEX[m]], tie_order(m)))
        self.representation = Representation().fit(clinical, movement)
        self.ztrain = self.representation.transform(clinical, movement)
        self.weights, self.intercepts, iterations = fit_bank(self.ztrain, y, self.representation.columns)
        self.ledger["maximum_solver_iterations"] = max(self.ledger["maximum_solver_iterations"], iterations)
        self._reachable_cache = {}
        return self

    def reachable(self, current, budget, fixed_sequence=None):
        key = current, budget, fixed_sequence
        if key not in self._reachable_cache:
            last = max(task_indices(current))
            allowed = [m for m in VALID if nominal_units(m) <= budget and (m & current) == current
                       and ((m ^ current) & ((1 << (last + 1)) - 1)) == 0]
            if fixed_sequence is not None:
                prefixes, prefix = [], 0
                for j in task_indices(fixed_sequence):
                    prefix |= 1 << j
                    prefixes.append(prefix)
                allowed = [m for m in allowed if m in prefixes]
            assert current in allowed
            self._reachable_cache[key] = allowed
        return self._reachable_cache[key]

    def recommend(self, current, observed, budget=7, fixed_sequence=None):
        """Only the currently observed feature vector enters this function."""
        cols = self.representation.columns[current]
        if observed.shape != (len(cols),):
            raise ValueError("Expected precisely the current-state feature vector")
        candidates = self.reachable(current, budget, fixed_sequence)
        if len(candidates) == 1:
            return None, {"current": current, "target": current, "reason": "no_reachable_extension"}
        difference = (self.ztrain[:, cols] - observed) * self.representation.distance_scale[cols]
        distance = np.einsum("ij,ij->i", difference, difference)
        neighbors = np.argsort(distance, kind="stable")[:min(25, len(distance))]
        candidate_index = [STATE_INDEX[m] for m in candidates]
        expected = np.average(self.loss[np.ix_(neighbors, candidate_index)], axis=0, weights=self.class_weights[neighbors])
        minimum = expected.min()
        best = next(i for i, value in enumerate(expected) if value <= minimum + 1e-12)
        target = candidates[best]
        next_task = None if target == current else min(task_indices(target ^ current))
        return next_task, {"current": current, "target": target, "expected_terminal_loss": float(expected[best]),
                           "expected_current_loss": float(expected[candidates.index(current)]),
                           "reason": "estimated_loss_stop" if next_task is None else "acquire"}

    def score_observed(self, current, observed):
        cols = self.representation.columns[current]
        if observed.shape != (len(cols),):
            raise ValueError("Unacquired features supplied to current-state scorer")
        state = STATE_INDEX[current]
        return float(expit(observed @ self.weights[state, cols] + self.intercepts[state]))

    def replay(self, z, budget=7, fixed_sequence=None):
        current, trace = 1, []
        for _ in range(11):
            observed = z[self.representation.columns[current]]
            next_task, step = self.recommend(current, observed, budget, fixed_sequence)
            trace.append(step)
            if next_task is None:
                break
            assert next_task > max(task_indices(current))
            current |= 1 << next_task
            assert current in VALID and nominal_units(current) <= budget
        else:
            raise RuntimeError("Replay failed to stop")
        score = self.score_observed(current, z[self.representation.columns[current]])
        return score, current, trace

    def predict_scenarios(self, clinical, movement):
        z = self.representation.transform(clinical, movement)
        n = len(z)
        results = {}
        static = {"clinical_only": 0, "rest_only": 1, "fixed_optimized": self.fixed[7],
                  "fixed_brier": self.fixed_brier, "full_protocol": FULL}
        static.update({f"fixed_budget_{b}": self.fixed[b] for b in range(2, 7)})
        for name, mask in static.items():
            cols = self.representation.columns[mask]
            state = STATE_INDEX[mask]
            results[name] = {"scores": expit(z[:, cols] @ self.weights[state, cols] + self.intercepts[state]),
                             "masks": np.full(n, mask, dtype=int), "traces": [None] * n}
        for budget in range(2, 8):
            name = "adaptive" if budget == 7 else f"adaptive_budget_{budget}"
            values = [self.replay(row, budget) for row in z]
            results[name] = {"scores": np.array([v[0] for v in values]), "masks": np.array([v[1] for v in values]),
                             "traces": [v[2] for v in values]}
        values = [self.replay(row, 7, self.fixed[7]) for row in z]
        results["fixed_sequence_stop"] = {"scores": np.array([v[0] for v in values]),
                                          "masks": np.array([v[1] for v in values]), "traces": [v[2] for v in values]}
        return results


def load_data(features_dir):
    path = features_dir / "PADS_motion_features.npz"
    if hashlib.sha256(path.read_bytes()).hexdigest() != FEATURE_HASH:
        raise ValueError("Unexpected feature file identity")
    fold_path = features_dir / "primary_outer_folds.csv"
    if hashlib.sha256(fold_path.read_bytes()).hexdigest() != FOLD_HASH:
        raise ValueError("Frozen split changed")
    data = np.load(path, allow_pickle=False)
    keep = data["primary_mask"]
    ids = data["subject_ids"][keep]
    y = data["primary_y"]
    with fold_path.open(newline="", encoding="utf-8") as h:
        manifest = {r["subject_id"]: r for r in csv.DictReader(h)}
    assert set(manifest) == set(ids) and len(ids) == 390
    assert all(int(manifest[sid]["y"]) == int(label) for sid, label in zip(ids, y))
    folds = np.array([int(manifest[sid]["outer_fold"]) for sid in ids])
    return {"ids": ids, "y": y, "folds": folds, "clinical": data["clinical_inputs"][keep],
            "movement": data["task_features"][keep], "timestamp": data["timestamp_estimated_seconds"][keep],
            "gap": data["gap_affected_records"][keep].any(axis=(1, 2))}


def run_outer(data, outer_fold, output, ablation=False):
    fold_path = output / f"fold_{outer_fold}"
    code_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if fold_path.exists():
        if (fold_path / "COMPLETE.json").exists():
            previous = json.loads((fold_path / "COMPLETE.json").read_text())
            if previous.get("code_sha256") != code_hash or previous.get("ablation_without_nms") != ablation:
                raise ValueError("Completed fold belongs to a different implementation or analysis")
            print(f"Reusing completed outer fold {outer_fold}", flush=True)
            return
        raise FileExistsError(f"Incomplete output exists; inspect before resuming: {fold_path}")
    fold_path.mkdir(parents=True)
    started = time.monotonic()
    ids, y, movement = data["ids"], data["y"], data["movement"]
    clinical = data["clinical"][:, :2] if ablation else data["clinical"]
    train = np.flatnonzero(data["folds"] != outer_fold)
    test = np.flatnonzero(data["folds"] == outer_fold)
    assert len(train) == 312 and len(test) == 78 and not set(ids[train]) & set(ids[test])
    threshold_scores, middle_ledger = {}, []
    for middle, (subtrain, subheld) in enumerate(StratifiedKFold(3, shuffle=True, random_state=SEED + 100 + outer_fold).split(clinical[train], y[train])):
        learning, validating = train[subtrain], train[subheld]
        system = AcquisitionSystem().fit(clinical[learning], movement[learning], y[learning], ids[learning],
                                        SEED + 1000 + 10 * outer_fold + middle)
        predictions = system.predict_scenarios(clinical[validating], movement[validating])
        for method, result in predictions.items():
            if method not in threshold_scores:
                threshold_scores[method] = np.full(len(train), np.nan)
            threshold_scores[method][subheld] = result["scores"]
        middle_ledger.append({"middle_fold": middle, "threshold_holdout_ids": ids[validating].tolist(),
                              "fit": system.ledger})
        print(f"Outer {outer_fold}: threshold cross-fit {middle + 1}/3 completed ({time.monotonic() - started:.1f}s)", flush=True)
    thresholds = {}
    for method, scores in threshold_scores.items():
        assert np.isfinite(scores).all()
        cutoff = select_binary_threshold(scores, y[train])
        margin = fit_indeterminate_threshold(scores, cutoff, .2)
        thresholds[method] = {"binary_threshold": cutoff, "indeterminate_margin": margin,
                              "threshold_training_n": len(train),
                              "threshold_training_coverage": float((np.abs(scores - cutoff) >= margin).mean())}
    system = AcquisitionSystem().fit(clinical[train], movement[train], y[train], ids[train], SEED + outer_fold)
    predictions = system.predict_scenarios(clinical[test], movement[test])
    rows, trace_rows = [], []
    for method, prediction in predictions.items():
        threshold = thresholds[method]
        binary, actual = predict_with_indeterminate(prediction["scores"], threshold["binary_threshold"], threshold["indeterminate_margin"])
        for i, index in enumerate(test):
            mask = int(prediction["masks"][i])
            tasks = task_indices(mask)
            nominal = nominal_units(mask) * 10.24
            timestamp = float(data["timestamp"][index, tasks].sum())
            if method not in {"full_protocol", "clinical_only"}:
                assert mask in VALID and nominal <= 71.68 + 1e-9
            rows.append({"subject_id": ids[index], "y": int(y[index]), "outer_fold": outer_fold, "method": method,
                         "score": float(prediction["scores"][i]), "binary_prediction": int(binary[i]),
                         "actual_output": int(actual[i]), "movement_seconds": nominal, "full_movement_seconds": 143.36,
                         "cost_basis": "protocol_defined", "timestamp_seconds": timestamp,
                         "full_timestamp_seconds": float(data["timestamp"][index].sum()), "task_mask": mask,
                         "n_tasks": len(tasks), "task_path": "|".join(TASKS[j] for j in tasks),
                         "binary_threshold": threshold["binary_threshold"], "indeterminate_margin": threshold["indeterminate_margin"],
                         "has_gap_record": bool(data["gap"][index])})
            if method in {"adaptive", "fixed_sequence_stop"}:
                trace_rows.append({"subject_id": ids[index], "method": method, "steps": prediction["traces"][i]})
    csv_write(fold_path / "held_out_predictions.csv", rows)
    json_write(fold_path / "acquisition_traces.json", trace_rows)
    json_write(fold_path / "thresholds.json", thresholds)
    json_write(fold_path / "training_ledger.json", {"outer_fold": outer_fold, "outer_test_ids": ids[test].tolist(),
                                                  "middle_fits": middle_ledger, "final_fit": system.ledger})
    json_write(fold_path / "fixed_protocols.json", {"by_budget": {str(b): {"mask": m, "tasks": [TASKS[j] for j in task_indices(m)],
                                                                  "inner_ba_at_half": float(system.state_ba[STATE_INDEX[m]])}
                                                                 for b, m in system.fixed.items()},
                                                      "brier_selected_mask": int(system.fixed_brier)})
    np.savez_compressed(fold_path / "threshold_training_scores.npz", subject_ids=ids[train], y=y[train], **threshold_scores)
    joblib.dump(system, fold_path / "fitted_system.joblib", compress=3)
    json_write(fold_path / "COMPLETE.json", {"outer_fold": outer_fold, "ablation_without_nms": ablation,
                                           "code_sha256": code_hash,
                                           "n_test": len(test), "n_train": len(train), "elapsed_seconds": time.monotonic() - started,
                                           "methods": list(predictions)})
    print(f"Outer {outer_fold} saved; {time.monotonic() - started:.1f}s", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--without-nms", action="store_true")
    parser.add_argument("--outer-fold", type=int)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    warnings.filterwarnings("error", category=ConvergenceWarning)
    data = load_data(args.features_dir)
    with threadpool_limits(limits=1):
        for fold in range(5) if args.outer_fold is None else [args.outer_fold]:
            run_outer(data, fold, args.output, args.without_nms)


if __name__ == "__main__":
    # Import by stable module name so saved model classes can be loaded normally.
    from run_analysis import main as imported_main
    imported_main()
