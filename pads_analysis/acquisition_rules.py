"""Rules to reuse when the predictive and task-selection models are fitted.

These functions do not fit an acquisition policy or claim clinical validity.
Only training-fold out-of-sample predictions may be used to set thresholds.
"""

import numpy as np

TASKS = ("Relaxed", "RelaxedTask", "StretchHold", "LiftHold", "HoldWeight",
         "PointFinger", "DrinkGlas", "CrossArms", "TouchIndex", "TouchNose", "Entrainment")
PREREQUISITES = {3: {2}, 10: {9}}


def feasible_next_tasks(acquired, spent_seconds, task_costs, budget_seconds):
    """Enforce whole tasks, rest first, forward order and known task costs.

    `task_costs` must be available before collecting the next signal.
    Future timestamp measurements cannot be supplied as an inference input.
    When costs are estimates, actual elapsed time must be audited separately.
    """
    acquired = list(acquired)
    costs = np.asarray(task_costs, dtype=float)
    if costs.shape != (len(TASKS),) or not np.isfinite(costs).all() or (costs <= 0).any():
        raise ValueError("Provide one positive, finite cost per original task")
    if not np.isfinite([spent_seconds, budget_seconds]).all() or min(spent_seconds, budget_seconds) < 0:
        raise ValueError("Time values must be non-negative and finite")
    if len(set(acquired)) != len(acquired) or acquired != sorted(acquired):
        raise ValueError("Acquired tasks must be unique and follow source order")
    if any(j < 0 or j >= len(TASKS) for j in acquired):
        raise ValueError("Unknown task index")
    if acquired and acquired[0] != 0:
        raise ValueError("The common baseline task must be acquired first")
    if any(not PREREQUISITES.get(j, set()).issubset(set(acquired[:k])) for k, j in enumerate(acquired)):
        raise ValueError("An acquired task is missing a prerequisite")
    if not acquired:
        return [0] if costs[0] <= budget_seconds - spent_seconds + 1e-9 else []
    return [j for j in range(acquired[-1] + 1, len(TASKS))
            if PREREQUISITES.get(j, set()).issubset(set(acquired))
            and costs[j] <= budget_seconds - spent_seconds + 1e-9]


def select_binary_threshold(training_oof_scores, training_labels):
    """Choose a prespecified grid threshold by equal-weight class accuracy."""
    scores = np.asarray(training_oof_scores, dtype=float)
    labels = np.asarray(training_labels)
    if scores.shape != labels.shape or scores.ndim != 1 or not np.isfinite(scores).all():
        raise ValueError("Provide matching finite score and label vectors")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Both reference classes must occur in threshold-setting data")
    if ((scores < 0) | (scores > 1)).any():
        raise ValueError("Scores must lie between zero and one")
    candidates = np.linspace(.05, .95, 19)
    values = []
    for cutoff in candidates:
        prediction = scores >= cutoff
        balanced_accuracy = .5 * (prediction[labels == 1].mean() + (~prediction[labels == 0]).mean())
        values.append((float(balanced_accuracy), -abs(float(cutoff) - .5), -float(cutoff)))
    return float(candidates[max(range(len(candidates)), key=lambda i: values[i])])


def fit_indeterminate_threshold(training_oof_scores, binary_threshold, target=.20):
    """Set a simple rejection margin around an already selected threshold.

    Tied margins are treated together. The training fraction can therefore be
    below the target. Held-out coverage is estimated, never forced to a quota.
    Distance to a decision threshold is a ranking score, not a probability of
    a correct diagnosis. This is the initial simple comparator for rejection.
    """
    scores = np.asarray(training_oof_scores, dtype=float)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError("Provide finite training out-of-sample scores")
    if not 0 <= target < 1 or not 0 <= binary_threshold <= 1:
        raise ValueError("Invalid target or classification threshold")
    if ((scores < 0) | (scores > 1)).any():
        raise ValueError("Scores must lie between zero and one")
    margins = np.sort(np.abs(scores - binary_threshold))
    k = int(np.floor(target * len(margins)))
    return float(margins[k]) if k else 0.0


def predict_with_indeterminate(scores, binary_threshold, indeterminate_margin):
    """Return complete binary predictions and actual outputs (-1 = unresolved)."""
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Provide a finite score vector in [0, 1]")
    if not 0 <= binary_threshold <= 1 or not np.isfinite(indeterminate_margin) or indeterminate_margin < 0:
        raise ValueError("Invalid classification or rejection threshold")
    binary = (scores >= binary_threshold).astype(int)
    actual = binary.copy()
    actual[np.abs(scores - binary_threshold) < indeterminate_margin] = -1
    return binary, actual


def full_cohort_output_summary(labels, binary, actual):
    """Keep every eligible participant and show unresolved cases explicitly."""
    y = np.asarray(labels)
    binary, actual = np.asarray(binary), np.asarray(actual)
    if y.ndim != 1 or not (y.shape == binary.shape == actual.shape):
        raise ValueError("Outputs and labels must have matching one-dimensional shapes")
    if set(np.unique(y)) != {0, 1} or not set(np.unique(binary)).issubset({0, 1}) or not set(np.unique(actual)).issubset({-1, 0, 1}):
        raise ValueError("Unexpected label or output encoding")
    if not np.array_equal(binary[actual != -1], actual[actual != -1]):
        raise ValueError("Actual clear outputs must match the underlying binary predictions")
    result = {"n": len(y), "coverage": float(np.mean(actual != -1)),
              "indeterminate_fraction": float(np.mean(actual == -1)),
              "underlying_binary_sensitivity": float(np.mean(binary[y == 1] == 1)),
              "underlying_binary_specificity": float(np.mean(binary[y == 0] == 0))}
    result["underlying_binary_balanced_accuracy"] = .5 * (result["underlying_binary_sensitivity"] + result["underlying_binary_specificity"])
    result["actual_outputs_by_reference_group"] = {}
    for label, name in [(1, "PD"), (0, "differential_diagnosis")]:
        group = actual[y == label]
        result["actual_outputs_by_reference_group"][name] = {
            "denominator": len(group), "correct": int(np.sum(group == label)),
            "wrong": int(np.sum(group == 1 - label)), "indeterminate": int(np.sum(group == -1))}
    return result
