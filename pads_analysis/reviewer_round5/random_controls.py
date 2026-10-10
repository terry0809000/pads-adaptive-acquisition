"""Label-free acquisition distributions and weighted development rules."""
import numpy as np
from scipy.special import expit
import run_analysis as a

MASKS = np.array(a.VALID, dtype=int)
STATE_COLUMNS = np.array([a.STATE_INDEX[int(m)] for m in MASKS])
INDEX = {int(m): i for i, m in enumerate(MASKS)}
UNITS = np.array([a.nominal_units(int(m)) for m in MASKS])
COUNTS = np.array([int(m).bit_count() for m in MASKS])
TASK_MATRIX = ((MASKS[:, None] >> np.arange(11)) & 1).astype(float)
NEXT = {int(m): [int(m) | (1 << j) for j in range(max(a.task_indices(int(m))) + 1, 11)
                if (int(m) | (1 << j)) in INDEX]
        for m in MASKS}


def score_states(system, z):
    return expit(z @ system.weights[STATE_COLUMNS].T + system.intercepts[STATE_COLUMNS])


def next_distribution(system, current, observed):
    """Only current acquired values enter the stopping calculation."""
    task, _ = system.recommend(current, observed, 7)
    if task is None:
        return (), ()
    children = NEXT[current]
    assert children
    return children, np.full(len(children), 1 / len(children))


def random_terminal_distribution(system, z):
    """Integrate all randomized paths through the forward-order state DAG."""
    alive = np.zeros((len(z), len(MASKS)))
    terminal = np.zeros_like(alive)
    alive[:, INDEX[1]] = 1
    visits = 0
    for col, mask in enumerate(MASKS):
        current = int(mask)
        for row in np.flatnonzero(alive[:, col] > 0):
            observed = z[row, system.representation.columns[current]]
            children, probs = next_distribution(system, current, observed)
            mass = alive[row, col]
            visits += 1
            if not children:
                terminal[row, col] += mass
            else:
                for child, prob in zip(children, probs):
                    assert INDEX[child] > col
                    alive[row, INDEX[child]] += mass * prob
    assert np.allclose(terminal.sum(axis=1), 1, atol=1e-12)
    return terminal, visits


def permutation_distribution(adaptive_masks, matched):
    n = len(adaptive_masks)
    groups = {}
    for i, mask in enumerate(adaptive_masks):
        key = (a.nominal_units(int(mask)), int(mask).bit_count()) if matched else (0, 0)
        groups.setdefault(key, []).append(i)
    prob = np.zeros((n, len(MASKS)))
    audit = []
    for key, rows in groups.items():
        values, counts = np.unique(adaptive_masks[rows], return_counts=True)
        for m, count in zip(values, counts):
            prob[rows, INDEX[int(m)]] = count / len(rows)
        for row in rows:
            audit.append({'local_index': row, 'nominal_units_group': key[0], 'task_count_group': key[1],
                          'pool_n': len(rows), 'distinct_masks': len(values),
                          'mask_change_probability': 1 - prob[row, INDEX[int(adaptive_masks[row])]]})
    assert np.allclose(prob.sum(axis=1), 1)
    assert np.allclose((prob @ TASK_MATRIX).sum(axis=0),
                       TASK_MATRIX[[INDEX[int(m)] for m in adaptive_masks]].sum(axis=0))
    if matched:
        assert np.allclose(prob @ UNITS, [a.nominal_units(int(m)) for m in adaptive_masks])
        assert np.allclose(prob @ COUNTS, [int(m).bit_count() for m in adaptive_masks])
    return prob, list(groups.values()), audit


def select_weighted_threshold(scores, probabilities, labels):
    candidates = np.linspace(.05, .95, 19)
    values = []
    for t in candidates:
        correct = ((scores >= t) == labels[:, None]) * probabilities
        per_person = correct.sum(axis=1)
        ba = .5 * (per_person[labels == 1].mean() + per_person[labels == 0].mean())
        values.append((float(ba), -abs(float(t) - .5), -float(t)))
    best = max(range(len(candidates)), key=lambda j: values[j])
    threshold = float(candidates[best])
    active = probabilities > 0
    margins = np.abs(scores - threshold)[active]
    weights = probabilities[active] / len(labels)
    order = np.argsort(margins, kind='stable')
    cumulative = np.cumsum(weights[order])
    ix = min(np.searchsorted(cumulative, .2, side='right'), len(order) - 1)
    margin = float(margins[order[ix]])
    dev_rejection = float((probabilities * (np.abs(scores - threshold) < margin)).sum() / len(labels))
    assert dev_rejection <= .2 + 1e-10
    return {'binary_threshold': threshold, 'indeterminate_margin': margin,
            'development_expected_ba': values[best][0], 'development_expected_rejection': dev_rejection,
            'development_n': len(labels), 'candidate_ba': [v[0] for v in values]}


def outcome_matrices(scores, labels, threshold, margin):
    binary = scores >= threshold
    correct = binary == labels[:, None]
    reject = np.abs(scores - threshold) < margin
    return {'binary_correct': correct.astype(float), 'clear_correct': (correct & ~reject).astype(float),
            'clear_wrong': (~correct & ~reject).astype(float), 'deferred': reject.astype(float)}


def expected_outcomes(scores, probabilities, labels, threshold, margin, timestamp):
    result = {name: (value * probabilities).sum(axis=1)
              for name, value in outcome_matrices(scores, labels, threshold, margin).items()}
    result['nominal_seconds'] = probabilities @ UNITS * 10.24
    result['timestamp_seconds'] = (probabilities * (timestamp @ TASK_MATRIX.T)).sum(axis=1)
    result['n_tasks'] = probabilities @ COUNTS
    result['loss_0_5'] = result['clear_wrong'] + .5 * result['deferred']
    assert np.allclose(result['clear_correct'] + result['clear_wrong'] + result['deferred'], 1)
    return result
