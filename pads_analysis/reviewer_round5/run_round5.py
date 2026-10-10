"""Rebuild middle fits and evaluate random controls on the six saved partitions."""
from pathlib import Path
import sys, json, time, hashlib, warnings
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
sys.path.insert(0, str(BASE))
import run_analysis as a
import random_controls as r

SEEDS = list(range(20260913, 20260919))


def source_dir(seed):
    return BASE / 'results_primary' if seed == SEEDS[0] else BASE / 'reviewer_round2/results' / f'seed_{seed}'


def work(args):
    seed, fold = args
    a.SEED = seed
    started = time.monotonic()
    out = ROOT / 'results' / f'seed_{seed}' / f'fold_{fold}'
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'COMPLETE.json').exists():
        return json.loads((out / 'COMPLETE.json').read_text())
    data = a.load_data(BASE.parent / 'pads_features_received/PADS_features_return')
    if seed != SEEDS[0]:
        folds = pd.read_csv(source_dir(seed) / 'outer_folds.csv', dtype={'subject_id': str}).set_index('subject_id')
        data['folds'] = folds.loc[data['ids'], 'outer_fold'].to_numpy()
    train = np.flatnonzero(data['folds'] != fold)
    test = np.flatnonzero(data['folds'] == fold)
    c, m, y, ids = data['clinical'], data['movement'], data['y'], data['ids']
    src = source_dir(seed) / f'fold_{fold}'
    saved_ledger = json.loads((src / 'training_ledger.json').read_text())
    saved_dev = np.load(src / 'threshold_training_scores.npz')
    assert list(ids[test]) == saved_ledger['outer_test_ids']
    assert np.array_equal(ids[train], saved_dev['subject_ids'])
    dev_scores = np.zeros((len(train), len(r.MASKS)))
    dev_probs = np.zeros_like(dev_scores)
    dev_adaptive = np.zeros(len(train))
    dev_masks = np.zeros(len(train), dtype=int)
    middle_ledgers = []
    visits = 0
    warnings.filterwarnings('error', category=ConvergenceWarning)
    with threadpool_limits(limits=1):
        for middle, (subtrain, subheld) in enumerate(StratifiedKFold(3, shuffle=True, random_state=seed + 100 + fold).split(c[train], y[train])):
            learning, held = train[subtrain], train[subheld]
            system = a.AcquisitionSystem().fit(c[learning], m[learning], y[learning], ids[learning], seed + 1000 + 10 * fold + middle)
            original = saved_ledger['middle_fits'][middle]
            assert list(ids[held]) == original['threshold_holdout_ids']
            assert system.ledger == original['fit']
            assert not (set(ids[learning]) & set(ids[held]) or set(ids[learning]) & set(ids[test]))
            z = system.representation.transform(c[held], m[held])
            scores = r.score_states(system, z)
            probs, nvisits = r.random_terminal_distribution(system, z)
            visits += nvisits
            dev_scores[subheld], dev_probs[subheld] = scores, probs
            adaptive = [system.replay(row) for row in z]
            dev_adaptive[subheld] = [v[0] for v in adaptive]
            dev_masks[subheld] = [v[1] for v in adaptive]
            middle_ledgers.append({'middle_fold': middle, 'threshold_holdout_ids': ids[held].tolist(), 'fit': system.ledger})
        max_dev_delta = float(np.max(np.abs(dev_adaptive - saved_dev['adaptive'])))
        assert max_dev_delta < 1e-10
        threshold = r.select_weighted_threshold(dev_scores, dev_probs, y[train])
        system = joblib.load(src / 'fitted_system.joblib')
        assert np.array_equal(system.ids, ids[train])
        z = system.representation.transform(c[test], m[test])
        scores = r.score_states(system, z)
        probs_random, nvisits = r.random_terminal_distribution(system, z)
        visits += nvisits
        saved = pd.read_csv(src / 'held_out_predictions.csv', dtype={'subject_id': str})
        adaptive = saved[saved.method == 'adaptive'].set_index('subject_id').loc[ids[test]]
        masks = adaptive.task_mask.to_numpy(dtype=int)
        mask_indices = np.array([r.INDEX[int(v)] for v in masks])
        max_outer_delta = float(np.max(np.abs(scores[np.arange(len(test)), mask_indices] - adaptive.score.to_numpy())))
        assert max_outer_delta < 1e-10
        replayed = [system.replay(row) for row in z]
        assert np.array_equal([v[1] for v in replayed], masks)
        probs_fold, _, audit_fold = r.permutation_distribution(masks, False)
        probs_matched, _, audit_matched = r.permutation_distribution(masks, True)
        thresholds = json.loads((src / 'thresholds.json').read_text())
        th_adaptive = thresholds['adaptive']
        entries = {'random_next_stop': (probs_random, threshold),
                   'permute_fold': (probs_fold, th_adaptive),
                   'permute_time_count': (probs_matched, th_adaptive)}
        for method in ['adaptive', 'fixed_optimized', 'fixed_sequence_stop']:
            selected = saved[saved.method == method].set_index('subject_id').loc[ids[test]]
            prob = np.zeros_like(scores)
            cols = [r.INDEX[int(mask)] for mask in selected.task_mask]
            prob[np.arange(len(test)), cols] = 1
            entries[method] = prob, thresholds[method]
        rows, task_rows = [], []
        for method, (prob, th) in entries.items():
            values = r.expected_outcomes(scores, prob, y[test], th['binary_threshold'], th['indeterminate_margin'], data['timestamp'][test])
            tasks = prob @ r.TASK_MATRIX
            for j, index in enumerate(test):
                row = {'seed': seed, 'outer_fold': fold, 'subject_id': ids[index], 'y': int(y[index]), 'method': method}
                row.update({key: float(value[j]) for key, value in values.items()})
                rows.append(row)
                task_rows.append({'seed': seed, 'outer_fold': fold, 'subject_id': ids[index], 'method': method,
                                  **{task: float(tasks[j, k]) for k, task in enumerate(a.TASKS)}})
            if method in ['adaptive', 'fixed_optimized', 'fixed_sequence_stop']:
                selected = saved[saved.method == method].set_index('subject_id').loc[ids[test]]
                assert np.array_equal(values['binary_correct'], (selected.binary_prediction.to_numpy() == y[test]).astype(float))
                assert np.array_equal(values['deferred'], (selected.actual_output.to_numpy() == -1).astype(float))
                assert np.allclose(values['nominal_seconds'], selected.movement_seconds.to_numpy())
        # A state changes only when an acquired value changes, never when unseen blocks change.
        current = 1
        altered = z[0].copy()
        unobserved = np.setdiff1d(np.arange(z.shape[1]), system.representation.columns[current])
        altered[unobserved] += 10000
        assert r.next_distribution(system, current, z[0, system.representation.columns[current]])[0] == r.next_distribution(system, current, altered[system.representation.columns[current]])[0]
    pd.DataFrame(rows).to_csv(out / 'participant_expectations.csv', index=False)
    pd.DataFrame(task_rows).to_csv(out / 'participant_task_probabilities.csv', index=False)
    for method, audit in [('permute_fold', audit_fold), ('permute_time_count', audit_matched)]:
        for row in audit:
            row.update(seed=seed, outer_fold=fold, subject_id=ids[test[row['local_index']]], method=method)
    pd.DataFrame(audit_fold + audit_matched).to_csv(out / 'permutation_pools.csv', index=False)
    np.savez_compressed(out / 'outer_state_distributions.npz', ids=ids[test], y=y[test], masks=r.MASKS, scores=scores,
                        random_next_stop=probs_random, permute_fold=probs_fold, permute_time_count=probs_matched,
                        adaptive_masks=masks, timestamp=data['timestamp'][test])
    np.savez_compressed(out / 'development_state_distributions.npz', ids=ids[train], y=y[train], masks=r.MASKS,
                        scores=dev_scores, random_next_stop=dev_probs, adaptive_scores=dev_adaptive, adaptive_masks=dev_masks)
    a.json_write(out / 'thresholds.json', {'random_next_stop': threshold, 'inherited_adaptive': th_adaptive})
    a.json_write(out / 'training_ledger.json', {'seed': seed, 'outer_fold': fold, 'outer_test_ids': ids[test].tolist(),
                                              'middle_fits': middle_ledgers, 'reused_outer_model': str(src.relative_to(BASE) / 'fitted_system.joblib')})
    result = {'seed': seed, 'outer_fold': fold, 'seconds': time.monotonic() - started,
              'new_logistic_fits': 3 * 4 * len(a.STATES), 'visited_random_states': visits,
              'maximum_middle_adaptive_score_difference': max_dev_delta, 'maximum_outer_adaptive_score_difference': max_outer_delta,
              'source_ledger_matched': True, 'original_outputs_matched': True, 'unobserved_value_check': True}
    a.json_write(out / 'COMPLETE.json', result)
    print(json.dumps(result), flush=True)
    return result


if __name__ == '__main__':
    freeze = json.loads((ROOT / 'FREEZE.json').read_text())
    for name, digest in freeze['code_hashes'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
    with ProcessPoolExecutor(max_workers=2) as pool:
        completed = list(pool.map(work, [(seed, fold) for seed in SEEDS for fold in range(5)]))
    a.json_write(ROOT / 'COMPUTATION_LEDGER.json', {'folds': completed, 'new_middle_systems': 90,
                  'reused_outer_systems': 30, 'new_logistic_fits': sum(v['new_logistic_fits'] for v in completed)})
