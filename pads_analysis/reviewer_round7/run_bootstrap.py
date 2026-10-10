"""Run the frozen, identity-grouped bootstrap of the complete PADS procedure."""
from pathlib import Path
import argparse
import gzip
import hashlib
import json
import sys
import time
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
sys.path[:0] = [str(ROOT), str(BASE / 'reviewer_round5'), str(BASE)]
import grouped_analysis as a
import random_controls as r

B = 500
MASTER_SEED = 2026091707
METHODS = ['adaptive', 'fixed_optimized', 'fixed_sequence_stop', 'random_next_stop']
FEATURES = BASE.parent / 'pads_features_received/PADS_features_return'
ORIGINAL_HASH = '34ba6614d209d88215f764a44a650133624b002e4cb6128dd225dccc0adbc056'
ORIGINAL_BANK = a.fit_bank
FIT_COUNT = 0


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def counted_bank(*args):
    global FIT_COUNT
    result = ORIGINAL_BANK(*args)
    FIT_COUNT += len(a.STATES)
    return result


a.fit_bank = counted_bank


def model_predictions(system, c, movement, ids):
    # Repeated held-out copies have identical features; score each identity once.
    _, first, inverse = np.unique(ids, return_index=True, return_inverse=True)
    z = system.representation.transform(c[first], movement[first])
    adaptive = [system.replay(row) for row in z]
    stopping = [system.replay(row, 7, system.fixed[7]) for row in z]
    fixed_mask = system.fixed[7]
    state = a.STATE_INDEX[fixed_mask]
    fixed = expit(z @ system.weights[state] + system.intercepts[state])
    scores = np.array([[v[0] for v in adaptive], fixed, [v[0] for v in stopping]])[:, inverse]
    masks = np.array([[v[1] for v in adaptive], np.full(len(z), fixed_mask), [v[1] for v in stopping]], dtype=int)[:, inverse]
    state_scores = r.score_states(system, z)[inverse]
    probabilities, visits = r.random_terminal_distribution(system, z)
    return scores, masks, state_scores, probabilities[inverse], visits


def audit_system(system, excluded):
    training = set(system.ids)
    assert not training & set(excluded)
    for split in system.ledger['state_crossfit']:
        learning, held = set(split['train_ids']), set(split['held_out_ids'])
        assert not learning & held
        assert learning | held == training
        assert not (learning | held) & set(excluded)


def fit_outer(data, train, test, seed, fold):
    ids, y = data['ids'], data['y']
    c, movement = data['clinical'], data['movement']
    assert not set(ids[train]) & set(ids[test])
    dev_scores = np.full((3, len(train)), np.nan)
    dev_states = np.full((len(train), len(r.MASKS)), np.nan)
    dev_probs = np.full_like(dev_states, np.nan)
    ledgers, visits = [], 0
    for middle, (learning, held) in enumerate(a.grouped_stratified_splits(ids[train], y[train], 3, seed + 100 + fold)):
        subtrain, subheld = train[learning], train[held]
        system = a.AcquisitionSystem().fit(c[subtrain], movement[subtrain], y[subtrain], ids[subtrain], seed + 1000 + 10 * fold + middle)
        audit_system(system, np.r_[ids[subheld], ids[test]])
        scores, masks, state_scores, probabilities, count = model_predictions(system, c[subheld], movement[subheld], ids[subheld])
        dev_scores[:, held] = scores
        dev_states[held], dev_probs[held] = state_scores, probabilities
        visits += count
        ledgers.append({'middle_fold': middle, 'threshold_holdout_ids': ids[subheld].tolist(), 'fit': system.ledger})
    assert np.isfinite(dev_scores).all() and np.isfinite(dev_states).all() and np.isfinite(dev_probs).all()
    thresholds = {}
    for j, name in enumerate(METHODS[:3]):
        cutoff = a.select_binary_threshold(dev_scores[j], y[train])
        margin = a.fit_indeterminate_threshold(dev_scores[j], cutoff, .2)
        rejection = float(np.mean(np.abs(dev_scores[j] - cutoff) < margin))
        assert rejection <= .2 + 1e-12
        thresholds[name] = {'binary_threshold': cutoff, 'indeterminate_margin': margin, 'development_n': len(train), 'development_rejection': rejection}
    thresholds[METHODS[3]] = r.select_weighted_threshold(dev_states, dev_probs, y[train])
    system = a.AcquisitionSystem().fit(c[train], movement[train], y[train], ids[train], seed + fold)
    audit_system(system, ids[test])
    scores, masks, states, probs, count = model_predictions(system, c[test], movement[test], ids[test])
    visits += count
    outputs = {key: np.zeros((4, len(test))) for key in ['binary_correct', 'clear_correct', 'clear_wrong', 'deferred', 'nominal_seconds', 'binary_positive', 'n_tasks']}
    actual = np.zeros((3, len(test)), dtype=np.int8)
    for j, name in enumerate(METHODS[:3]):
        th = thresholds[name]
        binary, actual[j] = a.predict_with_indeterminate(scores[j], th['binary_threshold'], th['indeterminate_margin'])
        outputs['binary_correct'][j] = binary == y[test]
        outputs['binary_positive'][j] = binary
        outputs['clear_correct'][j] = actual[j] == y[test]
        outputs['clear_wrong'][j] = actual[j] == 1 - y[test]
        outputs['deferred'][j] = actual[j] == -1
        outputs['nominal_seconds'][j] = [a.nominal_units(int(mask)) * 10.24 for mask in masks[j]]
        outputs['n_tasks'][j] = [int(mask).bit_count() for mask in masks[j]]
        assert all(int(mask) in a.VALID for mask in masks[j])
    th = thresholds[METHODS[3]]
    random_values = r.expected_outcomes(states, probs, y[test], th['binary_threshold'], th['indeterminate_margin'], data['timestamp'][test])
    for key in outputs:
        outputs[key][3] = (probs * (states >= th['binary_threshold'])).sum(axis=1) if key == 'binary_positive' else random_values[key]
    assert np.allclose(outputs['clear_correct'] + outputs['clear_wrong'] + outputs['deferred'], 1)
    assert np.all(outputs['nominal_seconds'] <= 71.68 + 1e-9)
    ledger = {'outer_fold': fold, 'outer_test_ids': ids[test].tolist(), 'middle_fits': ledgers, 'final_fit': system.ledger,
              'final_bank_sha256': hashlib.sha256(system.weights.tobytes() + system.intercepts.tobytes()).hexdigest()}
    detail = {'outer_fold': fold, 'train_rows': len(train), 'test_rows': len(test), 'unique_train': len(set(ids[train])), 'unique_test': len(set(ids[test])),
              'fixed_mask': int(system.fixed[7]), 'thresholds': thresholds, 'visited_random_states': visits}
    development = {'subject_ids': ids[train], 'y': y[train], 'scores': dev_scores,
                   'random_candidate_ba': np.array(thresholds[METHODS[3]]['candidate_ba'])}
    return outputs, scores, masks, actual, ledger, detail, development


def data_subset(data, rows):
    return {key: value[rows] for key, value in data.items()}


def reproduce():
    global FIT_COUNT
    FIT_COUNT = 0
    data = a.load_data(FEATURES)
    train, test = np.flatnonzero(data['folds'] != 0), np.flatnonzero(data['folds'] == 0)
    with threadpool_limits(limits=1):
        outputs, scores, masks, actual, ledger, detail, dev = fit_outer(data, train, test, a.SEED, 0)
    old = pd.read_csv(BASE / 'results_primary/fold_0/held_out_predictions.csv', dtype={'subject_id': str})
    old_random = pd.read_csv(BASE / 'reviewer_round5/results/seed_20260913/fold_0/participant_expectations.csv', dtype={'subject_id': str})
    max_difference = 0.
    for j, name in enumerate(METHODS[:3]):
        reference = old[old.method == name].set_index('subject_id').loc[data['ids'][test]]
        max_difference = max(max_difference, float(np.max(np.abs(reference.score.to_numpy() - scores[j]))))
        assert np.array_equal(reference.task_mask.to_numpy(), masks[j])
        assert np.array_equal(reference.actual_output.to_numpy(), actual[j])
    random_ref = old_random[old_random.method == METHODS[3]].set_index('subject_id').loc[data['ids'][test]]
    for key in ['binary_correct', 'clear_correct', 'clear_wrong', 'deferred', 'nominal_seconds', 'n_tasks']:
        delta = np.max(np.abs(random_ref[key].to_numpy() - outputs[key][3]))
        max_difference = max(max_difference, float(delta))
    assert max_difference < 1e-10 and FIT_COUNT == 4144
    a.json_write(ROOT / 'BASE_REPRODUCTION.json', {'passed': True, 'methods': METHODS, 'held_out_participants': len(test),
                  'maximum_numeric_difference': max_difference, 'logistic_fits': FIT_COUNT, 'identity_grouping_equivalent_without_duplicates': True})
    print('Original fold reproduced for all four methods.', flush=True)


def work(rep):
    global FIT_COUNT
    FIT_COUNT = 0
    output = ROOT / 'results' / f'replicate_{rep:04d}'
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'COMPLETE.json').exists():
        return json.loads((output / 'COMPLETE.json').read_text())
    start = time.monotonic()
    try:
        original = a.load_data(FEATURES)
        seed = MASTER_SEED + 10000 * rep
        # Keep seeds in the uint32 range for scikit-learn.
        assert seed + 2000 < 2 ** 32
        rng = np.random.default_rng(seed)
        counts = np.zeros(len(original['y']), dtype=int)
        for label in [0, 1]:
            candidates = np.flatnonzero(original['y'] == label)
            counts += np.bincount(rng.choice(candidates, len(candidates), replace=True), minlength=len(counts))
        rows = np.repeat(np.arange(len(counts)), counts)
        data = data_subset(original, rows)
        assert len(rows) == 390 and np.sum(data['y']) == 276
        n = len(rows)
        oof = {key: np.full((4, n), np.nan) for key in ['binary_correct', 'clear_correct', 'clear_wrong', 'deferred', 'nominal_seconds', 'binary_positive', 'n_tasks']}
        scores = np.full((3, n), np.nan)
        masks = np.full((3, n), -1, dtype=np.int16)
        actual = np.full((3, n), -2, dtype=np.int8)
        folds = np.full(n, -1, dtype=np.int8)
        ledgers, details, development = [], [], {}
        warnings.filterwarnings('error', category=ConvergenceWarning)
        with threadpool_limits(limits=1):
            for fold, (train, test) in enumerate(a.grouped_stratified_splits(data['ids'], data['y'], 5, seed)):
                values, sc, ms, ac, ledger, detail, dev = fit_outer(data, train, test, seed, fold)
                for key in oof:
                    oof[key][:, test] = values[key]
                scores[:, test], masks[:, test], actual[:, test], folds[test] = sc, ms, ac, fold
                ledgers.append(ledger)
                details.append(detail)
                development.update({f'fold_{fold}_{key}': value for key, value in dev.items()})
        assert FIT_COUNT == 20720 and (folds >= 0).all()
        assert all(np.isfinite(v).all() for v in oof.values()) and np.isfinite(scores).all()
        for sid in np.unique(data['ids']):
            selected = data['ids'] == sid
            assert len(np.unique(folds[selected])) == 1
            assert np.all(scores[:, selected] == scores[:, selected][:, :1])
        metrics = {}
        for j, name in enumerate(METHODS):
            y = data['y']
            per_class = {str(label): {key: float(oof[key][j, y == label].mean()) for key in ['binary_correct', 'clear_correct', 'clear_wrong', 'deferred']} for label in [0, 1]}
            ba = .5 * (per_class['0']['binary_correct'] + per_class['1']['binary_correct'])
            metrics[name] = {'balanced_accuracy': ba, 'nominal_seconds': float(oof['nominal_seconds'][j].mean()),
                'deferral': float(oof['deferred'][j].mean()), 'clear_wrong': float(oof['clear_wrong'][j].mean()),
                'mean_class_loss_0_5': .5 * sum(per_class[k]['clear_wrong'] + .5 * per_class[k]['deferred'] for k in ['0', '1']), 'by_class': per_class}
        np.savez_compressed(output / 'held_out_predictions.npz', subject_ids=data['ids'], original_index=rows, counts=counts, y=data['y'], outer_fold=folds,
                            scores=scores, task_masks=masks, actual_outputs=actual, **oof)
        np.savez_compressed(output / 'threshold_development.npz', **development)
        with gzip.open(output / 'training_ledger.json.gz', 'wt', encoding='utf-8') as handle:
            json.dump(ledgers, handle, separators=(',', ':'))
        a.json_write(output / 'thresholds_and_protocols.json', details)
        record = {'replicate': rep, 'seed': seed, 'rows': n, 'unique_participants': int((counts > 0).sum()),
                  'unique_pd': int(((counts > 0) & (original['y'] == 1)).sum()), 'unique_dd': int(((counts > 0) & (original['y'] == 0)).sum()),
                  'logistic_fits': FIT_COUNT, 'elapsed_seconds': time.monotonic() - start, 'metrics': metrics,
                  'identity_disjoint_all_levels': True, 'fit_code_sha256': sha(ROOT / 'grouped_analysis.py'), 'runner_sha256': sha(__file__),
                  'prediction_sha256': sha(output / 'held_out_predictions.npz')}
        a.json_write(output / 'COMPLETE.json', record)
        return record
    except Exception:
        (output / 'FAILED.txt').write_text(traceback.format_exc())
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--stop', type=int, default=B)
    parser.add_argument('--reproduce-only', action='store_true')
    args = parser.parse_args()
    assert sha(BASE / 'run_analysis.py') == ORIGINAL_HASH
    sources = [BASE / 'run_analysis.py', BASE / 'acquisition_rules.py', BASE / 'summary/held_out_predictions_primary.csv', BASE / 'summary/analysis_primary.json',
               BASE / 'reviewer_round5/random_controls.py', ROOT / 'ROUND7_SPEC.md', ROOT / 'grouped_analysis.py', Path(__file__), FEATURES / 'PADS_motion_features.npz']
    hashes = {str(p.relative_to(BASE.parent)): sha(p) for p in sources}
    freeze = {'replicates': B, 'master_seed': MASTER_SEED, 'input_and_code_hashes': hashes,
              'recorded_utc': __import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()}
    if (ROOT / 'FREEZE.json').exists():
        assert json.loads((ROOT / 'FREEZE.json').read_text())['input_and_code_hashes'] == hashes
    else:
        a.json_write(ROOT / 'FREEZE.json', freeze)
    if not (ROOT / 'BASE_REPRODUCTION.json').exists():
        reproduce()
    if args.reproduce_only:
        return
    assert 0 <= args.start < args.stop <= B
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(work, rep): rep for rep in range(args.start, args.stop)}
        for future in as_completed(futures):
            record = future.result()
            print(json.dumps({key: record[key] for key in ['replicate', 'unique_participants', 'logistic_fits', 'elapsed_seconds']}), flush=True)
    assert {str(p.relative_to(BASE.parent)): sha(p) for p in sources} == hashes
    completed = sorted((ROOT / 'results').glob('replicate_*/COMPLETE.json'))
    if len(completed) == B:
        records = [json.loads(p.read_text()) for p in completed]
        assert [r['replicate'] for r in records] == list(range(B))
        a.json_write(ROOT / 'COMPUTATION_LEDGER.json', {'complete': True, 'replicates': B, 'logistic_fits': sum(r['logistic_fits'] for r in records),
                      'additional_reproduction_fits': 4144, 'failures': [str(p.relative_to(ROOT)) for p in (ROOT / 'results').glob('*/FAILED.txt')],
                      'original_inputs_unchanged': True, 'results': records})


if __name__ == '__main__':
    main()
