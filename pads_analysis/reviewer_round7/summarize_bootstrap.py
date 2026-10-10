"""Audit all 500 refits and summarize empirical stability, not external validation."""
from pathlib import Path
import gzip
import hashlib
import json
import sys
import numpy as np
import pandas as pd
from scipy.stats import binom

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
sys.path.insert(0, str(BASE))
import run_analysis as a

METHODS = ['adaptive', 'fixed_optimized', 'fixed_sequence_stop', 'random_next_stop']
CONTRASTS = [('adaptive','fixed_optimized'), ('adaptive','fixed_sequence_stop'), ('adaptive','random_next_stop'), ('fixed_sequence_stop','fixed_optimized')]
NUMERICAL_TIE = 1e-12


def interval(values):
    values = np.asarray(values)
    return {'mean': float(np.mean(values)), 'median': float(np.median(values)),
            'low': float(np.quantile(values, .025)), 'high': float(np.quantile(values, .975)),
            'minimum': float(np.min(values)), 'maximum': float(np.max(values))}


def main():
    source = a.load_data(BASE.parent / 'pads_features_received/PADS_features_return')
    output = ROOT / 'tables'
    output.mkdir(exist_ok=True)
    files = sorted((ROOT / 'results').glob('replicate_*/COMPLETE.json'))
    assert len(files) == 500, f'All 500 complete refits are required; found {len(files)}.'
    assert not list((ROOT / 'results').glob('*/FAILED.txt')), 'Report and resolve failures before summarizing.'
    originals = pd.read_csv(BASE / 'summary/held_out_predictions_primary.csv', dtype={'subject_id': str})
    arrays = {key: np.full((500, 3, 390), np.nan) for key in ['score', 'binary_positive', 'actual_output', 'task_mask', 'time']}
    metric_rows, threshold_rows, count_rows, audits, sample_sizes = [], [], [], [], []
    frozen = json.loads((ROOT/'FREEZE.json').read_text())['input_and_code_hashes']
    total_splits = 0
    for b, path in enumerate(files):
        record = json.loads(path.read_text())
        assert record['replicate'] == b and record['logistic_fits'] == 20720
        assert record['fit_code_sha256'] == frozen['pads_analysis/reviewer_round7/grouped_analysis.py']
        assert record['runner_sha256'] == frozen['pads_analysis/reviewer_round7/run_bootstrap.py']
        p = path.parent
        assert hashlib.sha256((p/'held_out_predictions.npz').read_bytes()).hexdigest() == record['prediction_sha256']
        d = np.load(p / 'held_out_predictions.npz', allow_pickle=False)
        y, ids, indices = d['y'], d['subject_ids'], d['original_index']
        assert len(y) == 390 and np.sum(y) == 276
        assert np.array_equal(ids, source['ids'][indices]) and np.array_equal(y, source['y'][indices])
        assert np.array_equal(np.bincount(indices, minlength=390), d['counts'])
        assert np.all((d['task_masks'][2] & d['task_masks'][1]) == d['task_masks'][2])
        assert np.all(d['nominal_seconds'][2] <= d['nominal_seconds'][1] + 1e-10)
        unique, first = np.unique(indices, return_index=True)
        for key, values in [('score',d['scores']),('binary_positive',d['binary_positive'][:3]),('actual_output',d['actual_outputs']),('task_mask',d['task_masks']),('time',d['nominal_seconds'][:3])]:
            arrays[key][b][:, unique] = values[:, first]
        with gzip.open(p / 'training_ledger.json.gz', 'rt') as handle:
            ledgers = json.load(handle)
        details = json.loads((p / 'thresholds_and_protocols.json').read_text())
        for fold, (ledger, detail) in enumerate(zip(ledgers, details)):
            held = set(ledger['outer_test_ids'])
            training = set(ledger['final_fit']['training_ids'])
            assert held == set(ids[d['outer_fold'] == fold])
            assert np.all(d['task_masks'][1,d['outer_fold'] == fold] == detail['fixed_mask'])
            assert not held & training and held | training == set(ids)
            total_splits += 1
            fits = [ledger['final_fit']] + [m['fit'] for m in ledger['middle_fits']]
            for middle in ledger['middle_fits']:
                mheld, mtrain = set(middle['threshold_holdout_ids']), set(middle['fit']['training_ids'])
                assert not mheld & mtrain and mheld | mtrain == training
                assert not (mheld | mtrain) & held
                total_splits += 1
            for system_index, fit in enumerate(fits):
                stage = 'outer_final' if system_index == 0 else 'middle_system'
                sample_sizes.append({'replicate':b,'outer_fold':fold,'stage':stage,'rows':len(fit['training_ids']),'unique_participants':len(set(fit['training_ids']))})
                for split in fit['state_crossfit']:
                    left, right = set(split['train_ids']), set(split['held_out_ids'])
                    assert not left & right and left | right == set(fit['training_ids'])
                    assert not (left | right) & held
                    total_splits += 1
                    sample_sizes.append({'replicate':b,'outer_fold':fold,'stage':stage+'_state_crossfit','rows':len(split['train_ids']),'unique_participants':len(left)})
            dev = np.load(p / 'threshold_development.npz', allow_pickle=False)
            for j, name in enumerate(METHODS):
                th = detail['thresholds'][name]
                row = {'replicate': b, 'outer_fold': fold, 'method': name, 'binary_threshold': th['binary_threshold'],
                       'rejection_margin': th['indeterminate_margin'], 'development_rejection': th.get('development_rejection',th.get('development_expected_rejection')),
                       'development_n': th['development_n'], 'unique_development_n': detail['unique_train'], 'held_out_n': detail['test_rows'], 'unique_held_out_n': detail['unique_test'], 'fixed_mask': detail['fixed_mask']}
                threshold_rows.append(row)
                assert row['development_rejection'] <= .2 + 1e-10
                held_rows = d['outer_fold'] == fold
                if j < 3:
                    sc = dev[f'fold_{fold}_scores'][j]
                    yl = dev[f'fold_{fold}_y']
                    t = a.select_binary_threshold(sc, yl)
                    m = a.fit_indeterminate_threshold(sc, t, .2)
                    assert t == th['binary_threshold'] and m == th['indeterminate_margin']
                    binary, actual = a.predict_with_indeterminate(d['scores'][j, held_rows], t, m)
                    assert np.array_equal(binary, d['binary_positive'][j, held_rows])
                    assert np.array_equal(actual, d['actual_outputs'][j, held_rows])
                else:
                    candidates = np.linspace(.05, .95, 19)
                    candidate_ba = dev[f'fold_{fold}_random_candidate_ba']
                    assert np.array_equal(candidate_ba, th['candidate_ba'])
                    selected = max(range(19), key=lambda k: (candidate_ba[k], -abs(float(candidates[k])-.5), -float(candidates[k])))
                    assert float(candidates[selected]) == th['binary_threshold']
                    assert float(candidate_ba[selected]) == th['development_expected_ba']
        for j, name in enumerate(METHODS):
            corr = d['binary_correct'][j]
            ba = .5 * (corr[y == 0].mean() + corr[y == 1].mean())
            loss = .5 * sum((d['clear_wrong'][j, y == label] + .5*d['deferred'][j, y == label]).mean() for label in [0,1])
            metrics = {'balanced_accuracy': ba, 'nominal_seconds': d['nominal_seconds'][j].mean(), 'deferral': d['deferred'][j].mean(),
                       'clear_wrong': d['clear_wrong'][j].mean(), 'mean_class_loss_0_5': loss}
            assert np.allclose(d['clear_correct'][j] + d['clear_wrong'][j] + d['deferred'][j],1)
            for key,value in metrics.items():
                assert abs(value - record['metrics'][name][key]) < 1e-12
            row = {'replicate': b, 'method': name, **metrics}
            for label, group in [(1,'pd'),(0,'dd')]:
                for key in ['clear_correct','clear_wrong','deferred']:
                    row[group+'_'+key] = d[key][j, y == label].mean()
            metric_rows.append(row)
        count_rows.append({key: record[key] for key in ['replicate','seed','unique_participants','unique_pd','unique_dd','elapsed_seconds']})
        audits.append({'replicate':b, 'passed':True, 'prediction_hash_matched':True, 'all_identity_splits_disjoint':True, 'deterministic_thresholds_recomputed':15, 'random_binary_threshold_selections_checked':5})
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(output / 'replicate_metrics.csv', index=False)
    thresholds = pd.DataFrame(threshold_rows)
    thresholds.to_csv(output / 'fold_thresholds_and_protocols.csv', index=False)
    counts = pd.DataFrame(count_rows)
    counts.to_csv(output / 'resample_counts.csv', index=False)
    sizes = pd.DataFrame(sample_sizes)
    sizes.to_csv(output / 'fit_sample_sizes.csv', index=False)
    absolute = []
    for name in METHODS:
        d = metrics[metrics.method == name]
        for metric in d.columns.difference(['replicate','method']):
            absolute.append({'method': name, 'metric': metric, **interval(d[metric])})
    pd.DataFrame(absolute).to_csv(output / 'absolute_stability.csv', index=False)
    contrasts, replica_contrasts = [], []
    for left,right in CONTRASTS:
        dl,dr = (metrics[metrics.method == name].set_index('replicate') for name in [left,right])
        diffs = dl.select_dtypes('number') - dr.select_dtypes('number')
        for b,row in diffs.iterrows():
            replica_contrasts.append({'replicate':int(b),'method':left,'comparator':right,**row.to_dict()})
        for key in diffs:
            v = diffs[key].to_numpy()
            contrasts.append({'method':left,'comparator':right,'metric':key, **interval(v), 'fraction_positive':float(np.mean(v>NUMERICAL_TIE)), 'fraction_negative':float(np.mean(v<-NUMERICAL_TIE)), 'fraction_zero':float(np.mean(np.abs(v)<=NUMERICAL_TIE))})
    contrast_df = pd.DataFrame(contrasts)
    contrast_df.to_csv(output / 'paired_stability.csv', index=False)
    rc = pd.DataFrame(replica_contrasts)
    rc.to_csv(output / 'replicate_paired_contrasts.csv', index=False)
    joint = []
    for (left,right),d in rc.groupby(['method','comparator'],sort=False):
        success = (d.balanced_accuracy>NUMERICAL_TIE) & (d.nominal_seconds<-NUMERICAL_TIE)
        frac = success.mean()
        joint.append({'method':left,'comparator':right,'higher_ba_count':int((d.balanced_accuracy>NUMERICAL_TIE).sum()),
                      'lower_time_count':int((d.nominal_seconds<-NUMERICAL_TIE).sum()),'joint_count':int(success.sum()),'replicates':500,
                      'joint_fraction':float(frac),'joint_monte_carlo_se':float(np.sqrt(frac*(1-frac)/500))})
    pd.DataFrame(joint).to_csv(output/'joint_accuracy_time.csv',index=False)
    person_rows = []
    for j,name in enumerate(METHODS[:3]):
        old = originals[originals.method == name].set_index('subject_id').loc[source['ids']]
        for i,sid in enumerate(source['ids']):
            keep = np.isfinite(arrays['score'][:,j,i])
            n = int(keep.sum())
            sc = arrays['score'][keep,j,i]
            actual = arrays['actual_output'][keep,j,i]
            masks = arrays['task_mask'][keep,j,i]
            _,cnt = np.unique(actual,return_counts=True)
            person_rows.append({'subject_id':sid,'y':int(source['y'][i]),'method':name,'n_refit_predictions':n,
                'original_score':float(old.iloc[i].score),'score_median':float(np.median(sc)), 'score_low':float(np.quantile(sc,.025)), 'score_high':float(np.quantile(sc,.975)),
                'mean_absolute_score_change':float(np.mean(np.abs(sc-old.iloc[i].score))),
                'binary_disagreement_with_original':float(np.mean(arrays['binary_positive'][keep,j,i] != old.iloc[i].binary_prediction)),
                'output_disagreement_with_original':float(np.mean(actual != old.iloc[i].actual_output)),
                'task_disagreement_with_original':float(np.mean(masks != old.iloc[i].task_mask)),
                'refit_deferral_fraction':float(np.mean(actual==-1)), 'modal_output_fraction':float(np.max(cnt)/n),
                'distinct_task_masks':int(len(np.unique(masks)))})
    people = pd.DataFrame(person_rows)
    people.to_csv(output / 'participant_stability.csv', index=False)
    np.savez_compressed(ROOT / 'participant_refit_arrays.npz', subject_ids=source['ids'],y=source['y'],**arrays)
    mc = []
    for left,right in CONTRASTS:
        d = rc[(rc.method==left)&(rc.comparator==right)].sort_values('replicate')
        for metric in ['balanced_accuracy','nominal_seconds']:
            for n in [100,250,500]:
                mc.append({'method':left,'comparator':right,'metric':metric,'replicates':n,**interval(d[metric].to_numpy()[:n])})
    pd.DataFrame(mc).to_csv(output/'monte_carlo_prefix_diagnostics.csv',index=False)
    tail_mc = []
    for left,right in CONTRASTS:
        d=rc[(rc.method==left)&(rc.comparator==right)]
        for metric in ['balanced_accuracy','nominal_seconds']:
            ordered=np.sort(d[metric].to_numpy())
            for p in [.025,.5,.975]:
                lo=max(0,int(binom.ppf(.025,500,p))-1)
                hi=min(499,int(binom.ppf(.975,500,p)))
                tail_mc.append({'method':left,'comparator':right,'metric':metric,'quantile':p,'estimate':float(np.quantile(ordered,p)),
                                'mc_order_stat_low':float(ordered[lo]),'mc_order_stat_high':float(ordered[hi]),'low_order_index':lo,'high_order_index':hi})
    pd.DataFrame(tail_mc).to_csv(output/'quantile_monte_carlo_diagnostics.csv',index=False)
    summary = {'replicates':500,'methods':METHODS,'rows_per_replicate':390,'distinct_original_participants':390,'new_logistic_fits':10360000,
               'unique_per_resample':interval(counts.unique_participants),'unique_training_per_fold':interval(thresholds[thresholds.method==METHODS[0]].unique_development_n),
               'training_sample_sizes':{stage:{key:interval(d[key]) for key in ['rows','unique_participants']} for stage,d in sizes.groupby('stage')},
               'absolute':absolute,'contrasts':contrasts,'joint_accuracy_time':joint,
               'person_prediction_counts':interval(people[people.method==METHODS[0]].n_refit_predictions),
               'participant_diagnostics':{name:{key:interval(people.loc[people.method==name,key]) for key in ['mean_absolute_score_change','binary_disagreement_with_original','output_disagreement_with_original','task_disagreement_with_original','modal_output_fraction','distinct_task_masks']} for name in METHODS[:3]},
               'numerical_tie_tolerance':NUMERICAL_TIE,
               'interval_definition':'Central 95% empirical percentiles across complete grouped participant-bootstrap refits; not a corrected primary confidence interval or external validation.'}
    a.json_write(ROOT/'ROUND7_RESULTS.json',summary)
    a.json_write(ROOT/'VALIDATION.json',{'passed':True,'complete_replicates':500,'logistic_fits':10360000,'identity_boundaries_audited':total_splits,
                  'deterministic_thresholds_recomputed':7500,'random_binary_threshold_selections_checked':2500,'all_metrics_recomputed':True,'paired_all_methods':True,'per_replicate':audits})
    print(json.dumps({'complete_replicates':500,'identity_boundaries_audited':total_splits,'unique_per_resample':summary['unique_per_resample'],'joint':joint},indent=2))


if __name__ == '__main__':
    main()
