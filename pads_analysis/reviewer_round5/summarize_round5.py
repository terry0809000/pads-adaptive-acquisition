"""Conditional paired uncertainty and whole-cohort randomization realizations."""
from pathlib import Path
import sys, json, hashlib
import numpy as np
import pandas as pd
import joblib
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
sys.path.insert(0, str(BASE))
import run_analysis as a
import random_controls as r
from run_round5 import SEEDS, source_dir

METHODS = ['adaptive','fixed_optimized','fixed_sequence_stop','random_next_stop','permute_fold','permute_time_count']
RANDOM = METHODS[3:]
METRICS = ['balanced_accuracy','pd_sensitivity','dd_specificity','nominal_seconds','timestamp_seconds','deferred_fraction',
           'balanced_clear_correct','balanced_clear_wrong','balanced_deferred','balanced_loss_0_5']


def metrics(frame):
    g = frame.groupby('y')
    means = g.mean(numeric_only=True)
    return {'balanced_accuracy': float(means.binary_correct.mean()),
            'pd_sensitivity': float(means.loc[1,'binary_correct']), 'dd_specificity': float(means.loc[0,'binary_correct']),
            'nominal_seconds': float(frame.nominal_seconds.mean()), 'timestamp_seconds': float(frame.timestamp_seconds.mean()),
            'deferred_fraction': float(frame.deferred.mean()), 'balanced_clear_correct': float(means.clear_correct.mean()),
            'balanced_clear_wrong': float(means.clear_wrong.mean()), 'balanced_deferred': float(means.deferred.mean()),
            'balanced_loss_0_5': float(means.loss_0_5.mean())}


def boot_metrics(frame, i0, i1):
    out = {}
    idx = np.concatenate([i0,i1], axis=1)
    col = lambda k: frame[k].to_numpy()
    balanced = lambda k: .5 * (col(k)[i0].mean(axis=1) + col(k)[i1].mean(axis=1))
    out['balanced_accuracy'] = balanced('binary_correct')
    out['pd_sensitivity'] = col('binary_correct')[i1].mean(axis=1)
    out['dd_specificity'] = col('binary_correct')[i0].mean(axis=1)
    out['nominal_seconds'] = col('nominal_seconds')[idx].mean(axis=1)
    out['timestamp_seconds'] = col('timestamp_seconds')[idx].mean(axis=1)
    out['deferred_fraction'] = col('deferred')[idx].mean(axis=1)
    for name, key in [('balanced_clear_correct','clear_correct'),('balanced_clear_wrong','clear_wrong'),
                      ('balanced_deferred','deferred'),('balanced_loss_0_5','loss_0_5')]: out[name] = balanced(key)
    return out


def main():
    frames = [pd.read_csv(p, dtype={'subject_id':str}) for p in sorted((ROOT/'results').glob('seed_*/fold_*/participant_expectations.csv'))]
    assert len(frames) == 30
    all_rows = pd.concat(frames, ignore_index=True)
    assert len(all_rows) == 6*6*390
    assert all_rows.groupby(['seed','method']).subject_id.nunique().eq(390).all()
    all_rows.to_csv(ROOT/'tables/participant_expectations.csv',index=False)
    summary=[]; classes=[]; folds=[]
    for (seed,method), frame in all_rows.groupby(['seed','method']):
        summary.append({'seed':seed,'method':method,**metrics(frame)})
        for label,g in frame.groupby('y'):
            classes.append({'seed':seed,'method':method,'class':'PD' if label else 'DD','n':len(g),
                            **{k:float(g[k].sum()) for k in ['binary_correct','clear_correct','clear_wrong','deferred']},
                            'clear_wrong_fraction':float(g.clear_wrong.mean()),'deferred_fraction':float(g.deferred.mean())})
        for fold,g in frame.groupby('outer_fold'): folds.append({'seed':seed,'method':method,'outer_fold':fold,**metrics(g)})
    summary=pd.DataFrame(summary)
    summary.to_csv(ROOT/'tables/partition_metrics.csv',index=False)
    pd.DataFrame(classes).to_csv(ROOT/'tables/class_outcomes.csv',index=False)
    pd.DataFrame(folds).to_csv(ROOT/'tables/fold_metrics.csv',index=False)
    original={m:all_rows[(all_rows.seed==SEEDS[0]) & (all_rows.method==m)].sort_values('subject_id').reset_index(drop=True) for m in METHODS}
    labels=original['adaptive'].y.to_numpy()
    rng=np.random.default_rng(2026091505)
    c0,c1=np.flatnonzero(labels==0),np.flatnonzero(labels==1)
    i0=rng.choice(c0,(5000,len(c0)),replace=True);i1=rng.choice(c1,(5000,len(c1)),replace=True)
    boot={m:boot_metrics(frame,i0,i1) for m,frame in original.items()}
    contrasts=[]; absolute=[]; draws={}
    for m in METHODS:
        est=metrics(original[m])
        for key in METRICS:
            low,high=np.quantile(boot[m][key],[.025,.975])
            absolute.append({'method':m,'metric':key,'estimate':est[key],'low':low,'high':high})
            draws[f'{m}__{key}']=boot[m][key]
    for seed in SEEDS:
        s=summary[summary.seed==seed].set_index('method')
        for m in METHODS[1:]:
            for key in METRICS:
                value=float(s.loc['adaptive',key]-s.loc[m,key])
                row={'seed':seed,'comparator':m,'metric':key,'adaptive_minus_comparator':value}
                if seed==SEEDS[0]: row.update(zip(['low','high'],np.quantile(boot['adaptive'][key]-boot[m][key],[.025,.975])))
                contrasts.append(row)
    contrasts=pd.DataFrame(contrasts)
    contrasts.to_csv(ROOT/'tables/paired_contrasts.csv',index=False)
    pd.DataFrame(absolute).to_csv(ROOT/'tables/original_conditional_intervals.csv',index=False)
    np.savez_compressed(ROOT/'results/conditional_bootstrap.npz',subject_ids=original['adaptive'].subject_id.to_numpy(dtype=str),i0=i0,i1=i1,**draws)
    repeat_summary=[]
    for (comparator,metric),g in contrasts[contrasts.seed!=SEEDS[0]].groupby(['comparator','metric']):
        v=g.adaptive_minus_comparator
        repeat_summary.append({'comparator':comparator,'metric':metric,'mean':v.mean(),'minimum':v.min(),'maximum':v.max(),
                               'positive_partitions':int((v>0).sum()),'negative_partitions':int((v<0).sum()),'n_partitions':len(v)})
    pd.DataFrame(repeat_summary).to_csv(ROOT/'tables/repeat_contrast_ranges.csv',index=False)
    pools=pd.concat([pd.read_csv(p,dtype={'subject_id':str}) for p in sorted((ROOT/'results').glob('seed_*/fold_*/permutation_pools.csv'))])
    pools.to_csv(ROOT/'tables/permutation_pools.csv',index=False)
    pool_summary=[]
    for (seed,method),g in pools.groupby(['seed','method']):
        pool_summary.append({'seed':seed,'method':method,'n':len(g),'single_mask_participants':int((g.distinct_masks==1).sum()),
                             'mean_mask_change_probability':g.mask_change_probability.mean(),'median_pool_size':g.pool_n.median(),
                             'smallest_pool_size':g.pool_n.min(),'median_distinct_masks':g.distinct_masks.median()})
    pd.DataFrame(pool_summary).to_csv(ROOT/'tables/permutation_pool_summary.csv',index=False)
    task_rows=pd.concat([pd.read_csv(p,dtype={'subject_id':str}) for p in sorted((ROOT/'results').glob('seed_*/fold_*/participant_task_probabilities.csv'))])
    task_rows.groupby(['seed','method'])[a.TASKS].mean().reset_index().to_csv(ROOT/'tables/task_frequencies.csv',index=False)
    thresholds=[]
    for seed in SEEDS:
        for fold in range(5):
            t=json.loads((ROOT/f'results/seed_{seed}/fold_{fold}/thresholds.json').read_text())
            for method,th in t.items():
                thresholds.append({'seed':seed,'outer_fold':fold,'method':method,**{k:v for k,v in th.items() if not isinstance(v,list)}})
    pd.DataFrame(thresholds).to_csv(ROOT/'tables/selected_thresholds.csv',index=False)

    # Monte Carlo ranges describe realization variability, not participant-sampling uncertainty.
    mc_rows=[];mc_summary=[];checks=[]
    for sidx,seed in enumerate(SEEDS):
        rng=np.random.default_rng(2026091506+sidx)
        dataset={m:[] for m in RANDOM}
        masksets={m:[] for m in RANDOM};idsets=[];ysets=[]
        for fold in range(5):
            folder=ROOT/f'results/seed_{seed}/fold_{fold}'
            d=np.load(folder/'outer_state_distributions.npz')
            t=json.loads((folder/'thresholds.json').read_text())
            n=len(d['ids']);idsets.extend(d['ids']);ysets.extend(d['y'])
            base_masks=d['adaptive_masks'];sc=d['scores'];labels=d['y']
            for method in RANDOM:
                if method=='random_next_stop':
                    u=rng.random((1000,n,1))
                    indices=(u>np.cumsum(d[method],axis=1)[None,:,:]).sum(axis=2)
                    assert indices.max()<len(r.MASKS)
                    masks=r.MASKS[indices]
                    th=t['random_next_stop']
                else:
                    _,groups,_=r.permutation_distribution(base_masks,method=='permute_time_count')
                    masks=np.zeros((1000,n),dtype=int)
                    for group in groups:
                        for draw in range(1000): masks[draw,group]=rng.permutation(base_masks[group])
                    indices=np.array([[r.INDEX[int(v)] for v in row] for row in masks])
                    th=t['inherited_adaptive']
                    original_indices=[r.INDEX[int(v)] for v in base_masks]
                    assert np.allclose(r.TASK_MATRIX[indices].sum(axis=1),r.TASK_MATRIX[original_indices].sum(axis=0)[None,:])
                    assert np.allclose(r.UNITS[indices].sum(axis=1),r.UNITS[original_indices].sum())
                    if method=='permute_time_count':
                        assert np.all(r.UNITS[indices]==r.UNITS[original_indices][None,:])
                        assert np.all(r.COUNTS[indices]==r.COUNTS[original_indices][None,:])
                selected=sc[np.arange(n)[None,:],indices]
                correct=(selected>=th['binary_threshold'])==labels[None,:]
                deferred=np.abs(selected-th['binary_threshold'])<th['indeterminate_margin']
                wrong=~correct & ~deferred
                values={'binary_correct':correct.astype(float),'deferred':deferred.astype(float),'clear_correct':(correct & ~deferred).astype(float),
                        'clear_wrong':wrong.astype(float),'loss_0_5':wrong+.5*deferred,'nominal_seconds':r.UNITS[indices]*10.24,
                        'timestamp_seconds':(d['timestamp']@r.TASK_MATRIX.T)[np.arange(n)[None,:],indices]}
                dataset[method].append(values);masksets[method].append(masks)
        labels=np.array(ysets);idx0=labels==0;idx1=labels==1
        for method in RANDOM:
            values={k:np.concatenate([v[k] for v in dataset[method]],axis=1) for k in dataset[method][0]}
            bal=lambda k:.5*(values[k][:,idx0].mean(axis=1)+values[k][:,idx1].mean(axis=1))
            measured={'balanced_accuracy':bal('binary_correct'),'nominal_seconds':values['nominal_seconds'].mean(axis=1),
                      'deferred_fraction':values['deferred'].mean(axis=1),'balanced_loss_0_5':bal('loss_0_5')}
            for draw in range(1000):mc_rows.append({'seed':seed,'method':method,'draw':draw,**{k:float(v[draw]) for k,v in measured.items()}})
            for k,v in measured.items():
                exact=float(summary[(summary.seed==seed)&(summary.method==method)][k].iloc[0])
                lo,hi=np.quantile(v,[.025,.975])
                se=float(v.std(ddof=1)/np.sqrt(1000))
                mc_summary.append({'seed':seed,'method':method,'metric':k,'exact_expected':exact,'mc_mean':v.mean(),
                                   'mc_mean_standard_error':se,'reference_low':lo,'reference_high':hi})
                checks.append(abs(v.mean()-exact)<max(6*se,1e-8))
            np.savez_compressed(ROOT/f'results/mc_masks_{seed}_{method}.npz',ids=np.array(idsets),y=labels,
                                task_masks=np.concatenate(masksets[method],axis=1).astype(np.uint16))
    assert all(checks)
    pd.DataFrame(mc_rows).to_csv(ROOT/'tables/monte_carlo_realizations.csv',index=False)
    pd.DataFrame(mc_summary).to_csv(ROOT/'tables/monte_carlo_reference_ranges.csv',index=False)

    # Independent recursive enumeration of three held-out people verifies DAG propagation.
    system=joblib.load(source_dir(SEEDS[0])/'fold_0/fitted_system.joblib')
    data=a.load_data(BASE.parent/'pads_features_received/PADS_features_return')
    test=np.flatnonzero(data['folds']==0)
    z=system.representation.transform(data['clinical'][test],data['movement'][test])
    expected=np.load(ROOT/f'results/seed_{SEEDS[0]}/fold_0/outer_state_distributions.npz')['random_next_stop']
    errors=[]
    with threadpool_limits(limits=1):
        for row in [0,1,2]:
            accumulated=np.zeros(len(r.MASKS))
            def recurse(mask,mass):
                task,_=system.recommend(mask,z[row,system.representation.columns[mask]],7)
                if task is None: accumulated[r.INDEX[mask]]+=mass;return
                tasks=a.task_indices(mask)
                children=[]
                for j in range(tasks[-1]+1,11):
                    candidate=mask | (1<<j)
                    if a.nominal_units(candidate)>7:continue
                    if j==3 and 2 not in tasks:continue
                    if j==10 and 9 not in tasks:continue
                    children.append(candidate)
                for child in children: recurse(child,mass/len(children))
            recurse(1,1.0)
            errors.append(float(np.max(np.abs(accumulated-expected[row]))))
    assert max(errors)<1e-12
    prior=[]
    for row in json.loads((ROOT/'INPUT_MANIFEST.json').read_text()):
        p=BASE.parent/row['path']
        prior.append(hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256'])
    assert all(prior)
    validation={'completed_outer_folds':30,'unique_participants':390,'partitions':6,'participant_expectation_rows':len(all_rows),
                'paired_bootstrap_draws':5000,'mc_realizations_per_partition_control':1000,
                'original_inputs_unchanged':all(prior),'recursive_enumeration_max_difference':max(errors),
                'mc_means_within_six_simulation_standard_errors':all(checks),'permutation_task_frequencies_exact':True,
                'matched_per_person_nominal_cost_and_count_exact':True,'all_terminal_masks_admissible':True,
                'nominal_cap_seconds':71.68,'expected_counts_not_observed_counts':True,
                'intervals_condition_on_fixed_models_thresholds_and_control_distributions':True}
    a.json_write(ROOT/'VALIDATION.json',validation)
    result={'original':summary[summary.seed==SEEDS[0]].to_dict('records'),
                 'original_contrasts':contrasts[contrasts.seed==SEEDS[0]].to_dict('records'),
                 'repeated_contrast_ranges':repeat_summary,'pool_summary':pool_summary}
    a.json_write(ROOT/'ROUND5_RESULTS.json',json.loads(json.dumps(result,default=lambda value:value.item())))
    print(summary[summary.seed==SEEDS[0]].to_string(index=False))
    print(contrasts[(contrasts.seed==SEEDS[0]) & (contrasts.metric=='balanced_accuracy')].to_string(index=False))
    print(pd.DataFrame(repeat_summary).query("metric=='balanced_accuracy'").to_string(index=False))


if __name__=='__main__':main()
