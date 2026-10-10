from pathlib import Path
import sys,json,hashlib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent
sys.path[:0]=[str(ROOT),str(BASE)]
import parameterized_analysis as a
from run_sensitivity import CONFIGS

def metrics(d):
    y=d.y.to_numpy(); pred=d.binary_prediction.to_numpy(); out=d.actual_output.to_numpy()
    t=d.movement_seconds.to_numpy(); r=out==-1
    return dict(n=len(y),pd_n=int((y==1).sum()),dd_n=int((y==0).sum()),
                balanced_accuracy=.5*((pred[y==1]==1).mean()+(pred[y==0]==0).mean()),
                nominal_seconds=t.mean(),deferred_fraction=r.mean(),
                pd_sensitivity=(pred[y==1]==1).mean(),dd_specificity=(pred[y==0]==0).mean(),
                pd_clear_correct=int(((y==1)&(out==1)).sum()),pd_clear_wrong=int(((y==1)&(out==0)).sum()),pd_defer=int(((y==1)&r).sum()),
                dd_clear_correct=int(((y==0)&(out==0)).sum()),dd_clear_wrong=int(((y==0)&(out==1)).sum()),dd_defer=int(((y==0)&r).sum()))

if __name__=='__main__':
    ledger=json.loads((ROOT/'COMPUTATION_LEDGER.json').read_text())
    data=a.load_data(BASE.parent/'pads_features_received/PADS_features_return')
    ids=sorted(data['ids'])
    allrows=[];checks=[]
    original=pd.read_csv(BASE/'summary/held_out_predictions_primary.csv',dtype={'subject_id':str})
    original['configuration']='original'
    allrows.append(original)
    for name,pca,c,k in CONFIGS:
        frames=[]
        for f in range(5):
            p=ROOT/'results'/name/f'fold_{f}'
            frame=pd.read_csv(p/'held_out_predictions.csv',dtype={'subject_id':str})
            log=json.loads((p/'training_ledger.json').read_text())
            test=set(log['outer_test_ids']);train=set(log['final_fit']['training_ids'])
            assert not test&train and len(test)==78 and len(train)==312
            for mid in log['middle_fits']:
                held=set(mid['threshold_holdout_ids']); learn=set(mid['fit']['training_ids'])
                assert not held&learn and held|learn==train and not (held|learn)&test
                for inner in mid['fit']['state_crossfit']:
                    itr=set(inner['train_ids']);ihe=set(inner['held_out_ids'])
                    assert not itr&ihe and itr|ihe==learn
            for inner in log['final_fit']['state_crossfit']:
                itr=set(inner['train_ids']);ihe=set(inner['held_out_ids'])
                assert not itr&ihe and itr|ihe==train
            assert log['final_fit']['maximum_solver_iterations']<2000
            assert np.isfinite(frame.score).all()
            for method,d in frame.groupby('method'):
                assert len(d)==78 and set(d.subject_id)==test
                if method not in ['clinical_only','full_protocol']:
                    assert d.task_mask.isin(a.VALID).all() and (d.movement_seconds<=71.68+1e-9).all()
            frames.append(frame)
        df=pd.concat(frames,ignore_index=True);df['configuration']=name;allrows.append(df)
        for m,d in df.groupby('method'):assert sorted(d.subject_id)==ids
        checks.append(dict(configuration=name,folds=5,n=390,scenarios=17,all_ledgers_disjoint=True,finite_scores=True,feasible_paths=True))
    full=pd.concat(allrows,ignore_index=True)
    full.to_csv(ROOT/'tables/held_out_predictions.csv',index=False)
    ms=[];fs=[]
    for (cfg,m),d in full.groupby(['configuration','method'],sort=False):
        ms.append(dict(configuration=cfg,method=m,**metrics(d)))
        for fold,g in d.groupby('outer_fold'):fs.append(dict(configuration=cfg,method=m,outer_fold=fold,**metrics(g)))
    pd.DataFrame(ms).to_csv(ROOT/'tables/configuration_metrics.csv',index=False)
    pd.DataFrame(fs).to_csv(ROOT/'tables/fold_metrics.csv',index=False)
    y=original[original.method=='adaptive'].set_index('subject_id').loc[ids].y.to_numpy()
    rng=np.random.default_rng(2026091606)
    i1=rng.choice(np.flatnonzero(y==1),(5000,int((y==1).sum())),replace=True)
    i0=rng.choice(np.flatnonzero(y==0),(5000,int((y==0).sum())),replace=True)
    both=np.concatenate([i1,i0],axis=1)
    rows=[]
    for cfg,d in full.groupby('configuration',sort=False):
        aa=d[d.method=='adaptive'].set_index('subject_id').loc[ids]
        ff=d[d.method=='fixed_optimized'].set_index('subject_id').loc[ids]
        for metric in ['balanced_accuracy','nominal_seconds','deferred_fraction']:
            if metric=='balanced_accuracy':
                val=(aa.binary_prediction.to_numpy()==y).astype(float)-(ff.binary_prediction.to_numpy()==y)
                point=.5*(val[y==1].mean()+val[y==0].mean())
                boot=.5*(val[i1].mean(1)+val[i0].mean(1))
            else:
                val=(aa.movement_seconds.to_numpy()-ff.movement_seconds.to_numpy()) if metric=='nominal_seconds' else ((aa.actual_output.to_numpy()==-1).astype(float)-(ff.actual_output.to_numpy()==-1))
                point=val.mean();boot=val[both].mean(1)
            low,high=np.quantile(boot,[.025,.975])
            rows.append(dict(configuration=cfg,metric=metric,adaptive_minus_fixed=point,low=low,high=high,bootstrap_seed=2026091606))
    pd.DataFrame(rows).to_csv(ROOT/'tables/paired_contrasts.csv',index=False)
    np.savez_compressed(ROOT/'conditional_bootstrap.npz',subject_ids=ids,y=y,pd_indices=i1,dd_indices=i0)
    # k changes the acquisition rule only. Verify fixed models and threshold trajectories.
    for cfg in ['k15','k40']:
        p=full[(full.configuration==cfg)&(full.method=='fixed_optimized')].set_index('subject_id').loc[ids]
        q=original[original.method=='fixed_optimized'].set_index('subject_id').loc[ids]
        for col in ['score','binary_prediction','actual_output','task_mask','binary_threshold','indeterminate_margin']:
            assert np.allclose(p[col],q[col],rtol=0,atol=1e-12),(cfg,col)
    result=dict(configurations=ms,contrasts=rows,validation=checks,n_unique_participants=390,
                new_sensitivity_fits=124320,total_new_fits_including_reproduction=128464,
                neighbourhood_fixed_comparator_identical=True,
                all_original_primary_files_unchanged=ledger['original_source_outputs_unchanged'])
    a.json_write(ROOT/'ROUND6_RESULTS.json',result)
    a.json_write(ROOT/'VALIDATION.json',dict(passed=True,checks=checks,n_unique_participants=390,
                 n_new_sensitivity_rows=6*390*17,neighbourhood_fixed_comparator_identical=True))
    print(pd.DataFrame(ms).query("method in ['adaptive','fixed_optimized','fixed_sequence_stop']")[['configuration','method','balanced_accuracy','nominal_seconds','deferred_fraction']].to_string(index=False))
    print(pd.DataFrame(rows).query("metric=='balanced_accuracy'").to_string(index=False))
