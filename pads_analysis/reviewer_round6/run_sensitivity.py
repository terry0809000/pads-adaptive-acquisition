"""Fixed, exploratory one-factor sensitivities; preserve original primary outputs."""
from pathlib import Path
import sys, json, hashlib, time, warnings
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
sys.path[:0] = [str(ROOT), str(BASE)]
import parameterized_analysis as a

CONFIGS = [('pca4',4,1.0,25),('pca12',12,1.0,25),
           ('c0p1',8,0.1,25),('c10',8,10.0,25),
           ('k15',8,1.0,15),('k40',8,1.0,40)]
BASE_HASH = '34ba6614d209d88215f764a44a650133624b002e4cb6128dd225dccc0adbc056'

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def identity(config):
    name,pca,c,k = config
    return dict(name=name,pca_components=pca,logistic_c=c,neighbours=k,
                original_sha256=sha(BASE/'run_analysis.py'),
                parameterized_sha256=sha(ROOT/'parameterized_analysis.py'),
                wrapper_sha256=sha(Path(__file__)),spec_sha256=sha(ROOT/'ROUND6_SPEC.md'))

def run(config, folds=range(5)):
    assert sha(BASE/'run_analysis.py') == BASE_HASH
    name,a.PCA_COMPONENTS,a.LOGISTIC_C,a.NEIGHBOURS = config
    data = a.load_data(BASE.parent/'pads_features_received/PADS_features_return')
    out = ROOT/'results'/name
    out.mkdir(parents=True,exist_ok=True)
    id_path = out/'IDENTITY.json'
    ident = identity(config)
    if id_path.exists(): assert json.loads(id_path.read_text()) == ident
    else: a.json_write(id_path,ident)
    warnings.filterwarnings('error',category=ConvergenceWarning)
    started=time.monotonic()
    with threadpool_limits(limits=1):
        for fold in folds: a.run_outer(data,fold,out,False)
    result=dict(config=ident,folds=list(folds),elapsed_seconds=time.monotonic()-started,
                new_logistic_fits=4144*len(folds))
    a.json_write(out/'CONFIG_COMPLETE.json',result)
    return result

def reproduce():
    run(('base_reproduction',8,1.0,25),[0])
    source=BASE/'results_primary/fold_0'
    dest=ROOT/'results/base_reproduction/fold_0'
    old=pd.read_csv(source/'held_out_predictions.csv').sort_values(['method','subject_id']).reset_index(drop=True)
    new=pd.read_csv(dest/'held_out_predictions.csv').sort_values(['method','subject_id']).reset_index(drop=True)
    assert list(old.columns)==list(new.columns)
    assert len(old)==len(new)==78*17
    numeric=[c for c in old if pd.api.types.is_numeric_dtype(old[c])]
    deltas={c:float(np.max(np.abs(old[c].astype(float)-new[c].astype(float)))) for c in numeric}
    for c in numeric: assert deltas[c] < 1e-10,(c,deltas[c])
    for c in set(old.columns)-set(numeric): assert old[c].equals(new[c]),c
    assert json.loads((source/'thresholds.json').read_text())==json.loads((dest/'thresholds.json').read_text())
    result=dict(passed=True,n_scenario_rows=len(old),max_numeric_difference=max(deltas.values()),differences=deltas,
                identical_thresholds=True,identical_paths_and_decisions=True)
    a.json_write(ROOT/'BASE_REPRODUCTION.json',result)
    return result

if __name__=='__main__':
    sources=[BASE/'run_analysis.py',BASE/'ANALYSIS_PLAN.md',
             BASE/'summary/held_out_predictions_primary.csv',BASE/'summary/analysis_primary.json']
    frozen={str(p.relative_to(BASE)):sha(p) for p in sources}
    freeze=dict(configurations=[identity(c) for c in CONFIGS],input_hashes=frozen,
                recorded_utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat())
    path=ROOT/'FREEZE.json'
    if not path.exists():a.json_write(path,freeze)
    else:assert json.loads(path.read_text())['configurations']==freeze['configurations']
    reproduction=reproduce()
    with ProcessPoolExecutor(max_workers=3) as pool:
        results=list(pool.map(run,CONFIGS))
    assert {str(p.relative_to(BASE)):sha(p) for p in sources}==frozen
    a.json_write(ROOT/'COMPUTATION_LEDGER.json',dict(results=results,base_reproduction=reproduction,
                 new_sensitivity_fits=124320,additional_reproduction_fits=4144,
                 total_new_logistic_fits=128464,original_source_outputs_unchanged=True))
