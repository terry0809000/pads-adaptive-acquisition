"""Re-run the frozen PADS procedure on five newly specified outer splits."""
from pathlib import Path
import sys, os, hashlib, json, warnings, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent
sys.path.insert(0,str(BASE))
import run_analysis as a
SEEDS=[20260914,20260915,20260916,20260917,20260918]
HASH='34ba6614d209d88215f764a44a650133624b002e4cb6128dd225dccc0adbc056'

def run(seed):
    assert hashlib.sha256((BASE/'run_analysis.py').read_bytes()).hexdigest()==HASH
    data=a.load_data(BASE.parent/'pads_features_received/PADS_features_return')
    folds=np.full(len(data['y']),-1)
    for f,(_,test) in enumerate(StratifiedKFold(5,shuffle=True,random_state=seed).split(data['clinical'],data['y'])):
        folds[test]=f
    assert sorted(np.bincount(folds).tolist())==[78]*5
    data['folds']=folds
    out=ROOT/'results'/f'seed_{seed}'
    out.mkdir(parents=True,exist_ok=True)
    identity={'base_seed':seed,'original_code_sha256':HASH,'wrapper_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'spec_sha256':hashlib.sha256((ROOT/'ROUND2_SPEC.md').read_bytes()).hexdigest(),
              'participant_order':data['ids'].tolist(),'y':data['y'].tolist(),'new_folds':folds.tolist()}
    p=out/'REPEAT_IDENTITY.json'
    if p.exists():
        assert json.loads(p.read_text())==identity,'Existing repeat has a different identity'
    else:
        a.json_write(p,identity)
        a.csv_write(out/'outer_folds.csv',[{'subject_id':s,'y':int(y),'outer_fold':int(f)} for s,y,f in zip(data['ids'],data['y'],folds)])
    a.SEED=seed
    warnings.filterwarnings('error',category=ConvergenceWarning)
    started=time.monotonic()
    with threadpool_limits(limits=1):
        for fold in range(5):a.run_outer(data,fold,out,False)
    result={'seed':seed,'seconds':time.monotonic()-started,'folds_completed':5,'new_logistic_fits':20720}
    a.json_write(out/'REPEAT_COMPLETE.json',result)
    print(json.dumps(result),flush=True)
    return result

if __name__=='__main__':
    with ProcessPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(run,SEEDS))
    a.json_write(ROOT/'REPEATS_COMPLETE.json',{'results':results,'total_new_logistic_fits':103600,'original_primary_unchanged':True})
