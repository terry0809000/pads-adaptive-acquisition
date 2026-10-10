"""Conditional bootstrap intervals for descriptive calibration coefficients."""
from pathlib import Path
import csv
import json
import numpy as np
from scipy.special import expit, logit

R = Path(__file__).resolve().parent
T = R/"tables"
def read(name):
    with (T/name).open() as f:
        return list(csv.DictReader(f))
def write(name,rows):
    with (T/name).open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader();w.writerows(rows)

rows=read("held_out_probability_scores.csv")
methods=["adaptive","fixed_optimized","fixed_sequence_stop"]
groups={m:sorted([r for r in rows if r["seed"]=="20260913" and r["method"]==m],key=lambda r:r["subject_id"]) for m in methods}
y=np.array([int(r["y"]) for r in groups[methods[0]]])
rng=np.random.default_rng(2026091403)
parts=[np.flatnonzero(y==c) for c in [0,1]]
bi=np.concatenate([ix[rng.integers(0,len(ix),size=(5000,len(ix)))] for ix in parts],axis=1)
weights=np.zeros((5000,390),dtype=float)
np.add.at(weights,(np.arange(5000)[:,None],bi),1.)
pm={(r["method"],r["version"]):r for r in read("probability_metrics.csv") if r["seed"]=="20260913"}

def solve(x,w,initial,offset=False):
    a=np.full(len(w),initial[0]);b=np.full(len(w),1. if offset else initial[1])
    for it in range(60):
        p=expit(a[:,None]+b[:,None]*x)
        residual=w*(p-y)
        curvature=w*p*(1-p)
        ga=residual.sum(1);h0=curvature.sum(1)
        if offset:
            da=ga/h0;db=np.zeros(len(w))
            grad=np.abs(ga)/390
        else:
            gb=(residual*x).sum(1)
            h1=(curvature*x).sum(1);h2=(curvature*x*x).sum(1)
            determinant=h0*h2-h1*h1
            assert (determinant>1e-12).all()
            da=(h2*ga-h1*gb)/determinant
            db=(h0*gb-h1*ga)/determinant
            grad=np.maximum(np.abs(ga),np.abs(gb))/390
        if grad.max()<1e-9:
            return a,b,it,float(grad.max())
        scale=np.maximum(1.,np.maximum(np.abs(da),np.abs(db))/2.)
        a-=da/scale;b-=db/scale
    raise RuntimeError("Descriptive calibration bootstrap did not converge")

out=[];drawrows=[];validation=[]
for m in methods:
    assert [r["y"] for r in groups[m]]==[r["y"] for r in groups[methods[0]]]
    for version,field in [("raw","raw_probability"),("calibrated","calibrated_probability")]:
        x=logit(np.clip(np.array([float(r[field]) for r in groups[m]]),1e-6,1-1e-6))
        point=pm[m,version]
        start=[float(point["diagnostic_free_intercept"]),float(point["diagnostic_slope"])]
        # Independent Hessian solver must reproduce the original SciPy diagnostic fit.
        fulla,fullb,_,_=solve(x,np.ones((1,390)),start)
        assert abs(fulla[0]-start[0])<1e-5 and abs(fullb[0]-start[1])<1e-5
        a,b,nit,g=solve(x,weights,start)
        citl,_,citnit,citg=solve(x,weights,[float(point["citl_intercept"]),1.],offset=True)
        for name,v,estimate in [("diagnostic_free_intercept",a,start[0]),("diagnostic_slope",b,start[1]),("citl_intercept",citl,float(point["citl_intercept"]))]:
            lo,hi=np.quantile(v,[.025,.975])
            out.append({"method":m,"version":version,"metric":name,"estimate":estimate,
                        "lower_95":float(lo),"upper_95":float(hi),"draws":5000,
                        "scope":"original partition; diagnostic only; conditional on fitted scores"})
        for j in range(5000):
            drawrows.append({"method":m,"version":version,"draw":j,"free_intercept":float(a[j]),"slope":float(b[j]),"citl_intercept":float(citl[j])})
        validation.append({"method":m,"version":version,"n_converged_draws":5000,"joint_iterations":nit,
                           "offset_iterations":citnit,"maximum_average_gradient":max(g,citg),"full_sample_scipy_match":True})
write("calibration_diagnostic_intervals.csv",out)
write("calibration_diagnostic_bootstrap.csv",drawrows)
(R/"DIAGNOSTIC_VALIDATION.json").write_text(json.dumps(validation,indent=2)+"\n")
print(json.dumps(out,indent=2))
