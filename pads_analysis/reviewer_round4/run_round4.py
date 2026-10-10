"""Develop and evaluate diagnosis-balanced rejection allocations."""
from pathlib import Path
import csv, hashlib, json, inspect
import numpy as np
from scipy.special import expit, logit
from allocation_rules import apply_allocation, candidate_allocations, choose_allocation

R=Path(__file__).resolve().parent;B=R.parent;W=B.parent
T=R/"tables";T.mkdir(exist_ok=True)
SEEDS=list(range(20260913,20260919))
METHODS=["adaptive","fixed_optimized","fixed_sequence_stop"]
CONTROLS=["original_margin","raw_label_confidence","calibrated_label_confidence"]
NEW=["mean_class_loss","worst_class_loss"]
TARGETS=[0.,.1,.2,.3,.4]
COSTS=[0.,.25,.5,.75,1.]
NBOOT=5000;BOOTSEED=2026091404
def read(p):
    with p.open() as f:return list(csv.DictReader(f))
def write(name,rows):
    assert rows
    with (T/name).open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def js(p,data):p.write_text(json.dumps(data,indent=2,allow_nan=False,ensure_ascii=False)+"\n")
def recalibrate(p,fit):return expit(float(fit["intercept"])+float(fit["slope"])*logit(np.clip(p,1e-6,1-1e-6)))
def metrics(y,d,o):
    y,d,o=map(np.asarray,(y,d,o))
    clear=o!=-1;err=clear&(o!=y)
    n=len(y)
    z={"n":n,"pd":int((y==1).sum()),"dd":int((y==0).sum()),
       "rejection":float((~clear).mean()),"coverage":float(clear.mean()),
       "n_clear":int(clear.sum()),"n_deferred":int((~clear).sum()),"n_clear_wrong":int(err.sum()),
       "selective_error":float(err.sum()/clear.sum()) if clear.any() else None}
    for c,label in [(1,"pd"),(0,"dd")]:
        ix=y==c
        for name,mask in [("clear_correct",clear&(o==y)),("clear_wrong",err),("deferred",~clear)]:
            z[f"{label}_{name}"]=int((ix&mask).sum())
            z[f"{label}_{name}_rate"]=float(mask[ix].mean()) if ix.any() else None
    both=z["pd"]>0 and z["dd"]>0
    z["binary_ba"]=float(.5*((d[y==1]==1).mean()+(d[y==0]==0).mean())) if both else None
    z["clear_error_gap"]=abs(z["pd_clear_wrong_rate"]-z["dd_clear_wrong_rate"]) if both else None
    z["deferral_gap"]=abs(z["pd_deferred_rate"]-z["dd_deferred_rate"]) if both else None
    for cost in COSTS:
        losses=[]
        for label in ["pd","dd"]:
            v=z[f"{label}_clear_wrong_rate"]+cost*z[f"{label}_deferred_rate"] if z[label] else None
            z[f"{label}_loss_{cost:g}"]=v;losses.append(v)
        z[f"mean_loss_{cost:g}"]=float(np.mean(losses)) if both else None
        z[f"worst_loss_{cost:g}"]=float(np.max(losses)) if both else None
    return z

inputs={}
def track(p):inputs[str(p.relative_to(W))]=sha(p)
params=B/"reviewer_round3/tables/calibration_parameters.csv";track(params)
fits={(int(r["seed"]),int(r["outer_fold"]),r["method"]):r for r in read(params)}
probpath=B/"reviewer_round3/tables/held_out_probability_scores.csv";track(probpath)
oldprob={(int(r["seed"]),r["method"],r["subject_id"]):r for r in read(probpath)}
oldpath=B/"reviewer_round3/tables/held_out_selective_outputs.csv";track(oldpath)
old={(int(r["seed"]),r["method"],r["rule"],float(r["target"]),r["subject_id"]):r for r in read(oldpath)}
metapath=W/"pads_features_received/PADS_features_return/participant_metadata.csv";track(metapath)
meta={r["subject_id"]:r for r in read(metapath)}
labpath=W/"pads_features_received/PADS_features_return/primary_outer_folds.csv";track(labpath)
all_labels={r["subject_id"]:int(r["y"]) for r in read(labpath)}
frozen_hashes={"run_analysis.py":"34ba6614d209d88215f764a44a650133624b002e4cb6128dd225dccc0adbc056",
               "ANALYSIS_PLAN.md":"402b6efb13229cd13829b28fc1d9c2a582bb783f4b2a3d3eb14fb2f33549c79a"}
for name,h in frozen_hashes.items():assert sha(B/name)==h
assert list(inspect.signature(apply_allocation).parameters)==["binary","calibrated_probability","cutoff_predicted_dd","cutoff_predicted_pd"]

output=[];selected=[];candidate_rows=[];foldrows=[];checks=[]
for seed in SEEDS:
    base=B/"results_primary" if seed==SEEDS[0] else B/"reviewer_round2/results"/f"seed_{seed}"
    for fold in range(5):
        root=base/f"fold_{fold}"
        for name in ["training_ledger.json","threshold_training_scores.npz","held_out_predictions.csv","thresholds.json"]:
            track(root/name)
        ledger=json.loads((root/"training_ledger.json").read_text())
        thresh=json.loads((root/"thresholds.json").read_text())
        with np.load(root/"threshold_training_scores.npz",allow_pickle=False) as z:
            dev_ids=z["subject_ids"].astype(str);dy=z["y"].astype(int)
            devp={m:z[m].copy() for m in METHODS}
        held=set(ledger["outer_test_ids"])
        assert len(held)==78 and len(dev_ids)==312 and not set(dev_ids)&held
        assert list(dev_ids)==ledger["final_fit"]["training_ids"]
        assert all(all_labels[sid]==int(y) for sid,y in zip(dev_ids,dy))
        midheld=[]
        for middle in ledger["middle_fits"]:
            a=set(middle["threshold_holdout_ids"]);b=set(middle["fit"]["training_ids"])
            assert len(a)==104 and len(b)==208 and not a&b and a|b==set(dev_ids)
            midheld.extend(a)
        assert len(midheld)==len(set(midheld))==312
        src=read(root/"held_out_predictions.csv")
        for method in METHODS:
            rr=sorted([r for r in src if r["method"]==method],key=lambda r:r["subject_id"])
            ids=[r["subject_id"] for r in rr];assert set(ids)==held
            y=np.array([int(r["y"]) for r in rr]);d=np.array([int(r["binary_prediction"]) for r in rr])
            p=np.array([float(r["score"]) for r in rr]);dp=devp[method]
            c=float(thresh[method]["binary_threshold"])
            dd=(dp>=c).astype(int);assert np.array_equal(d,(p>=c).astype(int))
            fit=fits[seed,fold,method];cp=recalibrate(p,fit);dcp=recalibrate(dp,fit)
            assert all(abs(cp[i]-float(oldprob[seed,method,sid]["calibrated_probability"]))<1e-12 for i,sid in enumerate(ids))
            configurations=[]
            for control in CONTROLS:
                for q in TARGETS:
                    oo=np.array([int(old[seed,method,control,q,sid]["actual_output"]) for sid in ids])
                    configurations.append((control,q,-1.,oo,None))
            for q in TARGETS:
                costs=COSTS if q==.2 else [.5]
                for cost in costs:
                    candidates=candidate_allocations(dd,dcp,dy,q,cost)
                    if q==.2 and cost==.5:
                        candidate_rows.extend({"seed":seed,"outer_fold":fold,"method":method,"target":q,"selection_lambda":cost,**r} for r in candidates)
                    for objective in NEW:
                        s=choose_allocation(candidates,objective)
                        do=apply_allocation(dd,dcp,s["cutoff_predicted_dd"],s["cutoff_predicted_pd"])
                        dm=metrics(dy,dd,do)
                        assert int((do==-1).sum())==s["development_rejected"]
                        assert s["development_rejected"]<=int(np.floor(q*len(dy)))
                        assert abs(dm[f"mean_loss_{cost:g}"]-s["development_mean_loss"])<1e-12
                        assert abs(dm[f"worst_loss_{cost:g}"]-s["development_worst_loss"])<1e-12
                        # All candidates are retained for independent focal-optimality review.
                        primary="development_mean_loss" if objective==NEW[0] else "development_worst_loss"
                        assert s[primary]<=min(x[primary] for x in candidates)+1e-12
                        oo=apply_allocation(d,cp,s["cutoff_predicted_dd"],s["cutoff_predicted_pd"])
                        selected.append({"seed":seed,"outer_fold":fold,"method":method,"rule":objective,"target":q,
                            "selection_lambda":cost,"feasible_allocations":len(candidates),
                            "n_predicted_dd_development":int((dd==0).sum()),"n_predicted_pd_development":int((dd==1).sum()),
                            **s})
                        configurations.append((objective,q,cost,oo,s))
            assert len(configurations)==33
            for rule,q,cost,oo,s in configurations:
                mm=metrics(y,d,oo)
                foldrows.append({"seed":seed,"outer_fold":fold,"method":method,"rule":rule,"target":q,"selection_lambda":cost,
                    "target_error_pp":100*(mm["rejection"]-q),
                    "development_mean_loss":None if s is None else s["development_mean_loss"],
                    "development_worst_loss":None if s is None else s["development_worst_loss"],**mm})
                if q==0:assert np.array_equal(oo,d)
                for i,sid in enumerate(ids):
                    output.append({"seed":seed,"outer_fold":fold,"method":method,"rule":rule,"target":q,"selection_lambda":cost,
                        "subject_id":sid,"y":int(y[i]),"binary_prediction":int(d[i]),"actual_output":int(oo[i]),
                        "calibrated_probability":float(cp[i]),"movement_seconds":float(rr[i]["movement_seconds"]),
                        "task_mask":int(rr[i]["task_mask"])})
        checks.append({"seed":seed,"outer_fold":fold,"n_development":312,"n_test":78,
            "development_scores_and_labels_verified":True,"nested_participant_separation":True})
    print(f"Completed allocation evaluation for partition {seed}",flush=True)

write("held_out_allocation_outputs.csv",output)
write("selected_allocations.csv",selected)
write("focal_development_candidates.csv",candidate_rows)
write("fold_allocation_metrics.csv",foldrows)
groups={}
for r in output:groups.setdefault((r["seed"],r["method"],r["rule"],r["target"],r["selection_lambda"]),[]).append(r)
repeat=[];subgroup=[];transitions=[]
for key,rr in groups.items():
    seed,method,rule,q,cost=key
    rr.sort(key=lambda r:r["subject_id"])
    assert len(rr)==len({r["subject_id"] for r in rr})==390
    y=np.array([r["y"] for r in rr]);d=np.array([r["binary_prediction"] for r in rr]);o=np.array([r["actual_output"] for r in rr])
    mm=metrics(y,d,o)
    repeat.append({"seed":seed,"method":method,"rule":rule,"target":q,"selection_lambda":cost,
                   "target_error_pp":100*(mm["rejection"]-q),**mm})
    if q==.2 and cost in [-1.,.5]:
        for label,select in [
            ("age_under65",lambda t:float(t["age"])<65),("age_65plus",lambda t:float(t["age"])>=65),
            ("female",lambda t:int(t["gender_male"])==0),("male",lambda t:int(t["gender_male"])==1),
            ("Atypical Parkinsonism",lambda t:t["condition"]=="Atypical Parkinsonism"),
            ("Essential Tremor",lambda t:t["condition"]=="Essential Tremor"),
            ("Multiple Sclerosis",lambda t:t["condition"]=="Multiple Sclerosis"),
            ("Other Movement Disorders",lambda t:t["condition"]=="Other Movement Disorders")]:
            ix=np.array([select(meta[r["subject_id"]]) for r in rr])
            assert ix.any()
            subgroup.append({"seed":seed,"method":method,"rule":rule,"target":q,"selection_lambda":cost,
                             "group":label,**metrics(y[ix],d[ix],o[ix])})
        if rule in NEW:
            prior=np.array([int(old[seed,method,CONTROLS[2],q,r["subject_id"]]["actual_output"]) for r in rr])
            for cls in [0,1]:
                for before in [-1,0,1]:
                    for after in [-1,0,1]:
                        transitions.append({"seed":seed,"method":method,"rule":rule,"reference_class":cls,
                            "from_calibrated_output":before,"to_allocated_output":after,
                            "n":int(((y==cls)&(prior==before)&(o==after)).sum())})
write("repeat_allocation_metrics.csv",repeat)
write("subgroup_allocation_metrics.csv",subgroup)
write("output_transitions.csv",transitions)

# Original-partition conditional bootstrap; shared participant draws across every rule.
FOCAL=CONTROLS+NEW
def focalkey(seed,method,rule):return seed,method,rule,.2,(-1. if rule in CONTROLS else .5)
base=groups[focalkey(SEEDS[0],METHODS[0],FOCAL[0])]
y=np.array([r["y"] for r in base])
rng=np.random.default_rng(BOOTSEED)
parts=[np.flatnonzero(y==c) for c in [0,1]]
draws=[ix[rng.integers(0,len(ix),size=(NBOOT,len(ix)))] for ix in parts]
bi=np.concatenate(draws,axis=1)
intervals=[];bootrows=[]
index={(r["seed"],r["method"],r["rule"],r["target"],r["selection_lambda"]):r for r in repeat}
def add_ci(method,contrast,name,point,boot):
    lo,hi=np.quantile(boot,[.025,.975])
    intervals.append({"method":method,"comparison":contrast,"metric":name,"estimate":float(point),
        "lower_95":float(lo),"upper_95":float(hi),"draws":NBOOT,"scope":"original partition; conditional on fitted procedures"})
for method in METHODS:
    mb={}
    for rule in FOCAL:
        rr=groups[focalkey(SEEDS[0],method,rule)]
        assert [r["subject_id"] for r in rr]==[r["subject_id"] for r in base]
        o=np.array([r["actual_output"] for r in rr]);defer=o==-1;wrong=(o!=-1)&(o!=y)
        b={"rejection":defer[bi].mean(1)}
        for c,label in [(0,"dd"),(1,"pd")]:
            b[f"{label}_clear_wrong_rate"]=wrong[draws[c]].mean(1)
            b[f"{label}_deferred_rate"]=defer[draws[c]].mean(1)
            b[f"{label}_loss_0.5"]=b[f"{label}_clear_wrong_rate"]+.5*b[f"{label}_deferred_rate"]
        b["mean_loss_0.5"]=.5*(b["pd_loss_0.5"]+b["dd_loss_0.5"])
        b["worst_loss_0.5"]=np.maximum(b["pd_loss_0.5"],b["dd_loss_0.5"])
        b["deferral_gap"]=np.abs(b["pd_deferred_rate"]-b["dd_deferred_rate"])
        b["clear_error_gap"]=np.abs(b["pd_clear_wrong_rate"]-b["dd_clear_wrong_rate"])
        mb[rule]=b
        point=index[focalkey(SEEDS[0],method,rule)]
        for name,v in b.items():add_ci(method,rule,name,point[name],v)
    pairs=[(new,control) for new in NEW for control in [CONTROLS[0],CONTROLS[2]]]+[(NEW[1],NEW[0])]
    for first,second in pairs:
        a=index[focalkey(SEEDS[0],method,first)];b=index[focalkey(SEEDS[0],method,second)]
        for name in mb[first]:
            add_ci(method,first+"_minus_"+second,name,a[name]-b[name],mb[first][name]-mb[second][name])
        for j in range(NBOOT):
            bootrows.append({"method":method,"comparison":first+"_minus_"+second,"draw":j,
                "delta_mean_loss_half":float(mb[first]["mean_loss_0.5"][j]-mb[second]["mean_loss_0.5"][j]),
                "delta_worst_loss_half":float(mb[first]["worst_loss_0.5"][j]-mb[second]["worst_loss_0.5"][j]),
                "delta_rejection":float(mb[first]["rejection"][j]-mb[second]["rejection"][j]),
                "delta_pd_clear_wrong_rate":float(mb[first]["pd_clear_wrong_rate"][j]-mb[second]["pd_clear_wrong_rate"][j]),
                "delta_dd_clear_wrong_rate":float(mb[first]["dd_clear_wrong_rate"][j]-mb[second]["dd_clear_wrong_rate"][j])})
write("conditional_intervals.csv",intervals)
write("conditional_bootstrap_contrasts.csv",bootrows)
def describe(a):return {"mean":float(np.mean(a)),"min":float(np.min(a)),"max":float(np.max(a))}
summary={"unique_participants":390,"partitions":SEEDS,"new_acquisition_fits":0,"new_calibrator_fits":0,
    "selected_cutoff_pairs":len(selected),"positive_target_cutoff_pairs":sum(r["target"]>0 for r in selected),
    "held_out_output_rows":len(output),"focal_development_candidate_rows":len(candidate_rows),
    "focal_target":.2,"focal_illustrative_lambda":.5,"original_partition":{},"five_new_partitions":{}}
for method in METHODS:
    summary["original_partition"][method]={rule:index[focalkey(SEEDS[0],method,rule)] for rule in FOCAL}
    summary["five_new_partitions"][method]={}
    for rule in FOCAL:
        summary["five_new_partitions"][method][rule]={k:describe([index[focalkey(s,method,rule)][k] for s in SEEDS[1:]])
            for k in ["rejection","mean_loss_0.5","worst_loss_0.5","pd_clear_wrong_rate","dd_clear_wrong_rate",
                      "pd_deferred_rate","dd_deferred_rate"]}
    summary["five_new_partitions"][method]["contrasts"]={}
    for first,second in [(n,c) for n in NEW for c in [CONTROLS[0],CONTROLS[2]]]+[(NEW[1],NEW[0])]:
        summary["five_new_partitions"][method]["contrasts"][first+"_minus_"+second]={
            k:describe([index[focalkey(s,method,first)][k]-index[focalkey(s,method,second)][k] for s in SEEDS[1:]])
            for k in ["mean_loss_0.5","worst_loss_0.5","rejection","pd_clear_wrong_rate","dd_clear_wrong_rate"]}
js(R/"ROUND4_RESULTS.json",summary)
js(R/"INPUT_MANIFEST.json",inputs)
js(R/"VALIDATION.json",{
    "folds":checks,"inference_signature_has_no_reference_label":True,
    "all_90_calibrated_score_vectors_reproduced":True,"original_binary_and_task_paths_preserved":True,
    "all_three_control_rules_reused_exactly":True,"zero_target_full_coverage":True,
    "selected_pairs_optimal_within_enumeration":True,
    "selected_pairs_below_development_count_cap":True,
    "code_sha256":{n:sha(R/n) for n in ["run_round4.py","allocation_rules.py"]},
    "spec_sha256":sha(R/"ROUND4_SPEC.md"),"no_new_acquisition_or_probability_fitting":True,
})
print(json.dumps({"completed":True,"summary_counts":{k:v for k,v in summary.items() if k not in ["original_partition","five_new_partitions"]},
                  "original_adaptive":summary["original_partition"]["adaptive"]},indent=2))
