"""Outer-held-out evaluation of probability calibration and rejection targets."""
from pathlib import Path
import csv
import hashlib
import json
import sys
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit

R = Path(__file__).resolve().parent
B = R.parent
W = B.parent
sys.path.insert(0, str(B))
import acquisition_rules as rules

SEEDS = list(range(20260913, 20260919))
METHODS = ["adaptive", "fixed_optimized", "fixed_sequence_stop"]
RULES = ["original_margin", "raw_label_confidence", "calibrated_label_confidence"]
TARGETS = [0, .1, .2, .3, .4]
EPS = 1e-6
NDRAW = 5000
BOOTSEED = 2026091403
T = R / "tables"
T.mkdir(exist_ok=True)

def read(p):
    with p.open() as f:
        return list(csv.DictReader(f))

def write(name, rows):
    assert rows
    with (T / name).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

def js(p, d):
    p.write_text(json.dumps(d, indent=2, ensure_ascii=False, allow_nan=False) + "\n")

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def fit_logistic(p, y, positive=True, offset=False):
    x = logit(np.clip(p, EPS, 1-EPS))
    y = np.asarray(y)
    def objective(v):
        z = v[0] + x if offset else v[0] + v[1]*x
        residual = expit(z)-y
        gradient = [residual.mean()] if offset else [residual.mean(), (residual*x).mean()]
        return float(np.mean(np.logaddexp(0, z)-y*z)), np.asarray(gradient)
    result = minimize(
        objective, [0.] if offset else [0., 1.], jac=True, method="L-BFGS-B",
        bounds=None if offset or not positive else [(None, None), (1e-6, None)],
        options={"maxiter":2000, "ftol":1e-12, "gtol":1e-8},
    )
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError(f"Calibration optimizer did not converge: {result.message}")
    return {"intercept":float(result.x[0]), "slope":1. if offset else float(result.x[1]),
            "iterations":int(result.nit), "success":bool(result.success),
            "objective":float(result.fun)}

def calibrated(p, fit):
    return expit(fit["intercept"] + fit["slope"]*logit(np.clip(p, EPS, 1-EPS)))

def nll(p, y):
    q = np.clip(p, EPS, 1-EPS)
    return -y*np.log(q)-(1-y)*np.log1p(-q)

def ece(p, y):
    bins = np.minimum((np.asarray(p)*10).astype(int), 9)
    return float(sum(abs(np.sum((p-y)[bins==j])) for j in range(10))/len(y))

def wilson(k, n):
    z = 1.959963984540054
    p = k/n
    den = 1+z*z/n
    mid = (p+z*z/(2*n))/den
    radius = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return float(mid-radius), float(mid+radius)

def selective_metrics(y, binary, actual):
    y, binary, actual = map(np.asarray, (y, binary, actual))
    clear = actual != -1
    wrong = clear & (actual != y)
    pd, dd = y==1, y==0
    result = {
        "n":len(y), "pd":int(pd.sum()), "dd":int(dd.sum()),
        "coverage":float(clear.mean()), "rejection":float((~clear).mean()),
        "n_clear":int(clear.sum()), "n_deferred":int((~clear).sum()),
        "n_clear_wrong":int(wrong.sum()), "n_clear_correct":int((clear&~wrong).sum()),
        "selective_error":float(wrong.sum()/clear.sum()) if clear.any() else None,
        "binary_ba":float(.5*((binary[pd]==1).mean()+(binary[dd]==0).mean())) if pd.any() and dd.any() else None,
        "balanced_clear_wrong":float(.5*(wrong[pd].mean()+wrong[dd].mean())) if pd.any() and dd.any() else None,
        "balanced_deferral":float(.5*((~clear)[pd].mean()+(~clear)[dd].mean())) if pd.any() and dd.any() else None,
    }
    for c, name in [(1, "pd"), (0, "dd")]:
        group = y==c
        n = int(group.sum())
        for kind, mask in [("clear_correct", clear & (actual==y)), ("clear_wrong", wrong), ("deferred", ~clear)]:
            k = int((group & mask).sum())
            result[f"{name}_{kind}"] = k
            result[f"{name}_{kind}_rate"] = k/n if n else None
        result[f"{name}_selective_error"] = float(wrong[group].sum()/clear[group].sum()) if clear[group].any() else None
    for lam in [0, .25, .5, .75, 1]:
        result[f"balanced_loss_lambda_{lam:g}"] = result["balanced_clear_wrong"]+lam*result["balanced_deferral"] if pd.any() and dd.any() else None
    return result

def quantile_cut(confidence, target):
    return 0. if target == 0 else float(np.sort(confidence)[int(np.floor(target*len(confidence)))])

def interval(v):
    return [float(x) for x in np.quantile(v, [.025, .975])]

original_hashes = {
    "analysis_code":sha(B/"run_analysis.py"), "analysis_plan":sha(B/"ANALYSIS_PLAN.md"),
    "original_folds":sha(W/"pads_features_received/PADS_features_return/primary_outer_folds.csv"),
    "features":sha(W/"pads_features_received/PADS_features_return/PADS_motion_features.npz"),
}
assert original_hashes["analysis_code"] == "34ba6614d209d88215f764a44a650133624b002e4cb6128dd225dccc0adbc056"
assert original_hashes["analysis_plan"] == "402b6efb13229cd13829b28fc1d9c2a582bb783f4b2a3d3eb14fb2f33549c79a"
assert original_hashes["original_folds"] == "972ff7af4c4d4946f1166d43fcf18e5e6528915597870d741b608f9cf061377a"
assert original_hashes["features"] == "55d40eec48a3a7cfa5d891bd1cf9a9f56f187260f5487c2b7a8e037c013791c3"
metadata = {r["subject_id"]:r for r in read(W/"pads_features_received/PADS_features_return/participant_metadata.csv")}
development_labels = {r["subject_id"]:int(r["y"]) for r in read(
    W/"pads_features_received/PADS_features_return/primary_outer_folds.csv")}
score_rows, output_rows, parameter_rows, threshold_rows = [], [], [], []
fold_rows, validation, inputs = [], [], {}
arrays = {}

for seed in SEEDS:
    base = B/"results_primary" if seed==SEEDS[0] else B/"reviewer_round2/results"/f"seed_{seed}"
    for fold in range(5):
        d = base/f"fold_{fold}"
        for name in ["held_out_predictions.csv", "threshold_training_scores.npz", "training_ledger.json", "thresholds.json"]:
            inputs[str((d/name).relative_to(W))] = sha(d/name)
        ledger = json.loads((d/"training_ledger.json").read_text())
        saved_thresholds = json.loads((d/"thresholds.json").read_text())
        with np.load(d/"threshold_training_scores.npz", allow_pickle=False) as npz:
            dev_ids, dev_y = npz["subject_ids"].astype(str), npz["y"].astype(int)
            dev_scores = {m:npz[m].copy() for m in METHODS}
        test_ids = set(ledger["outer_test_ids"])
        assert len(dev_ids)==312 and len(test_ids)==78 and not set(dev_ids)&test_ids
        assert list(dev_ids)==ledger["final_fit"]["training_ids"]
        assert all(development_labels[sid]==int(y) for sid,y in zip(dev_ids,dev_y))
        mids = []
        for mid in ledger["middle_fits"]:
            held = set(mid["threshold_holdout_ids"])
            train = set(mid["fit"]["training_ids"])
            assert len(held)==104 and len(train)==208 and not held&train
            assert held|train==set(dev_ids)
            mids.extend(held)
        assert len(mids)==len(set(mids))==312
        source = read(d/"held_out_predictions.csv")
        for method in METHODS:
            rows = sorted([r for r in source if r["method"]==method], key=lambda r:r["subject_id"])
            ids = [r["subject_id"] for r in rows]
            assert set(ids)==test_ids
            y = np.array([int(r["y"]) for r in rows])
            p = np.array([float(r["score"]) for r in rows])
            binary = np.array([int(r["binary_prediction"]) for r in rows])
            original_actual = np.array([int(r["actual_output"]) for r in rows])
            dp = dev_scores[method]
            c = saved_thresholds[method]["binary_threshold"]
            assert c == rules.select_binary_threshold(dp, dev_y)
            assert np.array_equal(binary, (p>=c).astype(int))
            assert all(development_labels[sid]==int(label) for sid,label in zip(ids,y))
            fit = fit_logistic(dp, dev_y)
            cp, cdp = calibrated(p, fit), calibrated(dp, fit)
            assert np.array_equal(binary, (cp>=calibrated(np.array([c]),fit)[0]).astype(int))
            dbinary = (dp>=c).astype(int)
            prevalence = float(dev_y.mean())
            parameter_rows.append({"seed":seed, "outer_fold":fold, "method":method,
                                   "n_development":312, "n_test":78, "development_prevalence":prevalence,
                                   "binary_threshold_raw":c, "binary_threshold_calibrated":float(calibrated(np.array([c]),fit)[0]),
                                   **fit})
            for i,sid in enumerate(ids):
                score_rows.append({"seed":seed, "outer_fold":fold, "method":method, "subject_id":sid,
                    "y":int(y[i]), "binary_prediction":int(binary[i]), "raw_probability":float(p[i]),
                    "calibrated_probability":float(cp[i]), "null_probability":prevalence,
                    "movement_seconds":float(rows[i]["movement_seconds"]), "task_mask":int(rows[i]["task_mask"]),
                    "original_actual_output":int(original_actual[i])})
            dev_conf = [np.abs(dp-c), np.where(dbinary==1,dp,1-dp), np.where(dbinary==1,cdp,1-cdp)]
            test_conf = [np.abs(p-c), np.where(binary==1,p,1-p), np.where(binary==1,cp,1-cp)]
            for rule, dc, tc in zip(RULES, dev_conf, test_conf):
                previous = np.zeros(len(y), dtype=bool)
                for target in TARGETS:
                    cut = quantile_cut(dc, target)
                    deferred = tc<cut if target else np.zeros(len(y), dtype=bool)
                    assert not (previous & ~deferred).any()
                    previous = deferred
                    actual = np.where(deferred,-1,binary)
                    if rule=="original_margin" and target==.2:
                        assert abs(cut-saved_thresholds[method]["indeterminate_margin"])<1e-12
                        assert np.array_equal(actual, original_actual)
                    if target==0:
                        assert np.array_equal(actual,binary)
                    threshold_rows.append({"seed":seed,"outer_fold":fold,"method":method,"rule":rule,
                        "target":target,"cutoff":cut,"n_development":312,
                        "development_rejection":float((dc<cut).mean()) if target else 0.,
                        "outer_rejection":float(deferred.mean())})
                    fold_rows.append({"seed":seed,"outer_fold":fold,"method":method,"rule":rule,"target":target,
                        "target_error_pp":100*(deferred.mean()-target),**selective_metrics(y,binary,actual)})
                    for i,sid in enumerate(ids):
                        output_rows.append({"seed":seed,"outer_fold":fold,"method":method,"rule":rule,"target":target,
                            "subject_id":sid,"y":int(y[i]),"binary_prediction":int(binary[i]),"actual_output":int(actual[i]),
                            "confidence":float(tc[i]),"rejection_cutoff":cut,
                            "movement_seconds":float(rows[i]["movement_seconds"]),"task_mask":int(rows[i]["task_mask"])})
        validation.append({"seed":seed,"outer_fold":fold,"n_development":312,"n_test":78,
                           "middle_scores_participant_separation":True,"original_outputs_reproduced":True})
    print(f"Completed calibration and rejection evaluation for partition {seed}", flush=True)

write("calibration_parameters.csv", parameter_rows)
write("held_out_probability_scores.csv", score_rows)
write("held_out_selective_outputs.csv", output_rows)
write("rejection_thresholds.csv", threshold_rows)
write("fold_rejection_metrics.csv", fold_rows)

# Index once; all following numerical summaries use these fixed outer predictions.
score_groups, output_groups = {}, {}
for r in score_rows:
    score_groups.setdefault((r["seed"],r["method"]),[]).append(r)
for r in output_rows:
    output_groups.setdefault((r["seed"],r["method"],r["rule"],r["target"]),[]).append(r)
for rr in list(score_groups.values())+list(output_groups.values()):
    rr.sort(key=lambda r:r["subject_id"])
    assert len(rr)==len({r["subject_id"] for r in rr})==390

calmetrics, repeatmetrics, binrows, subgroups = [], [], [], []
for (seed,method),rr in score_groups.items():
    y = np.array([r["y"] for r in rr])
    binary = np.array([r["binary_prediction"] for r in rr])
    null = np.array([r["null_probability"] for r in rr])
    for version,field in [("raw","raw_probability"),("calibrated","calibrated_probability")]:
        p = np.array([r[field] for r in rr])
        diagnostic = fit_logistic(p,y,positive=False)
        citl = fit_logistic(p,y,offset=True)
        brier = float(np.mean((p-y)**2))
        null_brier = float(np.mean((null-y)**2))
        calmetrics.append({"seed":seed,"method":method,"version":version,"n":390,"pd":276,"dd":114,
            "brier":brier,"log_loss":float(nll(p,y).mean()),"ece_10_equal_width":ece(p,y),
            "mean_probability":float(p.mean()),"observed_prevalence":float(y.mean()),
            "citl_intercept":citl["intercept"],"diagnostic_free_intercept":diagnostic["intercept"],
            "diagnostic_slope":diagnostic["slope"],"null_brier":null_brier,"brier_skill":1-brier/null_brier})
        for kind,q,z in [("disease_probability",p,y),("assigned_label_confidence",np.where(binary==1,p,1-p),(binary==y).astype(int))]:
            bi = np.minimum((q*10).astype(int),9)
            for j in range(10):
                take = bi==j
                n = int(take.sum())
                if n:
                    k = int(z[take].sum())
                    lo,hi = wilson(k,n)
                    binrows.append({"seed":seed,"method":method,"version":version,"kind":kind,"bin":j,
                        "lower":j/10,"upper":(j+1)/10,"n":n,"events":k,"mean_prediction":float(q[take].mean()),
                        "observed_fraction":k/n,"wilson_lower":lo,"wilson_upper":hi})

for (seed,method,rule,target),rr in output_groups.items():
    y = np.array([r["y"] for r in rr])
    binary = np.array([r["binary_prediction"] for r in rr])
    actual = np.array([r["actual_output"] for r in rr])
    metrics = selective_metrics(y,binary,actual)
    repeatmetrics.append({"seed":seed,"method":method,"rule":rule,"target":target,
        "target_error_pp":100*(metrics["rejection"]-target),**metrics})
    if target==.2:
        for group,select in [
            ("age_under65",lambda m:float(m["age"])<65),("age_65plus",lambda m:float(m["age"])>=65),
            ("female",lambda m:int(m["gender_male"])==0),("male",lambda m:int(m["gender_male"])==1)]:
            ix = np.array([select(metadata[r["subject_id"]]) for r in rr])
            subgroups.append({"seed":seed,"method":method,"rule":rule,"target":target,"group":group,
                              **selective_metrics(y[ix],binary[ix],actual[ix])})

write("probability_metrics.csv", calmetrics)
write("repeat_rejection_metrics.csv", repeatmetrics)
write("reliability_bins.csv", binrows)
write("subgroup_rejection_metrics.csv", subgroups)

# Conditional, paired, class-stratified participant bootstrap for the original partition.
base_y = np.array([r["y"] for r in score_groups[SEEDS[0],METHODS[0]]])
rng = np.random.default_rng(BOOTSEED)
class_index = [np.flatnonzero(base_y==c) for c in (0,1)]
draws = [ix[rng.integers(0,len(ix),size=(NDRAW,len(ix)))] for ix in class_index]
bi = np.concatenate(draws,axis=1)
interval_rows, boot_rows = [], []
def add_interval(method,contrast,metric,point,values):
    lo,hi = interval(values)
    interval_rows.append({"method":method,"comparison":contrast,"metric":metric,"estimate":float(point),
                         "lower_95":lo,"upper_95":hi,"draws":NDRAW,"scope":"original partition; conditional on fitted procedures"})

for method in METHODS:
    rr = score_groups[SEEDS[0],method]
    assert [r["y"] for r in rr] == base_y.tolist()
    y = base_y
    raw = np.array([r["raw_probability"] for r in rr])
    cal = np.array([r["calibrated_probability"] for r in rr])
    brier_delta = (cal-y)**2-(raw-y)**2
    ll_delta = nll(cal,y)-nll(raw,y)
    b = brier_delta[bi].mean(1)
    ll = ll_delta[bi].mean(1)
    add_interval(method,"calibrated_minus_raw","brier",brier_delta.mean(),b)
    add_interval(method,"calibrated_minus_raw","log_loss",ll_delta.mean(),ll)
    for version,p in [("raw",raw),("calibrated",cal)]:
        add_interval(method,version,"brier",np.mean((p-y)**2),((p-y)**2)[bi].mean(1))
        add_interval(method,version,"log_loss",nll(p,y).mean(),nll(p,y)[bi].mean(1))
        eb = np.array([ece(p[ix],y[ix]) for ix in bi])
        add_interval(method,version,"ece_10_equal_width",ece(p,y),eb)
    statistics = {}
    for rule in RULES:
        oo = output_groups[SEEDS[0],method,rule,.2]
        assert [r["subject_id"] for r in oo] == [r["subject_id"] for r in rr]
        o = np.array([r["actual_output"] for r in oo])
        deferred = o==-1
        wrong = (o!=-1)&(o!=y)
        reb = deferred[bi].mean(1)
        wcb = .5*sum(wrong[ix].mean(1) for ix in draws)
        rcb = .5*sum(deferred[ix].mean(1) for ix in draws)
        er = wrong[bi].sum(1)/(~deferred)[bi].sum(1)
        point = selective_metrics(y,np.array([r["binary_prediction"] for r in rr]),o)
        statistics[rule] = {"rejection":reb,"balanced_loss_lambda_0.5":wcb+.5*rcb,"selective_error":er}
        for metric,values in statistics[rule].items():
            add_interval(method,rule,metric,point[metric],values)
        for c,name in [(0,"dd"),(1,"pd")]:
            db = deferred[draws[c]].mean(1)
            add_interval(method,rule,f"{name}_deferred_rate",deferred[y==c].mean(),db)
    for control in RULES[:2]:
        for metric in ["rejection","balanced_loss_lambda_0.5","selective_error"]:
            points = []
            for rule in [RULES[2],control]:
                oo = output_groups[SEEDS[0],method,rule,.2]
                mm = selective_metrics(y,[r["binary_prediction"] for r in oo],[r["actual_output"] for r in oo])
                points.append(mm[metric])
            add_interval(method,f"calibrated_label_confidence_minus_{control}",metric,points[0]-points[1],
                         statistics[RULES[2]][metric]-statistics[control][metric])
    for j in range(NDRAW):
        boot_rows.append({"method":method,"draw":j,"delta_brier":float(b[j]),"delta_log_loss":float(ll[j]),
                          "delta_rejection_calibrated_minus_margin":float(statistics[RULES[2]]["rejection"][j]-statistics[RULES[0]]["rejection"][j]),
                          "delta_balanced_loss_half_calibrated_minus_margin":float(statistics[RULES[2]]["balanced_loss_lambda_0.5"][j]-statistics[RULES[0]]["balanced_loss_lambda_0.5"][j])})
write("conditional_intervals.csv",interval_rows)
write("conditional_bootstrap_contrasts.csv",boot_rows)

cm = {(r["seed"],r["method"],r["version"]):r for r in calmetrics}
rm = {(r["seed"],r["method"],r["rule"],r["target"]):r for r in repeatmetrics}
def describe(values):
    return {"mean":float(np.mean(values)),"min":float(np.min(values)),"max":float(np.max(values))}
summary = {
    "scope":"Exploratory postprocessing fitted only to development middle-layer OOF scores",
    "partitions":SEEDS,"unique_participants":390,"new_recalibrator_fits":90,
    "new_acquisition_model_fits":0,"held_out_score_rows":len(score_rows),"held_out_rejection_rows":len(output_rows),
    "original_hashes":original_hashes,"bootstrap_draws":NDRAW,"bootstrap_seed":BOOTSEED,
    "original_partition":{},"five_new_partitions":{},
}
for m in METHODS:
    summary["original_partition"][m] = {
        "probability":{v:cm[SEEDS[0],m,v] for v in ["raw","calibrated"]},
        "rejection_target_20":{rule:rm[SEEDS[0],m,rule,.2] for rule in RULES},
    }
    summary["five_new_partitions"][m] = {
        "delta_brier":describe([cm[s,m,"calibrated"]["brier"]-cm[s,m,"raw"]["brier"] for s in SEEDS[1:]]),
        "delta_log_loss":describe([cm[s,m,"calibrated"]["log_loss"]-cm[s,m,"raw"]["log_loss"] for s in SEEDS[1:]]),
        "rejection_target_20":{rule:{
            key:describe([rm[s,m,rule,.2][key] for s in SEEDS[1:]])
            for key in ["rejection","selective_error","pd_deferred_rate","dd_deferred_rate","balanced_loss_lambda_0.5"]}
            for rule in RULES},
    }
js(R/"ROUND3_RESULTS.json",summary)
js(R/"INPUT_MANIFEST.json",inputs)
js(R/"VALIDATION.json",{
    "spec_sha256":sha(R/"ROUND3_SPEC.md"),"code_sha256":sha(Path(__file__)),
    "original_hashes":original_hashes,"folds":validation,
    "all_90_calibrators_converged":all(r["success"] for r in parameter_rows),
    "slope_min":min(r["slope"] for r in parameter_rows),"slope_max":max(r["slope"] for r in parameter_rows),
    "slope_at_lower_bound_count":sum(r["slope"]<=1.01e-6 for r in parameter_rows),
    "all_original_20_percent_outputs_reproduced":True,"zero_target_full_coverage":True,
    "monotonic_rejection_sets":True,"all_binary_decisions_and_costs_preserved":True,
    "outer_labels_never_used_to_fit_postprocessing":True,
    "held_out_diagnostic_fits_are_descriptive_only":True,
})
print(json.dumps({"completed":True,"n_recalibrators":90,"score_rows":len(score_rows),"selective_output_rows":len(output_rows),
                  "original_adaptive":summary["original_partition"]["adaptive"]},indent=2))
