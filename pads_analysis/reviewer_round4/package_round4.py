"""Validate inference and saved outputs, then package the cumulative snapshot."""
from pathlib import Path
import hashlib
import inspect
import importlib.metadata
import json
import platform
import shutil
import sys
import zipfile
import numpy as np
import pandas as pd
from allocation_rules import apply_allocation, choose_allocation

R=Path(__file__).resolve().parent;B=R.parent;W=B.parent
D=W/"deliverables";D.mkdir(exist_ok=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(name,obj):(R/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False)+"\n")
def csv(name):return pd.read_csv(R/"tables"/name,dtype={"subject_id":str})
for name,h in json.loads((R/"FREEZE.json").read_text())["sha256"].items():
    assert sha(R/name)==h
for name,h in json.loads((R/"INPUT_MANIFEST.json").read_text()).items():
    assert sha(W/name)==h
v=json.loads((R/"VALIDATION.json").read_text())
assert len(v["folds"])==30 and all(x["nested_participant_separation"] for x in v["folds"])
assert all(v[k] for k in ["all_90_calibrated_score_vectors_reproduced","original_binary_and_task_paths_preserved",
    "all_three_control_rules_reused_exactly","zero_target_full_coverage","selected_pairs_optimal_within_enumeration",
    "selected_pairs_below_development_count_cap","no_new_acquisition_or_probability_fitting"])
assert list(inspect.signature(apply_allocation).parameters)==[
    "binary","calibrated_probability","cutoff_predicted_dd","cutoff_predicted_pd"]

o=csv("held_out_allocation_outputs.csv");assert len(o)==231660 and o.subject_id.nunique()==390
base=pd.read_csv(B/"reviewer_round3/tables/held_out_probability_scores.csv",dtype={"subject_id":str})
key=["seed","method","subject_id"]
merged=o.merge(base,on=key,suffixes=("_new","_old"),validate="many_to_one")
for name in ["y","binary_prediction","task_mask","movement_seconds","calibrated_probability"]:
    assert np.allclose(merged[f"{name}_new"],merged[f"{name}_old"],rtol=0,atol=1e-12)
ctrl=["original_margin","raw_label_confidence","calibrated_label_confidence"]
old=pd.read_csv(B/"reviewer_round3/tables/held_out_selective_outputs.csv",dtype={"subject_id":str})
controls=o[o.rule.isin(ctrl)].merge(old,on=key+["rule","target"],suffixes=("_new","_old"),validate="one_to_one")
assert len(controls)==105300
assert (controls.actual_output_new==controls.actual_output_old).all()
assert (o[o.target==0].actual_output==o[o.target==0].binary_prediction).all()
assert set(o.actual_output.unique())=={-1,0,1}

s=csv("selected_allocations.csv");assert len(s)==1620 and (s.target>0).sum()==1440
c=csv("focal_development_candidates.csv");assert len(c)==5670
focal=s[np.isclose(s.target,.2)&np.isclose(s.selection_lambda,.5)]
assert len(focal)==180 and (focal.development_rejected==62).all()
reject_all_members=0
for _,x in focal.iterrows():
    candidates=c[(c.seed==x.seed)&(c.outer_fold==x.outer_fold)&(c.method==x.method)].to_dict("records")
    chosen=choose_allocation(candidates,x.rule)
    for name in ["k_predicted_dd","k_predicted_pd","cutoff_predicted_dd","cutoff_predicted_pd"]:
        assert abs(chosen[name]-x[name])<1e-12
    reject_all_members+=int(x.k_predicted_dd==x.n_predicted_dd_development)
    reject_all_members+=int(x.k_predicted_pd==x.n_predicted_pd_development)
assert reject_all_members==0

new=o[~o.rule.isin(ctrl)].merge(s,on=["seed","outer_fold","method","rule","target","selection_lambda"],
                              validate="many_to_one")
recomputed=apply_allocation(new.binary_prediction,new.calibrated_probability,
                           new.cutoff_predicted_dd.to_numpy(),new.cutoff_predicted_pd.to_numpy())
assert np.array_equal(recomputed,new.actual_output)
assert (s.development_rejected<=s.nominal_rejected_count).all()
assert (s.k_predicted_dd+s.k_predicted_pd==s.development_rejected).all()

# Independently reconstruct all class counts, losses and binary BA from person rows.
repeats=csv("repeat_allocation_metrics.csv");assert len(repeats)==594
groupkeys=["seed","method","rule","target","selection_lambda"]
lookup=repeats.set_index(groupkeys)
for keys,x in o.groupby(groupkeys):
    saved=lookup.loc[keys]
    assert len(x)==x.subject_id.nunique()==390 and (x.y==1).sum()==276
    clear=x.actual_output!=-1
    class_losses=[]
    for y,label in [(0,"dd"),(1,"pd")]:
        keep=x.y==y;n=int(keep.sum())
        correct=int((keep&clear&(x.actual_output==x.y)).sum())
        wrong=int((keep&clear&(x.actual_output!=x.y)).sum())
        deferred=int((keep&~clear).sum())
        assert correct+wrong+deferred==n==int(saved[label])
        for kind,val in [("clear_correct",correct),("clear_wrong",wrong),("deferred",deferred)]:
            assert int(saved[f"{label}_{kind}"])==val
        losses=[(wrong+cost*deferred)/n for cost in [0,.25,.5,.75,1]]
        class_losses.append(losses)
    class_losses=np.array(class_losses)
    for j,cost in enumerate([0,.25,.5,.75,1]):
        assert abs(saved[f"mean_loss_{cost:g}"]-class_losses[:,j].mean())<1e-12
        assert abs(saved[f"worst_loss_{cost:g}"]-class_losses[:,j].max())<1e-12
    ba=.5*((x[x.y==0].binary_prediction==0).mean()+(x[x.y==1].binary_prediction==1).mean())
    assert abs(saved.binary_ba-ba)<1e-12
    assert abs(saved.rejection-(~clear).mean())<1e-12
assert len(list((R/"tables").glob("*.csv")))==9
for ext in ["png","pdf","svg"]:
    assert len(list((R/"figures").glob(f"*.{ext}")))==4

boots=csv("conditional_bootstrap_contrasts.csv")
intervals=csv("conditional_intervals.csv")
assert len(boots)==75000
for (method,comp),x in boots.groupby(["method","comparison"]):
    assert len(x)==x.draw.nunique()==5000
    for source,metric in [("delta_mean_loss_half","mean_loss_0.5"),("delta_worst_loss_half","worst_loss_0.5"),
                          ("delta_rejection","rejection")]:
        iv=intervals[(intervals.method==method)&(intervals.comparison==comp)&(intervals.metric==metric)].iloc[0]
        assert np.allclose(np.quantile(x[source],[.025,.975]),[iv.lower_95,iv.upper_95],rtol=0,atol=1e-12)

validation={
    "unique_participants":390,"outer_partitions":6,"new_acquisition_fits":0,"new_calibrator_fits":0,
    "selected_cutoff_pairs":1620,"positive_target_pairs":1440,"focal_candidates":5670,
    "held_out_configuration_rows":231660,"new_csv_tables":9,"new_figure_groups":4,"cumulative_figure_groups":21,
    "spec_and_analysis_hashes_retained":True,"input_hashes_retained":True,
    "all_original_controls_reproduced":True,"all_new_outer_outputs_reproduced_without_reference_labels":True,
    "all_focal_lexicographic_selections_verified":True,"all_partition_class_counts_and_losses_recomputed":True,
    "all_binary_predictions_and_movement_costs_preserved":True,"bootstrap_intervals_match_saved_draws":True,
    "focal_reject_all_predicted_class_member_cutoffs":reject_all_members,
    "visual_inspection":["R13","R14","R15","R16"],"no_label_overlap_or_clipping":True,
    "word_status":"Original v1 retained; later English additions supplied separately",
    "evaluation_status":"Exploratory reuse of previously examined cohort; not independent external validation"
}
write("PACKAGE_VALIDATION.json",validation)
write("RUNTIME_VERSIONS.json",{"python":platform.python_version(),**{
    x:importlib.metadata.version(x) for x in ["numpy","pandas","scipy","scikit-learn","matplotlib"]}})
if "--validate-only" in sys.argv:
    print(json.dumps(validation,indent=2));sys.exit(0)

prior=D/"PADS_round3_calibration_package.zip"
assert sha(prior)=="b21e21ea26b70677ea016819620260dd152f1e37e60bd3f107c810a10b9c49b0"
target=D/"PADS_round4_allocation_package.zip";manifest={}
with zipfile.ZipFile(prior) as oldzip,zipfile.ZipFile(target,"w",zipfile.ZIP_DEFLATED) as out:
    old_manifest=json.loads(oldzip.read("ROUND3_PACKAGE_MANIFEST.json"))
    for name,record in old_manifest.items():
        blob=oldzip.read(name)
        assert hashlib.sha256(blob).hexdigest()==record["sha256"]
        out.writestr(name,blob);manifest[name]=record
    for p in sorted(R.rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts:
            name=str(p.relative_to(W));assert name not in manifest
            out.write(p,name)
            manifest[name]={"bytes":p.stat().st_size,"sha256":sha(p)}
    out.writestr("ROUND4_PACKAGE_MANIFEST.json",json.dumps(manifest,indent=2)+"\n")
    out.writestr("START_HERE.md",(R/"README.md").read_text())
with zipfile.ZipFile(target) as z:
    assert z.testzip() is None
    for name,record in manifest.items():
        assert hashlib.sha256(z.read(name)).hexdigest()==record["sha256"]
report=D/"PADS_round4_results_ZH.md"
shutil.copyfile(R/report.name,report)
print(json.dumps({"validation":validation,"files_in_archive":len(manifest)+2,
    "deliverables":[{"path":str(p),"bytes":p.stat().st_size,"sha256":sha(p)} for p in [report,target]]},indent=2))
