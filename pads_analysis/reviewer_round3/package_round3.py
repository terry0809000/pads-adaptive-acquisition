"""Validate and package a cumulative, source-traceable result snapshot."""
from pathlib import Path
import csv
import hashlib
import json
import shutil
import zipfile

R=Path(__file__).resolve().parent
B=R.parent
W=B.parent
D=W/"deliverables";D.mkdir(exist_ok=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):
    with p.open() as f:return list(csv.DictReader(f))
frozen=json.loads((R/"FREEZE.json").read_text())
for name,h in frozen["sha256"].items():
    assert sha(R/name)==h
inputs=json.loads((R/"INPUT_MANIFEST.json").read_text())
for name,h in inputs.items():
    assert sha(W/name)==h
v=json.loads((R/"VALIDATION.json").read_text())
assert len(v["folds"])==30 and v["all_90_calibrators_converged"]
assert v["slope_at_lower_bound_count"]==0
assert v["all_original_20_percent_outputs_reproduced"]
assert v["monotonic_rejection_sets"]
assert len(list((R/"tables").glob("*.csv")))==13
for ext in ["png","pdf","svg"]:
    assert len(list((R/"figures").glob(f"*.{ext}")))==4
scores=read(R/"tables/held_out_probability_scores.csv")
assert len(scores)==7020
ix={(r["seed"],r["method"],r["subject_id"]):r for r in scores}
outputs=read(R/"tables/held_out_selective_outputs.csv")
assert len(outputs)==105300
for r in outputs:
    base=ix[r["seed"],r["method"],r["subject_id"]]
    assert all(r[k]==base[k] for k in ["y","binary_prediction","movement_seconds","task_mask"])
    if r["rule"]=="original_margin" and float(r["target"])==.2:
        assert r["actual_output"]==base["original_actual_output"]
for r in read(R/"tables/repeat_rejection_metrics.csv"):
    for name in ["pd","dd"]:
        assert sum(int(r[f"{name}_{k}"]) for k in ["clear_correct","clear_wrong","deferred"])==int(r[name])
    assert int(r["n_clear"])+int(r["n_deferred"])==390
diagnostic=json.loads((R/"DIAGNOSTIC_VALIDATION.json").read_text())
assert len(diagnostic)==6
assert all(r["n_converged_draws"]==5000 and r["full_sample_scipy_match"] for r in diagnostic)
validation={
    "new_recalibrator_fits":90,"new_acquisition_fits":0,"unique_participants":390,
    "outer_partitions":6,"new_probability_rows":7020,"new_selective_output_rows":105300,
    "new_csv_tables":13,"new_figure_groups":4,"cumulative_distinct_figure_groups":17,
    "pre_fit_spec_and_implementation_hashes_retained":True,"all_input_hashes_retained":True,
    "binary_predictions_and_acquisition_costs_preserved":True,
    "all_original_margin_outputs_reproduced":True,"all_class_denominators_reconcile":True,
    "visual_inspection":["R9","R10","R11","R12"],"no_label_overlap_or_clipping":True,
    "diagnostic_bootstrap_converged":True,
    "word_manuscript_status":"Original v1 retained; English additions supplied separately",
}
(R/"PACKAGE_VALIDATION.json").write_text(json.dumps(validation,indent=2)+"\n")

prior=D/"PADS_round2_stability_package.zip"
assert sha(prior)=="be415ad1f1e802db1aa989dd68fbb812e5b6776ca7f3d97cd94d08045d432064"
target=D/"PADS_round3_calibration_package.zip"
manifest={}
with zipfile.ZipFile(prior) as old, zipfile.ZipFile(target,"w",zipfile.ZIP_DEFLATED) as out:
    old_manifest=json.loads(old.read("ROUND2_PACKAGE_MANIFEST.json"))
    for name,record in old_manifest.items():
        blob=old.read(name)
        assert hashlib.sha256(blob).hexdigest()==record["sha256"]
        out.writestr(name,blob)
        manifest[name]=record
    for p in sorted(R.rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts:
            name=str(p.relative_to(W))
            assert name not in manifest
            out.write(p,name)
            manifest[name]={"bytes":p.stat().st_size,"sha256":sha(p)}
    out.writestr("ROUND3_PACKAGE_MANIFEST.json",json.dumps(manifest,indent=2)+"\n")
    out.writestr("START_HERE.md",(R/"README.md").read_text())
with zipfile.ZipFile(target) as z:
    assert z.testzip() is None
    for name,record in manifest.items():
        assert hashlib.sha256(z.read(name)).hexdigest()==record["sha256"]
report=D/"PADS_round3_results_ZH.md"
shutil.copyfile(R/report.name,report)
print(json.dumps({"validation":validation,"files_in_archive":len(manifest)+2,
    "deliverables":[{"path":str(p),"bytes":p.stat().st_size,"sha256":sha(p)} for p in [report,target]]},indent=2))
