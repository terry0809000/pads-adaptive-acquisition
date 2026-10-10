"""Package the second-round result snapshot with its reproducibility inputs."""
from pathlib import Path
import hashlib
import json
import shutil
import zipfile

R = Path(__file__).resolve().parent
B = R.parent
W = B.parent
D = W / "deliverables"
D.mkdir(exist_ok=True)

expected = {
    B / "ANALYSIS_PLAN.md": "402b6efb13229cd13829b28fc1d9c2a582bb783f4b2a3d3eb14fb2f33549c79a",
    B / "run_analysis.py": "34ba6614d209d88215f764a44a650133624b002e4cb6128dd225dccc0adbc056",
    W / "pads_features_received/PADS_features_return/PADS_motion_features.npz":
        "55d40eec48a3a7cfa5d891bd1cf9a9f56f187260f5487c2b7a8e037c013791c3",
    W / "pads_features_received/PADS_features_return/primary_outer_folds.csv":
        "972ff7af4c4d4946f1166d43fcf18e5e6528915597870d741b608f9cf061377a",
}
for p, h in expected.items():
    assert hashlib.sha256(p.read_bytes()).hexdigest() == h

ledger = json.loads((R / "VALIDATION.json").read_text())
assert len(ledger["all_25_nested_ledgers_passed"]) == 25
assert all(x["nested_isolation_verified"] for x in ledger["all_25_nested_ledgers_passed"])
summary = json.loads((R / "STABILITY_RESULTS.json").read_text())
assert summary["n_unique_participants"] == 390
assert summary["n_new_prediction_rows"] == 33150
assert len(list((R / "tables").glob("*.csv"))) == 8
for ext in ["png", "pdf", "svg"]:
    assert len(list((R / "figures").glob(f"*.{ext}"))) == 3
    assert len(list((B / "reviewer_round1/figures").glob(f"*.{ext}"))) == 10

validation = {
    "new_repeats": 5, "new_outer_folds": 25, "independent_participants": 390,
    "new_figure_groups": 3, "cumulative_distinct_figure_groups": 13,
    "new_csv_tables": 8, "all_original_hashes_retained": True,
    "new_figures_visually_inspected": ["R6", "R7", "R8"],
    "r8_legend_overlap_corrected": True,
    "all_new_nested_ledgers_checked": True,
    "external_evaluation_completed": False,
    "manuscript_word_status": "Original v1; second-round English additions supplied separately",
}
(R / "PACKAGE_VALIDATION.json").write_text(json.dumps(validation, indent=2) + "\n")

files = []
for folder in [
    R, B / "reviewer_round1", B / "summary", B / "figures",
    B / "results_primary", B / "results_without_nms",
    W / "pads_features_received/PADS_features_return",
    W / "pads_audit_received/PADS_audit_return",
]:
    files.extend(p for p in folder.rglob("*")
                 if p.is_file() and "__pycache__" not in p.parts)
files.extend(p for p in B.iterdir() if p.is_file())
files.extend(p for p in (B / "manuscript").glob("*")
             if p.is_file() and p.suffix in (".docx", ".md", ".pdf"))
files = sorted(set(files))
manifest = {
    str(p.relative_to(W)): {
        "bytes": p.stat().st_size,
        "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
    } for p in files
}
target = D / "PADS_round2_stability_package.zip"
with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as z:
    for p in files:
        z.write(p, p.relative_to(W))
    z.writestr("ROUND2_PACKAGE_MANIFEST.json", json.dumps(manifest, indent=2) + "\n")
    z.writestr("START_HERE.md", (R / "README.md").read_text())
with zipfile.ZipFile(target) as z:
    assert z.testzip() is None
    for name, row in manifest.items():
        assert hashlib.sha256(z.read(name)).hexdigest() == row["sha256"]
report = D / "PADS_round2_results_ZH.md"
shutil.copyfile(R / report.name, report)
print(json.dumps({
    "validation": validation, "files_in_archive": len(files) + 2,
    "deliverables": [
        {"path": str(p), "bytes": p.stat().st_size,
         "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
        for p in [report, target]
    ]
}, indent=2))
