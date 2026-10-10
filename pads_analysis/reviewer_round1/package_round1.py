from pathlib import Path
import csv, hashlib, json, shutil, zipfile
import numpy as np

R=Path(__file__).resolve().parent
B=R.parent
W=B.parent
D=W/'deliverables'
D.mkdir(exist_ok=True)
old=json.loads((B/'summary/ALL_RESULTS.json').read_text())['primary']['extended_primary_contrast']
new=json.loads((R/'ROUND1_RESULTS.json').read_text())['joint_contrast']
assert np.allclose(new['conditional_interval_ba_pp'],np.array(old['balanced_accuracy']['conditional_interval_95'])*100,atol=1e-10)
assert np.allclose(new['conditional_interval_seconds'],old['mean_seconds']['conditional_interval_95'],atol=1e-10)
expected={'ANALYSIS_PLAN.md':'402b6efb13229cd13829b28fc1d9c2a582bb783f4b2a3d3eb14fb2f33549c79a',
          'run_analysis.py':'34ba6614d209d88215f764a44a650133624b002e4cb6128dd225dccc0adbc056'}
for file,sha in expected.items():assert hashlib.sha256((B/file).read_bytes()).hexdigest()==sha
report=(R/'PADS_reviewer_audit_round1_ZH.md').read_text()
for i in range(1,21):assert f'| {i} |' in report
for token in ['AGE_YOUNG','AGE_OLD','FEMALE','MALE','ATYPICAL_CI']:assert token not in report
for ext in ('png','pdf','svg'):assert len(list((R/'figures').glob(f'*.{ext}')))==10
assert len(list((R/'tables').glob('*.csv')))==7
with (R/'tables/score_reliability_bins.csv').open() as f:
    bins=list(csv.DictReader(f))
for m in ('adaptive','fixed_optimized','fixed_sequence_stop'):
    assert sum(int(r['n']) for r in bins if r['method']==m)==390
with (R/'tables/dd_subtype_intervals_and_outputs.csv').open() as f:
    for r in csv.DictReader(f):
        assert sum(int(r[x]) for x in ('clear_correct','clear_wrong','indeterminate'))==int(r['n'])

validation={'primary_hashes_unchanged':True,'primary_paired_intervals_reproduced':True,
'review_dimensions':20,'figure_groups':10,'new_figure_groups':5,'new_csv_tables':7,
'participant_pairing_n':390,'all_score_bins_reconcile':True,'dd_threeway_counts_reconcile':True,
'visual_review':'All five new PNG figures inspected; figures readable with no clipping or overlapping labels.',
'exploratory_analysis_labeled':True,'new_model_training':False}
(R/'ROUND1_VALIDATION.json').write_text(json.dumps(validation,indent=2)+'\n')
files=[p for p in R.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
files += [B/f for f in ('ANALYSIS_PLAN.md','ANALYSIS_FREEZE.json','run_analysis.py','requirements_analysis.txt','make_figures.py')]
files += [p for p in (B/'summary').glob('*') if p.is_file()]
files += list((B/'figures').glob('*'))
files += [W/'pads_features_received/PADS_features_return/participant_metadata.csv']
files=sorted(set(files))
manifest={str(p.relative_to(W)):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in files}
manifest_text=json.dumps(manifest,indent=2)+'\n'
with zipfile.ZipFile(D/'PADS_reviewer_round1_package.zip','w',zipfile.ZIP_DEFLATED) as z:
    for p in files:z.write(p,p.relative_to(W))
    z.writestr('ROUND1_PACKAGE_MANIFEST.json',manifest_text)
shutil.copy2(R/'PADS_reviewer_audit_round1_ZH.md',D/'PADS_reviewer_audit_round1_ZH.md')
with zipfile.ZipFile(D/'PADS_reviewer_round1_package.zip') as z:
    assert z.testzip() is None
    for name,r in manifest.items():assert hashlib.sha256(z.read(name)).hexdigest()==r['sha256']
print(json.dumps({'validation':validation,'files_in_package':len(files)+1,'deliverables':[
    {'path':str(D/n),'bytes':(D/n).stat().st_size,'sha256':hashlib.sha256((D/n).read_bytes()).hexdigest()}
    for n in ('PADS_reviewer_audit_round1_ZH.md','PADS_reviewer_round1_package.zip')]},indent=2))
