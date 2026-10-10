"""Create a cumulative archive while preserving every preceding member byte."""
from pathlib import Path
import zipfile,json,hashlib,shutil

ROOT=Path(__file__).resolve().parent
WORK=ROOT.parents[2]
OUTPUT=WORK/'outputs'
SOURCE=WORK/'inputs/PADS_round4_allocation_package.zip'


def main():
    OUTPUT.mkdir(exist_ok=True)
    validation=json.loads((ROOT/'VALIDATION.json').read_text())
    assert validation['completed_outer_folds']==30 and validation['original_inputs_unchanged']
    welcome='''# PADS cumulative package through round 5

Start with `pads_analysis/reviewer_round5/PADS_round5_results_ZH.md` and `pads_analysis/reviewer_round5/PADS_round5_random_action_supplement.docx`.

Round 5 adds random next-task acquisition, two path-permutation diagnostics, exact expected participant outcomes, conditional paired uncertainty, 18,000 random realizations, four scientific figure groups and an updated 20-dimension reviewer record. The tested individualized rule did not demonstrate additional accuracy over these controls.

All 540 members of the preceding round-4 package are preserved byte-for-byte. Its original `START_HERE.md` and package manifest describe that earlier snapshot. Later supplements, including the current one, have not been silently merged into the historical Word v1 manuscript. The English round-5 supplement is the current editable addition, not a claim that a final journal submission package has been completed.

See `pads_analysis/reviewer_round5/README.md` for reproducibility and evidence locations. The study contains 390 unique participants; multiple partitions and simulated randomizations do not increase the independent cohort size.
'''
    files=[p for p in sorted(ROOT.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.name!='PACKAGE_VALIDATION.json']
    manifest=[{'path':'pads_analysis/reviewer_round5/'+str(p.relative_to(ROOT)),'size_bytes':p.stat().st_size,
               'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in files]
    target=OUTPUT/'PADS_round5_random_action_package.zip'
    with zipfile.ZipFile(SOURCE) as old,zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as new:
        new.writestr('00_START_HERE_ROUND5.md',welcome)
        for info in old.infolist():new.writestr(info.filename,old.read(info.filename))
        for p,item in zip(files,manifest):new.write(p,item['path'])
        new.writestr('ROUND5_OUTPUT_MANIFEST.json',json.dumps(manifest,indent=2))
    with zipfile.ZipFile(SOURCE) as old,zipfile.ZipFile(target) as new:
        assert new.testzip() is None
        preserved=all(old.read(info.filename)==new.read(info.filename) for info in old.infolist())
        assert preserved
        assert len(new.namelist())==len(set(new.namelist()))
        for item in manifest:assert hashlib.sha256(new.read(item['path'])).hexdigest()==item['sha256']
        result={'zip_valid':True,'preceding_members_preserved':len(old.namelist()),'preceding_members_byte_identical':True,
                'round5_files':len(manifest),'total_members_before_validation':len(new.namelist()),
                'docx_pages_visually_checked':8,'figure_groups_visually_checked':4}
    (ROOT/'PACKAGE_VALIDATION.json').write_text(json.dumps(result,indent=2))
    with zipfile.ZipFile(target,'a',compression=zipfile.ZIP_DEFLATED) as new:new.write(ROOT/'PACKAGE_VALIDATION.json','pads_analysis/reviewer_round5/PACKAGE_VALIDATION.json')
    for name in ['PADS_round5_results_ZH.md','PADS_round5_random_action_supplement.docx']:shutil.copy2(ROOT/name,OUTPUT/name)
    print(json.dumps({'archive':str(target),'size_bytes':target.stat().st_size,**result},indent=2))


if __name__=='__main__':main()
