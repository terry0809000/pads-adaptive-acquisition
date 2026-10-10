"""Snapshot completed refits without touching a running worker's files."""
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
import hashlib,json
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parent
DEST=ROOT.parents[2]/'outputs/PADS_round7_computation_checkpoint.zip'

def main():
    completed=sorted((ROOT/'results').glob('*/COMPLETE.json'))
    files=[p for p in ROOT.iterdir() if p.is_file() and p.name!='run.log']
    for marker in completed:files.extend(p for p in marker.parent.iterdir() if p.is_file())
    manifest={'status':'Incomplete computation checkpoint; not the final manuscript package','created_utc':datetime.now(timezone.utc).isoformat(),
              'specified_replicates':500,'complete_replicates':len(completed),
              'completed_replicate_ids':[json.loads(p.read_text())['replicate'] for p in completed],
              'requires_predecessor':'PADS_round6_integrated_package.zip',
              'predecessor_sha256':'8a06720345ec9e0b1073257cd13a23608e2029310628ab6e3f1c9b893ceaf84a','entries':[]}
    with ZipFile(DEST,'w',ZIP_DEFLATED,compresslevel=6) as z:
        z.writestr('00_CHECKPOINT_NOT_FINAL.md',f'''# Incomplete computation checkpoint

This snapshot preserves {len(completed)} of the fixed 500 participant-bootstrap refits. It does not contain final inference, updated manuscripts or final figures. Completed replicate folders are immutable; unfinished folders were excluded.

To resume, extract PADS_round6_integrated_package.zip, then overlay this archive. Restore the feature-return directory from the predecessor package as documented there. Run reviewer_round7/run_bootstrap.py with the original environment and fixed code. It reuses COMPLETE.json folders and computes the remaining replicates. Do not change B, seeds, protected source hashes or outcome rules. Final interpretation requires all 500 completed replicates and validation.
''')
        for p in sorted(set(files)):
            name='pads_analysis/reviewer_round7/'+p.relative_to(ROOT).as_posix();data=p.read_bytes()
            z.writestr(name,data)
            manifest['entries'].append({'path':name,'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)})
        z.writestr('CHECKPOINT_MANIFEST.json',json.dumps(manifest,indent=2)+'\n')
    with ZipFile(DEST) as z:
        assert z.testzip() is None
        for entry in manifest['entries']:assert hashlib.sha256(z.read(entry['path'])).hexdigest()==entry['sha256']
    print(json.dumps({'path':str(DEST),'complete_replicates':len(completed),'bytes':DEST.stat().st_size,'sha256':hashlib.sha256(DEST.read_bytes()).hexdigest()},indent=2))

if __name__=='__main__':main()
