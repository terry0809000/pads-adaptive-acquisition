from pathlib import Path
import hashlib,json
root=Path(__file__).resolve().parent
target=root/'pads_analysis/reviewer_round5/INPUT_MANIFEST.json'
if target.exists():
    raise SystemExit('An input manifest already exists; use a fresh reproduction directory. Never replace an existing audit ledger.')
paths=json.loads((root/'historical_audit/round5_input_manifest.json').read_text())
rows=[]
for row in paths:
    p=root/row['path']
    rows.append(dict(path=row['path'],bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
target.write_text(json.dumps(rows,indent=2))
print('Recorded current reproduction inputs before random-control fitting:',len(rows))
