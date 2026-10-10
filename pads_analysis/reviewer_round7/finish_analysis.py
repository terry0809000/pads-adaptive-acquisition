"""Run result-dependent integration only after the frozen computation completes."""
from pathlib import Path
import json,subprocess,sys,time

ROOT=Path(__file__).resolve().parent

def main():
    last=-1
    while not (ROOT/'COMPUTATION_LEDGER.json').exists():
        n=len(list((ROOT/'results').glob('*/COMPLETE.json')))
        if n//25!=last:
            print(f'Completed {n}/500 refits; integration waits for the complete computation ledger.',flush=True)
            last=n//25
        time.sleep(20)
    ledger=json.loads((ROOT/'COMPUTATION_LEDGER.json').read_text())
    assert ledger['complete'] and ledger['replicates']==500 and not ledger['failures']
    for name in ['summarize_bootstrap.py','plot_bootstrap.py','assemble_round7.py','build_review_audit.py']:
        print('Running '+name,flush=True)
        subprocess.run([sys.executable,str(ROOT/name)],check=True)
    print('Numeric integration complete. DOCX authoring and rendered-page review remain.',flush=True)

if __name__=='__main__':main()
