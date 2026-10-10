"""Check numerical provenance, citation continuity and editable document structure."""
from pathlib import Path
import hashlib,json,re
from zipfile import ZipFile
import numpy as np
import pandas as pd
from docx import Document
from assemble_round7 import ci

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'integrated'
CHECKS=[]

def check(name,condition,details=None):
    assert condition,name
    CHECKS.append({'check':name,'passed':True,'details':details})

def tables(text):
    return [b for b in text.split('\n\n') if b.startswith('|')]

def main():
    computation=json.loads((ROOT/'COMPUTATION_LEDGER.json').read_text())
    audit=json.loads((ROOT/'VALIDATION.json').read_text())
    result=json.loads((ROOT/'ROUND7_RESULTS.json').read_text())
    original=(ROOT.parent/'reviewer_round6/integrated/PADS_manuscript_v1.md').read_text()
    manuscript=(OUT/'PADS_manuscript_v1.md').read_text()
    supplement=(OUT/'PADS_integrated_supplement.md').read_text()
    review=(OUT/'PADS_20_dimension_review_ZH.md').read_text()
    refs=json.loads((ROOT/'REFERENCES_WITH_LOCATIONS.json').read_text())
    check('500 specified attempts completed without discarded failures',computation['complete'] and computation['replicates']==500 and computation['failures']==[])
    check('Complete fit count',computation['logistic_fits']==10360000 and computation['additional_reproduction_fits']==4144)
    check('Original protected inputs unchanged',computation['original_inputs_unchanged'])
    check('All nested identity boundaries checked',audit['passed'] and audit['identity_boundaries_audited']==40000)
    check('Threshold fitting checked',audit['deterministic_thresholds_recomputed']==7500 and audit['random_binary_threshold_selections_checked']==2500)
    check('Original primary model script unchanged',hashlib.sha256((ROOT.parent/'run_analysis.py').read_bytes()).hexdigest()=='34ba6614d209d88215f764a44a650133624b002e4cb6128dd225dccc0adbc056')
    check('Original four main data tables preserved',tables(manuscript)[:4]==tables(original)[:4])
    check('Original primary interval retained','−5.97 to 4.20' in manuscript and '−0.83' in manuscript and '70.99%' in manuscript and '71.82%' in manuscript)
    check('All complete-refit paired intervals in main Table 5',all(ci(r,100 if r['metric']=='balanced_accuracy' else 1,True) in manuscript for r in result['contrasts'] if r['metric'] in ['balanced_accuracy','nominal_seconds']))
    check('Empirical interval scope explained','empirical stability intervals' in manuscript and 'not replacements for the archived primary confidence intervals' in manuscript)
    check('Unrefitted extensions distinguished','does not refit the calibration, class-aware allocation or mask-permutation extensions' in manuscript)
    check('No outdated full-refit absence claim',not any(x in manuscript+supplement for x in ['absence of full retraining in the bootstrap','No final all-cohort deployment model, live clinical test, compatible external PD/DD validation or full-development participant bootstrap is supplied','full-development participant bootstrap is absent']))
    intro=manuscript.split('## Introduction\n\n')[1].split('## Methods')[0]
    check('Research question placed near introduction opening',0<intro.index('Our primary research question')<1000)
    check('Four research questions stated',all('RQ'+str(i) in intro for i in range(1,5)))
    body=manuscript.split('## References')[0]
    citation_ids=[int(v) for m in re.findall(r'\[(\d+(?:,\s*\d+)*)\]',body) for v in m.split(',')]
    unique=list(dict.fromkeys(citation_ids))
    check('48 references substantively cited in main text',unique==list(range(1,49)))
    check('All references have provenance and claim boundaries',len(refs)==48 and all(r['inspection_level'] and r['support_boundary'] and r['main_locations'] for r in refs))
    check('DOIs distinct',len([r['doi'].lower() for r in refs if r.get('doi')])==len({r['doi'].lower() for r in refs if r.get('doi')}))
    check('NMSQuest DOI and published acquisition reference verified',any(r['id']==7 and r['doi']=='10.1002/mds.20844' for r in refs) and any(r['id']==10 and 'jmlr.org/papers/v26/23-1635.html' in r['inspected_url'] for r in refs))
    check('Twenty review questions retained',len(re.findall(r'^### Q\d\d',review,re.M))==20)
    check('Review no longer proposes completed priority as next work','推荐下一轮进行按人分层' not in review and '下一轮可优先补全流程按人重抽样' not in review)
    image_paths=[]
    for text,path in [(manuscript,OUT/'PADS_manuscript_v1.md'),(supplement,OUT/'PADS_integrated_supplement.md')]:
        for name,link in re.findall(r'!\[([^\]]*)\]\(([^)]+)\)',text):
            p=(path.parent/link).resolve();image_paths.append(p)
            check('Figure file '+p.stem,p.exists() and p.with_suffix('.pdf').exists() and p.with_suffix('.svg').exists())
    check('Twenty-five complete figure sets',len(image_paths)==25 and len(set(image_paths))==25)
    for name,ntables,nfigures in [('PADS_manuscript_v1.docx',5,7),('PADS_integrated_supplement.docx',8,18)]:
        doc=Document(OUT/name)
        check(name+' editable table and figure structure',len(doc.tables)==ntables and len(doc.inline_shapes)==nfigures)
        with ZipFile(OUT/name) as z:
            rel=z.read('word/_rels/document.xml.rels').decode()
            check(name+' genuine citation hyperlinks','relationships/hyperlink' in rel and 'https://doi.org/' in rel)
        if name=='PADS_manuscript_v1.docx':
            targets={r.target_ref for r in doc.part.rels.values() if r.reltype.endswith('/hyperlink')}
            expected={'https://doi.org/'+r['doi'] if r.get('doi') else r['inspected_url'] for r in refs}
            check('All 48 Word reference URLs match audited targets, including DOI parentheses',expected.issubset(targets))
    checks={'passed':True,'checks':CHECKS,'documents':{name:hashlib.sha256((OUT/name).read_bytes()).hexdigest() for name in ['PADS_manuscript_v1.docx','PADS_integrated_supplement.docx']}}
    (ROOT/'INTEGRATION_VALIDATION.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'checks':len(CHECKS),'passed':True,'references':48,'figure_sets':25},indent=2))

if __name__=='__main__':main()
