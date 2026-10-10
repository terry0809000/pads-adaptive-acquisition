"""Create claim-specific citation records without redistributing source articles."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parent
CACHE = ROOT.parents[2] / 'reference_cache_round7'

SUPPORT = {
25: ('Development stability', 'All model-building steps should be repeated when examining development instability; our grouped nested implementation is an extension, not their exact estimator.'),
26: ('Nested evaluation', 'Selection must be separated from evaluation; this paper does not certify our implementation or remove all small-sample uncertainty.'),
27: ('Feature learning and limited samples', 'Feature selection and validation design can bias small-sample assessments; no numerical effect from its simulations is transferred to PADS.'),
28: ('Leakage audit', 'Leakage requires explicit checks throughout the data and modeling workflow; successful local checks are not proof against every conceivable bias.'),
30: ('Internal versus external evaluation', 'Model-building evaluation and new-population validation answer different questions; our new bootstrap is not optimism correction or external validation.'),
31: ('Calibration and transportability', 'Discrimination, calibration, population compatibility, and decision context need separate assessment; source guidance motivates, but does not validate, our protocol.'),
32: ('Precision and future validation', 'External-validation sample size should target precision for the intended model and population; no external sample-size calculation is claimed here.'),
33: ('Internal validation', 'Apparent performance is optimistic and resampling can assess model development; the original paper studies a different logistic model and bootstrap estimator.'),
34: ('Correlated validation errors', 'Fold errors are correlated and naive variance estimates can be misleading; no universal variance correction is claimed.'),
35: ('Class balance and probability calibration', 'Resampling class-balance corrections in logistic regression can impair calibration; this is related evidence, not a direct replication of our class-weighted method.'),
36: ('Choice of comparator model', 'Algorithmic complexity does not automatically improve clinical prediction; its review does not imply that logistic regression is universally optimal.'),
37: ('Penalization and limited information', 'Penalization does not guarantee stable predictions in small samples; our C values are fixed, so their tuning-parameter mechanism is not asserted as our proven cause.'),
38: ('Reference diagnosis', 'Historical clinicopathological diagnostic accuracy varies; this meta-analysis does not estimate the error rate of PADS labels.'),
39: ('Reference diagnosis and disease stage', 'A later neuropathological series found better clinical diagnostic accuracy, varying by expertise and stage; no direct numerical comparison with PADS is made.'),
40: ('Mobile data collection', 'mPower illustrates repeated remote movement and symptom measurement; its controls and collection scheme differ from PADS differential diagnoses.'),
41: ('Measurement reliability', 'Active smartphone measures were examined for feasibility, reliability and clinical association; this is not validation of our descriptors or an abbreviated PADS examination.'),
42: ('Remote assessment and burden', 'Unsupervised smartwatch measurements have been evaluated for adherence, reliability, and clinical association; actual completion and use differ from nominal task duration.'),
43: ('Digital PD assessment', 'WATCH-PD assessed early untreated PD versus matched controls using multiple devices; its task and population are not a PADS diagnostic benchmark.'),
44: ('Minimal sensing and symptom monitoring', 'A wrist accelerometer can support specific tremor/bradykinesia measures in its evaluated setting; symptom monitoring does not establish differential diagnostic utility.'),
45: ('Measurement validation', 'Verification, analytical validation, and clinical validation are separate requirements; finite feature values and model accuracy alone do not satisfy V3.'),
47: ('Translation to clinical practice', 'Wearable technical capability and clinical usefulness need distinct evidence, including adherence and integration; this is a task-force perspective rather than trial evidence for our policy.'),
48: ('Spectral estimation', 'Attribution of Welch spectral estimation only; the window, overlap and aggregation choices belong to our implementation.'),
49: ('Principal component analysis', 'PCA selects data-dependent variance directions; retained variance does not guarantee preservation of diagnostic information.'),
50: ('Diagnostic accuracy reporting', 'STARD-AI motivates transparent description of the AI index test, evaluation data, and potential bias; no formal checklist-compliance claim is made. The July 2026 correction adds a consortium member and does not revise the cited reporting principles.'),
}


def main():
    previous = json.loads((ROOT.parent / 'reviewer_round6/REFERENCES_VERIFIED.json').read_text())
    records = []
    for ref in previous:
        item = dict(ref)
        item['legacy_round6_id'] = item['id']
        item['id'] = item['manuscript_number']
        item.pop('manuscript_number')
        if item['id'] == 7:
            nms = json.loads((CACHE / 'nmsquest.txt').read_text())['resultList']['result'][0]
            assert nms['id'] == '16547944' and nms['doi'] == '10.1002/mds.20844'
            item.update(doi=nms['doi'], inspected_url='https://pubmed.ncbi.nlm.nih.gov/16547944/',
                        inspection_level='Published abstract and bibliographic record via Europe PMC, plus official instrument page',
                        support_boundary='A 30-item symptom screen intended to prompt further investigation; not a graded disability scale or a PD/DD diagnostic test.')
        if item['id'] == 10:
            item.update(inspected_url='https://www.jmlr.org/papers/v26/23-1635.html',
                        inspection_level='Published JMLR abstract and bibliographic record; prior preprint record also retained in the historical archive')
        records.append(item)
    for key, (role, boundary) in SUPPORT.items():
        if key == 34:
            item = {'id': key, 'authors': 'Bengio Y, Grandvalet Y.', 'title': 'No Unbiased Estimator of the Variance of K-Fold Cross-Validation.',
                    'publication': 'Journal of Machine Learning Research. 2004;5:1089–1105.', 'doi': '',
                    'inspected_url': 'https://www.jmlr.org/papers/v5/grandvalet04a.html', 'inspection_level': 'Publisher abstract and bibliographic record'}
        else:
            path = CACHE / f'ref_{key}.json'
            source = json.loads(path.read_text())
            cr = source.get('crossref', {}).get('message', {})
            results = source.get('epmc', {}).get('resultList', {}).get('result', [])
            ep = results[0] if results else {}
            if ep:
                assert ep['doi'].lower() == source['doi'].lower()
                info = ep['journalInfo']
                authors = ep['authorString'].rstrip('.').split(', ')
                author_text = ', '.join(authors[:3]) + (', et al.' if len(authors) > 3 else '.')
                journal = info['journal'].get('medlineAbbreviation') or info['journal']['title']
                journal = {'Biom J':'Biometrical Journal','PLoS One':'PLOS ONE','Patterns (N Y)':'Patterns',
                           'J Clin Epidemiol':'Journal of Clinical Epidemiology','J Am Med Inform Assoc':'Journal of the American Medical Informatics Association',
                           'Mov Disord':'Movement Disorders','Sci Data':'Scientific Data','NPJ Digit Med':'npj Digital Medicine',
                           'Nat Med':'Nature Medicine',
                           'NPJ Parkinsons Dis':"npj Parkinson's Disease",'Philos Trans A Math Phys Eng Sci':'Philosophical Transactions of the Royal Society A'}.get(journal,journal)
                issue = '(' + info['issue'] + ')' if info.get('issue') else ''
                publication = f"{journal}. {info['yearOfPublication']};{info['volume']}{issue}:{ep['pageInfo'].replace('-', '–')}."
                title = ep['title'].rstrip('.') + '.'
                url = 'https://pubmed.ncbi.nlm.nih.gov/' + ep['id'] + '/'
            else:
                assert key == 48 and cr['DOI'].lower() == source['doi'].lower()
                author_text = 'Welch PD.'
                publication = 'IEEE Transactions on Audio and Electroacoustics. 1967;15(2):70–73.'
                title = cr['title'][0].rstrip('.') + '.'
                url = 'https://ieeexplore.ieee.org/document/1161901/'
            level = 'Published abstract and bibliographic metadata via Europe PMC; DOI matched to registry when available'
            if key in [30,31,32]:
                level = 'Relevant evaluation, calibration, or precision sections of the full article via Europe PMC XML, plus bibliographic metadata'
                url = 'https://europepmc.org/articles/' + ep['pmcid']
            if key == 25:
                level = 'Published abstract and selected methodological sections on the publisher page, plus Europe PMC metadata'
                url = 'https://onlinelibrary.wiley.com/doi/10.1002/bimj.202200302'
            if key == 48:
                level = 'Publisher method description and DOI metadata; attribution only'
            if key == 50:
                level = 'Published abstract and official bibliographic record; full author-correction text checked (consortium membership only); full checklist article was not accessible'
                url = 'https://www.nature.com/articles/s41591-025-03953-8'
            item = {'id': key, 'authors': author_text, 'title': title, 'publication': publication, 'doi': source['doi'],
                    'inspected_url': url, 'inspection_level': level, 'source_metadata_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        item['title']=item['title'].replace('?.','?').replace('!.','!')
        item.update(role=role, support_boundary=boundary, checked_date='2026-09-13')
        if key == 50:
            item['correction_doi']='10.1038/s41591-026-04570-9'
            item['correction_url']='https://www.nature.com/articles/s41591-026-04570-9'
            item['correction_scope']='A missing consortium member was added; no extra substantive reference counted.'
        records.append(item)
    assert len(records) == 48 and len({r['id'] for r in records}) == 48
    (ROOT / 'REFERENCES_AUDITED.json').write_text(json.dumps(sorted(records, key=lambda x: x['id']), ensure_ascii=False, indent=2) + '\n')
    exclusions = [{'doi': '10.1136/bmj.m441', 'reason': 'Metadata verified but full text was not obtained; excluded from new claims because accessible sources already support the required sample-size argument.'}]
    (ROOT / 'REFERENCE_SCREENING_NOTES.json').write_text(json.dumps(exclusions, ensure_ascii=False, indent=2) + '\n')
    print('48 distinct reference records assembled; claim support and access levels retained.')


if __name__ == '__main__':
    main()
