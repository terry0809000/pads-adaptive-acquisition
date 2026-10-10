"""Integrate real complete-refit results with the preserved primary evidence."""
from pathlib import Path
import json
import re
import numpy as np
import pandas as pd
import manuscript_revisions as prose
import methods_revisions as methods

ROOT=Path(__file__).resolve().parent
PREV=ROOT.parent/'reviewer_round6/integrated'
OUT=ROOT/'integrated'
METHODS=['adaptive','fixed_optimized','fixed_sequence_stop','random_next_stop']
NAMES={'adaptive':'Adaptive','fixed_optimized':'Optimized fixed','fixed_sequence_stop':'Fixed sequence + stopping','random_next_stop':'Random next task'}
CONTRASTS=[('adaptive','fixed_optimized'),('adaptive','fixed_sequence_stop'),('adaptive','random_next_stop'),('fixed_sequence_stop','fixed_optimized')]
CITATION=re.compile(r'\[(\d+(?:,\s*\d+)*)\]')

def ci(r,scale=1,signed=False,digits=2):
    f=('+' if signed else '')+'.'+str(digits)+'f'
    return f"{format(r['median']*scale,f)} ({format(r['low']*scale,f)}, {format(r['high']*scale,f)})"

def paragraph_replace(text,starts,replacement):
    blocks=text.split('\n\n')
    matches=[i for i,b in enumerate(blocks) if b.startswith(starts)]
    assert len(matches)==1,(starts,matches)
    blocks[matches[0]]=replacement
    return '\n\n'.join(blocks)

def main():
    result=json.loads((ROOT/'ROUND7_RESULTS.json').read_text())
    validation=json.loads((ROOT/'VALIDATION.json').read_text())
    assert result['replicates']==500 and validation['passed']
    OUT.mkdir(exist_ok=True)
    old=(PREV/'PADS_manuscript_v1.md').read_text()
    supplement=(PREV/'PADS_integrated_supplement.md').read_text()
    paired={(r['method'],r['comparator'],r['metric']):r for r in result['contrasts']}
    absolute={(r['method'],r['metric']):r for r in result['absolute']}
    joint={(r['method'],r['comparator']):r for r in result['joint_accuracy_time']}
    a=paired['adaptive','fixed_optimized','balanced_accuracy']
    t=paired['adaptive','fixed_optimized','nominal_seconds']
    ar=paired['adaptive','random_next_stop','balanced_accuracy']
    art=paired['adaptive','random_next_stop','nominal_seconds']
    sf=paired['fixed_sequence_stop','fixed_optimized','balanced_accuracy']
    st=paired['fixed_sequence_stop','fixed_optimized','nominal_seconds']
    unique=result['unique_per_resample']
    training=result['unique_training_per_fold']
    pdg=result['participant_diagnostics']['adaptive']
    people=result['person_prediction_counts']
    # Interpretation is checked again against the complete tables before delivery.
    supported=(a['low']>0 and t['high']<0)
    if supported:
        refit_claim='The full-refit accuracy and time intervals favored adaptive acquisition, a finding that differs from the original primary comparison and requires independent assessment.'
    else:
        refit_claim='The complete-refit analysis did not establish a consistent joint accuracy and acquisition-time advantage for adaptive selection.'
    if a['low']<0<a['high']:
        accuracy_scope='The accuracy interval included both gain and loss.'
    elif a['high']<0:
        accuracy_scope='The central accuracy interval lay below zero under this resampling procedure.'
    else:
        accuracy_scope='The central accuracy interval lay above zero under this resampling procedure.'
    abstract=f'''## Abstract

**Objective:** To determine whether individualized selection of smartwatch movement tasks improves balanced accuracy over an optimized fixed examination for Parkinson disease (PD) versus differential diagnoses (DD) under a common acquisition-time cap.

**Methods:** We analyzed 276 participants with PD and 114 with DD from PADS using nested participant-level cross-validation. Both strategies used the same clinical inputs, task representations and state classifiers, with a nominal cap of 71.68 seconds. The primary endpoint included every participant's binary prediction, including those whose actual output was indeterminate. Exploratory analyses examined stopping, random acquisition, model settings, probability calibration and class-specific deferral. A further 500 class-stratified participant-bootstrap replicates repeated representation learning, model fitting, policy development and threshold selection, keeping duplicate identities together at every validation level.

**Results:** Primary balanced accuracy was 70.99% for adaptive and 71.82% for fixed assessment; the paired difference was −0.83 percentage points (conditional 95% interval −5.97 to 4.20). Mean nominal times were 57.45 and 49.15 seconds, and deferral occurred in 26.4% and 24.1%. Random controls showed no consistent advantage from individualized task identity. Across complete refits, the median adaptive-minus-fixed accuracy difference was {ci(a,100,True)} percentage points and the time difference was {ci(t,1,True)} seconds; parentheses denote central 95% empirical stability intervals. Recalibration improved Brier score but shifted clear errors between PD and DD.

**Conclusions:** The specified policy did not demonstrate a consistent benefit from individualized task selection in this cohort. Complete refitting extends the assessment of development uncertainty, without establishing equivalence, external validity or clinical utility. Strong fixed and stopping comparators remain necessary when evaluating personalization.
'''
    if supported:
        abstract=abstract.replace('The specified policy did not demonstrate a consistent benefit from individualized task selection in this cohort.', 'The primary comparison did not favor adaptive selection, although the exploratory full-refit analysis favored it under the specified resampling procedure.')

    results=old.split('## Results\n\n',1)[1].split('## Discussion\n\n',1)[0]
    results='## Results\n\n'+results
    results=results.replace('### Random controls did not reveal useful task personalization','### Random acquisition and task identity')
    results=results.replace('### Calibration improved probability error but altered rejection consequences','### Probability quality and deferral consequences')
    results=results.replace('All 390 eligible participants contributed one outer-held-out prediction to every method and sensitivity scenario.', 'In the original evaluation, all 390 eligible participants contributed one outer-held-out prediction to every method and sensitivity scenario. Age ranged from 40 to 85 years.')
    result_edits={
        '### Cohort and completed evaluation':'### Cohort characteristics',
        'Recorded female gender occurred in 81 of 276 PD participants (29.3%) and 57 of 114 DD participants (50.0%).':'The dataset recorded female gender for 81 of 276 PD participants (29.3%) and 57 of 114 DD participants (50.0%).',
        'The clinical-only, Relaxed-only, and full-protocol benchmarks':'The clinical-only, clinical-plus-Relaxed, and full-protocol benchmarks',
        'Figure S1 shows the prespecified budget curves; their non-monotonicity was retained rather than selecting a favorable cap after evaluation.':'Accuracy did not increase monotonically across the prespecified time caps (Figure S1).',
        'Figure S3 retains the paired joint accuracy–time distribution.':'Figure S3 shows paired accuracy and time differences under fixed-prediction resampling.',
        'Class-specific expected outputs, acquisition randomness and assignment support are shown in Figures S11–S13.':'Figure S11 reports class-specific expected outputs. Figures S12–S13 show variation from random acquisition and the available task-mask exchanges.',
        'Target-transfer ranges and selective-error curves are retained in Figures S6–S7.':'Figures S6–S7 show how development rejection targets transferred to the held-out samples and how errors varied with coverage.',
        'in the recorded-female stratum':'among participants recorded as female',
        'in the recorded-male stratum':'among participants recorded as male',
    }
    for before,after in result_edits.items():
        assert before in results,before
        results=results.replace(before,after)
    results=paragraph_replace(results,'The original analysis produced 13,260',
      'The original evaluation produced complete paired predictions across 17 scenarios and two input configurations. Checks confirmed separation of participants at every nested fitting level, whole-task feasibility and agreement with the frozen folds. Perturbing unacquired test features left the current acquisition decision unchanged. All clinical model fits completed without convergence warnings. Detailed fitting and prediction counts are provided in Supplementary Methods S2.')
    results=paragraph_replace(results,'All six one-factor configurations completed,',
      'All six one-factor configurations completed with finite predictions, feasible paths, complete method pairing and converged fits. Adaptive-minus-fixed balanced accuracy ranged from −5.82 percentage points with four components to +0.18 points with fifteen neighbors; five of six differences were negative. The four-component contrast was −5.82 points (conditional interval −9.79 to −2.02), an exploratory marginal result. Adaptive acquisition took longer in all six settings, by 0.45–13.52 seconds (Figure 4C–D; Table S1). Figure S14 gives absolute accuracy, time and deferral. Fixed-model outputs were identical across neighborhood sizes because this parameter does not enter fixed-subset selection. Supplementary Methods S3 records the additional fits and reproduction checks.')
    counts=joint['adaptive','fixed_optimized']
    boot_results=f'''### Participant resampling with complete refitting

Across all 500 specified complete refits, the median adaptive-minus-fixed balanced-accuracy difference was {ci(a,100,True)} percentage points, and the nominal-time difference was {ci(t,1,True)} seconds (central 95% empirical stability intervals; Figure 7 and Table 5). {accuracy_scope} Adaptive acquisition had higher accuracy in {counts['higher_ba_count']}/500 replicates and lower acquisition time in {counts['lower_time_count']}/500; both occurred in {counts['joint_count']}/500 ({counts['joint_fraction']*100:.1f}%; Figure S15). These frequencies summarize refits of the observed cohort, rather than probabilities of clinical superiority.

Against random next-task acquisition, the adaptive accuracy difference was {ci(ar,100,True)} percentage points and the time difference was {ci(art,1,True)} seconds. Fixed-sequence stopping versus fixed acquisition yielded {ci(sf,100,True)} accuracy points and {ci(st,1,True)} seconds. Absolute performance and class-specific outcomes appear in Tables S5–S6.

Development variation was also apparent at the participant level. Each person contributed a median of {people['median']:.0f} held-out refit predictions (range {people['minimum']:.0f}–{people['maximum']:.0f}), counting at most one prediction per person per replicate. For adaptive acquisition, the median participant disagreed with their original binary decision in {pdg['binary_disagreement_with_original']['median']*100:.1f}% of available refits; disagreement for the three-way output was {pdg['output_disagreement_with_original']['median']*100:.1f}% and for the acquired task set {pdg['task_disagreement_with_original']['median']*100:.1f}%. These comparisons concern procedural variation, not changes in a person's clinical state (Figure S16; Table S7). Threshold and fixed-protocol distributions are shown in Figure S17 and Table S8.

No attempted replicate was discarded, and the identity and threshold audits passed (Supplementary Methods S9). Each resample contained 390 rows but a median of {unique['median']:.0f} unique participants (range {unique['minimum']:.0f}–{unique['maximum']:.0f}); final outer training sets contained a median of {training['median']:.0f} unique people (Figure S18). This reduction in distinct training information precludes treating the refit medians as corrected versions of the original point estimates.
'''
    discussion_boot=f'''Complete participant refitting addresses a limitation of resampling a fixed set of held-out predictions: the fitted representation, selected protocol, prediction coefficients and decision thresholds can all change with the development sample [25]. Here, the adaptive-minus-fixed accuracy difference had a median of {a['median']*100:+.2f} percentage points and a central interval of {a['low']*100:+.2f} to {a['high']*100:+.2f}, with higher accuracy and lower time together in {counts['joint_count']}/500 refits. {refit_claim} Participant-level disagreement further shows why one cohort-level estimate cannot describe the stability of every decision.

The resource findings also depended on the comparator. Adaptive acquisition was shorter than optimized fixed acquisition in {counts['lower_time_count']}/500 refits, so its original excess time did not persist through redevelopment. Against random next-task acquisition, adaptive acquisition remained slower in all 500 refits, while the accuracy interval included both gain and loss. These findings distinguish an unstable comparison with optimized fixed acquisition from a more consistent time disadvantage against random next-task acquisition.

The scope of that experiment is nevertheless specific. Resampling retained class totals, combined participant multiplicities with new fold assignments and held the model family fixed. Keeping repeated identities together prevented their appearing in both training and evaluation, but left fewer distinct people in each training sample. Repeated copies also entered the nearest-neighbor calculation with their sampled multiplicity. Consequently, the empirical intervals combine several sources of development variation and have not been shown to provide nominal 95% coverage for the original estimand. They are distinct from conventional optimism-corrected bootstrap estimates [30,33] and from external evaluation [31]. The 500 refits improve the description of uncertainty; they do not supply new patients or undo the outcome-informed sequence of exploratory analyses.
'''
    conclusion=prose.CONCLUSION.replace('Complete participant-bootstrap refitting quantified substantial development variation without providing independent validation.',
       'Complete participant refitting quantified variation in comparative performance, thresholds and individual outputs; the original adaptive-versus-fixed time ordering was not stable across refits. These analyses do not provide independent validation.')
    if supported:
        conclusion=conclusion.replace('did not demonstrate a consistent advantage over optimized fixed and random acquisition controls.',
           'did not improve the original primary estimate, whereas exploratory complete refits favored it under the specified resampling procedure. The difference between these evaluations requires independent investigation.')
    main_body='\n\n'.join(['# '+prose.TITLE,abstract.strip(),prose.INTRODUCTION.strip(),methods.METHODS.strip(),prose.BOOTSTRAP_METHODS.strip(),methods.EVIDENCE_STATUS.strip(),results.strip(),boot_results.strip(),prose.DISCUSSION_START.strip(),discussion_boot.strip(),prose.DISCUSSION_END.strip(),conclusion.strip(),methods.DATA_ETHICS.strip()])

    tables=old.split('## Tables\n\n',1)[1].split('## Figures\n\n',1)[0]
    table5='''Table 5. Paired differences after complete participant-bootstrap refitting.

| Comparison | Δ BA, pp | Δ nominal time, s | Higher BA and less time |
| --- | --- | --- | --- |
'''
    for left,right in CONTRASTS:
        q=joint[left,right]
        table5+=f"| {NAMES[left]} minus {NAMES[right]} | {ci(paired[left,right,'balanced_accuracy'],100,True)} | {ci(paired[left,right,'nominal_seconds'],1,True)} | {q['joint_count']}/500 ({q['joint_fraction']*100:.1f}%) |\n"
    table5+='\nDifferences are medians with central 95% empirical stability intervals across the same 500 complete refits. Positive BA differences and negative time differences favor the first method. Joint frequencies are descriptive, not p values or probabilities of clinical superiority. Resamples preserve 276 PD and 114 DD rows, retain bootstrap multiplicities and group duplicate identities at every fitting level. These intervals do not replace the primary conditional interval or establish equivalence. Stopping selects a prefix of the fixed examination, so non-increasing acquisition time is partly structural.\n'
    figures=old.split('## Figures\n\n',1)[1].strip()
    figures=figures.replace('Cohort flow, participant isolation and information available during sequential acquisition.',
      'Original-analysis cohort flow, participant isolation and information available during sequential acquisition. The complete-refit sample sizes differ and are shown in Figure S18.')
    figures+='''

### Figure 7. Participant resampling with complete refitting

![Figure 7. Participant resampling with complete refitting](../figures/Figure_7_complete_refit.png)

Paired differences across 500 class-stratified participant-bootstrap replicates, with complete refitting of representations, state classifiers, acquisition policies and output thresholds. Circles denote medians and bars denote central 95% empirical intervals; diamonds mark original-partition point estimates for orientation. A, balanced-accuracy differences include all sampled participants, including deferred outputs. B, differences in mean nominal signal-acquisition time. Fixed, stopping and random abbreviate optimized fixed acquisition, fixed-sequence stopping and random next-task acquisition. Every replicate used the same sampled cohort and folds for all four strategies, with repeated copies of an identity kept together at all nested levels. The displayed intervals describe development stability under this resampling scheme; they are not corrected primary confidence intervals. The smaller distinct training samples and retained nearest-neighbor multiplicities limit direct comparison of refit medians with original estimates. Values are supplied in Table 5.
'''

    refs=json.loads((ROOT/'REFERENCES_AUDITED.json').read_text())
    lookup={r['id']:r for r in refs}
    mapping={}
    for m in CITATION.finditer(main_body):
        for v in m.group(1).split(','):
            oid=int(v.strip());assert oid in lookup,oid
            if oid not in mapping:mapping[oid]=len(mapping)+1
    assert len(mapping)==48,('References missing from main argument',set(lookup)-set(mapping))
    def renumber(text):
        return CITATION.sub(lambda m:'['+','.join(str(mapping[int(v.strip())]) for v in m.group(1).split(','))+']',text)
    def entry(oldid):
        r=lookup[oldid];doi=r.get('doi','');url='https://doi.org/'+doi if doi else r['inspected_url']
        label='doi:'+doi if doi else ('Preprint' if oldid==8 else 'Source record')
        title=r['title'].replace('?.','?').replace('!.','!')
        return f"{mapping[oldid]}. {r['authors']} {title} {r['publication']} [{label}]({url})."
    bibliography='\n\n'.join(entry(oldid) for oldid in mapping)
    manuscript=renumber(main_body)+'\n\n## References\n\n'+bibliography+'\n\n## Tables\n\n'+renumber(tables.strip()+'\n\n'+table5)+'\n\n## Figures\n\n'+renumber(figures)+'\n'
    (OUT/'PADS_manuscript_v1.md').write_text(manuscript)

    supplement=supplement.split('## Supplementary references\n\n',1)[0]+'## Supplementary figures\n\n'+supplement.split('## Supplementary figures\n\n',1)[1]
    supplement=supplement.replace('Adaptive versus fixed movement assessment for Parkinson disease differential classification: a retrospective PADS study',prose.TITLE)
    supplement=supplement.replace('Integrated supplement • 13 September 2026 • All evaluations reuse 390 participants\n\n','')
    supplement=supplement.replace('Entry points and file-to-claim mappings are in the round-6 README and evidence ledger.', 'Entry points and file-to-claim mappings are in the cumulative README and round-specific evidence ledgers.')
    supplement=supplement.replace('No final all-cohort deployment model, live clinical test, compatible external PD/DD validation or full-development participant bootstrap is supplied.',
      'Complete participant-bootstrap refitting is supplied in reviewer_round7, with per-replicate predictions, threshold-development scores and identity ledgers. Bootstrap model coefficients are represented by reproducibility hashes and can be regenerated from the fixed code and seeds; all fitted bootstrap objects are not serialized. A final all-cohort deployment model, live clinical test and compatible external PD/DD evaluation remain unavailable.')
    supplement=supplement.replace('| Primary comparison | Frozen original split, complete nesting, paired conditional intervals | Full-development sampling uncertainty absent |',
      '| Primary comparison | Frozen original split, complete nesting, archived paired conditional intervals | Original interval remains conditional; separate full-refit intervals have different scope |')
    supplement=supplement.replace('| Robustness | Five new full development partitions; six one-factor settings; original input/cap checks | No learning curve, nonlinear-family comparison or factor interaction study |',
      '| Robustness | Five additional partitions, six one-factor settings and 500 complete participant-bootstrap refits | Fixed family and class totals; fewer distinct training identities; no external evaluation |')
    s9=f'''### S9. Participant resampling and complete redevelopment

The later specification fixed 500 class-stratified bootstrap replicates before their fits. A deterministic seed of 2026091707 + 10000b was used for replicate b=0,…,499. Within each reference class, sampling with replacement produced 276 PD and 114 DD rows. All clinical inputs, task blocks, labels and timestamps from one identity were sampled together. Bootstrap row multiplicities were retained in fitting and evaluation.

Five outer folds were generated on the unique sampled identities, stratified by reference class, and broadcast back to their repeated rows. This identity rule was repeated for the three middle folds and for each system's three state cross-fitting folds. The same original person could therefore never appear on opposite sides of any train–test boundary. Every replicate evaluated all sampled rows, with paired strategies and bootstrap-weighted class denominators. This is a grouped nested assessment within each resample, not an out-of-bag assessment of a single final model.

Each replicate fitted five outer procedures. Each outer procedure required three middle acquisition systems and one final system; each system required three cross-fitted state banks and one final bank. With 259 classifiers per bank, the count was 5 × 4 × 4 × 259 = 20,720 logistic fits per replicate and 10,360,000 across 500 replicates. An additional 4,144 fits reproduced the original first fold before the experiment. All feature scaling, task PCA, classifiers, nearest-neighbor loss estimates, fixed-subset choices, binary cutoffs and rejection margins were developed anew. C=1, eight task components, 25 neighbors, the original cutoff grid and tie rules, and the 71.68-second cap remained fixed. No hyperparameter search or selection among replicates was performed.

Four core strategies were evaluated: adaptive, optimized fixed, fixed-sequence stopping and random next-task acquisition. The random policy used its own middle-developed cutoff and rejection margin; exact terminal-path hard-decision expectations avoided additional simulation noise. Calibration, class-aware deferral allocation and empirical mask-permutation controls were not refitted within this experiment. Their previously reported uncertainty remains conditional on their fitted components.

The implementation made a mechanical identity-grouping change to the state-model cross-fit while retaining the original acquisition and prediction calculations. Original-fold reproduction checked deterministic scores, masks and three-way outputs and random-policy expected outcomes. The frozen input hashes, code hashes, seeds, every completed replicate and all fitting warnings are retained. Convergence warnings were treated as errors. All 500 attempts completed; no result-dependent stopping, filtering or replacement of unfavorable replicates occurred. The final audit checked 40,000 nested identity boundaries, recomputed 7,500 deterministic cutoff-and-margin pairs and verified 2,500 random cutoff selections against their stored candidate accuracies. Random rejection margins were fitted by the stored weighted routine; its full development terminal-state arrays were not retained for a separate post-run reconstruction.

For each contrast, paired differences were calculated within each replicate before their median and 2.5th and 97.5th percentiles were taken. They are termed empirical stability intervals. Fractions of replicates with higher BA, shorter time or both are descriptive Monte Carlo frequencies, not p values or posterior probabilities. Differences within 10⁻¹² of zero in their native units were counted as numerical ties to prevent floating-point noise from defining direction; this is not a clinical-equivalence margin. The maximum Monte Carlo standard error of a frequency based on 500 replicates is 0.0224, obtained from the binomial variance at a frequency of 0.5. Percentile tails each contain approximately 12–13 ordered replicates and are correspondingly less precise than the central estimate. Saved prefix diagnostics compare 100, 250 and 500 replicates; binomial order-statistic calculations describe Monte Carlo uncertainty in the reported quantiles. These are computational diagnostics rather than another clinical confidence interval.

The bootstrap retained a median of {unique['median']:.0f} unique people among 390 rows (range {unique['minimum']:.0f}–{unique['maximum']:.0f}). The median distinct final training sample was {training['median']:.0f}, compared with 312 in the original outer evaluation. Grouping duplicates is essential for identity separation, but changes the distinct information available to nested training; duplicated neighbors also retain their multiplicity. Class totals, model family, original cohort and acquisition assumptions are conditioned upon. The procedure therefore does not isolate one component of uncertainty, correct primary point estimates or establish nominal confidence-interval coverage. It implements the principle of repeating model-building steps to examine instability [25,26], with a grouped acquisition-specific design that differs from optimism-corrected validation [30,33]. Fold dependence and limited information remain relevant [19,27,34].

Individual stability uses one held-out prediction for each identity when that identity appears in a replicate. Duplicate copies have the same fitted score and are collapsed for this diagnostic, although they retain their weights in cohort metrics. Omitted identities contribute no prediction for that replicate. Summaries include score percentiles, disagreement with the original held-out binary and three-way outputs, deferral frequency and changes in task masks. These compare learned procedures for the same fixed observations; the raw scores are not calibrated disease probabilities, and their ranges are not credible intervals for an individual's clinical diagnosis. Random-policy individual score stability is not reported because expected path decisions do not define an equivalent single-score prediction.
'''
    supplement=supplement.replace('## Supplementary tables\n\n',s9+'\n## Supplementary tables\n\n')
    s5='''Table S5. Absolute performance across complete participant refits.

| Strategy | Binary BA, % | Nominal time, s | Deferral, % |
| --- | --- | --- | --- |
'''
    for name in METHODS:s5+=f"| {NAMES[name]} | {ci(absolute[name,'balanced_accuracy'],100)} | {ci(absolute[name,'nominal_seconds'])} | {ci(absolute[name,'deferral'],100)} |\n"
    s5+='\nValues are medians (central 95% empirical intervals) across 500 complete refits. All-participant binary BA includes deferred participants. These are resampling summaries with fewer distinct training identities, not corrected original-cohort estimates or external-validation performance.\n'
    s6='''Table S6. Reference-class output consequences after complete refitting.

| Strategy / class | Clear correct, % | Clear wrong, % | Deferred, % |
| --- | --- | --- | --- |
'''
    for name in METHODS:
        for cl in ['pd','dd']:
            values=[ci(absolute[name,cl+'_'+k],100,False,1) for k in ['clear_correct','clear_wrong','deferred']]
            s6+=f"| {NAMES[name]} / {cl.upper()} | "+' | '.join(values)+' |\n'
    s6+='\nThe three proportions sum to 100% within each class in every replicate; marginal medians need not sum exactly to 100%. Denominators retain all 276 PD and 114 DD sampled rows, including duplicates and deferred outputs. Random-policy outcomes are exact expectations of terminal decisions. Parentheses contain central 95% empirical intervals across refits.\n'
    s7='''Table S7. Participant-level disagreement with the original held-out procedure.

| Strategy | Binary decision, % | Three-way output, % | Task set, % |
| --- | --- | --- | --- |
'''
    for name in METHODS[:3]:
        d=result['participant_diagnostics'][name]
        s7+=f"| {NAMES[name]} | "+' | '.join(ci(d[k],100,False,1) for k in ['binary_disagreement_with_original','output_disagreement_with_original','task_disagreement_with_original'])+' |\n'
    s7+=f'\nFor each person, disagreement was calculated across their available held-out refit predictions (median {people["median"]:.0f}; range {people["minimum"]:.0f}–{people["maximum"]:.0f}). Table cells report the median participant and central 95% of participant-level disagreement fractions across 390 people. These ranges summarize heterogeneity between people; they are not confidence intervals, and differ from intervals across the 500 cohort refits. Figures S16A–C display within-person score variation.\n'
    freq=pd.read_csv(ROOT/'tables/selected_fixed_mask_frequency.csv')
    tasks=['Relaxed','RelaxedTask','StretchHold','LiftHold','HoldWeight','PointFinger','DrinkGlas','CrossArms','TouchIndex','TouchNose','Entrainment']
    units=[2,2,1,1,1,1,1,1,1,1,2]
    freq['tasks']=['; '.join(task for i,task in enumerate(tasks) if int(mask)&(1<<i)) for mask in freq.task_mask]
    freq['nominal_seconds']=[10.24*sum(u for i,u in enumerate(units) if int(mask)&(1<<i)) for mask in freq.task_mask]
    freq.to_csv(ROOT/'tables/selected_fixed_mask_frequency.csv',index=False)
    s8='''Table S8. Most frequent fixed examinations selected during complete refitting.

| Mask | Acquired tasks | Time, s | Outer fits, n (%) |
| --- | --- | --- | --- |
'''
    for r in freq.head(8).itertuples():s8+=f'| {r.task_mask} | {r.tasks} | {r.nominal_seconds:.2f} | {r.outer_fits} ({r.fraction*100:.1f}) |\n'
    s8+=f'\nAcross 2,500 outer development fits, {len(freq)} distinct fixed task masks were selected. Masks encode the ordered tasks listed in Methods S1; they identify protocols, not participant groups. The complete frequency table, including all less frequent masks, is provided as selected_fixed_mask_frequency.csv. Repeated outer fits share data and are not independent study populations.\n'
    supp_tables='\n\n'.join([s5,s6,s7,s8])
    supplement=supplement.replace('## Supplementary figures\n\n',supp_tables+'\n## Supplementary figures\n\n')
    supplement+='''

### Figure S15. Joint accuracy and acquisition-time variation after complete refitting

![Figure S15. Joint full-refit differences](../figures/Figure_S15_joint_refit.png)

Each point is a paired contrast within one of the 500 complete participant-bootstrap refits. Fixed, stopping and random denote optimized fixed acquisition, fixed-sequence stopping and random next-task acquisition, respectively. The shaded quadrant denotes higher all-participant binary accuracy and lower nominal acquisition time for the first method; the annotated count reports how often both occurred. Diamonds denote original-partition point estimates. All four panels use the same axis limits. Resamples reuse the observed cohort, and direction frequencies are descriptive rather than formal probabilities of superiority. The stopping-versus-fixed time ordering follows partly from the prefix constraint. This distribution includes complete redevelopment, unlike the fixed-prediction cloud in Figure S3.

### Figure S16. Individual predictions and decisions across complete refits

![Figure S16. Participant-level refit stability](../figures/Figure_S16_participant_refit.png)

A–C, each of 390 points compares a participant's original outer-held-out score with the median of their available held-out refit scores; thin vertical lines show their central 95% score range. The dashed line denotes equality. D–F, empirical cumulative distributions of each participant's fraction of refits disagreeing with the original binary decision, actual three-way output or task set. An identity contributes only when drawn in a replicate and is collapsed to one prediction within that replicate. These are changes in fitted procedures applied to fixed observations, not longitudinal clinical changes or uncertainty intervals for clinical disease risk. The number of available refit predictions varies by participant; Table S7 gives the range. Random-policy expectations do not define a comparable single score and are excluded from this figure.

### Figure S17. Threshold selection, deferral and fixed-protocol variation

![Figure S17. Thresholds and protocol variation](../figures/Figure_S17_thresholds.png)

A, frequencies of selected binary cutoffs among 2,500 outer development fits for each strategy. B, empirical distributions of fitted rejection margins across those fits. C, distributions of held-out deferral across 500 complete cohort refits; the dashed line marks the approximately 20% development target, not a required test fraction. D, the eight most frequently selected fixed task masks and the pooled remainder; Table S8 identifies their tasks. Rejection thresholds are fitted from middle-held-out scores and transferred unchanged to outer evaluation. All numerical outputs, including less frequent protocols, are available in the supplementary tables. The plotted folds overlap in their training data and are not independent samples.

### Figure S18. Distinct participant information under grouped resampling

![Figure S18. Resampling multiplicities and nested sample sizes](../figures/Figure_S18_sample_information.png)

A, unique identities among each replicate's 390 sampled rows; the dashed line marks the median. B, unique training identities for final outer systems, middle systems and the innermost state cross-fits. Boxes show the interquartile range, central lines the median and whiskers the central 95% empirical range across the relevant fits. Diamonds show original-analysis sample sizes of 312 and 208, while the short segment denotes 138–139 identities in the original innermost training folds. Bootstrap multiplicities are retained for learning and evaluation but do not create new identities. The figure makes explicit why grouping duplicates changes distinct training information and why the full-refit medians cannot be treated as corrected primary point estimates.
'''
    cited_supp=set()
    for m in CITATION.finditer(supplement):cited_supp.update(int(v.strip()) for v in m.group(1).split(','))
    supp_refs='## Supplementary references\n\nNumbering follows the main manuscript.\n\n'+'\n\n'.join(entry(i) for i in sorted(cited_supp,key=lambda i:mapping[i]))+'\n\n'
    supplement=renumber(supplement)
    supplement=supplement.replace('## Supplementary figures\n\n',supp_refs+'## Supplementary figures\n\n')
    (OUT/'PADS_integrated_supplement.md').write_text(supplement)

    locations={i:[] for i in lookup};section=''
    for block in main_body.split('\n\n'):
        if block.startswith('#'):section=block.lstrip('#').strip()
        for m in CITATION.finditer(block):
            for v in m.group(1).split(','):
                oldid=int(v.strip())
                if section not in locations[oldid]:locations[oldid].append(section)
    for ref in refs:ref.update(manuscript_number=mapping[ref['id']],main_locations=locations[ref['id']])
    (ROOT/'REFERENCES_WITH_LOCATIONS.json').write_text(json.dumps(sorted(refs,key=lambda r:r['manuscript_number']),ensure_ascii=False,indent=2)+'\n')
    (ROOT/'REFERENCE_NUMBER_MAP.json').write_text(json.dumps(mapping,indent=2)+'\n')
    (ROOT/'INTERPRETATION_CHECK.json').write_text(json.dumps({'refit_joint_interval_favors_adaptive':supported,'accuracy_interval_crosses_zero':a['low']<0<a['high'],'all_500_used':True,'claim':refit_claim},indent=2)+'\n')
    ledger=json.loads((ROOT.parent/'reviewer_round6/EVIDENCE_LEDGER.json').read_text())
    for evidence in ledger:
        if evidence['id']=='E03':evidence['manuscript_locations']=['Table 2','Table 3','Figure 2','Table 5']
        if evidence['id']=='E11':
            evidence.update(evidence='48 substantively cited references and 20 source-informed research questions',
              source_files=['reviewer_round7/REFERENCES_WITH_LOCATIONS.json','reviewer_round7/REFERENCE_NUMBER_MAP.json','reviewer_round7/integrated/PADS_20_dimension_review_ZH.md'],
              code=['reviewer_round7/build_review_audit.py'])
    ledger.extend([
      {'id':'E12','claim':'Complete participant-bootstrap refitting of four acquisition strategies',
       'evidence':{'replicates':500,'new_logistic_fits':10360000,'paired_contrasts':[r for r in result['contrasts'] if r['metric'] in ['balanced_accuracy','nominal_seconds']],'joint':result['joint_accuracy_time']},
       'source_files':['reviewer_round7/ROUND7_SPEC.md','reviewer_round7/ROUND7_RESULTS.json','reviewer_round7/tables/replicate_metrics.csv','reviewer_round7/tables/paired_stability.csv','reviewer_round7/tables/joint_accuracy_time.csv'],
       'code':['reviewer_round7/run_bootstrap.py','reviewer_round7/grouped_analysis.py','reviewer_round7/summarize_bootstrap.py'],
       'manuscript_locations':['Figure 7','Table 5','Figure S15','Tables S5–S6','Supplementary Methods S9'],
       'limits':'Empirical grouped-refit stability, not corrected primary intervals, independent cases or external validation. Calibration, class-aware allocation and mask permutation are not refitted.'},
      {'id':'E13','claim':'Participant predictions, thresholds and protocol instability',
       'evidence':result['participant_diagnostics'],
       'source_files':['reviewer_round7/tables/participant_stability.csv','reviewer_round7/tables/fold_thresholds_and_protocols.csv','reviewer_round7/tables/selected_fixed_mask_frequency.csv'],
       'code':['reviewer_round7/summarize_bootstrap.py','reviewer_round7/plot_bootstrap.py'],
       'manuscript_locations':['Figures S16–S17','Tables S7–S8'],
       'limits':'Raw-score procedural variation; one prediction per sampled identity per replicate; no clinical-risk interval or longitudinal change.'},
      {'id':'E14','claim':'Identity separation and effective training information',
       'evidence':{'unique_per_resample':result['unique_per_resample'],'training_sample_sizes':result['training_sample_sizes'],'identity_boundaries':40000},
       'source_files':['reviewer_round7/VALIDATION.json','reviewer_round7/COMPUTATION_LEDGER.json','reviewer_round7/tables/resample_counts.csv','reviewer_round7/tables/fit_sample_sizes.csv'],
       'code':['reviewer_round7/run_bootstrap.py','reviewer_round7/summarize_bootstrap.py'],
       'manuscript_locations':['Figure S18','Supplementary Methods S9'],
       'limits':'Fewer distinct people than sampled rows; duplicate neighbor multiplicities retained; numerical audit is not independent clinical adjudication.'}
    ])
    (ROOT/'EVIDENCE_LEDGER.json').write_text(json.dumps(ledger,ensure_ascii=False,indent=2)+'\n')
    (OUT/'AUTHOR_ITEMS.md').write_text('''# Author information required before submission

The manuscript remains a research draft. The authors need to supply names, affiliations, contributions, funding and competing-interest declarations; confirm any institutional determination applicable to this secondary analysis; identify the target journal and repository-release plan; and prepare a factual disclosure of AI assistance in accordance with that journal's policy. These facts cannot be inferred from the public dataset. The dataset's source ethics approval and registration are reported only as provenance. The clearer academic wording does not alter the need for an accurate assistance disclosure.
''')
    (OUT/'00_START_HERE_ROUND7.md').write_text(f'''# PADS round 7 integrated research package

The current article is `pads_analysis/reviewer_round7/integrated/PADS_manuscript_v1.docx`; the accompanying supplement and Chinese 20-dimension research audit are in the same directory. Earlier files remain as historical versions. This archive preserves every round-6 member unchanged and adds the completed round-7 work.

All 500 specified participant-bootstrap replicates completed. Each repeated representation learning, state fitting, acquisition development and binary/rejection threshold selection for four strategies, keeping duplicate identities together at every nested level. These completed refits required 10,360,000 logistic fits, with 4,144 additional original-fold reproduction fits.

The median adaptive-minus-fixed difference was {ci(a,100,True)} balanced-accuracy percentage points and {ci(t,1,True)} nominal seconds. Parentheses denote central 95% empirical full-refit stability intervals. Higher accuracy and shorter time jointly occurred in {counts['joint_count']}/500 replicates. The original primary comparison remains −0.83 percentage points with its archived conditional interval of −5.97 to +4.20.

The manuscript states its primary research question near the beginning and integrates 48 distinct, claim-mapped references. Seven main figures and eighteen supplementary figures are supplied in PNG, PDF and SVG. All new plots use audited numerical tables. Complete refitting adds internal uncertainty evidence; it does not correct the primary interval, create new participants or establish external validity or clinical utility.

Begin with the article, then the supplement and the Chinese question–answer audit. `reviewer_round7/README.md` describes reproduction, files, uncertainty definitions and remaining limitations. `ROUND7_ARCHIVE_MANIFEST.json` verifies uncompressed member hashes. `ROUND7_PACKAGE_VALIDATION.json` records package and document verification.
''')
    print(json.dumps({'references':len(mapping),'main_words':len(manuscript.split()),'supplement_words':len(supplement.split()),'interpretation':refit_claim},indent=2))

if __name__=='__main__':main()
