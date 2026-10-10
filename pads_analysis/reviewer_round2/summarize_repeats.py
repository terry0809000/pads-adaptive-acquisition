"""Validate complete repeated development and report split sensitivity."""
from pathlib import Path
import sys,csv,json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent;BASE=ROOT.parent;WORK=BASE.parent
sys.path.insert(0,str(BASE))
import run_analysis as a
SEEDS=[20260914,20260915,20260916,20260917,20260918]
ALLOWED={budget:set(a.admissible_masks(budget)) for budget in range(2,8)}
CORE=['adaptive','fixed_optimized','fixed_sequence_stop']
LABELS=['Adaptive selection','Optimized fixed','Fixed sequence + stopping']
COLORS=['#2364A2','#C76A20','#24766D']
FIG=ROOT/'figures';TAB=ROOT/'tables'
FIG.mkdir(exist_ok=True);TAB.mkdir(exist_ok=True)
def read(p):
    with p.open() as f:return list(csv.DictReader(f))
def write(name,rows):a.csv_write(TAB/name,rows)
def metrics(rows):
    y=np.array([int(r['y']) for r in rows]);p=np.array([int(r['binary_prediction']) for r in rows]);o=np.array([int(r['actual_output']) for r in rows]);sec=np.array([float(r['movement_seconds']) for r in rows])
    sens=float((p[y==1]==1).mean());spec=float((p[y==0]==0).mean())
    d={'n':len(y),'pd':int(y.sum()),'dd':int((y==0).sum()),'ba':.5*(sens+spec),'sensitivity':sens,'specificity':spec,
    'coverage':float((o!=-1).mean()),'mean_seconds':float(sec.mean()),'mean_tasks':float(np.mean([int(r['n_tasks']) for r in rows]))}
    for c,label in [(0,'dd'),(1,'pd')]:
        for name,m in [('clear_correct',(o==c)),('clear_wrong',(o==1-c)),('deferred',o==-1)]:d[f'{label}_{name}']=int((m&(y==c)).sum())
    return d
orig=read(BASE/'summary/held_out_predictions_primary.csv')
meta={r['subject_id']:r for r in read(WORK/'pads_features_received/PADS_features_return/participant_metadata.csv')}
data={};allrows=[];methodrows=[];foldrows=[];checks=[]
for seed in [20260913]+SEEDS:
    if seed==20260913:rr=orig
    else:
        rep=ROOT/'results'/f'seed_{seed}'
        assert json.loads((rep/'REPEAT_COMPLETE.json').read_text())['folds_completed']==5
        manifest={r['subject_id']:r for r in read(rep/'outer_folds.csv')}
        rr=[]
        for f in range(5):
            d=rep/f'fold_{f}';completion=json.loads((d/'COMPLETE.json').read_text())
            assert completion['code_sha256']=='34ba6614d209d88215f764a44a650133624b002e4cb6128dd225dccc0adbc056'
            ledger=json.loads((d/'training_ledger.json').read_text())
            held=set(ledger['outer_test_ids']);train=set(ledger['final_fit']['training_ids'])
            assert len(held)==78 and len(train)==312 and not held&train and held|train==set(manifest)
            midheld=[]
            for mid in ledger['middle_fits']:
                t=set(mid['fit']['training_ids']);h=set(mid['threshold_holdout_ids'])
                assert len(t)==208 and len(h)==104 and not t&h and t|h==train
                midheld.extend(h)
            assert len(midheld)==len(set(midheld))==312
            for fit in [ledger['final_fit']]+[m['fit'] for m in ledger['middle_fits']]:
                ids=set(fit['training_ids']);innerheld=[]
                assert not ids&held
                for cf in fit['state_crossfit']:
                    t=set(cf['train_ids']);h=set(cf['held_out_ids'])
                    assert not t&h and t|h==ids
                    innerheld.extend(h)
                assert len(innerheld)==len(set(innerheld))==len(ids)
                assert fit['maximum_solver_iterations']<2000
            new=read(d/'held_out_predictions.csv')
            for row in new:
                sid=row['subject_id'];mask=int(row['task_mask']);m=row['method']
                assert row['y']==manifest[sid]['y'] and row['outer_fold']==manifest[sid]['outer_fold']
                assert abs(float(row['movement_seconds'])-a.nominal_units(mask)*10.24)<1e-8
                if m not in ('full_protocol','clinical_only'):
                    cap=int(m.split('_')[-1]) if '_budget_' in m else 7
                    assert mask in ALLOWED[cap]
            rr.extend(new)
            checks.append({'seed':seed,'outer_fold':f,'nested_isolation_verified':True,'n_test':78,'n_train':312})
        allrows.extend([{'seed':seed,**r} for r in rr])
    data[seed]={}
    for m in sorted({r['method'] for r in rr}):
        mr=sorted([r for r in rr if r['method']==m],key=lambda r:r['subject_id'])
        assert len(mr)==len({r['subject_id'] for r in mr})==390
        data[seed][m]=mr
        methodrows.append({'seed':seed,'role':'original_primary' if seed==20260913 else 'new_repeat','method':m,**metrics(mr)})
        for f in range(5):foldrows.append({'seed':seed,'outer_fold':f,'method':m,**metrics([r for r in mr if int(r['outer_fold'])==f])})
write('held_out_predictions_new_repeats.csv',allrows);write('repeat_metrics.csv',methodrows);write('fold_metrics.csv',foldrows)
index={(r['seed'],r['method']):r for r in methodrows}
contrasts=[]
for seed in [20260913]+SEEDS:
    for first in ('adaptive','fixed_sequence_stop'):
        x,z=index[seed,first],index[seed,'fixed_optimized']
        contrasts.append({'seed':seed,'role':x['role'],'comparison':first+'_minus_fixed',
        'delta_ba_pp':100*(x['ba']-z['ba']),'delta_seconds':x['mean_seconds']-z['mean_seconds'],
        'delta_coverage_pp':100*(x['coverage']-z['coverage']),
        'delta_sensitivity_pp':100*(x['sensitivity']-z['sensitivity']),'delta_specificity_pp':100*(x['specificity']-z['specificity'])})
write('repeat_contrasts.csv',contrasts)
stability=[];freq=[]
for m in CORE:
    assert all([r['subject_id'] for r in data[s][m]]==[r['subject_id'] for r in data[SEEDS[0]][m]] for s in SEEDS)
    for j,row in enumerate(data[SEEDS[0]][m]):
        pred=[int(data[s][m][j]['binary_prediction']) for s in SEEDS]
        out=[int(data[s][m][j]['actual_output']) for s in SEEDS]
        mask=[int(data[s][m][j]['task_mask']) for s in SEEDS]
        stability.append({'method':m,'subject_id':row['subject_id'],'y':int(row['y']),
         'n_distinct_binary':len(set(pred)),'n_distinct_actual_outputs':len(set(out)),
         'n_distinct_task_sets':len(set(mask)),'n_pd_predictions':sum(pred),'n_deferrals':sum(x==-1 for x in out)})
    for s in SEEDS:
        for j,t in enumerate(a.TASKS):freq.append({'seed':s,'method':m,'task':t,'fraction':float(np.mean([bool(int(r['task_mask'])&(1<<j)) for r in data[s][m]]))})
write('participant_stability.csv',stability);write('task_frequencies.csv',freq)
subgroups=[];subtypes=[]
for seed in SEEDS:
    for m in CORE:
        rr=data[seed][m]
        for name,filter_ in [('age_under65',lambda t:float(t['age'])<65),('age_65plus',lambda t:float(t['age'])>=65),('female',lambda t:int(t['gender_male'])==0),('male',lambda t:int(t['gender_male'])==1)]:
            z=[r for r in rr if filter_(meta[r['subject_id']])]
            subgroups.append({'seed':seed,'method':m,'group':name,**metrics(z)})
        for diag in ['Atypical Parkinsonism','Essential Tremor','Multiple Sclerosis','Other Movement Disorders']:
            z=[r for r in rr if meta[r['subject_id']]['condition']==diag]
            subtypes.append({'seed':seed,'method':m,'diagnosis':diag,'n':len(z),
              'specificity':sum(int(r['binary_prediction'])==0 for r in z)/len(z),
              'coverage':sum(int(r['actual_output'])!=-1 for r in z)/len(z)})
write('subgroup_repeat_metrics.csv',subgroups);write('dd_subtype_repeat_metrics.csv',subtypes)
summary={'user_constraint':'Public data only; no new clinical recruitment','n_new_repeats':5,'n_new_outer_folds':25,'n_unique_participants':390,
 'n_new_logistic_fits':103600,'n_new_prediction_rows':len(allrows),'estimation_scope':'Repeated internal development; descriptive split sensitivity, not independent replication or external validation',
 'contrasts':{},'participant_stability':{}}
for comp in ('adaptive_minus_fixed','fixed_sequence_stop_minus_fixed'):
    rr=[r for r in contrasts if r['role']=='new_repeat' and r['comparison']==comp]
    summary['contrasts'][comp]={key:{'mean':float(np.mean([r[key] for r in rr])),'median':float(np.median([r[key] for r in rr])),
     'min':min(r[key] for r in rr),'max':max(r[key] for r in rr)} for key in ['delta_ba_pp','delta_seconds','delta_coverage_pp']}
for m in CORE:
    rr=[r for r in stability if r['method']==m]
    summary['participant_stability'][m]={key:sum(r[key]>1 for r in rr) for key in ['n_distinct_binary','n_distinct_actual_outputs','n_distinct_task_sets']}
a.json_write(ROOT/'STABILITY_RESULTS.json',summary)
a.json_write(ROOT/'VALIDATION.json',{'all_25_nested_ledgers_passed':checks,'primary_code_sha256':hashlib.sha256((BASE/'run_analysis.py').read_bytes()).hexdigest(),
 'original_fold_sha256':hashlib.sha256((WORK/'pads_features_received/PADS_features_return/primary_outer_folds.csv').read_bytes()).hexdigest(),
 'original_plan_sha256':hashlib.sha256((BASE/'ANALYSIS_PLAN.md').read_bytes()).hexdigest(),'paired_rows_verified':True})

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
def save(fig,name):
    for ext in ['png','pdf','svg']:fig.savefig(FIG/f'{name}.{ext}',dpi=350,bbox_inches='tight',facecolor='white')
    plt.close(fig)
fig,axs=plt.subplots(1,3,figsize=(11,4.7),sharey=True)
for ax,key,factor,title in zip(axs,['ba','mean_seconds','coverage'],[100,1,100],['A  Binary balanced accuracy (%)','B  Mean nominal motion time (s)','C  Clear-output coverage (%)']):
    ax.axhspan(-.43,.43,color='#EEF1F4',zorder=0)
    for i,m in enumerate(CORE):
        xx=[index[s,m][key]*factor for s in [20260913]+SEEDS]; yy=np.arange(6)+(i-1)*.19
        ax.plot(xx,yy,'o',color=COLORS[i],ms=4,label=LABELS[i])
    ax.grid(axis='x',alpha=.15);ax.set_title(title,loc='left',fontsize=10,weight='bold');ax.set_ylim(5.6,-.7)
axs[0].set_yticks(range(6),['Original primary']+[f'New repeat {i+1}' for i in range(5)])
axs[0].set_xlim(62,80);axs[1].set_xlim(15,72);axs[2].set_xlim(62,86)
for i,s in enumerate([20260913]+SEEDS):
    for ax,key,factor in zip(axs,['ba','mean_seconds','coverage'],[100,1,100]):
        for j,m in enumerate(CORE):
            v=index[s,m][key]*factor
            ax.annotate(f'{v:.1f}',(v,i+(j-1)*.19),xytext=(4,0),textcoords='offset points',va='center',fontsize=7,color=COLORS[j])
handles,labs=axs[0].get_legend_handles_labels();fig.legend(handles,labs,loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,.04))
fig.text(.04,.012,'Same 390 participants in every repeat; complete nested development refitted. Repeats are not independent cohorts.',fontsize=8,color='#657582')
fig.tight_layout(rect=(0,.13,1,1),w_pad=1.5);save(fig,'Figure_R6_repeat_performance')

fig,(ax,bx)=plt.subplots(1,2,figsize=(9.6,4.3))
for k,(comp,c,label) in enumerate([('adaptive_minus_fixed',COLORS[0],'Adaptive − fixed'),('fixed_sequence_stop_minus_fixed',COLORS[2],'Fixed + stopping − fixed')]):
    rr=[r for r in contrasts if r['comparison']==comp]
    for j,r in enumerate(rr):
        ax.scatter(r['delta_seconds'],r['delta_ba_pp'],marker='D' if j==0 else 'o',color=c,s=45 if j==0 else 30,label=label if j==0 else None)
        ax.annotate('P' if j==0 else str(j),(r['delta_seconds'],r['delta_ba_pp']),xytext=(4,4),textcoords='offset points',fontsize=8,color=c)
    bx.plot([r['delta_ba_pp'] for r in rr],np.arange(6)+(k-.5)*.18,'o',color=c,label=label)
ax.axhline(0,color='gray',ls='--',lw=1);ax.axvline(0,color='gray',ls='--',lw=1)
ax.set(xlabel='Difference in mean nominal acquisition time (s)',ylabel='Difference in binary BA (percentage points)')
ax.set_title('A  Paired resource and accuracy contrasts',loc='left',weight='bold',fontsize=10)
ax.legend(frameon=False,fontsize=8);ax.grid(alpha=.1)
bx.axvline(0,color='gray',ls='--',lw=1);bx.set_yticks(range(6),['Original primary']+[f'Repeat {i+1}' for i in range(5)]);bx.set_ylim(5.5,-.5)
bx.set_xlabel('Difference in binary BA (percentage points)');bx.set_title('B  Direction across complete repeats',loc='left',weight='bold',fontsize=10);bx.grid(axis='x',alpha=.15)
fig.text(.04,.015,'P = original primary; 1–5 = seeds 20260914–20260918. No fold-level significance test or independence assumption.',fontsize=8,color='#657582')
fig.tight_layout(rect=(0,.06,1,1),w_pad=2);save(fig,'Figure_R7_repeat_contrasts')

fig,(ax,bx)=plt.subplots(1,2,figsize=(11.8,5.3),gridspec_kw={'width_ratios':[1,1.8]})
for j,key in enumerate(['n_distinct_binary','n_distinct_actual_outputs','n_distinct_task_sets']):
    for k,m in enumerate(CORE):
        v=summary['participant_stability'][m][key];yy=j+(k-1)*.21
        ax.barh(yy,100*v/390,color=COLORS[k],height=.17,label=LABELS[k] if j==0 else None)
        ax.text(100*v/390+1,yy,f'{v}/390',va='center',fontsize=7)
ax.set_yticks(range(3),['Binary prediction','Three-way output','Acquired task set']);ax.set_ylim(2.65,-.65);ax.set_xlim(0,122)
ax.set_xticks([0,25,50,75,100])
ax.set_xlabel('Participants with any change across 5 new repeats (%)');ax.set_title('A  Within-participant changes',loc='left',weight='bold',fontsize=10)
handles,labs=ax.get_legend_handles_labels()
fig.legend(handles,labs,loc='lower center',bbox_to_anchor=(.5,.055),ncol=3,frameon=False,fontsize=8)
heat=np.array([[100*next(r['fraction'] for r in freq if r['seed']==s and r['method']==m and r['task']==t) for t in a.TASKS] for m in CORE[:2] for s in SEEDS])
im=bx.imshow(heat,cmap='Blues',vmin=0,vmax=100,aspect='auto')
bx.set_yticks(range(10),[f'{name}{i+1}' for name in ['A','F'] for i in range(5)],fontsize=8)
bx.set_xticks(range(11),a.TASKS,rotation=45,ha='right',fontsize=7)
for i in range(10):
    for j in range(11):bx.text(j,i,f'{heat[i,j]:.0f}',ha='center',va='center',fontsize=6.5,color='white' if heat[i,j]>55 else '#18334A')
bx.set_title('B  Task acquisition by repeat (%)',loc='left',weight='bold',fontsize=10)
bx.axhline(4.5,color='white',lw=2);bx.tick_params(length=0)
fig.colorbar(im,ax=bx,pad=.02,fraction=.04).set_label('Acquired (%)')
fig.text(.04,.015,'A1–A5: adaptive; F1–F5: optimized fixed. Changes reflect development sensitivity, not new patients or observed clinical deterioration.',fontsize=8,color='#657582')
fig.tight_layout(rect=(0,.15,1,1),w_pad=2);save(fig,'Figure_R8_policy_stability')
print(json.dumps(summary,indent=2))
