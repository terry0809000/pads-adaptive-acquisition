"""Exploratory reviewer plots; no model training or threshold modification."""
from pathlib import Path
import csv, hashlib, json, shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
WORK = BASE.parent
FIG = ROOT/'figures'
TAB = ROOT/'tables'
for d in (FIG, TAB): d.mkdir(exist_ok=True)
SEED, B = 2026091301, 5000
BLUE, ORANGE, GREEN, GRAY = '#2364A2', '#C76A20', '#24766D', '#657582'
METHODS = ['adaptive','fixed_optimized','fixed_sequence_stop']
LABELS = ['Adaptive selection','Optimized fixed','Fixed sequence + stopping']
COLORS = [BLUE,ORANGE,GREEN]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,
    'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,
    'ps.fonttype':42,'savefig.facecolor':'white'})

def read(path):
    with path.open() as f: return list(csv.DictReader(f))
def write(name, rows):
    with (TAB/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
def save(fig,name):
    for ext in ('png','pdf','svg'):
        fig.savefig(FIG/f'{name}.{ext}',dpi=350,bbox_inches='tight')
    plt.close(fig)
def wilson(k,n):
    z=1.959963984540054; p=k/n; den=1+z*z/n
    c=(p+z*z/(2*n))/den; d=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return float(c-d),float(c+d)
def ci(v): return [float(x) for x in np.quantile(v,[.025,.975])]

rows=read(BASE/'summary/held_out_predictions_primary.csv')
meta={r['subject_id']:r for r in read(WORK/'pads_features_received/PADS_features_return/participant_metadata.csv')}
rr={m:sorted([r for r in rows if r['method']==m],key=lambda r:r['subject_id']) for m in METHODS}
ids=[r['subject_id'] for r in rr[METHODS[0]]]
y=np.array([int(r['y']) for r in rr[METHODS[0]]])
assert len(ids)==len(set(ids))==390 and sum(y)==276
for m in METHODS:
    assert [r['subject_id'] for r in rr[m]]==ids
    assert np.array_equal([int(r['y']) for r in rr[m]],y)
pred={m:np.array([int(r['binary_prediction']) for r in rr[m]]) for m in METHODS}
out={m:np.array([int(r['actual_output']) for r in rr[m]]) for m in METHODS}
score={m:np.array([float(r['score']) for r in rr[m]]) for m in METHODS}
sec={m:np.array([float(r['movement_seconds']) for r in rr[m]]) for m in METHODS}
correct={m:pred[m]==y for m in METHODS}
rng=np.random.default_rng(SEED)
primary_rng=np.random.default_rng(20260913)
class_indices=[np.flatnonzero(y==c) for c in (0,1)]
boot=[ix[primary_rng.integers(0,len(ix),size=(B,len(ix)))] for ix in class_indices]
bi=np.concatenate(boot,axis=1)
ba_boot={m:.5*(correct[m][boot[0]].mean(1)+correct[m][boot[1]].mean(1)) for m in METHODS}
summary={'status':'Exploratory after primary results; no refitting', 'seed':SEED, 'primary_bootstrap_seed':20260913,'bootstrap_draws':B,
 'interval_scope':'Conditional on existing held-out predictions and fitted systems; not full-development uncertainty'}

# Cohort and development diagram. Counts describe the release, not original recruitment.
fig=plt.figure(figsize=(10,8))
ax=fig.add_axes([.03,.05,.94,.88]); ax.set_xlim(0,10); ax.set_ylim(0,10); ax.axis('off')
def box(x,y,w,h,text,color='#EDF3F8',fs=9):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.08,rounding_size=.08',
                              facecolor=color,edgecolor='#718493',lw=.8))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=fs)
def arrow(a,b):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=11,color=GRAY,lw=1.1))
ax.text(0,9.8,'A  Public release and primary population',weight='bold',fontsize=11)
box(.2,8.45,2.3,.9,'PADS public release\n469 participants')
box(3.8,8.45,2.6,.9,'Primary analysis: 390\n276 PD / 114 DD')
box(7.55,8.45,2.1,.9,'Five outer folds\n78 held out per fold')
box(.2,7.05,2.3,.8,'79 healthy controls\nExcluded from PD vs DD','#F4F1ED')
arrow((2.58,8.9),(3.7,8.9)); arrow((6.48,8.9),(7.45,8.9)); arrow((1.35,8.37),(1.35,7.92))
ax.text(0,6.65,'B  Development repeated separately inside each outer fold',weight='bold',fontsize=11)
box(.2,5.15,2.3,1,'312 development\nparticipants')
box(3.35,5.15,3,1,'Three middle folds\n208 learn / 104 score\nRebuild the entire procedure',fs=8.6)
box(7.15,5.15,2.5,1,'Cross-fitted scores\nSet decision cutoff\nand rejection margin',fs=8.6)
box(.2,3.5,2.3,.95,'Refit on all 312\nSame frozen settings')
box(3.35,3.5,3,.95,'Apply fitted policy + thresholds\n78 outer-held-out participants',fs=8.6)
box(7.15,3.5,2.5,.95,'Pool 390 test predictions\nOne per participant',fs=8.6)
arrow((2.58,5.65),(3.25,5.65)); arrow((6.43,5.65),(7.05,5.65))
arrow((1.35,5.07),(1.35,4.54)); arrow((2.58,3.98),(3.25,3.98))
arrow((8.4,5.07),(6.05,4.55)); arrow((6.43,3.98),(7.05,3.98))
ax.text(.15,2.85,'Each learning set: 3-fold state cross-fitting; fit scalers, task PCA and classifiers using its training participants only.',fontsize=8.1)
ax.text(0,2.35,'C  Retrospective acquisition and separate outputs',weight='bold',fontsize=11)
box(.2,.95,2.3,1,'Age, recorded gender, NMS\n+ mandatory Relaxed',fs=8.5)
box(3.35,.95,3,1,'Choose a feasible later task\nor stop using acquired data\nNominal cap: 71.68 s',fs=8.5)
box(7.15,.95,2.5,1,'Binary PD / DD for BA\nPD / DD / indeterminate\nfor service outputs',fs=8.5)
arrow((2.58,1.45),(3.25,1.45)); arrow((6.43,1.45),(7.05,1.45))
ax.text(.15,.24,'No independent external cohort or timed prospective workflow is represented in this diagram.',fontsize=9,color=GRAY)
fig.suptitle('Cohort, information boundaries and nested evaluation',x=.045,ha='left',fontsize=14,weight='bold',y=.98)
save(fig,'Figure_R1_design')

# Joint paired accuracy/time display.
delta_ba=100*(ba_boot['adaptive']-ba_boot['fixed_optimized'])
delta_sec=(sec['adaptive']-sec['fixed_optimized'])[bi].mean(1)
point_ba=100*.5*sum((correct['adaptive'][ix].mean()-correct['fixed_optimized'][ix].mean()) for ix in class_indices)
point_sec=float((sec['adaptive']-sec['fixed_optimized']).mean())
write('joint_bootstrap.csv',[{'draw':i,'delta_ba_percentage_points':delta_ba[i],'delta_nominal_seconds':delta_sec[i]} for i in range(B)])
fig,(ax,bx)=plt.subplots(1,2,figsize=(10,4.4),gridspec_kw={'width_ratios':[1,1.05]})
ax.scatter(delta_sec,delta_ba,s=3,color=BLUE,alpha=.085,rasterized=True)
ax.scatter([point_sec],[point_ba],marker='D',s=42,color=ORANGE,zorder=4,label='Observed paired difference')
ax.axhline(0,color=GRAY,ls='--',lw=1); ax.axvline(0,color=GRAY,ls='--',lw=1)
ax.set(xlim=(-.5,14),ylim=(-11,10),xlabel='Adaptive − fixed mean nominal time (s)',ylabel='Adaptive − fixed binary BA (percentage points)')
ax.set_title('A  Paired participant bootstrap',loc='left',weight='bold')
ax.legend(loc='upper left',fontsize=8,frameon=False)
all_results=json.loads((BASE/'summary/ALL_RESULTS.json').read_text())['primary']['methods']
short=['Adaptive','Optimized fixed','Fixed + stopping','Fixed, squared error','Relaxed only','Clinical only','Full protocol']
all_m=METHODS+['fixed_brier','rest_only','clinical_only','full_protocol']
for i,m in enumerate(all_m):
    v=all_results[m]; c=COLORS[i] if i<3 else GRAY
    bx.scatter(v['mean_seconds'],100*v['underlying_binary_balanced_accuracy'],s=28,color=c)
    offset={'adaptive':(4,-13),'fixed_optimized':(4,3),'fixed_sequence_stop':(-3,7),'fixed_brier':(4,-2),'rest_only':(-16,-13),'clinical_only':(4,2),'full_protocol':(-59,6)}[m]
    bx.annotate(short[i],(v['mean_seconds'],100*v['underlying_binary_balanced_accuracy']),xytext=offset,textcoords='offset points',fontsize=7.4,color=c)
bx.axvline(71.68,color=GRAY,ls=':',lw=1)
bx.text(73,68.6,'Abbreviated cap',rotation=90,color=GRAY,fontsize=8)
bx.set(xlim=(-7,157),ylim=(68.3,74.4),xlabel='Mean nominal motion acquisition time (s)',ylabel='Underlying binary balanced accuracy (%)')
bx.set_title('B  Realized time and discrimination',loc='left',weight='bold')
for a in (ax,bx): a.grid(alpha=.12); a.set_axisbelow(True)
fig.text(.06,.01,'Bootstrap uncertainty conditions on fitted systems. Shared time caps do not imply equal realized time; questionnaire time is unmeasured.',fontsize=8,color=GRAY)
fig.tight_layout(rect=(0,.05,1,1),w_pad=2.5)
save(fig,'Figure_R2_joint_tradeoff')
summary['joint_contrast']={'delta_ba_pp':point_ba,'conditional_interval_ba_pp':ci(delta_ba),'delta_seconds':point_sec,'conditional_interval_seconds':ci(delta_sec)}

# Pre-existing demographic strata: paired conditional intervals. DD Wilson intervals.
age=np.array([float(meta[s]['age']) for s in ids])
gender=np.array([int(meta[s]['gender_male']) for s in ids])
groups=[('All participants',np.ones(390,dtype=bool)),('Age < 65',age<65),('Age ≥ 65',age>=65),('Recorded female',gender==0),('Recorded male',gender==1)]
gr=[]
for name,mask in groups:
    ix=[np.flatnonzero(mask&(y==c)) for c in (0,1)]
    rb=[rng.choice(z,size=(B,len(z)),replace=True) for z in ix]
    d=.5*sum(correct['adaptive'][b].mean(1)-correct['fixed_optimized'][b].mean(1) for b in rb)*100
    val=.5*sum(correct['adaptive'][z].mean()-correct['fixed_optimized'][z].mean() for z in ix)*100
    lo,hi=ci(d)
    if name=='All participants':
        lo,hi=ci(100*(ba_boot['adaptive']-ba_boot['fixed_optimized']))
    gr.append({'group':name,'n':int(mask.sum()),'pd':len(ix[1]),'dd':len(ix[0]),'delta_ba_pp':val,'ci_low_pp':lo,'ci_high_pp':hi})
write('subgroup_paired_intervals.csv',gr)
diagnoses=['Atypical Parkinsonism','Essential Tremor','Multiple Sclerosis','Other Movement Disorders']
ddrows=[]
for diagnosis in diagnoses:
    mask=np.array([meta[s]['condition']==diagnosis for s in ids])&(y==0)
    for m in METHODS[:2]:
        n=int(mask.sum()); k=int((pred[m][mask]==0).sum()); lo,hi=wilson(k,n)
        ddrows.append({'diagnosis':diagnosis,'method':m,'n':n,'correct_binary_dd':k,'specificity':k/n,'wilson_low':lo,'wilson_high':hi,
        'clear_correct':int((out[m][mask]==0).sum()),'clear_wrong':int((out[m][mask]==1).sum()),'indeterminate':int((out[m][mask]==-1).sum())})
write('dd_subtype_intervals_and_outputs.csv',ddrows)
fig,(ax,bx)=plt.subplots(1,2,figsize=(11.4,4.8),gridspec_kw={'width_ratios':[1.05,1]})
for j,r in enumerate(gr):
    ax.errorbar(r['delta_ba_pp'],j,xerr=[[r['delta_ba_pp']-r['ci_low_pp']],[r['ci_high_pp']-r['delta_ba_pp']]],fmt='o',color=BLUE,capsize=3,ms=5)
ax.set_yticks(range(5),[f"{r['group']}\nPD {r['pd']} / DD {r['dd']}" for r in gr],fontsize=8.5)
ax.axvline(0,color=GRAY,ls='--',lw=1); ax.set_ylim(4.65,-.65)
ax.set_xlim(-21,22);ax.set_xlabel('Adaptive − fixed binary BA (percentage points)')
ax.set_title('A  Conditional paired intervals',loc='left',weight='bold')
for j,d in enumerate(diagnoses):
    for k,m in enumerate(METHODS[:2]):
        r=next(t for t in ddrows if t['diagnosis']==d and t['method']==m)
        p=100*r['specificity'];lo=100*r['wilson_low'];hi=100*r['wilson_high'];yy=j+(k-.5)*.22
        bx.errorbar(p,yy,xerr=[[p-lo],[hi-p]],fmt=['o','s'][k],color=COLORS[k],capsize=2,ms=4,label=LABELS[k] if j==0 else None)
        bx.text(103,yy,f"{r['correct_binary_dd']}/{r['n']}",fontsize=8,va='center',color=COLORS[k])
bx.set_yticks(range(4),['Atypical parkinsonism','Essential tremor','Multiple sclerosis','Other movement disorders'],fontsize=8.3)
bx.set_xlim(10,120);bx.set_xticks([20,40,60,80,100]);bx.set_ylim(3.65,-.65)
bx.set_xlabel('Underlying binary DD specificity (%)');bx.set_title('B  DD subtypes: Wilson intervals',loc='left',weight='bold')
bx.legend(loc='lower left',fontsize=8,frameon=False)
for a in (ax,bx): a.grid(axis='x',alpha=.12);a.set_axisbelow(True)
fig.text(.04,.015,'Descriptive strata; unadjusted 95% intervals. Binary metrics include deferred participants. Small strata do not establish fairness or effect modification.',fontsize=8,color=GRAY)
fig.tight_layout(rect=(0,.06,1,1),w_pad=3)
save(fig,'Figure_R3_heterogeneity')
summary['demographic_intervals']=gr

# Fixed-bin diagnostic reliability: scores are not calibrated clinical probabilities.
binrows=[];scorerows=[]
fig,axs=plt.subplots(1,3,figsize=(10.5,3.65),sharex=True,sharey=True)
for ax,m,label,color in zip(axs,METHODS,LABELS,COLORS):
    p=score[m];bucket=np.minimum((p*5).astype(int),4)
    for b in range(5):
        mask=bucket==b;n=int(mask.sum())
        if n==0:continue
        k=int(y[mask].sum());obs=k/n;lo,hi=wilson(k,n);mean=float(p[mask].mean())
        binrows.append({'method':m,'bin_lower':b/5,'bin_upper':(b+1)/5,'n':n,'pd':k,'mean_score':mean,'pd_fraction':obs,'wilson_low':lo,'wilson_high':hi})
        ax.errorbar(mean,obs,yerr=[[obs-lo],[hi-obs]],fmt='o',color=color,capsize=2,ms=4)
        ax.annotate(str(n),(mean,obs),xytext=(3,5),textcoords='offset points',fontsize=7.3,color=color)
    brier=float(np.mean((p-y)**2));auc=float(roc_auc_score(y,p))
    scorerows.append({'method':m,'n':390,'cohort_brier_score':brier,'pooled_held_out_auroc':auc})
    ax.plot([0,1],[0,1],ls='--',color=GRAY,lw=.8)
    ax.set_title(label,loc='left',weight='bold',fontsize=9.4)
    ax.text(.03,.97,f'Brier = {brier:.3f}\nAUROC = {auc:.3f}',transform=ax.transAxes,va='top',fontsize=8)
    ax.set(xlim=(-.03,1.03),ylim=(-.03,1.06),xlabel='Mean decision score in bin')
    ax.grid(alpha=.12)
axs[0].set_ylabel('Observed PD fraction in bin')
fig.text(.04,.025,'Numbers denote bin sizes; bars are descriptive 95% Wilson intervals. Class-weighted scores are unrecalibrated and cohort-specific.',fontsize=8,color=GRAY)
fig.tight_layout(rect=(0,.07,1,1),w_pad=1.2)
save(fig,'Figure_R4_score_reliability')
write('score_reliability_bins.csv',binrows);write('score_metrics.csv',scorerows)
summary['score_metrics']=scorerows

# Explicit hypothetical deferral-cost scenarios; no implied actual clinical valuation.
lossrows=[];losscomponents=[]
fig,ax=plt.subplots(figsize=(7.8,4.2))
for m,label,color in zip(METHODS,LABELS,COLORS):
    wrong=(out[m]!=-1)&(out[m]!=y);defer=out[m]==-1
    w=.5*sum(wrong[ix].mean() for ix in class_indices)
    d=.5*sum(defer[ix].mean() for ix in class_indices)
    wb=.5*sum(wrong[b].mean(1) for b in boot);db=.5*sum(defer[b].mean(1) for b in boot)
    lam=np.linspace(0,1,101);val=w+lam*d
    intervals=np.quantile(wb[:,None]+db[:,None]*lam,[.025,.975],axis=0)
    ax.plot(lam,val,color=color,label=label,lw=1.8)
    ax.fill_between(lam,*intervals,color=color,alpha=.08,lw=0)
    losscomponents.append({'method':m,'class_balanced_wrong_clear_fraction':float(w),'class_balanced_deferred_fraction':float(d)})
    for i,l in enumerate(lam):lossrows.append({'method':m,'lambda':float(l),'expected_class_balanced_loss':float(val[i]),'conditional_low':float(intervals[0,i]),'conditional_high':float(intervals[1,i])})
ax.set(xlabel='Assumed cost of a deferred output, λ (one wrong clear output costs 1)',ylabel='Class-balanced expected loss',xlim=(0,1),ylim=(.1,.55))
ax.set_title('Three-way outputs under hypothetical referral penalties',loc='left',weight='bold')
ax.grid(alpha=.15);ax.legend(loc='upper left',frameon=False,fontsize=9)
fig.text(.1,.015,'Same frozen decisions; 95% conditional bands. This is not measured clinical utility. Motion and questionnaire costs are excluded.',fontsize=8,color=GRAY)
fig.tight_layout(rect=(0,.05,1,1))
save(fig,'Figure_R5_deferral_loss')
write('deferral_loss_scenarios.csv',lossrows);write('deferral_loss_components.csv',losscomponents)
summary['deferral_loss_components']=losscomponents

# Include original figures intact, avoiding renumbering the frozen manuscript.
for p in (BASE/'figures').glob('*'):
    if p.suffix in ('.png','.svg','.pdf'): shutil.copy2(p,FIG/p.name)
(ROOT/'ROUND1_RESULTS.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
inputs=[BASE/'ANALYSIS_PLAN.md',BASE/'run_analysis.py',BASE/'summary/held_out_predictions_primary.csv',WORK/'pads_features_received/PADS_features_return/participant_metadata.csv']
ledger={'inputs':{str(p.relative_to(WORK)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        'new_analysis_code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'new_spec_sha256':hashlib.sha256((ROOT/'ROUND1_SPEC.md').read_bytes()).hexdigest(),
        'new_figures':5,'inherited_figures':5,'formats':['png','pdf','svg'],'n_participants':390,
        'no_refitting':True,'bootstrap_draws':B,'seed':SEED,'primary_bootstrap_seed':20260913}
(ROOT/'COMPUTATION_LEDGER.json').write_text(json.dumps(ledger,indent=2)+'\n')
print(json.dumps(summary,indent=2,ensure_ascii=False))
