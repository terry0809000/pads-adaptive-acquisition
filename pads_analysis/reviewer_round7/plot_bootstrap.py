"""Publication figures from the complete audited 500-replicate result tables."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parent
FIG = ROOT / 'figures'
TAB = ROOT / 'tables'
METHODS = ['adaptive','fixed_optimized','fixed_sequence_stop','random_next_stop']
LABELS = {'adaptive':'Adaptive','fixed_optimized':'Optimized fixed','fixed_sequence_stop':'Fixed + stopping','random_next_stop':'Random next task'}
COLORS = {'adaptive':'#2569A5','fixed_optimized':'#D07A25','fixed_sequence_stop':'#218477','random_next_stop':'#875BA7'}
CONTRASTS = [('adaptive','fixed_optimized'),('adaptive','fixed_sequence_stop'),('adaptive','random_next_stop'),('fixed_sequence_stop','fixed_optimized')]
CLABELS = ['Adaptive − fixed','Adaptive − stopping','Adaptive − random','Stopping − fixed']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':13,'axes.titlesize':14,'axes.labelsize':13,'xtick.labelsize':12,
                     'ytick.labelsize':12,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,
                     'savefig.facecolor':'white','axes.edgecolor':'#555555','text.color':'#222222'})
manifest=[]

def save(fig,name,source_files):
    paths=[]
    for ext in ['png','pdf','svg']:
        path=FIG/(name+'.'+ext)
        fig.savefig(path,dpi=320,bbox_inches='tight',pad_inches=.12)
        paths.append({'file':path.relative_to(ROOT).as_posix(),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    manifest.append({'figure':name,'data':[Path(os.path.relpath(p,ROOT)).as_posix() for p in source_files],'files':paths})
    plt.close(fig)

def ecdf(ax,values,**kwargs):
    values=np.sort(np.asarray(values))
    ax.step(values,np.arange(1,len(values)+1)/len(values),where='post',**kwargs)

def panel(ax,label,title):
    ax.set_title(label+'  '+title,loc='left',fontweight='bold',pad=12)

def main():
    assert json.loads((ROOT/'VALIDATION.json').read_text())['complete_replicates']==500
    FIG.mkdir(exist_ok=True)
    for path in (ROOT.parent/'reviewer_round6/figures').glob('*'):
        if path.suffix in ['.png','.pdf','.svg']:shutil.copy2(path,FIG/path.name)
    paired=pd.read_csv(TAB/'paired_stability.csv')
    diffs=pd.read_csv(TAB/'replicate_paired_contrasts.csv')
    metrics=pd.read_csv(TAB/'replicate_metrics.csv')
    persons=pd.read_csv(TAB/'participant_stability.csv')
    thresholds=pd.read_csv(TAB/'fold_thresholds_and_protocols.csv')
    counts=pd.read_csv(TAB/'resample_counts.csv')
    sizes=pd.read_csv(TAB/'fit_sample_sizes.csv')
    assert metrics.groupby('method').size().eq(500).all()
    original_ledger_path=ROOT.parent/'reviewer_round6/EVIDENCE_LEDGER.json'
    original_ledger={r['id']:r for r in json.loads(original_ledger_path.read_text())}
    original_random_path=ROOT.parent/'reviewer_round5/tables/paired_contrasts.csv'
    original_random=pd.read_csv(original_random_path)
    original_random=original_random[(original_random.seed==20260913)&(original_random.comparator=='random_next_stop')].set_index('metric')
    first=original_ledger['E02']['evidence'];stop=original_ledger['E03']['evidence']
    original=[(100*first['balanced_accuracy']['difference'],first['mean_seconds']['difference']),
              (100*(first['balanced_accuracy']['difference']-stop['balanced_accuracy']['difference']),first['mean_seconds']['difference']-stop['mean_seconds']['difference']),
              (100*original_random.loc['balanced_accuracy','adaptive_minus_comparator'],original_random.loc['nominal_seconds','adaptive_minus_comparator']),
              (100*stop['balanced_accuracy']['difference'],stop['mean_seconds']['difference'])]

    fig,axes=plt.subplots(1,2,figsize=(10.2,6.2),gridspec_kw={'wspace':.28})
    for ai,(metric,scale,title,xlabel) in enumerate([('balanced_accuracy',100,'Accuracy difference','Balanced accuracy difference\n(percentage points)'),('nominal_seconds',1,'Acquisition difference','Mean nominal time difference\n(seconds)')]):
        ax=axes[ai]
        ax.axvline(0,color='#888888',lw=1,zorder=0)
        for k,(left,right) in enumerate(CONTRASTS):
            r=paired[(paired.method==left)&(paired.comparator==right)&(paired.metric==metric)].iloc[0]
            y=3-k;color=COLORS[left]
            ax.errorbar(r['median']*scale,y,xerr=np.array([[r['median']-r.low],[r.high-r['median']]])*scale,
                        fmt='o',color=color,ms=7,lw=2,capsize=3,zorder=4)
            ax.scatter(original[k][ai],y+.15,marker='D',s=31,color='#444444',zorder=5)
            ax.text(.03,y-.25,f"{r['median']*scale:+.2f} [{r.low*scale:+.2f}, {r.high*scale:+.2f}]",
                    transform=ax.get_yaxis_transform(),fontsize=12,color=color,va='center',
                    bbox={'facecolor':'white','edgecolor':'none','pad':1})
        ax.set(ylim=(-.55,3.6),yticks=[3,2,1,0],xlabel=xlabel)
        ax.set_yticklabels(['Adaptive − fixed','Adaptive − stopping','Adaptive − random','Stopping − fixed'] if ai==0 else [])
        ax.tick_params(axis='y',length=0,pad=10)
        ax.spines['left'].set_visible(False)
        panel(ax,'AB'[ai],title)
    fig.legend(handles=[Line2D([],[],color=COLORS['adaptive'],marker='o',lw=2,label='Refit median and central 95% empirical interval'),
                        Line2D([],[],color='#444444',marker='D',lw=0,label='Original-partition estimate')],
               loc='lower center',bbox_to_anchor=(.48,-.06),ncol=1,frameon=False,fontsize=12)
    fig.suptitle('Participant resampling with complete refitting',x=.02,y=1.075,ha='left',fontsize=17,fontweight='bold')
    fig.text(.02,1.0,'500 resamples · 390 rows per resample · repeated identities grouped',ha='left',fontsize=12,color='#555555')
    fig.subplots_adjust(left=.21,right=.985,bottom=.21)
    save(fig,'Figure_7_complete_refit',[TAB/'paired_stability.csv',original_ledger_path,original_random_path])

    fig,axes=plt.subplots(2,2,figsize=(10.5,8.6))
    limx=max(10,float(np.ceil(abs(diffs.balanced_accuracy).max()*100/5)*5))
    limy=max(15,float(np.ceil(abs(diffs.nominal_seconds).max()/5)*5))
    for i,((left,right),ax) in enumerate(zip(CONTRASTS,axes.ravel())):
        d=diffs[(diffs.method==left)&(diffs.comparator==right)]
        ax.add_patch(Rectangle((0,-limy),limx,limy,facecolor='#EAF3EC',edgecolor='none',zorder=0))
        ax.scatter(d.balanced_accuracy*100,d.nominal_seconds,s=14,alpha=.36,color=COLORS[left],edgecolors='none')
        ax.scatter(*original[i],marker='D',s=42,color='#303030',zorder=4)
        ax.axvline(0,color='#777777',lw=.9);ax.axhline(0,color='#777777',lw=.9)
        count=int(((d.balanced_accuracy>1e-12)&(d.nominal_seconds<-1e-12)).sum())
        ax.text(.04,.96,f'Higher accuracy + less time: {count}/500',transform=ax.transAxes,va='top',fontsize=12,
                bbox={'facecolor':'white','edgecolor':'none','alpha':.8,'pad':3})
        ax.set(xlim=(-limx,limx),ylim=(-limy,limy),xlabel='Balanced accuracy difference (pp)',ylabel='Mean nominal time difference (s)')
        panel(ax,'ABCD'[i],CLABELS[i])
    fig.tight_layout(h_pad=2.8,w_pad=2.6)
    save(fig,'Figure_S15_joint_refit',[TAB/'replicate_paired_contrasts.csv',original_ledger_path,original_random_path])

    fig,axes=plt.subplots(2,3,figsize=(11,8.2),gridspec_kw={'hspace':.45,'wspace':.32})
    for j,method in enumerate(METHODS[:3]):
        d=persons[persons.method==method]
        ax=axes[0,j]
        ax.vlines(d.original_score,d.score_low,d.score_high,lw=.5,color=COLORS[method],alpha=.13)
        ax.scatter(d.original_score,d.score_median,s=8,color=COLORS[method],alpha=.65,edgecolors='none')
        ax.plot([0,1],[0,1],ls='--',color='#777777',lw=.9)
        ax.set(xlim=(0,1),ylim=(0,1),xlabel='Original held-out score',ylabel='Refit held-out score' if j==0 else '')
        panel(ax,'ABC'[j],LABELS[method])
        ax=axes[1,j]
        for key,color,style,label in [('binary_disagreement_with_original','#2569A5','-','Binary decision'),
                                     ('output_disagreement_with_original','#875BA7','--','Three-way output'),
                                     ('task_disagreement_with_original','#218477',':','Task set')]:
            ecdf(ax,d[key]*100,color=color,ls=style,lw=1.8,label=label)
        ax.set(xlim=(0,100),ylim=(0,1.02),xlabel='Refits differing\nfrom original (%)',ylabel='Fraction of participants' if j==0 else '')
        panel(ax,'DEF'[j],'Disagreement')
        if j==0:ax.legend(loc='upper left',bbox_to_anchor=(.05,-.25),frameon=False,ncol=3,fontsize=12)
    save(fig,'Figure_S16_participant_refit',[TAB/'participant_stability.csv'])

    fig,axes=plt.subplots(2,2,figsize=(11.2,8.2),gridspec_kw={'hspace':.48,'wspace':.32})
    grid=np.linspace(.05,.95,19)
    freq=[]
    for method in METHODS:
        v=thresholds.loc[thresholds.method==method,'binary_threshold'].to_numpy()
        assert len(v)==2500
        freq.append([np.isclose(v,t,atol=1e-12).mean()*100 for t in grid])
    ax=axes[0,0];im=ax.imshow(freq,aspect='auto',cmap='Blues',vmin=0)
    ax.set(xticks=range(0,19,3),yticks=range(4),xlabel='Selected binary cutoff')
    ax.set_xticklabels([f'{v:.2f}' for v in grid[::3]])
    ax.set_yticklabels([LABELS[m] for m in METHODS])
    fig.colorbar(im,ax=ax,fraction=.04,pad=.03,label='Outer fits (%)')
    panel(ax,'A','Binary threshold selection')
    ax=axes[0,1]
    for method in METHODS:ecdf(ax,thresholds.loc[thresholds.method==method,'rejection_margin'],color=COLORS[method],lw=1.8,label=LABELS[method])
    ax.set(xlabel='Selected rejection margin',ylabel='Fraction of outer fits',ylim=(0,1.02))
    panel(ax,'B','Rejection margins')
    ax=axes[1,0]
    for method in METHODS:ecdf(ax,metrics.loc[metrics.method==method,'deferral']*100,color=COLORS[method],lw=1.8)
    ax.axvline(20,color='#555555',lw=1,ls='--')
    ax.set(xlabel='Held-out deferral (%)',ylabel='Fraction of resamples',ylim=(0,1.02))
    panel(ax,'C','Deferral after refitting')
    ax=axes[1,1]
    v=thresholds.loc[thresholds.method==METHODS[0],'fixed_mask'].value_counts()
    pd.DataFrame({'task_mask':v.index,'outer_fits':v.values,'fraction':v.values/2500}).to_csv(TAB/'selected_fixed_mask_frequency.csv',index=False)
    top=v.iloc[:8];labels=[f'Mask {int(i)}' for i in top.index];values=list(top.values)
    if len(v)>8:labels.append('All other masks');values.append(int(v.iloc[8:].sum()))
    ax.barh(range(len(labels)),np.array(values)/2500*100,color=COLORS['fixed_optimized'],height=.65)
    ax.set(yticks=range(len(labels)),xlabel='Outer fits (%)');ax.set_yticklabels(labels);ax.invert_yaxis()
    panel(ax,'D','Selected fixed protocols')
    fig.legend(handles=[Line2D([],[],color=COLORS[m],lw=2,label=LABELS[m]) for m in METHODS],loc='lower center',bbox_to_anchor=(.5,-.03),ncol=4,frameon=False,fontsize=11)
    save(fig,'Figure_S17_thresholds',[TAB/'fold_thresholds_and_protocols.csv',TAB/'replicate_metrics.csv'])

    fig,axes=plt.subplots(1,2,figsize=(11,4.2),gridspec_kw={'wspace':.65})
    ax=axes[0];ax.hist(counts.unique_participants,bins=np.arange(counts.unique_participants.min()-.5,counts.unique_participants.max()+1.5,1),color='#527E9C',edgecolor='white',lw=.5)
    ax.set(xlabel='Unique participants\namong 390 sampled rows',ylabel='Number of resamples')
    ax.axvline(counts.unique_participants.median(),color='#303030',lw=1,ls='--')
    panel(ax,'A','Distinct participant information')
    ax=axes[1]
    levels=['outer_final','middle_system','middle_system_state_crossfit']
    level_names=['Outer final fit','Middle system fit','Innermost state fit']
    data=[sizes.loc[sizes.stage==k,'unique_participants'] for k in levels]
    parts=ax.boxplot(data,vert=False,tick_labels=level_names,showfliers=False,whis=(2.5,97.5),patch_artist=True,
                     medianprops={'color':'#222222','linewidth':1.4},boxprops={'facecolor':'#C7D9E6','edgecolor':'#527E9C'})
    for j,n in enumerate([312,208,138.5]):
        if j<2:ax.scatter(n,j+1,marker='D',color='#333333',s=35,zorder=4)
        else:ax.plot([138,139],[j+1,j+1],color='#333333',lw=3,zorder=4)
    ax.set(xlabel='Unique participants\navailable for training',xlim=(60,330));ax.invert_yaxis()
    panel(ax,'B','Nested training sample sizes')
    save(fig,'Figure_S18_sample_information',[TAB/'resample_counts.csv',TAB/'fit_sample_sizes.csv'])

    previous=json.loads((ROOT.parent/'reviewer_round6/FIGURE_MANIFEST.json').read_text())
    (ROOT/'FIGURE_MANIFEST.json').write_text(json.dumps({'new_figures':manifest,'historical_figures':previous,'historical_copy_policy':'Round-6 PNG, PDF and SVG assets copied without editing.'},indent=2)+'\n')
    print(f'{len(manifest)} new figures in PNG, PDF and SVG; 20 earlier figures retained.')

if __name__=='__main__':main()
