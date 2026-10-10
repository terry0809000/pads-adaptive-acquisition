"""Publication figures generated directly from the saved numerical tables."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

ROOT=Path(__file__).resolve().parent
T=ROOT/'tables';F=ROOT/'figures'
SEED=20260913
METHODS=['adaptive','fixed_optimized','fixed_sequence_stop','random_next_stop','permute_fold','permute_time_count']
LABELS={'adaptive':'Individualized tasks','fixed_optimized':'Optimized fixed tasks',
        'fixed_sequence_stop':'Fixed sequence + stopping','random_next_stop':'Random next task + stopping',
        'permute_fold':'Permuted paths', 'permute_time_count':'Time and count matched paths'}
SHORT={'adaptive':'Individualized','fixed_optimized':'Fixed','fixed_sequence_stop':'Fixed + stop',
       'random_next_stop':'Random + stop','permute_fold':'Permuted','permute_time_count':'Matched permutation'}
BLUE='#22577A';TEAL='#268B82';ORANGE='#C47239';RED='#B04C53';GRAY='#AEB8C1'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,'axes.labelsize':10,
                     'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,
                     'svg.fonttype':'none','axes.titleweight':'bold','savefig.facecolor':'white'})


def save(fig,name):
    for ext in ['png','pdf','svg']:
        fig.savefig(F/f'{name}.{ext}',dpi=350,bbox_inches='tight',pad_inches=.16)
    plt.close(fig)


def main():
    dif=pd.read_csv(T/'paired_contrasts.csv')
    ba=dif[dif.metric=='balanced_accuracy']
    fig,panels=plt.subplots(1,3,figsize=(13,4.6),gridspec_kw={'width_ratios':[1.5,1.05,1.1]},layout='constrained')
    axes=[panels[0],panels[2]]
    panels[1].axis('off');panels[1].set_ylim(-.65,4.65)
    panels[1].set_title('Difference [95% CI]',fontsize=10,fontweight='normal')
    comps=METHODS[3:]+METHODS[1:3]
    y=np.arange(len(comps))[::-1]
    for ax in axes:ax.axvline(0,color='#88949B',lw=1,ls='--',zorder=0);ax.grid(axis='x',color='#E5E8EB',lw=.7)
    for k,(m,yy) in enumerate(zip(comps,y)):
        row=ba[(ba.seed==SEED)&(ba.comparator==m)].iloc[0]
        val,lo,hi=100*np.array([row.adaptive_minus_comparator,row.low,row.high])
        axes[0].errorbar(val,yy,xerr=[[val-lo],[hi-val]],fmt='o',color=BLUE,capsize=3,lw=1.8,zorder=3)
        panels[1].text(.02,yy,f'{val:+.2f} [{lo:+.2f}, {hi:+.2f}]',va='center',fontsize=10)
        values=100*ba[(ba.seed!=SEED)&(ba.comparator==m)].sort_values('seed').adaptive_minus_comparator.to_numpy()
        for j,v in enumerate(values):axes[1].scatter(v,yy+(j-2)*.07,s=33,color=TEAL,alpha=.9,zorder=3)
        axes[1].plot([min(values),max(values)],[yy,yy],color=TEAL,lw=.8,zorder=1)
    axes[0].set_yticks(y,[LABELS[m] for m in comps]);axes[1].set_yticks(y,[])
    axes[0].set_title('A  Original partition',loc='left');axes[1].set_title('B  Alternative partitions',loc='left')
    axes[0].set_xlabel('Balanced accuracy difference (pp)')
    axes[1].set_xlabel('Balanced accuracy difference (pp)')
    for ax in axes:ax.set_ylim(-.65,4.65)
    save(fig,'Figure_R17_personalization_contrasts')

    classes=pd.read_csv(T/'class_outcomes.csv')
    classes=classes[classes.seed==SEED]
    fig,axes=plt.subplots(1,2,figsize=(12.5,5.1),sharey=True,layout='constrained')
    y=np.arange(len(METHODS))[::-1]
    for ax,cl in zip(axes,['PD','DD']):
        g=classes[classes['class']==cl].set_index('method').loc[METHODS]
        left=np.zeros(len(g))
        for field,color,name in [('clear_correct',TEAL,'Clear correct'),('deferred',GRAY,'Deferred'),('clear_wrong',RED,'Clear wrong')]:
            values=g[field].to_numpy()/g.n.to_numpy()*100
            ax.barh(y,values,left=left,height=.65,color=color,label=name)
            for j,(l,v) in enumerate(zip(left,values)):
                count=g[field].iloc[j]
                label=f'{count:.0f}' if METHODS[j] in METHODS[:3] else f'{count:.1f}'
                ax.text(l+v/2,y[j],label,ha='center',va='center',color='white' if field!='deferred' else '#1F2933',fontsize=9)
            left+=values
        ax.set_xlim(0,100);ax.set_xlabel('Participants in the disease class (%)')
        ax.set_title(f'{cl}   n = {int(g.n.iloc[0])}',loc='left');ax.set_yticks(y,[LABELS[m] for m in METHODS])
        ax.xaxis.set_major_formatter(PercentFormatter(100,decimals=0))
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=3,frameon=False)
    save(fig,'Figure_R18_random_control_class_outcomes')

    mc=pd.read_csv(T/'monte_carlo_realizations.csv')
    refs=pd.read_csv(T/'monte_carlo_reference_ranges.csv')
    original=pd.read_csv(T/'partition_metrics.csv').query('seed==20260913').set_index('method')
    fig,axes=plt.subplots(1,3,figsize=(12.4,4.1),layout='constrained')
    for ax,m,title in zip(axes,METHODS[3:],['A  Random next task','B  Permuted paths','C  Matched permutation']):
        values=mc[(mc.seed==SEED)&(mc.method==m)].balanced_accuracy*100
        ref=refs[(refs.seed==SEED)&(refs.method==m)&(refs.metric=='balanced_accuracy')].iloc[0]
        ax.hist(values,bins=22,color=BLUE,alpha=.8,edgecolor='white',linewidth=.4)
        ax.axvline(100*ref.exact_expected,color=ORANGE,lw=1.8,label='Exact random expectation')
        ax.axvline(100*original.loc['adaptive','balanced_accuracy'],color='#202A33',lw=1.8,ls='--',label='Individualized tasks')
        ax.set_title(title,loc='left');ax.set_xlabel('Balanced accuracy (%)')
        ax.text(.97,.96,f'Random reference range\n{100*ref.reference_low:.2f}–{100*ref.reference_high:.2f}%',transform=ax.transAxes,ha='right',va='top',fontsize=9)
        ax.set_ylim(0,ax.get_ylim()[1]*1.22)
    axes[0].set_ylabel('Random realizations')
    h,l=axes[0].get_legend_handles_labels();fig.legend(h,l,loc='outside lower center',ncol=2,frameon=False)
    save(fig,'Figure_R19_randomization_reference')

    task=pd.read_csv(T/'task_frequencies.csv').query('seed==20260913').set_index('method')
    pool=pd.read_csv(T/'permutation_pool_summary.csv').query("method=='permute_time_count'").sort_values('seed')
    fig,axes=plt.subplots(1,2,figsize=(12.2,5.6),gridspec_kw={'width_ratios':[1.1,1]},layout='constrained')
    tasks=list(task.columns[1:])
    fields=METHODS[:4]
    mat=100*task.loc[fields,tasks].to_numpy().T
    im=axes[0].imshow(mat,vmin=0,vmax=100,cmap='Blues',aspect='auto')
    axes[0].set_xticks(range(4),[SHORT[m] for m in fields],rotation=30,ha='right')
    axes[0].set_yticks(range(len(tasks)),tasks)
    for j in range(mat.shape[0]):
        for k in range(mat.shape[1]):axes[0].text(k,j,f'{mat[j,k]:.0f}',ha='center',va='center',fontsize=9,color='white' if mat[j,k]>60 else '#172B3A')
    axes[0].set_title('A  Original task acquisition frequency (%)',loc='left')
    yy=np.arange(6)[::-1]
    change=100*pool.mean_mask_change_probability.to_numpy()
    immobile=100*pool.single_mask_participants.to_numpy()/390
    axes[1].barh(yy+.15,change,height=.28,color=BLUE,label='Probability of a different task set')
    axes[1].barh(yy-.15,immobile,height=.28,color=GRAY,label='Participants in a single-mask pool')
    for j in range(6):
        axes[1].text(change[j]+1,yy[j]+.15,f'{change[j]:.1f}%',va='center',fontsize=9)
        axes[1].text(immobile[j]+1,yy[j]-.15,f'{immobile[j]:.1f}%',va='center',fontsize=9)
    axes[1].set_yticks(yy,['Original']+[f'Alternative {i}' for i in range(1,6)])
    axes[1].set_xlim(0,100);axes[1].set_xlabel('Percentage');axes[1].set_title('B  Scope of the matched permutation',loc='left')
    axes[1].legend(loc='lower center',bbox_to_anchor=(.5,-.34),frameon=False,fontsize=9)
    save(fig,'Figure_R20_action_assignment_audit')


if __name__=='__main__':main()
