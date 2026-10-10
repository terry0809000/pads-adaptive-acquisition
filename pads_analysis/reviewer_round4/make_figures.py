"""Round-4 scientific figures; all values read from saved evaluation tables."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

R=Path(__file__).resolve().parent
F=R/"figures";F.mkdir(exist_ok=True)
M=["adaptive","fixed_optimized","fixed_sequence_stop"]
ML=["Adaptive selection","Optimized fixed","Fixed sequence + stopping"]
RULES=["original_margin","raw_label_confidence","calibrated_label_confidence","mean_class_loss","worst_class_loss"]
RL=["Original margin","Raw confidence","Calibrated confidence","Mean-loss allocation","Worst-loss allocation"]
C=["#6D7984","#BC7728","#976AB4","#167D9A","#B4454D"]
S=list(range(20260913,20260919))
T=pd.read_csv(R/"tables/repeat_allocation_metrics.csv")
CI=pd.read_csv(R/"tables/conditional_intervals.csv")
A=pd.read_csv(R/"tables/selected_allocations.csv")
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.spines.top":False,
    "axes.spines.right":False,"pdf.fonttype":42,"ps.fonttype":42,"axes.titlesize":10})

def row(s,m,r,q=.2,cost=.5):
    v=T[(T.seed==s)&(T.method==m)&(T.rule==r)&np.isclose(T.target,q)&
        np.isclose(T.selection_lambda, cost if r in RULES[3:] else -1)]
    assert len(v)==1
    return v.iloc[0]

def save(fig,name):
    for ext in ["png","pdf","svg"]:
        fig.savefig(F/f"{name}.{ext}",dpi=350,bbox_inches="tight",facecolor="white")
    plt.close(fig)

# R13: all people in each reference class remain in the denominator.
fig,axs=plt.subplots(2,3,figsize=(12.4,7.4),sharex=True)
outcolors=["#307E79","#D9AE52","#B74D56"]
for j,(m,ml) in enumerate(zip(M,ML)):
    for k,cls in enumerate(["pd","dd"]):
        ax=axs[k,j]
        for y,r in enumerate(RULES):
            v=row(S[0],m,r);left=0
            for kind,color in zip(["clear_correct","deferred","clear_wrong"],outcolors):
                n=int(v[f"{cls}_{kind}"]);value=100*n/int(v[cls])
                ax.barh(y,value,left=left,color=color,height=.67)
                ax.text(left+value/2,y,str(n),ha="center",va="center",fontsize=8,
                        color="#27313A" if kind=="deferred" else "white")
                left+=value
        ax.set_yticks(range(5),RL if j==0 else [""]*5,fontsize=8)
        ax.set_ylim(4.7,-.6);ax.set_xlim(0,100);ax.set_xticks([0,25,50,75,100])
        ax.set_title(f"{chr(65+k*3+j)}  {ml}",loc="left",weight="bold")
        ax.set_xlabel(f"Reference {cls.upper()} class (%)")
        if j==0:ax.set_ylabel(f"{cls.upper()}: n = {276 if cls=='pd' else 114}",labelpad=10)
fig.legend(handles=[Patch(facecolor=c,label=l) for c,l in zip(outcolors,["Clear correct","Deferred","Clear wrong"])],
           loc="lower center",ncol=3,frameon=False,bbox_to_anchor=(.5,.038))
fig.text(.025,.01,"Original partition. Bar labels are participant counts. All rules target 20% rejection in development; allocation uses illustrative cost lambda = 0.5.",fontsize=8,color="#5A6772")
fig.tight_layout(rect=(0,.11,1,1),h_pad=2.1,w_pad=2)
save(fig,"Figure_R13_class_output_allocation")

# R14: new-partition points are repeated evaluations, not independent cohorts.
comparisons=[("mean_class_loss","original_margin"),("worst_class_loss","original_margin"),
             ("mean_class_loss","calibrated_label_confidence"),("worst_class_loss","calibrated_label_confidence")]
yl=["Mean − margin","Worst − margin","Mean − calibrated","Worst − calibrated"]
fig,axs=plt.subplots(2,3,figsize=(12.3,7.5))
for j,(m,ml) in enumerate(zip(M,ML)):
    for k,metric in enumerate(["mean_loss_0.5","worst_loss_0.5"]):
        ax=axs[k,j];ax.axvline(0,color="#8C959D",ls="--",lw=1)
        ax.axhspan(1.5,3.5,color="#F3F5F7",zorder=0)
        for y,(new,base) in enumerate(comparisons):
            color=C[RULES.index(new)]
            iv=CI[(CI.method==m)&(CI.comparison==f"{new}_minus_{base}")&(CI.metric==metric)].iloc[0]
            ax.errorbar(iv.estimate,y-.10,xerr=[[iv.estimate-iv.lower_95],[iv.upper_95-iv.estimate]],
                        color=color,marker="D",ms=4,capsize=2,lw=1,zorder=3)
            dx=[row(s,m,new)[metric]-row(s,m,base)[metric] for s in S[1:]]
            ax.scatter(dx,y+.14+np.linspace(-.055,.055,5),s=20,facecolors="none",edgecolors=color,lw=1,zorder=3)
        ax.set_yticks(range(4),yl if j==0 else [""]*4,fontsize=8)
        ax.set_ylim(3.65,-.55);ax.grid(axis="x",alpha=.1)
        ax.set_title(f"{chr(65+k*3+j)}  {ml}",loc="left",weight="bold")
        ax.set_xlabel("Change in mean class loss" if k==0 else "Change in worst-class loss")
        ax.set_xlim((-.10,.07) if k==0 else (-.29,.08))
fig.legend(handles=[Line2D([0],[0],color="#586570",marker="D",lw=1,label="Original: conditional 95% interval"),
                    Line2D([0],[0],color="#586570",marker="o",mfc="none",lw=0,label="Five new partitions: individual values")],
           loc="lower center",ncol=2,frameon=False,bbox_to_anchor=(.5,.038))
fig.text(.025,.01,"Lower favors the new allocation. Target = 20%; lambda = 0.5. Intervals: 5,000 paired class-stratified bootstrap draws, without refitting.",fontsize=8,color="#5A6772")
fig.tight_layout(rect=(0,.11,1,1),h_pad=2.3,w_pad=1.8)
save(fig,"Figure_R14_loss_contrasts")

# R15: cost at selection and evaluation is the same; no post-test cost choice.
fig,axs=plt.subplots(2,3,figsize=(12.2,7.5))
costs=[0,.25,.5,.75,1];targets=[0,.1,.2,.3,.4]
for j,(m,ml) in enumerate(zip(M,ML)):
    ax,bx=axs[:,j]
    for r,color in zip(RULES,C):
        vals=[row(S[0],m,r,cost=x)[f"mean_loss_{x:g}"] for x in costs]
        ax.plot(costs,vals,color=color,marker="o",ms=3,lw=1.15)
        yy=np.array([[row(s,m,r,q=x)["rejection"]*100 for s in S[1:]] for x in targets])
        bx.plot(np.array(targets)*100,yy.mean(1),color=color,marker="o",ms=3,lw=1.15)
        bx.fill_between(np.array(targets)*100,yy.min(1),yy.max(1),color=color,alpha=.07)
    ax.set_title(f"{chr(65+j)}  {ml}",loc="left",weight="bold")
    ax.set(xlabel="Hypothetical deferral cost (lambda)",xticks=costs,ylim=(.13,.5))
    bx.set_title(f"{chr(68+j)}  Rejection target transfer",loc="left",weight="bold")
    bx.plot([0,40],[0,40],color="#87949C",ls="--",lw=.8)
    bx.set(xlabel="Development rejection target (%)",xticks=[0,10,20,30,40],ylim=(-1,54))
    if j==0:ax.set_ylabel("Mean class loss: original partition");bx.set_ylabel("Test rejection: mean and range (%)")
    ax.grid(alpha=.1);bx.grid(alpha=.1)
fig.legend(handles=[Line2D([0],[0],color=c,lw=1.4,label=l) for c,l in zip(C,RL)],
           loc="lower center",ncol=3,frameon=False,bbox_to_anchor=(.5,.043),fontsize=8)
fig.text(.025,.012,"Top: target = 20%; new allocations are selected for each lambda and evaluated at that same lambda. Bottom: lambda = 0.5; five new partitions.",fontsize=8,color="#5A6772")
fig.tight_layout(rect=(0,.14,1,1),h_pad=2,w_pad=2)
save(fig,"Figure_R15_cost_and_target_sensitivity")

# R16: predicted class allocation must not be mislabeled reference-class rejection.
fig,axs=plt.subplots(2,3,figsize=(11.8,7.5),layout="constrained")
for j,(m,ml) in enumerate(zip(M,ML)):
    for k,r in enumerate(RULES[3:]):
        ax=axs[k,j]
        a=A[(A.method==m)&(A.rule==r)&np.isclose(A.target,.2)&np.isclose(A.selection_lambda,.5)]
        assert len(a)==30 and (a.development_rejected==62).all()
        values=a.pivot(index="seed",columns="outer_fold",values="k_predicted_pd").reindex(index=S).values/62*100
        im=ax.imshow(values,aspect="auto",vmin=0,vmax=100,cmap="Blues")
        for ii in range(6):
            for jj in range(5):
                ax.text(jj,ii,f"{values[ii,jj]:.0f}",ha="center",va="center",fontsize=8,
                        color="white" if values[ii,jj]>55 else "#263942")
        ax.set_yticks(range(6),["Original","Repeat 1","Repeat 2","Repeat 3","Repeat 4","Repeat 5"])
        ax.set_xticks(range(5),[1,2,3,4,5]);ax.set_xlabel("Outer fold")
        ax.set_title(f"{chr(65+k*3+j)}  {ml}",loc="left",weight="bold")
        if j==0:ax.set_ylabel("Mean-loss allocation" if k==0 else "Worst-loss allocation")
fig.colorbar(im,ax=axs,label="Development rejections allocated to predicted PD (%)",shrink=.85,pad=.025)
fig.suptitle("Allocation across predicted labels: 62 of 312 development participants rejected",fontsize=11)
save(fig,"Figure_R16_development_allocation_stability")
print("Saved four figure groups as PNG (350 dpi), PDF and SVG.")
