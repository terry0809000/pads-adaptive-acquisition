"""Publication figures from saved outer-held-out round-3 results."""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

R=Path(__file__).resolve().parent
F=R/"figures";F.mkdir(exist_ok=True)
def read(name):
    with (R/"tables"/name).open() as f:return list(csv.DictReader(f))
def save(fig,name):
    for ext in ["png","pdf","svg"]:
        fig.savefig(F/f"{name}.{ext}",dpi=350,bbox_inches="tight",facecolor="white")
    plt.close(fig)

METHODS=["adaptive","fixed_optimized","fixed_sequence_stop"]
LABELS=["Adaptive selection","Optimized fixed","Fixed sequence + stopping"]
MC=["#2364A2","#C76A20","#24766D"]
RULES=["original_margin","raw_label_confidence","calibrated_label_confidence"]
RL=["Original margin","Raw label confidence","Calibrated label confidence"]
RC=["#657582","#C76A20","#2364A2"]
SEEDS=list(range(20260913,20260919))
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.spines.top":False,
    "axes.spines.right":False,"pdf.fonttype":42,"ps.fonttype":42,"axes.titlesize":10})
bins=read("reliability_bins.csv")
metrics=read("probability_metrics.csv")
repeat=read("repeat_rejection_metrics.csv")
folds=read("fold_rejection_metrics.csv")
ci=read("conditional_intervals.csv")
dc=read("calibration_diagnostic_intervals.csv")
mi={(int(r["seed"]),r["method"],r["version"]):r for r in metrics}
ri={(int(r["seed"]),r["method"],r["rule"],float(r["target"])):r for r in repeat}

# R9: bin-level reliability with explicit sampling uncertainty.
fig,axs=plt.subplots(2,3,figsize=(11.2,7.2),sharex=True,sharey=True)
for col,(method,label) in enumerate(zip(METHODS,LABELS)):
    for row,kind in enumerate(["disease_probability","assigned_label_confidence"]):
        ax=axs[row,col]
        ax.plot([0,1],[0,1],color="#87949C",ls="--",lw=1,zorder=0)
        for version,color in [("raw",RC[1]),("calibrated",RC[2])]:
            b=sorted([r for r in bins if r["seed"]==str(SEEDS[0]) and r["method"]==method and r["kind"]==kind and r["version"]==version],key=lambda r:int(r["bin"]))
            x=np.array([float(r["mean_prediction"]) for r in b])
            y=np.array([float(r["observed_fraction"]) for r in b])
            lo=np.array([float(r["wilson_lower"]) for r in b]);hi=np.array([float(r["wilson_upper"]) for r in b])
            ax.errorbar(x,y,yerr=[np.maximum(0,y-lo),np.maximum(0,hi-y)],color=color,lw=1.05,elinewidth=.75,capsize=2,
                        marker="o",markersize=3.5,label=version.capitalize(),alpha=.9)
        ax.set_xlim(-.025,1.025);ax.set_ylim(-.025,1.025);ax.set_xticks([0,.25,.5,.75,1])
        ax.grid(alpha=.1)
        if row==0:ax.set_title(f"{chr(65+col)}  {label}",loc="left",weight="bold")
        else:ax.set_title(f"{chr(68+col)}  Confidence in assigned label",loc="left",weight="bold")
        ax.set_xlabel("Mean predicted PD probability" if row==0 else "Mean assigned-label confidence")
axs[0,0].set_ylabel("Observed PD proportion");axs[1,0].set_ylabel("Observed correct proportion")
fig.legend(handles=[Line2D([0],[0],color=RC[1],marker="o",label="Raw"),Line2D([0],[0],color=RC[2],marker="o",label="Calibrated")],
           loc="lower center",ncol=2,frameon=False,bbox_to_anchor=(.5,.035))
fig.text(.045,.008,"Original partition; 390 participants. Ten fixed probability bins; bars show pointwise Wilson intervals, conditional on fitted scores.",fontsize=8,color=RC[0])
fig.tight_layout(rect=(0,.105,1,1),h_pad=2,w_pad=1.8)
save(fig,"Figure_R9_probability_reliability")

# R10: development targets versus realized rejection; within-fold and repeat dispersion.
fig,axs=plt.subplots(2,3,figsize=(11.3,7.5))
for col,(m,label) in enumerate(zip(METHODS,LABELS)):
    ax,bx=axs[:,col]
    ax.plot([0,40],[0,40],color="#87949C",ls="--",lw=1)
    for j,(rule,color) in enumerate(zip(RULES,RC)):
        q=np.array([0,.1,.2,.3,.4])
        y=np.array([[100*float(ri[s,m,rule,t]["rejection"]) for s in SEEDS[1:]] for t in q])
        mid=y.mean(1)
        ax.errorbar(100*q+(j-1)*.5,mid,yerr=[mid-y.min(1),y.max(1)-mid],color=color,
                    marker="o",markersize=3.5,lw=1.1,elinewidth=.8,capsize=2)
    ax.set_title(f"{chr(65+col)}  {label}",loc="left",weight="bold")
    ax.set(xlabel="Development rejection target (%)",xlim=(-2,42),ylim=(-2,57))
    if col==0:ax.set_ylabel("Test rejection: mean and range (%)")
    ax.grid(alpha=.1)
    bx.axhline(20,color="#87949C",ls="--",lw=1)
    bx.axvspan(-.45,.45,color="#EDF2F6",zorder=0)
    for j,(rule,color) in enumerate(zip(RULES,RC)):
        xx=np.arange(6)+(j-1)*.21
        yy=[100*float(ri[s,m,rule,.2]["rejection"]) for s in SEEDS]
        for k,s in enumerate(SEEDS):
            fr=sorted([r for r in folds if int(r["seed"])==s and r["method"]==m and r["rule"]==rule and float(r["target"])==.2],key=lambda r:int(r["outer_fold"]))
            bx.scatter(xx[k]+np.linspace(-.048,.048,5),[100*float(r["rejection"]) for r in fr],
                       color=color,alpha=.3,s=10,zorder=2)
        bx.scatter(xx,yy,color=color,marker="D",s=25,edgecolors="white",linewidths=.4,zorder=3)
    bx.set_title(f"{chr(68+col)}  Transfer of the 20% target",loc="left",weight="bold")
    bx.set_xticks(range(6),["P","1","2","3","4","5"]);bx.set(xlabel="Original partition / new repeat",ylim=(-1,61),xlim=(-.5,5.5))
    if col==0:bx.set_ylabel("Test rejection (%)")
    bx.grid(axis="y",alpha=.12)
fig.legend(handles=[Line2D([0],[0],color=c,marker="o",label=l) for c,l in zip(RC,RL)],
           loc="lower center",ncol=3,frameon=False,bbox_to_anchor=(.5,.039))
fig.text(.045,.012,"Top: mean and min–max over five new repeats, not confidence intervals. Bottom: diamonds = 390-person repeats; small dots = 78-person folds.",fontsize=8,color=RC[0])
fig.tight_layout(rect=(0,.11,1,1),h_pad=2,w_pad=1.8)
save(fig,"Figure_R10_rejection_target_transfer")

# R11: selective error and retained full-class denominators.
fig,axs=plt.subplots(2,3,figsize=(11.5,7.7))
outcolors=["#24766D","#C99C42","#B95759"]
for col,(m,label) in enumerate(zip(METHODS,LABELS)):
    ax,bx=axs[:,col]
    for rule,color in zip(RULES,RC):
        rr=[ri[SEEDS[0],m,rule,t] for t in [0,.1,.2,.3,.4]]
        xx=[100*float(r["coverage"]) for r in rr]
        yy=[100*float(r["selective_error"]) for r in rr]
        ax.plot(xx,yy,color=color,marker="o",ms=3,lw=1.2)
        ax.scatter(xx[2],yy[2],color=color,marker="D",s=45,edgecolors="white",linewidths=.5,zorder=3)
    ax.set_title(f"{chr(65+col)}  {label}",loc="left",weight="bold")
    ax.set(xlabel="Actual clear-output coverage (%)",xlim=(45,102),ylim=(7,34))
    if col==0:ax.set_ylabel("Error among clear outputs (%)")
    ax.grid(alpha=.12)
    labels=[]
    for k,cls in enumerate(["pd","dd"]):
        for j,rule in enumerate(RULES):
            y=k*3.8+j
            r=ri[SEEDS[0],m,rule,.2]
            values=[float(r[f"{cls}_{kind}_rate"])*100 for kind in ["clear_correct","deferred","clear_wrong"]]
            left=0
            for value,color in zip(values,outcolors):
                bx.barh(y,value,left=left,color=color,height=.65)
                if value>=5:
                    bx.text(left+value/2,y,f"{value:.0f}",ha="center",va="center",fontsize=7.5,color="white" if color!=outcolors[1] else "#192F3B")
                left+=value
            labels.append(f"{cls.upper()} · {['Margin','Raw','Cal.'][j]}")
    bx.set_yticks([0,1,2,3.8,4.8,5.8],labels,fontsize=8)
    bx.set_ylim(6.5,-.7);bx.set_xlim(0,100);bx.set_xticks([0,25,50,75,100])
    bx.set_xlabel("All members of the reference class (%)")
    bx.set_title(f"{chr(68+col)}  Actual outputs at the 20% target",loc="left",weight="bold")
    bx.text(.0,-.24,"PD: n = 276; DD: n = 114",transform=bx.transAxes,fontsize=8,color=RC[0])
fig.legend(handles=[Line2D([0],[0],color=c,marker="o",label=l) for c,l in zip(RC,RL)],
           loc="lower center",ncol=3,frameon=False,bbox_to_anchor=(.5,.052))
fig.legend(handles=[Patch(facecolor=c,label=l) for c,l in zip(outcolors,["Clear correct","Deferred","Clear wrong"])],
           loc="lower center",ncol=3,frameon=False,bbox_to_anchor=(.5,.009))
fig.tight_layout(rect=(0,.15,1,1),h_pad=2.1,w_pad=2)
save(fig,"Figure_R11_rejection_consequences")

# R12: probability-error contrast and remaining calibration spread.
fig,(ax,bx)=plt.subplots(1,2,figsize=(10.7,5.2),gridspec_kw={"width_ratios":[1.25,1]})
for j,(m,color,label) in enumerate(zip(METHODS,MC,LABELS)):
    yy=np.arange(6)+(j-1)*.2
    delta=[float(mi[s,m,"calibrated"]["brier"])-float(mi[s,m,"raw"]["brier"]) for s in SEEDS]
    ax.scatter(delta,yy,color=color,s=27,label=label,zorder=3)
    r=next(r for r in ci if r["method"]==m and r["comparison"]=="calibrated_minus_raw" and r["metric"]=="brier")
    ax.errorbar(delta[0],yy[0],xerr=[[delta[0]-float(r["lower_95"])],[float(r["upper_95"])-delta[0]]],color=color,capsize=3,lw=1)
ax.axvline(0,color=RC[0],ls="--",lw=1);ax.axhspan(-.5,.5,color="#EDF2F6",zorder=0)
ax.set_yticks(range(6),["Original","Repeat 1","Repeat 2","Repeat 3","Repeat 4","Repeat 5"])
ax.set_ylim(5.5,-.55);ax.set_xlim(-.055,.005)
ax.set_xlabel("Brier score: calibrated − raw");ax.set_title("A  Probability-error changes",loc="left",weight="bold")
ax.legend(loc="lower left",bbox_to_anchor=(0,-.28),frameon=False,fontsize=8)
ax.grid(axis="x",alpha=.12)
for j,(version,color,label) in enumerate([("raw",RC[1],"Raw"),("calibrated",RC[2],"Calibrated")]):
    for k,m in enumerate(METHODS):
        r=next(r for r in dc if r["method"]==m and r["version"]==version and r["metric"]=="diagnostic_slope")
        value=float(r["estimate"]);lo=float(r["lower_95"]);hi=float(r["upper_95"])
        bx.errorbar(value,k+(j-.5)*.23,xerr=[[value-lo],[hi-value]],marker="o",markersize=4,color=color,capsize=3,
                    label=label if k==0 else None)
bx.axvline(1,color=RC[0],ls="--",lw=1)
bx.set_yticks(range(3),["Adaptive","Fixed","Fixed + stop"]);bx.set_ylim(2.55,-.55)
bx.set(xlabel="Diagnostic calibration slope (ideal = 1)",xlim=(0,2.8))
bx.set_title("B  Residual calibration after updating",loc="left",weight="bold")
bx.legend(loc="lower left",bbox_to_anchor=(0,-.2),frameon=False,ncol=2,fontsize=8)
bx.grid(axis="x",alpha=.12)
fig.text(.045,.008,"Intervals: 5,000 paired participant bootstrap draws, original partition only; conditional on fitted procedures. New-repeat points have no interval bars.",fontsize=8,color=RC[0])
fig.tight_layout(rect=(0,.1,1,1),w_pad=2)
save(fig,"Figure_R12_calibration_changes")
print("Created four figure groups in PNG, PDF and SVG.")
