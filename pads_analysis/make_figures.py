"""Publication figures drawn only from the saved held-out evaluation results."""
from pathlib import Path
import json
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
OUT = BASE / "figures"
OUT.mkdir(exist_ok=True)
D = json.loads((BASE / "summary/ALL_RESULTS.json").read_text())
R = D["primary"]["methods"]
BLUE, ORANGE, GREEN, GREY = "#2364A2", "#C76A20", "#24766D", "#657582"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.labelsize": 10, "axes.titlesize": 11,
                     "pdf.fonttype": 42, "ps.fonttype": 42,
                     "savefig.facecolor": "white"})

def save(fig, name):
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(OUT / f"{name}.{ext}", dpi=350, bbox_inches="tight")
    plt.close(fig)

def ba(r):
    return 100*r["underlying_binary_balanced_accuracy"]

methods = ["adaptive", "fixed_optimized", "fixed_sequence_stop", "fixed_brier",
           "rest_only", "clinical_only", "full_protocol"]
labels = ["Adaptive selection", "Optimized fixed", "Fixed sequence + stopping",
          "Fixed (squared-error loss)", "Relaxed only", "Clinical inputs only", "Full protocol"]
colors = [BLUE, ORANGE, GREEN, GREY, GREY, GREY, GREY]

fig, (ax, txt) = plt.subplots(1, 2, figsize=(9.1, 4.25), gridspec_kw={"width_ratios": [1.15, 1], "wspace": .08})
for j, (m, color) in enumerate(zip(methods, colors)):
    r = R[m]; point = ba(r); lo, hi = np.array(r["confidence_intervals_95"]["balanced_accuracy"])*100
    ax.errorbar(point, j, xerr=[[point-lo], [hi-point]], fmt="o", color=color, capsize=3, ms=6)
    txt.text(.01, j, f"{point:.2f} ({lo:.2f}–{hi:.2f})", va="center", fontsize=9.2)
    txt.text(.62, j, f"{r['mean_seconds']:.2f}", va="center", ha="center", fontsize=9.2)
    txt.text(.89, j, f"{100*r['coverage']:.1f}", va="center", ha="center", fontsize=9.2)
for a in [ax, txt]: a.set_ylim(6.6, -.8)
ax.set_yticks(range(7), labels); ax.set_xlim(63, 80)
ax.set_xlabel("Underlying binary balanced accuracy (%)")
ax.grid(axis="x", alpha=.18); ax.set_axisbelow(True)
txt.set_xlim(0, 1.03); txt.axis("off")
for x, s, align in [(.01, "BA (95% interval)", "left"), (.62, "Mean time\n(s)", "center"), (.89, "Coverage\n(%)", "center")]:
    txt.text(x, -.77, s, ha=align, va="bottom", weight="bold", fontsize=9)
fig.subplots_adjust(top=.86, bottom=.16)
save(fig, "Figure_1_performance")

fig, axs = plt.subplots(1, 2, figsize=(9, 3.85))
caps = np.arange(2, 8)*10.24
for prefix, label, color, marker, dx in [("adaptive", "Adaptive selection", BLUE, "o", -.32),
                                         ("fixed", "Optimized fixed", ORANGE, "s", .32)]:
    keys = [f"{prefix}_budget_{b}" for b in range(2, 7)] + ["adaptive" if prefix == "adaptive" else "fixed_optimized"]
    vals = np.array([ba(R[k]) for k in keys]); ci = np.array([R[k]["confidence_intervals_95"]["balanced_accuracy"] for k in keys])*100
    axs[0].errorbar(caps+dx, vals, yerr=[vals-ci[:,0], ci[:,1]-vals], marker=marker, color=color, label=label, lw=1.5, capsize=2.5, ms=4)
    axs[1].plot(caps, [R[k]["mean_seconds"] for k in keys], marker=marker, color=color, label=label, lw=1.5, ms=4)
axs[0].set_title("A  Discrimination", loc="left", weight="bold")
axs[0].set_ylabel("Underlying binary balanced accuracy (%)")
axs[0].set_ylim(60, 82)
axs[1].set_title("B  Movement acquisition", loc="left", weight="bold")
axs[1].set_ylabel("Mean nominal acquisition time (s)")
axs[1].plot(caps, caps, ls=":", lw=1, color=GREY, label="Allowed cap")
axs[1].set_ylim(15, 77)
for ax in axs:
    ax.set_xlabel("Allowed nominal time cap (s)")
    ax.set_xticks(caps, [f"{v:.2f}" for v in caps], rotation=35)
    ax.grid(alpha=.15); ax.set_axisbelow(True)
axs[1].legend(frameon=False, loc="upper left", fontsize=8.5)
fig.tight_layout(w_pad=2)
save(fig, "Figure_2_budget")

fig, axs = plt.subplots(1, 2, figsize=(9, 3.65))
for ax, group, label in zip(axs, ["PD", "differential_diagnosis"], ["A  Parkinson disease (n = 276)", "B  Differential diagnoses (n = 114)"]):
    selected = methods[:3]
    for j, m in enumerate(selected):
        r = R[m]["actual_outputs_by_reference_group"][group]; left=0
        for cat, color, legend in [("correct", GREEN, "Correct clear output"), ("wrong", ORANGE, "Wrong clear output"), ("indeterminate", "#CFD6DD", "Indeterminate")]:
            width=100*r[cat]/r["denominator"]
            ax.barh(j, width, left=left, color=color, label=legend if j==0 else None, height=.58)
            ax.text(left+width/2, j, str(r[cat]), ha="center", va="center", color="#1E2832" if cat=="indeterminate" else "white", weight="bold", fontsize=10)
            left+=width
    ax.set_xlim(0, 100); ax.set_ylim(2.65, -.65)
    ax.set_yticks(range(3), ["Adaptive", "Optimized fixed", "Fixed + stopping"] if group=="PD" else [])
    ax.set_xlabel("All participants in the reference group (%)")
    ax.set_title(label, loc="left", fontsize=10.5, weight="bold")
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
handles, labs=axs[0].get_legend_handles_labels()
fig.legend(handles, labs, ncol=3, frameon=False, loc="lower center", fontsize=9)
fig.tight_layout(rect=(0,.14,1,1), w_pad=2)
save(fig, "Figure_3_actual_outputs")

settings = [("Primary: 71.68 s", D["primary"]["extended_primary_contrast"], 390),
            ("Conservative cap: 61.44 s", D["primary"]["conservative_budget_contrast"], 390),
            ("Without NMS questionnaire", D["without_nms"]["extended_primary_contrast"], 390),
            ("Gap-affected participants omitted", D["primary"]["gap_scoring_contrast"], 386)]
fig, ax = plt.subplots(figsize=(8,3.2))
for j, (label, result, n) in enumerate(settings):
    r=result["balanced_accuracy"]; val=100*r["difference"]; lo,hi=np.array(r["conditional_interval_95"])*100
    ax.errorbar(val,j,xerr=[[val-lo],[hi-val]],fmt="o",color=BLUE,ms=5,capsize=3)
    ax.text(5.7,j,f"{val:+.2f} ({lo:+.2f}, {hi:+.2f})",va="center",fontsize=9)
ax.axvline(0,color=GREY,lw=1,ls="--"); ax.set_xlim(-11,12.8);ax.set_ylim(3.55,-.65)
ax.set_yticks(range(4),[f"{label} (n = {n})" for label,_,n in settings])
ax.set_xticks([-10,-5,0,5]);ax.set_xlabel("Adaptive − fixed balanced accuracy (percentage points)")
ax.text(5.7,-.55,"Difference (95% interval)",weight="bold",fontsize=9,va="bottom")
ax.grid(axis="x",alpha=.15)
fig.tight_layout()
save(fig,"Figure_S1_sensitivity")

with (BASE/'summary/task_acquisition_frequencies.csv').open() as f: rows=list(csv.DictReader(f))
tasks=[r['task'] for r in rows if r['method']=='adaptive']
heat=np.array([[100*float(next(r['fraction_acquired'] for r in rows if r['method']==m and r['task']==t)) for t in tasks] for m in methods[:3]])
fig,ax=plt.subplots(figsize=(9.2,3.1))
im=ax.imshow(heat,vmin=0,vmax=100,cmap='Blues',aspect='auto')
ax.set_yticks(range(3),['Adaptive','Optimized fixed','Fixed + stopping'])
ax.set_xticks(range(11),tasks,rotation=40,ha='right',fontsize=9)
for i in range(3):
 for j in range(11):ax.text(j,i,f'{heat[i,j]:.0f}',ha='center',va='center',color='white' if heat[i,j]>55 else '#182B3A',fontsize=9)
cbar=fig.colorbar(im,ax=ax,pad=.025,shrink=.85);cbar.set_label('Acquired (%)')
ax.tick_params(length=0);fig.tight_layout()
save(fig,'Figure_S2_task_frequency')
print('Saved five figures in PNG, PDF and SVG formats.')
