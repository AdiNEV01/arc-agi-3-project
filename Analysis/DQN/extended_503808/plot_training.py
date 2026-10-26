"""Static training curves; task and novelty rewards remain separate."""
import argparse
from collections import defaultdict
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--csv",type=Path,default=Path("results/dqn_extended_503808/curated_20261026/training_binned_5000.csv"))
    ap.add_argument("--out",type=Path,default=Path(__file__).parent/"figures")
    a=ap.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    groups=defaultdict(list)
    with a.csv.open(newline="") as f:
        for r in csv.DictReader(f): groups[(r["variant"],int(r["training_seed"]))].append(r)
    variants=("v0","v1","v2_history_only","v2")
    titles=("V0: baseline","V1: novelty","History only","V2: history + novelty")
    for column,name,label in [("cumulative_task_reward","training_task_reward","Cumulative level-delta task reward"),
                              ("cumulative_intrinsic_reward","training_intrinsic_reward","Cumulative intrinsic reward")]:
        fig,axs=plt.subplots(2,2,figsize=(10,6),sharex=True,sharey=True,layout="constrained")
        for ax,v,title in zip(axs.flat,variants,titles):
            for seed in (42,123,456):
                rows=groups[(v,seed)]
                ax.step([0]+[int(r["end_action"]) for r in rows],
                        [0]+[float(r[column]) for r in rows],where="post",label=f"seed {seed}",linewidth=1.5)
            ax.axvline(450000,color="0.4",linestyle="--",linewidth=.9)
            ax.axvspan(450000,503808,color="0.5",alpha=.08)
            ax.set_title(title)
            ax.grid(alpha=.2)
            ax.ticklabel_format(axis="x",style="sci",scilimits=(3,3))
        axs[0,0].legend(fontsize=8)
        fig.supxlabel("Selected training actions per model (shaded: continuation)")
        fig.supylabel(label)
        fig.suptitle("DQN training diagnostics: three independent seeds per variant",fontsize=12)
        fig.savefig(a.out/(name+".png"),dpi=180)
        fig.savefig(a.out/(name+".pdf"))
        plt.close(fig)


if __name__ == "__main__": main()
