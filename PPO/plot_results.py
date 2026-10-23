import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if not os.path.exists("evaluation_results.csv"):
    print("No evaluation_results.csv found. Run evaluate.py first.")
    exit()

df = pd.read_csv("evaluation_results.csv")
EXPS = ["baseline", "curiosity", "memory_only", "memory_curiosity"]
COLORS = {"baseline": "red", "curiosity": "blue", "memory_only": "orange", "memory_curiosity": "green"}

# --- 1. Learning Curve: Levels Completed vs Timesteps ---
plt.figure(figsize=(10, 6))
for exp in EXPS:
    sub = df[df.experiment == exp]
    if sub.empty: continue
    agg = sub.groupby("timestep")["levels"].agg(["mean", "std"]).reset_index()
    plt.plot(agg["timestep"], agg["mean"], label=exp, color=COLORS[exp])
    plt.fill_between(agg["timestep"], agg["mean"] - agg["std"], agg["mean"] + agg["std"], color=COLORS[exp], alpha=0.2)

plt.xlabel("Training Timesteps")
plt.ylabel("Mean Levels Completed (Held-out)")
plt.title("Held-out Level Completion over Training")
plt.legend()
plt.grid(True)
plt.savefig("learning_curves.png", dpi=300)
plt.close()

# --- 2. Bar Chart: Final RHAE Scores ---
final = df[df.timestep == df.timestep.max()]
plt.figure(figsize=(8, 5))
agg = final.groupby("experiment")["rhae"].agg(["mean", "std"]).reindex(EXPS).dropna()
agg["mean"].plot(kind="bar", yerr=agg["std"], color=[COLORS[e] for e in agg.index])
plt.ylabel("Mean Local RHAE (Final Checkpoint)")
plt.title("Interaction Efficiency on Held-out Games")
plt.xticks(rotation=0)
plt.grid(axis="y")
plt.savefig("rhae_scores.png", dpi=300)
plt.close()

# --- 3. LaTeX Table ---
table_rows = []
for exp in EXPS:
    sub = final[final.experiment == exp]
    if sub.empty: continue
    def ms(col): return f"{sub[col].mean():.2f} ± {sub[col].std():.2f}"
    table_rows.append({
        "Experiment": exp.replace("_", " ").title(),
        "Levels": ms("levels"),
        "Actions": ms("steps"),
        "RHAE": ms("rhae"),
        "Win rate": f"{(sub.levels > 0).mean()*100:.0f}\\%",
    })

latex = pd.DataFrame(table_rows).to_latex(index=False, escape=False, caption="Held-out evaluation across 3 seeds and 4 games (final checkpoint).", label="tab:results")
with open("results_table.tex", "w") as f: f.write(latex)
print("Wrote: learning_curves.png, rhae_scores.png, results_table.tex")