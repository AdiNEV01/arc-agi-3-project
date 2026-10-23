import os, json, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("evaluation_results.csv")
EXPS = ["baseline", "curiosity", "memory_only", "memory_curiosity"]
COLORS = {"baseline": "red", "curiosity": "blue",
          "memory_only": "orange", "memory_curiosity": "green"}

plt.figure(figsize=(8, 5))
for exp in EXPS:
    sub = df[df.experiment == exp]
    if sub.empty: continue
    agg = sub.groupby("timestep")["levels"].agg(["mean", "std"]).reset_index()
    plt.plot(agg["timestep"], agg["mean"], label=exp, color=COLORS[exp])
    plt.fill_between(agg["timestep"],
                     agg["mean"] - agg["std"],
                     agg["mean"] + agg["std"],
                     color=COLORS[exp], alpha=0.2)
plt.xlabel("Training timesteps"); plt.ylabel("Mean levels completed (held-out)")
plt.title("Sample efficiency on held-out games"); plt.legend(); plt.grid(True)
plt.savefig("efficiency_curve.png", dpi=300); plt.close()

final = df[df.timestep == df.timestep.max()]
plt.figure(figsize=(8, 5))
agg = final.groupby("experiment")["steps"].agg(["mean", "std"]).reindex(EXPS).dropna()
agg["mean"].plot(kind="bar", yerr=agg["std"], color=[COLORS[e] for e in agg.index])
plt.ylabel("Actions used (final checkpoint)"); plt.title("Interaction efficiency")
plt.xticks(rotation=0); plt.grid(axis="y")
plt.savefig("efficiency_actions.png", dpi=300); plt.close()

table_rows = []
for exp in EXPS:
    sub = final[final.experiment == exp]
    if sub.empty: continue
    def ms(col): 
        return f"{sub[col].mean():.2f} ± {sub[col].std():.2f}"
    table_rows.append({
        "Experiment": exp.replace("_", " ").title(),
        "Levels": ms("levels"),
        "Actions": ms("steps"),
        "RHAE": ms("rhae"),
        "Win rate": f"{(sub.levels > 0).mean()*100:.0f}\\%",
    })

latex = pd.DataFrame(table_rows).to_latex(
    index=False, escape=False,
    caption="Held-out evaluation across 3 seeds and 4 games (final checkpoint).",
    label="tab:results",
)
with open("results_table.tex", "w") as f:
    f.write(latex)

per_game = final.groupby(["experiment", "game"])[["levels", "steps", "rhae"]].mean().reset_index()
per_game.to_csv("per_game_breakdown.csv", index=False)

rows = []
for exp in EXPS:
    for seed in [0, 1, 2]:
        p = f"./logs/{exp}_seed{seed}/resources.json"
        if os.path.exists(p):
            with open(p) as f: rows.append(json.load(f))
if rows:
    pd.DataFrame(rows).to_csv("resources_table.csv", index=False)
    pd.DataFrame(rows).to_latex("resources_table.tex", index=False,
                                 caption="Compute cost per run.", label="tab:resources")

print("Wrote: efficiency_curve.png, efficiency_actions.png, results_table.tex, per_game_breakdown.csv, resources_table.tex")