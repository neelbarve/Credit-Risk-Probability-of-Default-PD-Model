"""
Step 8: Plot k vs model performance + training loss curves
==============================================================
Run after 04_kfold_train_eval.py.

Outputs: metrics_vs_k.png, loss_curves.png
"""
import json
import pandas as pd
import matplotlib.pyplot as plt

results_df = pd.read_csv("kfold_results.csv")
metrics = ["AUC", "KS", "Gini", "F1", "Recall", "Precision", "Accuracy", "Brier"]
models = results_df["model"].unique()

# ---- metrics vs k ----
fig, axes = plt.subplots(2, 4, figsize=(20, 8))
axes = axes.flatten()
for i, metric in enumerate(metrics):
    for model in models:
        sub = results_df[results_df["model"] == model].sort_values("k")
        axes[i].plot(sub["k"], sub[metric], marker="o", label=model)
    axes[i].set_title(metric)
    axes[i].set_xlabel("k (folds)")
    axes[i].legend(fontsize=7)
plt.tight_layout()
plt.savefig("metrics_vs_k.png", dpi=100)
plt.close()

# ---- training loss curves (fold 0), smallest and largest k for readability ----
with open("loss_curves.json") as f:
    loss_curves = json.load(f)

k_vals = sorted(results_df["k"].unique())
show_k = [k_vals[0], k_vals[-1]]

fig, axes = plt.subplots(1, len(show_k), figsize=(6 * len(show_k), 4))
if len(show_k) == 1:
    axes = [axes]
for ax, k in zip(axes, show_k):
    for model in models:
        key = f"{k}_{model}"
        if key in loss_curves:
            ax.plot(loss_curves[key], label=model)
    ax.set_title(f"Validation AUC per boosting round (k={k}, fold 0)")
    ax.set_xlabel("Boosting round")
    ax.set_ylabel("AUC")
    ax.legend(fontsize=7)
plt.tight_layout()
plt.savefig("loss_curves.png", dpi=100)
plt.close()

print("Saved: metrics_vs_k.png, loss_curves.png")
print("\nCHECK: metrics_vs_k.png -- AUC/KS/Gini should be roughly flat across k "
      "(if k barely matters, that's expected -- k mainly affects variance of the "
      "estimate, not its central value). If performance trends strongly with k, "
      "something's off (likely too little data per fold at high k, or instability).")
