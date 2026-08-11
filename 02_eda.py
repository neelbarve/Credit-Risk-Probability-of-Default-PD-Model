"""
Step 2: EDA - distributions & data types
==========================================
Run after 01 (preprocessing_pipeline.py) has produced train.csv / test.csv.

Outputs:
  - eda_summary.csv          (dtype, missing %, unique count per column)
  - eda_numeric_hists.png    (grid of histograms, numeric columns)
  - eda_categorical_bars.png (grid of bar charts, categorical columns)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train_df = pd.read_csv("train.csv")

assert "target" in train_df.columns, "target column missing - re-run 01 pipeline first"
print(f"Rows: {len(train_df)}  |  Columns: {train_df.shape[1]}")

# ---- 1. Summary table: dtype, missing %, n_unique ----
summary = pd.DataFrame({
    "dtype": train_df.dtypes.astype(str),
    "missing_pct": train_df.isnull().mean() * 100,
    "n_unique": train_df.nunique(),
}).sort_values("missing_pct", ascending=False)
summary.to_csv("eda_summary.csv")
print(summary)
print("\nCHECK: open eda_summary.csv - confirm missing_pct values look sane "
      "before step 03 (which drops columns >=35% missing).")

# ---- 2. Numeric distributions ----
numeric_cols = [c for c in train_df.select_dtypes(include=[np.number]).columns if c != "target"]
ncols = 6
nrows = int(np.ceil(len(numeric_cols) / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 2.5))
axes = axes.flatten()
for i, col in enumerate(numeric_cols):
    train_df[col].dropna().hist(ax=axes[i], bins=30)
    axes[i].set_title(col, fontsize=8)
for j in range(i + 1, len(axes)):
    axes[j].axis("off")
plt.tight_layout()
plt.savefig("eda_numeric_hists.png", dpi=100)
plt.close()

# ---- 3. Categorical distributions ----
cat_cols = train_df.select_dtypes(include=["category", "object"]).columns.tolist()
if cat_cols:
    ncols2 = 3
    nrows2 = int(np.ceil(len(cat_cols) / ncols2))
    fig, axes = plt.subplots(nrows2, ncols2, figsize=(ncols2 * 4, nrows2 * 3))
    axes = axes.flatten()
    for i, col in enumerate(cat_cols):
        train_df[col].value_counts(dropna=False).head(15).plot(kind="bar", ax=axes[i])
        axes[i].set_title(col, fontsize=8)
        axes[i].tick_params(axis='x', labelsize=6)
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    plt.tight_layout()
    plt.savefig("eda_categorical_bars.png", dpi=100)
    plt.close()

print("\nSaved: eda_summary.csv, eda_numeric_hists.png, eda_categorical_bars.png")
