"""
Step 3-5: Missing values -> Imputation -> Encoding
=====================================================
Run after 02_eda.py.

  3. Drop columns with >=35% missing (threshold computed on TRAIN, applied to both)
  4. Impute: categorical -> mode | skewed numeric (|skew|>1) -> median | other numeric -> mean
  5. Label-encode categoricals (fit on train; unseen test categories -> 'Unknown')

Outputs: train_imputed.csv, test_imputed.csv, label_encoders.pkl, dropped_cols.txt
"""
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

MISSING_THRESHOLD = 0.35

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
print(f"Before drop: train {train_df.shape}, test {test_df.shape}")

# ---- Step 3: drop high-missing columns ----
missing_pct = train_df.isnull().mean()
cols_to_drop = [c for c in missing_pct[missing_pct >= MISSING_THRESHOLD].index if c != "target"]
print(f"Dropping {len(cols_to_drop)} columns (>= {MISSING_THRESHOLD:.0%} missing): {cols_to_drop}")

train_df = train_df.drop(columns=cols_to_drop)
test_df = test_df.drop(columns=cols_to_drop)
with open("dropped_cols.txt", "w") as f:
    f.write("\n".join(cols_to_drop))

# ---- Step 4: impute ----
cat_cols = train_df.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = [c for c in train_df.select_dtypes(include=[np.number]).columns if c != "target"]

impute_values = {}
for col in cat_cols:
    mode_val = train_df[col].mode(dropna=True)
    impute_values[col] = mode_val.iloc[0] if len(mode_val) else "Unknown"
for col in num_cols:
    skew = train_df[col].skew()
    impute_values[col] = train_df[col].median() if abs(skew) > 1 else train_df[col].mean()

for col, val in impute_values.items():
    train_df[col] = train_df[col].fillna(val)
    test_df[col] = test_df[col].fillna(val)  # always use TRAIN statistic, never test's own

remaining_na = train_df.drop(columns=["target"]).isnull().sum().sum()
assert remaining_na == 0, f"Still {remaining_na} missing values in train after imputation"
print("CHECK PASSED: no missing values remain in train (excl. target)")

# ---- Step 5: encode categoricals ----
label_encoders = {}
for col in cat_cols:
    train_df[col] = train_df[col].astype(str)
    test_df[col] = test_df[col].astype(str)

    le = LabelEncoder()
    le.fit(list(train_df[col].unique()) + ["Unknown"])
    train_df[col] = le.transform(train_df[col])
    test_df[col] = test_df[col].apply(lambda x: x if x in le.classes_ else "Unknown")
    test_df[col] = le.transform(test_df[col])
    label_encoders[col] = le

with open("label_encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)

print(f"\nAfter processing: train {train_df.shape}, test {test_df.shape}")
assert list(train_df.columns) == list(test_df.columns), "Train/test columns mismatch!"
print("CHECK PASSED: train/test columns match")

train_df.to_csv("train_imputed.csv", index=False)
test_df.to_csv("test_imputed.csv", index=False)
print("Saved: train_imputed.csv, test_imputed.csv, label_encoders.pkl, dropped_cols.txt")
