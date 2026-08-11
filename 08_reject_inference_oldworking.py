"""
08: Reject Inference
========================
Run after 06_report_pd.py (needs kfold_results.csv, train_imputed.csv,
test_imputed.csv, label_encoders.pkl) and 07_reject_preprocessing.py
(needs rejected_aligned.csv).

WHAT THIS DOES
--------------
1. Trains a REDUCED-feature model on accepted data, using only the 6 features
   shared with the rejected file (loan_amnt, dti, zip_code, addr_state,
   emp_length, risk_score) - this is the "accepted-only baseline".
2. Scores every rejected applicant with that model -> P(bad) / P(good).
3. Fuzzy/parceling augmentation: each rejected applicant becomes TWO weighted
   rows - one labeled good (weight = P(good)), one labeled bad
   (weight = P(bad)) - rather than forcing one hard guess per applicant.
4. Retrains the reduced model on accepted + augmented-rejected (weighted).
5. Compares baseline vs. augmented model on the SAME accepted test set
   (reduced features), to check whether augmentation destabilizes what the
   model already knows to be true on labeled data.

CAVEAT (read before trusting the output): reject inference does not recover
real outcomes for rejected applicants - nobody will ever know if they'd have
defaulted. This is a documented bias-correction heuristic, not new ground
truth. Treat any resulting shift in the risk model as a hypothesis to monitor
in production, not a validated improvement - there is no labeled data to
validate it against.

Outputs:
  - rejected_scored.csv          (rejected applicants + P(good)/P(bad))
  - reduced_model_baseline.pkl
  - reduced_model_augmented.pkl
  - reject_inference_comparison.txt
"""
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

RANDOM_STATE = 42
HARD_CUTOFF = None   # e.g. 0.5 to also compute the hard-cutoff alternative; None = skip it

REDUCED_FEATURES = ["loan_amnt", "dti", "zip_code", "addr_state", "emp_length", "risk_score"]


def ks_stat(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(tpr - fpr))


def build_model(name, n_estimators=300):
    if name == "XGBoost":
        return xgb.XGBClassifier(n_estimators=n_estimators, max_depth=4,
                                  learning_rate=0.05, eval_metric="auc", n_jobs=-1)
    elif name == "LightGBM":
        return lgb.LGBMClassifier(n_estimators=n_estimators, max_depth=4,
                                   learning_rate=0.05, n_jobs=-1, verbosity=-1)
    else:
        return CatBoostClassifier(iterations=n_estimators, depth=4, learning_rate=0.05,
                                   eval_metric="AUC", verbose=False, allow_writing_files=False)


# ---------------------------------------------------------------------------
# 1. Load accepted data, pick model type (reuse the winner from 04's CV)
# ---------------------------------------------------------------------------
cv_results = pd.read_csv("kfold_results.csv")
model_name = cv_results.groupby("model")["AUC"].mean().idxmax()
print(f"Using model type: {model_name} (reused from 04's best-by-CV-AUC)")

train_df = pd.read_csv("train_imputed.csv")
test_df = pd.read_csv("test_imputed.csv")

for df_ in (train_df, test_df):
    assert {"fico_range_low", "fico_range_high"}.issubset(df_.columns), (
        "fico_range_low/high not found - they may have been dropped in 03 for high "
        "missingness. risk_score needs both; adjust MISSING_THRESHOLD in 03 or "
        "compute risk_score differently if this fires."
    )
    df_["risk_score"] = (df_["fico_range_low"] + df_["fico_range_high"]) / 2

X_train_acc = train_df[REDUCED_FEATURES]
y_train_acc = train_df["target"]
X_test_acc = test_df[REDUCED_FEATURES]
y_test_acc = test_df["target"]

# ---------------------------------------------------------------------------
# 2. Baseline: reduced model trained on accepted data ONLY
# ---------------------------------------------------------------------------
baseline_model = build_model(model_name)
baseline_model.fit(X_train_acc, y_train_acc)

baseline_test_pred = baseline_model.predict_proba(X_test_acc)[:, 1]
baseline_auc = roc_auc_score(y_test_acc, baseline_test_pred)
baseline_ks = ks_stat(y_test_acc, baseline_test_pred)
print(f"\nBaseline (accepted-only, reduced features) test AUC: {baseline_auc:.4f}  KS: {baseline_ks:.4f}")

with open("reduced_model_baseline.pkl", "wb") as f:
    pickle.dump(baseline_model, f)

# ---------------------------------------------------------------------------
# 3. Prepare rejected data: encode categoricals with the SAME encoders fit on
#    accepted data (03's label_encoders.pkl), impute missing values with
#    accepted-derived statistics (never rejected's own - keeps this leakage-safe
#    the same way test-set handling was in 03/06)
# ---------------------------------------------------------------------------
rejected_df = pd.read_csv("rejected_aligned.csv")
print(f"\nRejected applicants loaded: {rejected_df.shape}")

with open("label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

for col in ["zip_code", "emp_length"]:
    if col in label_encoders:
        print(f"NOTE: '{col}' unexpectedly found in label_encoders.pkl - your pipeline "
              f"may differ from the assumption below; re-check before trusting this fix.")

# addr_state (e.g. "PA") survived the 01->03 CSV round-trip as genuine text and was
# properly LabelEncoder'd in 03 - reuse that same encoder here
le = label_encoders["addr_state"]
rejected_df["addr_state"] = rejected_df["addr_state"].astype(str)
rejected_df["addr_state"] = rejected_df["addr_state"].apply(lambda x: x if x in le.classes_ else "Unknown")
rejected_df["addr_state"] = le.transform(rejected_df["addr_state"])

# zip_code and emp_length are already pure numeric on BOTH sides (3-digit zip prefix,
# 0-10 ordinal emp_length) - they were auto-inferred as int64 on the accepted side
# during 03's CSV read, so no encoder exists or is needed; just cast to match
rejected_df["zip_code"] = pd.to_numeric(rejected_df["zip_code"], errors="coerce")
rejected_df["emp_length"] = pd.to_numeric(rejected_df["emp_length"], errors="coerce")

# impute remaining numeric columns using ACCEPTED train statistics only
for col in ["loan_amnt", "dti", "risk_score", "zip_code", "emp_length"]:
    fill_val = X_train_acc[col].median() if abs(X_train_acc[col].skew()) > 1 else X_train_acc[col].mean()
    n_missing = rejected_df[col].isna().sum()
    if n_missing:
        print(f"Imputing {n_missing} missing '{col}' values in rejected data with {fill_val:.3f} (accepted stat)")
    rejected_df[col] = rejected_df[col].fillna(fill_val)

X_rejected = rejected_df[REDUCED_FEATURES]

# ---------------------------------------------------------------------------
# 4. Score rejected applicants
# ---------------------------------------------------------------------------
p_bad = baseline_model.predict_proba(X_rejected)[:, 1]
p_good = 1 - p_bad
rejected_df["P_bad"] = p_bad
rejected_df["P_good"] = p_good
rejected_df.to_csv("rejected_scored.csv", index=False)
print(f"\nScored rejected applicants. Mean P(bad): {p_bad.mean():.3f}  "
      f"(compare to accepted train's observed default rate: {y_train_acc.mean():.3f} - "
      f"rejected applicants scoring meaningfully higher is the expected/sane direction)")

# ---------------------------------------------------------------------------
# 5. Fuzzy / parceling augmentation: each reject -> two weighted rows
# ---------------------------------------------------------------------------
good_rows = X_rejected.copy()
good_rows["target"] = 0
good_rows["weight"] = p_good

bad_rows = X_rejected.copy()
bad_rows["target"] = 1
bad_rows["weight"] = p_bad

augmented_rejected = pd.concat([good_rows, bad_rows], ignore_index=True)

accepted_weighted = X_train_acc.copy()
accepted_weighted["target"] = y_train_acc.values
accepted_weighted["weight"] = 1.0

combined_df = pd.concat([accepted_weighted, augmented_rejected], ignore_index=True)
print(f"\nCombined training set: {len(accepted_weighted)} accepted rows + "
      f"{len(augmented_rejected)} augmented-reject rows = {len(combined_df)} total")

# ---------------------------------------------------------------------------
# 6. Retrain on combined, weighted data
# ---------------------------------------------------------------------------
X_combined = combined_df[REDUCED_FEATURES]
y_combined = combined_df["target"]
w_combined = combined_df["weight"]

augmented_model = build_model(model_name)
augmented_model.fit(X_combined, y_combined, sample_weight=w_combined)

with open("reduced_model_augmented.pkl", "wb") as f:
    pickle.dump(augmented_model, f)

# ---------------------------------------------------------------------------
# 7. Compare on the SAME accepted test set - this checks for stability, not
#    "truth", since there is no labeled ground truth for rejected applicants
# ---------------------------------------------------------------------------
augmented_test_pred = augmented_model.predict_proba(X_test_acc)[:, 1]
augmented_auc = roc_auc_score(y_test_acc, augmented_test_pred)
augmented_ks = ks_stat(y_test_acc, augmented_test_pred)

comparison = (
    f"Baseline  (accepted only)      - AUC: {baseline_auc:.4f}  KS: {baseline_ks:.4f}\n"
    f"Augmented (accepted + rejects) - AUC: {augmented_auc:.4f}  KS: {augmented_ks:.4f}\n"
    f"Delta AUC: {augmented_auc - baseline_auc:+.4f}   Delta KS: {augmented_ks - baseline_ks:+.4f}\n"
)
print("\n" + comparison)

with open("reject_inference_comparison.txt", "w") as f:
    f.write(comparison)
    f.write(
        "\nNOTE: this compares performance on ACCEPTED test data only - it shows "
        "whether adding inferred reject labels destabilized the model on data we "
        "actually have outcomes for. It cannot confirm the model got any BETTER at "
        "predicting rejected applicants specifically, since no true outcome exists "
        "for them. A small AUC/KS drop here is common and not automatically bad - "
        "it can mean the model is adjusting for a population the accepted-only "
        "model never saw. A large drop is a signal the augmentation assumptions "
        "(model quality on the 6 shared features, or the fuzzy-weighting method) "
        "need review before using this model in production.\n"
    )

# optional: hard-cutoff alternative, for comparison against fuzzy augmentation
if HARD_CUTOFF is not None:
    hard_labels = (p_bad >= HARD_CUTOFF).astype(int)
    hard_df = X_rejected.copy()
    hard_df["target"] = hard_labels
    hard_combined = pd.concat([accepted_weighted.drop(columns=["weight"]), hard_df], ignore_index=True)
    hard_model = build_model(model_name)
    hard_model.fit(hard_combined[REDUCED_FEATURES], hard_combined["target"])
    hard_pred = hard_model.predict_proba(X_test_acc)[:, 1]
    hard_auc = roc_auc_score(y_test_acc, hard_pred)
    print(f"\nHard-cutoff alternative (threshold={HARD_CUTOFF}) test AUC: {hard_auc:.4f} "
          f"(compare to fuzzy augmented AUC {augmented_auc:.4f})")

print("\nSaved: rejected_scored.csv, reduced_model_baseline.pkl, "
      "reduced_model_augmented.pkl, reject_inference_comparison.txt")
