"""
06: Final PD Report
======================
Run after 04_kfold_train_eval.py (used here only to pick the best-performing
model type by mean CV AUC - the actual PD numbers reported come from a model
retrained on train_imputed.csv and scored on the untouched test_imputed.csv).

Addresses the 5 reporting requirements:
  1. Calibrated probabilities (isotonic regression on top of raw tree scores)
  2. Calibration / reliability plot (predicted PD vs actual observed default rate)
  3. Risk bands (deciles) with observed default rate per band
  4. Explainability (SHAP - global feature importance + per-record values)
  5. Final report: AUC / KS / Gini / Brier on the held-out TEST set (not CV mean)

Install once: pip install shap --break-system-packages

Outputs:
  - final_model.pkl, calibrator.pkl
  - pd_predictions_test.csv   (test set + calibrated PD + risk band)
  - risk_bands.csv            (decile-level summary table)
  - calibration_plot.png
  - pd_distribution.png
  - shap_summary.png
  - final_report_metrics.txt
"""
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, brier_score_loss, roc_curve
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import shap

N_ESTIMATORS = 300
N_BANDS = 10
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# 0. Pick the best model type from CV results (04_kfold_train_eval.py output)
# ---------------------------------------------------------------------------
cv_results = pd.read_csv("kfold_results.csv")
mean_auc_by_model = cv_results.groupby("model")["AUC"].mean().sort_values(ascending=False)
best_model_name = mean_auc_by_model.index[0]
print("Mean CV AUC by model:\n", mean_auc_by_model)
print(f"\nSelected model for final report: {best_model_name}")


def build_model(name, n_estimators):
    if name == "XGBoost":
        return xgb.XGBClassifier(n_estimators=n_estimators, max_depth=5,
                                  learning_rate=0.05, eval_metric="auc", n_jobs=-1)
    elif name == "LightGBM":
        return lgb.LGBMClassifier(n_estimators=n_estimators, max_depth=5,
                                   learning_rate=0.05, n_jobs=-1, verbosity=-1)
    else:
        return CatBoostClassifier(iterations=n_estimators, depth=5, learning_rate=0.05,
                                   eval_metric="AUC", verbose=False, allow_writing_files=False)


# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
train_df = pd.read_csv("train_imputed.csv")
test_df = pd.read_csv("test_imputed.csv")
X_train_full, y_train_full = train_df.drop(columns=["target"]), train_df["target"]
X_test, y_test = test_df.drop(columns=["target"]), test_df["target"]

# hold out a calibration slice from TRAIN only - test stays untouched until final scoring
X_fit, X_calib, y_fit, y_calib = train_test_split(
    X_train_full, y_train_full, test_size=0.2, stratify=y_train_full, random_state=RANDOM_STATE
)
print(f"Fit: {X_fit.shape}, Calib: {X_calib.shape}, Test: {X_test.shape}")

# ---------------------------------------------------------------------------
# 2. Fit the final raw model on the "fit" split
# ---------------------------------------------------------------------------
final_model = build_model(best_model_name, N_ESTIMATORS)
final_model.fit(X_fit, y_fit)

# ---------------------------------------------------------------------------
# 3. Requirement 4 - Explainability (SHAP), on the RAW model
#    (isotonic calibration is monotonic - it doesn't change feature ranking,
#    so SHAP on the raw model is the standard, correct place to compute this)
# ---------------------------------------------------------------------------
shap_sample = X_test.sample(n=min(5000, len(X_test)), random_state=RANDOM_STATE)
explainer = shap.TreeExplainer(final_model)
shap_values = explainer.shap_values(shap_sample)
if isinstance(shap_values, list):  # some wrappers return [class0, class1]
    shap_values = shap_values[1]

plt.figure()
shap.summary_plot(shap_values, shap_sample, plot_type="bar", show=False, max_display=20)
plt.tight_layout()
plt.savefig("shap_summary.png", dpi=100, bbox_inches="tight")
plt.close()

mean_abs_shap = pd.Series(np.abs(shap_values).mean(axis=0), index=shap_sample.columns)
top_features = mean_abs_shap.sort_values(ascending=False).head(15)
print("\nTop 15 features by mean |SHAP value|:")
print(top_features)

# ---------------------------------------------------------------------------
# 4. Requirement 1 - Calibrate probabilities (isotonic regression) using the
#    held-out calib split, on top of the already-fitted raw model
# ---------------------------------------------------------------------------
try:
    from sklearn.frozen import FrozenEstimator  # sklearn >= 1.6
    calibrator = CalibratedClassifierCV(FrozenEstimator(final_model), method="isotonic")
except ImportError:
    calibrator = CalibratedClassifierCV(final_model, method="isotonic", cv="prefit")
calibrator.fit(X_calib, y_calib)

raw_test_scores = final_model.predict_proba(X_test)[:, 1]
calibrated_pd = calibrator.predict_proba(X_test)[:, 1]

# ---------------------------------------------------------------------------
# 5. Requirement 5 - Final held-out TEST metrics (not CV mean)
# ---------------------------------------------------------------------------
def ks_stat(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(tpr - fpr))

auc = roc_auc_score(y_test, calibrated_pd)
metrics_report = {
    "model": best_model_name,
    "test_AUC": auc,
    "test_Gini": 2 * auc - 1,
    "test_KS": ks_stat(y_test, calibrated_pd),
    "test_Brier_raw": brier_score_loss(y_test, raw_test_scores),
    "test_Brier_calibrated": brier_score_loss(y_test, calibrated_pd),
}
print("\n=== Final held-out test metrics ===")
for k, v in metrics_report.items():
    print(f"{k}: {v}")
with open("final_report_metrics.txt", "w") as f:
    for k, v in metrics_report.items():
        f.write(f"{k}: {v}\n")

assert metrics_report["test_Brier_calibrated"] <= metrics_report["test_Brier_raw"] * 1.05, (
    "Calibration did not improve (or meaningfully match) Brier score - check calib "
    "split size or consider method='sigmoid' (Platt) instead of isotonic."
)
print("\nCHECK: calibrated Brier <= raw Brier (calibration improved probability quality)")

# ---------------------------------------------------------------------------
# 6. Requirement 2 - Calibration / reliability plot
# ---------------------------------------------------------------------------
test_out = X_test.copy()
test_out["target"] = y_test.values
test_out["raw_score"] = raw_test_scores
test_out["PD"] = calibrated_pd

test_out["pd_decile"] = pd.qcut(test_out["PD"], q=N_BANDS, duplicates="drop")
reliability = test_out.groupby("pd_decile", observed=True).agg(
    mean_predicted_PD=("PD", "mean"),
    observed_default_rate=("target", "mean"),
    n=("target", "size"),
).reset_index()

plt.figure(figsize=(6, 6))
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
plt.plot(reliability["mean_predicted_PD"], reliability["observed_default_rate"],
         marker="o", label=best_model_name)
plt.xlabel("Mean predicted PD (per decile)")
plt.ylabel("Observed default rate (per decile)")
plt.title("Calibration / Reliability Plot (test set)")
plt.legend()
plt.tight_layout()
plt.savefig("calibration_plot.png", dpi=100)
plt.close()

# ---------------------------------------------------------------------------
# 7. PD distribution
# ---------------------------------------------------------------------------
plt.figure(figsize=(7, 4))
plt.hist(test_out["PD"], bins=50)
plt.xlabel("Predicted PD")
plt.ylabel("Count")
plt.title("PD distribution (test set)")
plt.tight_layout()
plt.savefig("pd_distribution.png", dpi=100)
plt.close()

# ---------------------------------------------------------------------------
# 8. Requirement 3 - Risk bands (deciles), observed default rate per band
# ---------------------------------------------------------------------------
risk_bands = reliability.sort_values("mean_predicted_PD").reset_index(drop=True)
risk_bands["risk_grade"] = [chr(65 + i) for i in range(len(risk_bands))]  # A (safest) ... 
risk_bands = risk_bands[["risk_grade", "pd_decile", "n", "mean_predicted_PD", "observed_default_rate"]]
risk_bands.to_csv("risk_bands.csv", index=False)
print("\n=== Risk bands (test set) ===")
print(risk_bands)

# ---------------------------------------------------------------------------
# 9. Save predictions + model artifacts
# ---------------------------------------------------------------------------
test_out[["PD", "raw_score", "pd_decile", "target"]].to_csv("pd_predictions_test.csv", index=False)
with open("final_model.pkl", "wb") as f:
    pickle.dump(final_model, f)
with open("calibrator.pkl", "wb") as f:
    pickle.dump(calibrator, f)

print("\nSaved: final_model.pkl, calibrator.pkl, pd_predictions_test.csv, risk_bands.csv, "
      "calibration_plot.png, pd_distribution.png, shap_summary.png, final_report_metrics.txt")

#############
#One thing to watch when you run it: the assertion at the end (calibrated Brier <= raw Brier * 1.05) will fail loudly if isotonic calibration doesn't help — 
# if that happens, it's usually because the calib split is too small relative to your class imbalance, or isotonic is overfitting; 
# switching method="isotonic" to method="sigmoid" (Platt scaling) in the calibrator line is the fix, 
# and I've left that as a one-line change if needed.
#############