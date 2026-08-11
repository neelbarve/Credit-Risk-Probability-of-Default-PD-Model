"""
Step 6-7: K-fold CV (k=2..16) x {XGBoost, LightGBM, CatBoost} + metrics
==========================================================================
Run after 03_missing_impute_encode.py.

NOTE ON METRICS: R^2, RMSE, MAE are regression metrics and aren't standard
for binary classification. Substituted with Brier score (mean squared error
between predicted probability and actual 0/1 outcome - the calibration
analog of RMSE for probabilistic classifiers; RMSE-equivalent = sqrt(Brier)).
AUC-ROC, KS, Gini, F1, Recall, Precision, Accuracy are computed directly.

COST WARNING: k=2..16 across 3 models = sum(k for k in 2..16) * 3 = 405
individual model fits. Set FAST_MODE=True first to smoke-test the pipeline
(few boosting rounds, runs in minutes), then switch to False for the real run.

Install once: pip install xgboost lightgbm catboost scikit-learn --break-system-packages

Outputs: kfold_results.csv, loss_curves.json
"""
import time
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, f1_score, recall_score, precision_score,
                              accuracy_score, brier_score_loss, roc_curve)
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

FAST_MODE = True          # True = quick smoke test, False = full run
K_RANGE = range(2, 17)    # k = 2..16 inclusive

train_df = pd.read_csv("train_imputed.csv")
X = train_df.drop(columns=["target"])
y = train_df["target"]
print(f"X: {X.shape}, y positive rate: {y.mean():.3%}")

n_estimators = 50 if FAST_MODE else 300


def ks_stat(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(tpr - fpr))


def get_models(n_estimators):
    return {
        "XGBoost": xgb.XGBClassifier(
            n_estimators=n_estimators, max_depth=5, learning_rate=0.05,
            eval_metric="auc", n_jobs=-1
        ),
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=n_estimators, max_depth=5, learning_rate=0.05,
            n_jobs=-1, verbosity=-1
        ),
        "CatBoost": CatBoostClassifier(
            iterations=n_estimators, depth=5, learning_rate=0.05,
            eval_metric="AUC", verbose=False, allow_writing_files=False
        ),
    }


results = []
loss_curves = {}  # "{k}_{model}" -> list of validation AUC per boosting round (fold 0 only)

start_all = time.time()
for k in K_RANGE:
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    oof_preds = {name: np.zeros(len(X)) for name in ["XGBoost", "LightGBM", "CatBoost"]}

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        models = get_models(n_estimators)  # fresh models every fold - no warm-start leakage
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        for name, model in models.items():
            if name == "XGBoost":
                model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
                hist = model.evals_result()["validation_0"]["auc"]
            elif name == "LightGBM":
                try:
                    model.fit(X_tr, y_tr, eval_X=X_val, eval_y=y_val,
                              eval_metric="auc", callbacks=[lgb.log_evaluation(0)])
                except TypeError:
                    # older LightGBM versions use eval_set instead of eval_X/eval_y
                    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                              eval_metric="auc", callbacks=[lgb.log_evaluation(0)])
                hist = model.evals_result_["valid_0"]["auc"]
            else:  # CatBoost
                model.fit(X_tr, y_tr, eval_set=(X_val, y_val))
                hist = model.get_evals_result()["validation"]["AUC"]

            oof_preds[name][val_idx] = model.predict_proba(X_val)[:, 1]
            if fold_idx == 0:
                loss_curves[f"{k}_{name}"] = hist

    for name in oof_preds:
        y_prob = oof_preds[name]
        y_pred = (y_prob >= 0.5).astype(int)
        auc = roc_auc_score(y, y_prob)
        results.append({
            "k": k, "model": name,
            "AUC": auc,
            "Gini": 2 * auc - 1,
            "KS": ks_stat(y, y_prob),
            "F1": f1_score(y, y_pred),
            "Recall": recall_score(y, y_pred),
            "Precision": precision_score(y, y_pred),
            "Accuracy": accuracy_score(y, y_pred),
            "Brier": brier_score_loss(y, y_prob),
        })
    print(f"k={k} done  ({time.time() - start_all:.0f}s elapsed)")

results_df = pd.DataFrame(results)
results_df.to_csv("kfold_results.csv", index=False)
with open("loss_curves.json", "w") as f:
    json.dump(loss_curves, f)

print("\nSaved: kfold_results.csv, loss_curves.json")
print("\nMean metrics by model (across all k):")
print(results_df.groupby("model")[["AUC", "KS", "Gini", "F1", "Brier"]].mean().round(4))
print("\nCHECK: AUC should be well above 0.5 (random) and below ~0.90 -- "
      "if AUC is suspiciously close to 1.0, re-check for leakage columns.")
