"""
Credit Default (PD) Model - Preprocessing Pipeline
====================================================
Input : accepted loans CSV (raw, ~160 cols)
Output: train.csv / test.csv, leakage-free, tree-model-ready (NaNs preserved
        for native handling by XGBoost / LightGBM / CatBoost)

Adjust ACCEPTED_CSV_PATH and the loan_status mapping below to match your data.
"""

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# 0. CONFIG
# ---------------------------------------------------------------------------
ACCEPTED_CSV_PATH = "accepted.csv"   # <-- update path
TEST_SPLIT_DATE   = "2018-01-01"     # loans issued on/after this date -> test set
                                       # (out-of-time split; tune based on your date range)

FINAL_FEATURES = [
    'loan_amnt','term','emp_length','home_ownership','annual_inc','verification_status',
    'purpose','zip_code','addr_state','dti','delinq_2yrs','earliest_cr_line',
    'fico_range_low','fico_range_high','inq_last_6mths','mths_since_last_delinq',
    'mths_since_last_record','open_acc','pub_rec','revol_bal','revol_util','total_acc',
    'initial_list_status','collections_12_mths_ex_med','mths_since_last_major_derog',
    'application_type','annual_inc_joint','dti_joint','verification_status_joint',
    'acc_now_delinq','tot_coll_amt','tot_cur_bal','open_acc_6m','open_act_il',
    'open_il_12m','open_il_24m','mths_since_rcnt_il','total_bal_il','il_util',
    'open_rv_12m','open_rv_24m','max_bal_bc','all_util','total_rev_hi_lim','inq_fi',
    'total_cu_tl','inq_last_12m','acc_open_past_24mths','avg_cur_bal','bc_open_to_buy',
    'bc_util','chargeoff_within_12_mths','delinq_amnt','mo_sin_old_il_acct',
    'mo_sin_old_rev_tl_op','mo_sin_rcnt_rev_tl_op','mo_sin_rcnt_tl','mort_acc',
    'mths_since_recent_bc','mths_since_recent_bc_dlq','mths_since_recent_inq',
    'mths_since_recent_revol_delinq','num_accts_ever_120_pd','num_actv_bc_tl',
    'num_actv_rev_tl','num_bc_sats','num_bc_tl','num_il_tl','num_op_rev_tl',
    'num_rev_accts','num_rev_tl_bal_gt_0','num_sats','num_tl_120dpd_2m','num_tl_30dpd',
    'num_tl_90g_dpd_24m','num_tl_op_past_12m','pct_tl_nvr_dlq','percent_bc_gt_75',
    'pub_rec_bankruptcies','tax_liens','tot_hi_cred_lim','total_bal_ex_mort',
    'total_bc_limit','total_il_high_credit_limit','revol_bal_joint',
    'sec_app_fico_range_low','sec_app_fico_range_high',
]

ID_DATE_COLS = ['issue_d']        # kept only for splitting, not used as a model feature
TARGET_COL   = 'loan_status'

DEFAULT_STATUSES     = ['Charged Off', 'Default']
NON_DEFAULT_STATUSES = ['Fully Paid']
# everything else (Current, Late, In Grace Period, Issued, etc.) is dropped -
# outcome not yet finalized

CATEGORICAL_COLS = [
    'term','emp_length','home_ownership','verification_status','purpose','addr_state',
    'initial_list_status','application_type','verification_status_joint',
]

# ---------------------------------------------------------------------------
# 1. LOAD
# ---------------------------------------------------------------------------
usecols = FINAL_FEATURES + ID_DATE_COLS + [TARGET_COL]
df = pd.read_csv(ACCEPTED_CSV_PATH, usecols=lambda c: c in usecols, low_memory=False)
print(f"Loaded: {df.shape}")

# guard rails: fail loudly here rather than silently passing an empty/malformed
# frame down the pipeline
assert df.shape[0] > 0, (
    "Loaded 0 rows. Likely cause: a junk header/disclaimer line before the real "
    "CSV header - try pd.read_csv(..., skiprows=1, ...). Run 00_diagnose.py to confirm."
)
missing_expected = set(usecols) - set(df.columns)
assert not missing_expected, (
    f"Expected columns not found in file: {missing_expected}. "
    "Column names may not match exactly (check for typos/casing) or the header "
    "row wasn't parsed correctly - run 00_diagnose.py to confirm."
)

# ---------------------------------------------------------------------------
# 2. TARGET DERIVATION
# ---------------------------------------------------------------------------
df[TARGET_COL] = df[TARGET_COL].astype(str).str.strip()  # guard against stray whitespace
df = df[df[TARGET_COL].isin(DEFAULT_STATUSES + NON_DEFAULT_STATUSES)].copy()
df['target'] = df[TARGET_COL].isin(DEFAULT_STATUSES).astype(int)
df.drop(columns=[TARGET_COL], inplace=True)

print(f"After filtering to resolved loans: {df.shape}")
assert df.shape[0] > 0, (
    "0 rows left after filtering on loan_status. Run 00_diagnose.py and check the "
    "printed value_counts - DEFAULT_STATUSES/NON_DEFAULT_STATUSES likely don't match "
    "the exact strings in your file."
)
print(df['target'].value_counts(normalize=True).rename('proportion'))

# ---------------------------------------------------------------------------
# 3. FEATURE ENGINEERING FROM RAW FIELDS
# ---------------------------------------------------------------------------
# term: "36 months" -> 36
df['term'] = df['term'].str.extract(r'(\d+)').astype(float)

# emp_length: "< 1 year".."10+ years" -> ordinal 0-10
emp_map = {
    '< 1 year': 0, '1 year': 1, '2 years': 2, '3 years': 3, '4 years': 4,
    '5 years': 5, '6 years': 6, '7 years': 7, '8 years': 8, '9 years': 9,
    '10+ years': 10
}
df['emp_length'] = df['emp_length'].map(emp_map)

# zip_code: keep 3-digit prefix only (already truncated in most LC exports, e.g. "945xx")
df['zip_code'] = df['zip_code'].astype(str).str.extract(r'(\d{3})')

# dates -> engineered numeric features, then drop raw dates
df['issue_d'] = pd.to_datetime(df['issue_d'], format='%b-%y', errors='coerce')
df['earliest_cr_line'] = pd.to_datetime(df['earliest_cr_line'], format='%b-%y', errors='coerce')

# guard rail: catch silent all-NaT parsing before it causes a 0/0 train-test split
n_bad_issue = df['issue_d'].isna().sum()
assert n_bad_issue < len(df) * 0.01, (
    f"{n_bad_issue}/{len(df)} issue_d values failed to parse - date format likely "
    f"wrong for this file. Check a raw sample, e.g. df['issue_d'] before parsing."
)
df['credit_history_months'] = (
    (df['issue_d'] - df['earliest_cr_line']).dt.days / 30.44
)
df.drop(columns=['earliest_cr_line'], inplace=True)

# ---------------------------------------------------------------------------
# 4. TIME-BASED TRAIN/TEST SPLIT (out-of-time, before any imputation/encoding
#    so nothing from the test period leaks into train statistics)
# ---------------------------------------------------------------------------
split_date = pd.Timestamp(TEST_SPLIT_DATE)
train_df = df[df['issue_d'] < split_date].copy()
test_df  = df[df['issue_d'] >= split_date].copy()
print(f"Train: {train_df.shape}  (issued before {TEST_SPLIT_DATE})")
print(f"Test : {test_df.shape}  (issued on/after {TEST_SPLIT_DATE})")

train_df.drop(columns=['issue_d'], inplace=True)
test_df.drop(columns=['issue_d'], inplace=True)

# ---------------------------------------------------------------------------
# 5. CATEGORICAL ENCODING
#    Tree models (LightGBM/CatBoost natively, XGBoost with enable_categorical=True)
#    can split directly on pandas 'category' dtype - no one-hot needed.
#    NaNs are left as-is; the model learns the optimal split direction for them.
# ---------------------------------------------------------------------------
for col in CATEGORICAL_COLS:
    if col in train_df.columns:
        train_df[col] = train_df[col].astype('category')
        test_df[col] = pd.Categorical(test_df[col], categories=train_df[col].cat.categories)

# ---------------------------------------------------------------------------
# 6. SAVE
# ---------------------------------------------------------------------------
train_df.to_csv("train.csv", index=False)
test_df.to_csv("test.csv", index=False)
print("Saved train.csv and test.csv")

# ---------------------------------------------------------------------------
# NOTES
# ---------------------------------------------------------------------------
# - NaNs deliberately NOT imputed: pass categorical columns as dtype='category'
#   with enable_categorical=True (XGBoost) or native categorical_feature list
#   (LightGBM) / cat_features (CatBoost).
# - Evaluate with AUC-ROC, KS statistic, and Gini - not raw accuracy, given
#   the class imbalance you printed above.
# - Calibrate output probabilities (Platt scaling / isotonic regression) after
#   training if you need well-calibrated PD values, not just rank-ordering.
