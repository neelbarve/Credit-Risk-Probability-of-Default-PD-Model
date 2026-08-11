"""
07: Reject data preprocessing - align to accepted schema
=============================================================
Input : rejected loans CSV (9 raw columns)
Output: rejected_aligned.csv - renamed/reformatted to match the accepted
        pipeline's column names and formats, ready for the later reject-
        inference step (scoring + combining with accepted data).

This mirrors 01_preprocessing_pipeline.py's conventions so downstream code
can treat both files consistently. No target/label exists here - rejected
applicants were never funded, so there's no observed outcome. That's the
whole point of reject inference: outcomes get *inferred* in a later step,
not read from this file.
"""
import pandas as pd
import numpy as np

REJECTED_CSV_PATH = "C:/Neel2025/Rutgers_NB/Projects/modelrisk_p1/rejected.csv"   # <-- update path

RENAME_MAP = {
    "Amount_Requested": "loan_amnt",
    "Application_Date": "application_date",   # NOT the same concept as accepted's issue_d -
                                                # this is true application timing; issue_d is
                                                # funding/origination timing. Kept as a distinct
                                                # column, not merged into issue_d.
    "Risk_Score": "risk_score",                # single FICO-type score; accepted side has
                                                # fico_range_low/high as a pair - reconciled
                                                # later, not force-renamed to fico_range_*
    "Debt-To-Income_Ratio": "dti",
    "Zip_Code": "zip_code",
    "State": "addr_state",
    "Employment_Length": "emp_length",
}

DROP_COLS = ["Loan_Title", "Policy_Code"]

# same ordinal mapping used in 01_preprocessing_pipeline.py - keep identical so
# emp_length means the same thing on both sides
EMP_MAP = {
    '< 1 year': 0, '1 year': 1, '2 years': 2, '3 years': 3, '4 years': 4,
    '5 years': 5, '6 years': 6, '7 years': 7, '8 years': 8, '9 years': 9,
    '10+ years': 10
}

# ---------------------------------------------------------------------------
# 1. Load
# ---------------------------------------------------------------------------
df = pd.read_csv(REJECTED_CSV_PATH, low_memory=False)
print(f"Loaded: {df.shape}")
assert df.shape[0] > 0, "Loaded 0 rows - check REJECTED_CSV_PATH (see 00_diagnose.py pattern)."

# ---------------------------------------------------------------------------
# 2. Drop zero-variance / free-text columns
# ---------------------------------------------------------------------------
missing_expected = set(DROP_COLS) - set(df.columns)
assert not missing_expected, f"Expected columns not found: {missing_expected}"

if "Policy_Code" in df.columns:
    print("Policy_Code unique values:", df["Policy_Code"].unique())
    # confirmed zero-variance per earlier check - safe to drop unconditionally
df = df.drop(columns=DROP_COLS)

# ---------------------------------------------------------------------------
# 3. Rename to match accepted schema
# ---------------------------------------------------------------------------
missing_rename_cols = set(RENAME_MAP.keys()) - set(df.columns)
assert not missing_rename_cols, f"Expected columns not found: {missing_rename_cols}"
df = df.rename(columns=RENAME_MAP)

# ---------------------------------------------------------------------------
# 4. Reformat dti: "10%" (string) -> 10.0 (float), same scale as accepted's dti
# ---------------------------------------------------------------------------
df["dti"] = (
    df["dti"].astype(str).str.replace("%", "", regex=False).str.strip()
)
df["dti"] = pd.to_numeric(df["dti"], errors="coerce")
n_bad_dti = df["dti"].isna().sum()
print(f"dti: {n_bad_dti} values failed to parse to numeric ({n_bad_dti / len(df):.2%})")

# ---------------------------------------------------------------------------
# 5. Reformat zip_code: keep 3-digit prefix only, matching 01's
#    df['zip_code'].str.extract(r'(\d{3})') convention exactly
# ---------------------------------------------------------------------------
df["zip_code"] = df["zip_code"].astype(str).str.extract(r"(\d{3})")
n_bad_zip = df["zip_code"].isna().sum()
print(f"zip_code: {n_bad_zip} values failed to parse ({n_bad_zip / len(df):.2%})")

# ---------------------------------------------------------------------------
# 6. emp_length: same ordinal mapping as accepted
# ---------------------------------------------------------------------------
unmapped = set(df["emp_length"].dropna().unique()) - set(EMP_MAP.keys())
if unmapped:
    print(f"WARNING: emp_length values not in EMP_MAP (will become NaN): {unmapped}")
df["emp_length"] = df["emp_length"].map(EMP_MAP)

# ---------------------------------------------------------------------------
# 7. risk_score: sanity range check (FICO is 300-850)
# ---------------------------------------------------------------------------
df["risk_score"] = pd.to_numeric(df["risk_score"], errors="coerce")
out_of_range = ((df["risk_score"] < 300) | (df["risk_score"] > 850)).sum()
print(f"risk_score: {out_of_range} values outside plausible FICO range 300-850 "
      f"({out_of_range / len(df):.2%}) - inspect if this is more than a rounding edge case")

# ---------------------------------------------------------------------------
# 8. application_date: parse, keep separate from accepted's issue_d
# ---------------------------------------------------------------------------
df["application_date"] = pd.to_datetime(df["application_date"], errors="coerce")
n_bad_date = df["application_date"].isna().sum()
print(f"application_date: {n_bad_date} values failed to parse ({n_bad_date / len(df):.2%})")
if n_bad_date > len(df) * 0.05:
    print("WARNING: >5% unparsed dates - check the raw date format and adjust "
          "pd.to_datetime(..., format=...) explicitly, similar to the issue_d fix "
          "in 01_preprocessing_pipeline.py.")

# ---------------------------------------------------------------------------
# 9. Flag source + save
# ---------------------------------------------------------------------------
df["source"] = "rejected"   # useful once combined with accepted data later

print(f"\nFinal shape: {df.shape}")
print(df.dtypes)
print("\nMissing value % by column:")
print(df.isnull().mean().sort_values(ascending=False))

df.to_csv("rejected_aligned.csv", index=False)
print("\nSaved: rejected_aligned.csv")
print("\nNEXT STEP: compare zip_code / dti / emp_length distributions against "
      "train.csv (accepted) before combining, to confirm formats truly line up - "
      "e.g. pd.read_csv('train.csv')['zip_code'].head() vs "
      "pd.read_csv('rejected_aligned.csv')['zip_code'].head()")
