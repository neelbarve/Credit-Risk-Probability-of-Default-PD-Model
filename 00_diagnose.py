"""
Diagnostic: find where the data disappears
==============================================
"""
import pandas as pd

PATH = "accepted.csv"   # <-- update path

# ---- 1. Peek at raw lines - is line 0 a real CSV header or junk text? ----
print("=== First 3 raw lines ===")
with open(PATH) as f:
    for i, line in zip(range(3), f):
        print(i, line[:150])

# ---- 2. Load with no filtering at all, check shape ----
raw = pd.read_csv(PATH, nrows=5)
print("\n=== Columns pandas sees (first 5 rows) ===")
print(raw.columns.tolist()[:10], "...")
print(f"n_columns detected: {len(raw.columns)}")

# If n_columns == 1 and it's one giant string -> you have the junk-header-line problem.
# Fix: pd.read_csv(PATH, skiprows=1, ...)

# ---- 3. Check loan_status values exactly as they appear ----
try:
    status_col = pd.read_csv(PATH, usecols=["loan_status"])["loan_status"]
    print("\n=== loan_status raw value_counts ===")
    print(status_col.value_counts(dropna=False))
except ValueError as e:
    print(f"\nCouldn't read loan_status directly - column name mismatch: {e}")
    print("This confirms the header-row problem above (try skiprows=1).")
