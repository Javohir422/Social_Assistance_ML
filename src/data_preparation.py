import os
os.chdir(r'C:\ML_Projects\Social_Assistance_ML')

import pandas as pd

file_path = r"C:\ML_Projects\Social_Assistance_ML\data\raw\gov_data.xlsx"
df = pd.read_excel(file_path)
print(df.head())
print(df.shape)
print(df.describe())
print(df.info())


# The main objective is to predict StatusID (APPROVED vs WAITING LIST). Before modeling, we analyze how other features relate to the target.
# A key first step is to check the balance of PovertyLevelId — its distribution and its relationship with the target.

counts = pd.crosstab(df["PovertyLevelId"], df["StatusID"])
print(counts)

percentages = pd.crosstab(df["PovertyLevelId"], df["StatusID"], normalize="index") * 100
print(percentages.round(2))


waiting_col = [col for col in df["StatusID"].dropna().unique() if "WAIT" in str(col).upper()][0]

for level in sorted(df["PovertyLevelId"].dropna().unique()):
    pct = percentages.loc[level, waiting_col]
    print(f"PovertyLevelId = {int(level)} → {pct:.2f}% Waiting")

# PovertyLevelId is a STRONG feature** — clear relationship with target
# Level 2 → highest waiting rate (10.17%), main driver of the queue (83% of all waiting)
# Level 1 → low waiting rate (3.24%) — 3x below average
# Level 3 → 0% waiting — anomaly, needs investigation

cols_to_drop = ['ProgrammeID']
df = df.drop(columns=cols_to_drop)

# ProgrammeID — state-level confidential information.
# Excluded from features due to privacy and data governance policy.
# Kept in raw data for audit only, NOT used in training/inference.

df = pd.read_excel('data/raw/gov_data.xlsx')
df.to_parquet('data/processed/raw.parquet')
print("Parquet yaratildi: data/processed/raw.parquet")

print(f"Total rows: {len(df)}")
print(f"Unique PinFL: {df['PinFL'].nunique()}")
print(f"Difference: {len(df) - df['PinFL'].nunique()}")

pinfl_counts = df['PinFL'].value_counts()
repeated = pinfl_counts[pinfl_counts > 1]
print(f"Repeated PinFL count: {len(repeated)}")

if len(repeated) > 0:
    print("\nTop 5 most repeated:")
    print(repeated.head())

# PinFL uniqueness check
# Purpose: detect duplicate persons (affects train/test split)
# Note: nunique() may fail on float64 — use value_counts() to confirm


# Analyze repeated PinFL
# Purpose: understand scope before deciding strategy
pinfl_counts = df['PinFL'].value_counts()
repeated = pinfl_counts[pinfl_counts > 1]
print(f"Total rows: {len(df)}")
print(f"Unique PinFL: {len(pinfl_counts)}")
print(f"Repeated PinFL count: {len(repeated)}")
print(f"Rows involved in repeats: {repeated.sum()}")
print(f"Percentage: {repeated.sum() / len(df) * 100:.2f}%")

# Distribution of repeat counts
print("\nRepeat count distribution:")
print(repeated.value_counts().sort_index())

# Inspect one repeated PinFL
# Purpose: check if rows are duplicates or different applications

top_pinfl = 43007786120035
rows = df[df['PinFL'] == top_pinfl]
print(rows.T)  # transpose for readability

# ============================================================
# Keep all rows — repeated PinFL is real, not duplicate
# ============================================================
# Repeated PinFL means same person in different families.
# Not a duplicate — a real life event (moved, remarried, etc.)
# Strategy: keep all rows, use GroupShuffleSplit by PinFL
# to prevent data leakage.
# ============================================================

# No deletion. Just document.
print(f"Total rows: {len(df)}")
print(f"Unique PinFL: {df['PinFL'].nunique()}")
print(f"Rows with repeated PinFL: {len(df) - df['PinFL'].nunique()}")

# Verify: same PinFL, different families
example_pinfl = 43007786120035
example = df[df['PinFL'] == example_pinfl][['PinFL', 'Ариза рақами', 'LocalityName', 'IsMainApplicant', 'AdditionDate']]
print(f"\nExample PinFL: {example_pinfl}")
print(example)

# Repeated PinFL: real (person in multiple families), not duplicates.
# Decision: keep all rows.
# Train/test split must use GroupShuffleSplit by PinFL to avoid leakage.