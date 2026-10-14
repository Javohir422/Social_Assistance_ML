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


