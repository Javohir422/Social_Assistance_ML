# EDA Notes — Social Assistance ML

**Project:** Predicting StatusID (APPROVED vs WAITING LIST) for social assistance applications in Uzbekistan
**Date started:** 2026-10-14
**Dataset:** data/raw/gov_data.xlsx
**Author:** [your name]

---

## 1. Dataset Overview

- **Rows:** 215,216
- **Columns:** 16
- **Unique families (`Ариза рақами`):** 51,621
- **Average family size:** ~4.17 people
- **File format:** Excel (.xlsx)

## 2. Target: StatusID

| Status | Count | % |
|---|---|---|
| APPROVED | ~199,366 | 92.64% |
| WAITING LIST | ~15,850 | 7.36% |

**Imbalance ratio:** ~1 : 12.5

**Key implication:** Accuracy is misleading — must use **PR-AUC**, **F1**, **Precision@K**. Need class weights during training.

## 3. Column Findings

| Column | Type | Notes |
|---|---|---|
| `DistrictName` | str | Region/district |
| `LocalityName` | str | Local community (MFY) |
| `Ариза рақами` | int | Family ID (51,621 unique) |
| `ProgrammeID` | int | Confidential — excluded |
| `StatusID` | str | Target: APPROVED / WAITING LIST |
| `IsMainApplicant` | int | 0/1 flag |
| `PinFL` | int | Person identifier |
| `Surname` | str | — |
| `FirstName` | str | — |
| `Patronymic` | str | — |
| `DateOfBirth` | str | Needs conversion to datetime |
| `PhoneNumber` | str | ~77% missing |
| `Document` | str | ~77% missing |
| `AdditionDate` | str | Needs conversion to datetime |
| `PovertyLevelId` | int | Values 1, 2, 3 |
| `FamilyMemberID` | int | Unique per row → person ID |

## 4. Missing Values (CRITICAL)

| Column | Missing | % |
|---|---|---|
| `PhoneNumber` | 165,387 | ~77% |
| `Document` | 165,200 | ~77% |
| `Surname` | 2 | ~0% |
| `FirstName` | 2 | ~0% |
| `Patronymic` | 9 | ~0% |

**Implication:** PhoneNumber and Document are unusable as direct features. But "has phone / has document" might be a signal.

## 5. Duplicates

- Full duplicate rows: **0**
- Duplicate `FamilyMemberID`: **0**

## 6. PovertyLevelId × StatusID

| Level | APPROVED | WAITING | TOTAL | % WAITING |
|---|---|---|---|---|
| 1 | 79,374 | 2,654 | 82,028 | 3.24% |
| 2 | 116,573 | 13,196 | 129,769 | 10.17% |
| 3 | 3,419 | 0 | 3,419 | 0.00% |

**Findings:**
- `PovertyLevelId` is a **STRONG feature** — clear relationship with target
- **Level 2** → main driver of the queue (83% of all waiting)
- **Level 1** → low waiting rate (3.24%) — 3× below average
- **Level 3** → 0% waiting — **anomaly**, needs investigation
  - Possible: privileged category, separate programme, or data issue

## 7. Excluded Features

### `ProgrammeID` — state-level confidential information

`ProgrammeID` represents internal classification of social assistance programmes and is considered **state-level confidential information**. The mapping between programme codes and their real-world meaning is not publicly available.

**Decision:** Excluded from feature set due to privacy and data governance policy. Retained in raw data for audit purposes only — **NOT used in training or inference**.

**Impact:** Minimal — the feature is near-constant (7 dominates), and its removal avoids reliance on undisclosed categories, keeping the model explainable and compliant.

## 8. Key Takeaways

1. **Target is imbalanced** (7.36%) → use PR-AUC, class weights
2. **PovertyLevelId is a top feature** — must keep
3. **ProgrammeID is confidential** — excluded
4. **PhoneNumber/Document ~77% empty** — handle carefully
5. **FamilyMemberID = person ID** → build family features via `Ариза рақами`
6. **Dates need parsing** — currently strings
7. **No duplicates** — data is clean at row level

## 9. PinFL Uniqueness Check

**Purpose:** Verify each person appears once. Repeated PinFL affects train/test split (data leakage).

**Method:** Count rows, count unique PinFL (`nunique` + `value_counts`), find repeated values.

**Finding:** `nunique()` returned 215,216, but `value_counts()` revealed repeated values (e.g. `43007786120035` × 3). Likely caused by `float64` precision loss on 14-digit integers.

**Next:** Check dtype, cast to `int64` if needed.

## 10. Repeated PinFL Analysis

**Finding:** 804 PinFL values repeat (1,609 rows, 0.75%). Inspection shows these are NOT duplicates — the same person appears in different families with different `Ариза рақами`, `LocalityName`, `IsMainApplicant`, and `AdditionDate`.

**Example:** Person `43007786120035` appears 3 times:
- Family `29586334`, `IsMainApplicant=1`, added 2026-01-18
- Family `30272010`, `IsMainApplicant=0`, added 2026-03-23
- Family `30735873`, `IsMainApplicant=0`, added 2026-04-08

**Decision:** Keep all rows. Use `GroupShuffleSplit` by `PinFL` to prevent data leakage.

**Implication:** Same person must not appear in both train and test sets.