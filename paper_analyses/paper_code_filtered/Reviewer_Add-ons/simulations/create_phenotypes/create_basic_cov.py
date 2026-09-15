#!/usr/bin/env python3

import pandas as pd
from pathlib import Path

# =========================
# Input files
# =========================
covariates_file = Path("/path/to/your/project//UKBB/phenotypes/covariates.txt")
bmi_file = Path("/path/to/your/project//UKBB/phenotypes/bmi")

# =========================
# Output file
# =========================
out_file = Path("/path/to/your/project//UKBB/phenotypes/covariates_age_sex_bmi.txt")

# =========================
# Columns to keep from covariates file
# =========================
age_col = "age_when_attended_assessment_centre_f21003_0_0"
sex_col = "genetic_sex_f22001_0_0"   # use genetic sex

# =========================
# Read files
# =========================
cov = pd.read_csv(covariates_file, sep=r"\s+", usecols=["FID", "IID", age_col, sex_col])
bmi = pd.read_csv(bmi_file, sep=r"\s+")

# =========================
# Rename columns
# =========================
cov = cov.rename(columns={
    age_col: "age",
    sex_col: "sex"
})

cov["sex"] = (
    cov["sex"]
    .astype(str)
    .str.strip()
    .str.lower()
    .map({
        "female": 0,
        "male": 1
    })
)

# =========================
# Merge age, sex, BMI
# =========================
cov_matrix = cov.merge(
    bmi[["FID", "IID", "bmi"]],
    on=["FID", "IID"],
    how="inner"
)

# =========================
# Optional: remove missing values
# =========================
cov_matrix = cov_matrix.dropna(subset=["age", "sex", "bmi"])

# =========================
# Save covariate matrix
# =========================
cov_matrix.to_csv(out_file, sep="\t", index=False)

print(f"Saved covariate matrix to: {out_file}")
print(cov_matrix.head())
print(f"Number of individuals: {len(cov_matrix)}")