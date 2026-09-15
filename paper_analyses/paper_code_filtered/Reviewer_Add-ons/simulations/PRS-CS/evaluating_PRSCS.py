#!/usr/bin/env python3

import os
import sys
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    average_precision_score,
    log_loss,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


###############################################################################
# Paths
###############################################################################

BASE = Path(
    "/path/to/your/project/"
)

PRSCS_RESULTS_BASE = (
    BASE
    / "data"
    / "simulations"
    / "PRScs"
    / "PRScs_results"
)

PHENOTYPE_BASE = (
    BASE
    / "data"
    / "simulations"
    / "combined_DRIP_phenotypes_by_rep"
)

# This is the covariate file used in the simulated GWAS.
COVARIATE_FILE = Path(
    "/path/to/your/project//UKBB/phenotypes/"
    "cov_matrix_MinMax_scaled_no_missing.txt"
)

OUTPUT_BASE = (
    BASE
    / "data"
    / "simulations"
    / "PRScs"
    / "evaluation"
)

OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

METRICS_OUTPUT = OUTPUT_BASE / "PRSCS_metrics_summary.csv"
ERROR_OUTPUT = OUTPUT_BASE / "PRSCS_evaluation_errors.csv"


###############################################################################
# Settings
###############################################################################

TEST_SIZE = 0.30
RANDOM_STATE = 42

ID_COLUMNS = {"FID", "IID", "#FID"}


###############################################################################
# Reading functions
###############################################################################

def read_whitespace_file(path: Path) -> pd.DataFrame:
    """Read a whitespace-delimited file and standardize column names."""

    df = pd.read_csv(
        path,
        sep=r"\s+",
        low_memory=False,
    )

    df.columns = df.columns.astype(str).str.strip()

    if "#FID" in df.columns and "FID" not in df.columns:
        df = df.rename(columns={"#FID": "FID"})

    for id_col in ["FID", "IID"]:
        if id_col in df.columns:
            df[id_col] = df[id_col].astype(str)

    return df


def detect_score_column(profile_df: pd.DataFrame) -> str:
    """
    Detect the score column produced by PLINK1 or PLINK2.

    PLINK1 commonly produces:
        SCORESUM

    PLINK2 commonly produces:
        SCORE1_SUM
    """

    preferred_columns = [
        "SCORESUM",
        "SCORE1_SUM",
        "SCORE",
        "SCORE1_AVG",
    ]

    for column in preferred_columns:
        if column in profile_df.columns:
            return column

    possible_columns = [
        column
        for column in profile_df.columns
        if "SCORE" in column.upper()
        and column not in {"SCORECNT"}
    ]

    if len(possible_columns) == 1:
        return possible_columns[0]

    raise ValueError(
        "Could not identify the PRS score column. "
        f"Available columns: {profile_df.columns.tolist()}"
    )


def identify_phenotype_column(
    phenotype_df: pd.DataFrame,
    phenotype_name: str,
    rep: int,
) -> str:
    """
    Match the PRS-CS phenotype directory to a column in the combined phenotype
    file.

    It supports both:
        chr1_N100_...
    and:
        rep1_chr1_N100_...
    """

    candidates = [
        phenotype_name,
        f"rep{rep}_{phenotype_name}",
    ]

    for candidate in candidates:
        if candidate in phenotype_df.columns:
            return candidate

    # Case-insensitive fallback
    lower_to_original = {
        column.lower(): column
        for column in phenotype_df.columns
    }

    for candidate in candidates:
        if candidate.lower() in lower_to_original:
            return lower_to_original[candidate.lower()]

    raise ValueError(
        f"No phenotype column matched '{phenotype_name}'. "
        f"Tried: {candidates}"
    )


###############################################################################
# Data preparation
###############################################################################

def load_prs(profile_path: Path) -> pd.DataFrame:
    """Read one chromosome-1 PLINK profile file."""

    profile_df = read_whitespace_file(profile_path)

    if "FID" not in profile_df.columns or "IID" not in profile_df.columns:
        raise ValueError(
            f"FID or IID is missing from {profile_path}"
        )

    score_column = detect_score_column(profile_df)

    prs_df = profile_df[
        ["FID", "IID", score_column]
    ].copy()

    prs_df = prs_df.rename(
        columns={score_column: "PRS"}
    )

    prs_df["PRS"] = pd.to_numeric(
        prs_df["PRS"],
        errors="coerce",
    )

    prs_df = prs_df.drop_duplicates(
        subset=["FID", "IID"],
        keep="first",
    )

    return prs_df


def prepare_covariates(covariate_df: pd.DataFrame) -> pd.DataFrame:
    """Keep IDs and numeric covariates."""

    covariate_df = covariate_df.copy()

    if "FID" not in covariate_df.columns:
        if "IID" in covariate_df.columns:
            covariate_df.insert(
                0,
                "FID",
                covariate_df["IID"],
            )
        else:
            raise ValueError(
                "The covariate file contains neither FID nor IID."
            )

    if "IID" not in covariate_df.columns:
        covariate_df.insert(
            1,
            "IID",
            covariate_df["FID"],
        )

    covariate_columns = [
        column
        for column in covariate_df.columns
        if column not in ID_COLUMNS
    ]

    for column in covariate_columns:
        covariate_df[column] = pd.to_numeric(
            covariate_df[column],
            errors="coerce",
        )

    # Remove completely empty or constant columns
    valid_covariate_columns = []

    for column in covariate_columns:
        if covariate_df[column].notna().sum() == 0:
            continue

        if covariate_df[column].nunique(dropna=True) <= 1:
            continue

        valid_covariate_columns.append(column)

    return covariate_df[
        ["FID", "IID"] + valid_covariate_columns
    ].copy()


###############################################################################
# Model evaluation
###############################################################################

def make_pipeline(binary: bool):
    """Create a standardized regression model."""

    if binary:
        model = LogisticRegression(
            random_state=RANDOM_STATE,
            max_iter=2000,
            solver="lbfgs",
        )
    else:
        model = LinearRegression()

    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "model",
                model,
            ),
        ]
    )


def evaluate_binary_model(
    data: pd.DataFrame,
    feature_columns: List[str],
    phenotype_column: str,
    phenotype_name: str,
    rep: int,
    model_name: str,
) -> Dict:
    """Evaluate one binary phenotype model."""

    model_data = data[
        ["FID", "IID", phenotype_column] + feature_columns
    ].copy()

    model_data[phenotype_column] = pd.to_numeric(
        model_data[phenotype_column],
        errors="coerce",
    )

    model_data = model_data.dropna(
        subset=[phenotype_column]
    )

    # Simulated labels are 1/2, so convert:
    # 1 -> 0
    # 2 -> 1
    unique_values = sorted(
        model_data[phenotype_column]
        .dropna()
        .unique()
        .tolist()
    )

    if set(unique_values).issubset({1, 2}):
        y = (
            model_data[phenotype_column] > 1
        ).astype(int)

    elif set(unique_values).issubset({0, 1}):
        y = model_data[phenotype_column].astype(int)

    else:
        raise ValueError(
            f"Unexpected binary labels: {unique_values}"
        )

    X = model_data[feature_columns].copy()

    for column in feature_columns:
        X[column] = pd.to_numeric(
            X[column],
            errors="coerce",
        )

    X = X.replace([np.inf, -np.inf], np.nan)

    if y.nunique() != 2:
        raise ValueError(
            f"Phenotype contains only {y.nunique()} class."
        )

    class_counts = y.value_counts()

    if class_counts.min() < 2:
        raise ValueError(
            f"Too few observations in one class: "
            f"{class_counts.to_dict()}"
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    pipeline = make_pipeline(binary=True)
    pipeline.fit(X_train, y_train)

    predicted_probability = pipeline.predict_proba(
        X_test
    )[:, 1]

    return {
        "rep": rep,
        "phenotype": phenotype_name,
        "phenotype_column": phenotype_column,
        "phenotype_type": "binary",
        "model": model_name,
        "n_total": len(model_data),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_cases_total": int(y.sum()),
        "n_controls_total": int((y == 0).sum()),
        "n_features": len(feature_columns),
        "roc_auc": roc_auc_score(
            y_test,
            predicted_probability,
        ),
        "log_loss": log_loss(
            y_test,
            predicted_probability,
            labels=[0, 1],
        ),
        "average_precision": average_precision_score(
            y_test,
            predicted_probability,
        ),
        "r2": np.nan,
        "mse": np.nan,
    }


def evaluate_continuous_model(
    data: pd.DataFrame,
    feature_columns: List[str],
    phenotype_column: str,
    phenotype_name: str,
    rep: int,
    model_name: str,
) -> dict:
    """Evaluate one continuous phenotype model."""

    model_data = data[
        ["FID", "IID", phenotype_column] + feature_columns
    ].copy()

    model_data[phenotype_column] = pd.to_numeric(
        model_data[phenotype_column],
        errors="coerce",
    )

    model_data = model_data.dropna(
        subset=[phenotype_column]
    )

    y = model_data[phenotype_column]
    X = model_data[feature_columns].copy()

    for column in feature_columns:
        X[column] = pd.to_numeric(
            X[column],
            errors="coerce",
        )

    X = X.replace([np.inf, -np.inf], np.nan)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    pipeline = make_pipeline(binary=False)
    pipeline.fit(X_train, y_train)

    predicted_value = pipeline.predict(X_test)

    return {
        "rep": rep,
        "phenotype": phenotype_name,
        "phenotype_column": phenotype_column,
        "phenotype_type": "continuous",
        "model": model_name,
        "n_total": len(model_data),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_cases_total": np.nan,
        "n_controls_total": np.nan,
        "n_features": len(feature_columns),
        "roc_auc": np.nan,
        "log_loss": np.nan,
        "average_precision": np.nan,
        "r2": r2_score(
            y_test,
            predicted_value,
        ),
        "mse": mean_squared_error(
            y_test,
            predicted_value,
        ),
    }


def calculate_raw_prs_auc(
    merged_data: pd.DataFrame,
    phenotype_column: str,
) -> float:
    """
    Calculate AUC directly from PRS without fitting a logistic regression.

    Because PRS direction can occasionally be reversed by allele orientation,
    both AUC and 1-AUC are returned separately by the main script.
    """

    data = merged_data[
        [phenotype_column, "PRS"]
    ].copy()

    data[phenotype_column] = pd.to_numeric(
        data[phenotype_column],
        errors="coerce",
    )

    data["PRS"] = pd.to_numeric(
        data["PRS"],
        errors="coerce",
    )

    data = data.replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()

    unique_values = set(
        data[phenotype_column].unique()
    )

    if unique_values.issubset({1, 2}):
        y = (
            data[phenotype_column] > 1
        ).astype(int)

    elif unique_values.issubset({0, 1}):
        y = data[phenotype_column].astype(int)

    else:
        return np.nan

    if y.nunique() != 2:
        return np.nan

    return roc_auc_score(y, data["PRS"])


###############################################################################
# Evaluate one phenotype
###############################################################################

def evaluate_phenotype(
    rep: int,
    phenotype_name: str,
    profile_path: Path,
    phenotype_df: pd.DataFrame,
    covariate_df: pd.DataFrame,
) -> Tuple[List[Dict], pd.DataFrame]:

    phenotype_column = identify_phenotype_column(
        phenotype_df=phenotype_df,
        phenotype_name=phenotype_name,
        rep=rep,
    )

    prs_df = load_prs(profile_path)

    phenotype_subset = phenotype_df[
        ["FID", "IID", phenotype_column]
    ].copy()

    merged_data = pd.merge(
        prs_df,
        phenotype_subset,
        on=["FID", "IID"],
        how="inner",
        validate="one_to_one",
    )

    merged_data = pd.merge(
        merged_data,
        covariate_df,
        on=["FID", "IID"],
        how="left",
        validate="one_to_one",
    )

    if merged_data.empty:
        raise ValueError(
            "No individuals remained after merging PRS, phenotype, "
            "and covariates."
        )

    covariate_columns = [
        column
        for column in covariate_df.columns
        if column not in ID_COLUMNS
        and column in merged_data.columns
    ]

    phenotype_values = pd.to_numeric(
        merged_data[phenotype_column],
        errors="coerce",
    ).dropna()

    unique_values = set(
        phenotype_values.unique()
    )

    is_binary = (
        unique_values.issubset({0, 1})
        or unique_values.issubset({1, 2})
    )

    model_definitions = {
        "PRS_only": ["PRS"],
        "covariates_only": covariate_columns,
        "PRS_plus_covariates": ["PRS"] + covariate_columns,
    }

    metrics = []

    for model_name, feature_columns in model_definitions.items():

        if len(feature_columns) == 0:
            continue

        if is_binary:
            result = evaluate_binary_model(
                data=merged_data,
                feature_columns=feature_columns,
                phenotype_column=phenotype_column,
                phenotype_name=phenotype_name,
                rep=rep,
                model_name=model_name,
            )
        else:
            result = evaluate_continuous_model(
                data=merged_data,
                feature_columns=feature_columns,
                phenotype_column=phenotype_column,
                phenotype_name=phenotype_name,
                rep=rep,
                model_name=model_name,
            )

        metrics.append(result)

    raw_prs_auc = calculate_raw_prs_auc(
        merged_data,
        phenotype_column,
    )

    for result in metrics:
        result["raw_PRS_auc"] = raw_prs_auc

        if pd.notna(raw_prs_auc):
            result["raw_PRS_auc_direction_free"] = max(
                raw_prs_auc,
                1.0 - raw_prs_auc,
            )
        else:
            result["raw_PRS_auc_direction_free"] = np.nan

        result["profile_file"] = str(profile_path)

    return metrics, merged_data


###############################################################################
# Find profile files
###############################################################################

def find_profile_files(rep: int) -> List[Tuple[str, Path]]:
    """
    Find all chromosome-1 PRS profile files for one repetition.
    """

    rep_directory = PRSCS_RESULTS_BASE / f"rep{rep}"

    if not rep_directory.exists():
        return []

    results = []

    for phenotype_directory in sorted(rep_directory.iterdir()):

        if not phenotype_directory.is_dir():
            continue

        phenotype_name = phenotype_directory.name
        score_directory = phenotype_directory / "PRS_scores"

        expected_profile = (
            score_directory
            / (
                f"PRS_{phenotype_name}_rep{rep}_chr1.profile"
            )
        )

        if expected_profile.exists():
            results.append(
                (phenotype_name, expected_profile)
            )
            continue

        # Flexible fallback
        candidates = sorted(
            score_directory.glob("*.profile")
        ) if score_directory.exists() else []

        if len(candidates) == 1:
            results.append(
                (phenotype_name, candidates[0])
            )

        elif len(candidates) > 1:
            chromosome_1_candidates = [
                path
                for path in candidates
                if "chr1" in path.name
            ]

            if chromosome_1_candidates:
                results.append(
                    (
                        phenotype_name,
                        chromosome_1_candidates[0],
                    )
                )

    return results


###############################################################################
# Main
###############################################################################

def main():

    if not COVARIATE_FILE.exists():
        raise FileNotFoundError(
            f"Covariate file not found: {COVARIATE_FILE}"
        )

    print(f"Reading covariates: {COVARIATE_FILE}")

    covariate_df = prepare_covariates(
        read_whitespace_file(COVARIATE_FILE)
    )

    print(
        f"Covariate rows: {len(covariate_df):,}; "
        f"covariate columns: {len(covariate_df.columns) - 2}"
    )

    all_metrics = []
    all_errors = []

    for rep in range(1, 6):

        phenotype_file = (
            PHENOTYPE_BASE
            / f"rep{rep}_test_phenotypes.txt"
        )

        if not phenotype_file.exists():
            print(
                f"WARNING: missing phenotype file: "
                f"{phenotype_file}"
            )
            continue

        print()
        print("=" * 80)
        print(f"Processing repetition {rep}")
        print(f"Phenotype file: {phenotype_file}")
        print("=" * 80)

        phenotype_df = read_whitespace_file(
            phenotype_file
        )

        profile_files = find_profile_files(rep)

        print(
            f"Found {len(profile_files)} PRS profile files "
            f"for rep{rep}"
        )

        for phenotype_name, profile_path in profile_files:

            print()
            print("-" * 80)
            print(f"Phenotype: {phenotype_name}")
            print(f"Profile:   {profile_path}")
            print("-" * 80)

            try:
                metrics, merged_data = evaluate_phenotype(
                    rep=rep,
                    phenotype_name=phenotype_name,
                    profile_path=profile_path,
                    phenotype_df=phenotype_df,
                    covariate_df=covariate_df,
                )

                all_metrics.extend(metrics)

                phenotype_output_directory = (
                    OUTPUT_BASE
                    / f"rep{rep}"
                    / phenotype_name
                )

                phenotype_output_directory.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                final_prs_output = (
                    phenotype_output_directory
                    / "final_PRS.csv"
                )

                merged_data.to_csv(
                    final_prs_output,
                    index=False,
                )

                for result in metrics:
                    if result["phenotype_type"] == "binary":
                        print(
                            f"{result['model']}: "
                            f"AUC={result['roc_auc']:.4f}, "
                            f"AP={result['average_precision']:.4f}, "
                            f"log loss={result['log_loss']:.4f}"
                        )
                    else:
                        print(
                            f"{result['model']}: "
                            f"R²={result['r2']:.4f}, "
                            f"MSE={result['mse']:.4f}"
                        )

                print(
                    f"Raw PRS AUC: "
                    f"{metrics[0]['raw_PRS_auc']:.4f}"
                    if pd.notna(metrics[0]["raw_PRS_auc"])
                    else "Raw PRS AUC: not applicable"
                )

            except Exception as error:

                print(
                    f"ERROR for rep{rep}, "
                    f"{phenotype_name}: {error}"
                )

                all_errors.append(
                    {
                        "rep": rep,
                        "phenotype": phenotype_name,
                        "profile_file": str(profile_path),
                        "error": str(error),
                    }
                )

    ###########################################################################
    # Save combined results
    ###########################################################################

    if all_metrics:

        metrics_df = pd.DataFrame(all_metrics)

        metrics_df = metrics_df.sort_values(
            ["rep", "phenotype", "model"]
        )

        metrics_df.to_csv(
            METRICS_OUTPUT,
            index=False,
        )

        print()
        print("=" * 80)
        print(
            f"Saved {len(metrics_df)} metric rows to:"
        )
        print(METRICS_OUTPUT)

        phenotype_count = (
            metrics_df[
                ["rep", "phenotype"]
            ]
            .drop_duplicates()
            .shape[0]
        )

        print(
            f"Successfully evaluated "
            f"{phenotype_count} repetition-phenotype combinations."
        )

    else:
        print("No phenotypes were evaluated successfully.")

    if all_errors:

        errors_df = pd.DataFrame(all_errors)

        errors_df.to_csv(
            ERROR_OUTPUT,
            index=False,
        )

        print()
        print(
            f"Saved {len(errors_df)} errors to:"
        )
        print(ERROR_OUTPUT)


if __name__ == "__main__":
    main()
