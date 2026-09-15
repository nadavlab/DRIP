#!/usr/bin/env python3

import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# Input paths
# =============================================================================

DRIP_RESULTS_PATTERN = (
    "/path/to/your/project/simulations/LR_results_chr1_only"
    "/metrics_by_pheno/*.csv"
)

PRSCS_RESULTS_FILE = (
    "/path/to/your/project/simulations/PRScs/evaluation/"
    "PRSCS_metrics_summary.csv"
)

OUTPUT_DIR = (
    "/path/to/your/project/simulations/comparison_figures/"
    "DRIP_vs_PRSCS_chro1_only"
    
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# Settings
# =============================================================================

# PRS-CS is currently available only for repetitions 1-3.
REPETITIONS_TO_USE = [1, 2, 3, 4, 5]

# Metric to compare.
METRIC = "ROC_AUC"

# For the heritability-scale figure, keep the number of causal SNPs fixed.
# Change this if your heritability simulations used another value.
HERITABILITY_SCALE_N_CAUSAL = 100

# For the causal-SNP-scale figure, keep heritability fixed.
CAUSAL_SNP_SCALE_H2 = 0.3

# Fixed simulation settings.
FIXED_COVARIATE_VARIANCE = 0.2
FIXED_NOISE_VARIANCE = 0.5

# Figure settings.
FIGURE_DPI = 300


# =============================================================================
# Helper functions
# =============================================================================

def parse_number_token(token):
    """
    Convert filename tokens such as:
        0p3 -> 0.3
        0p15 -> 0.15
        1p0 -> 1.0
    """
    if pd.isna(token):
        return np.nan

    return float(str(token).replace("p", "."))


def extract_simulation_parameters(phenotype_name):
    """
    Extract simulation parameters from phenotype names such as:

        rep1_chr1_N10000_h2_0p3_cov_0p2_noise_0p5

    or:

        chr1_N10000_h2_0p3_cov_0p2_noise_0p5

    Returns:
        dictionary containing:
            n_causal
            h2
            cov_variance
            noise_variance
    """
    phenotype_name = str(phenotype_name)

    n_match = re.search(r"(?:^|_)N(\d+)(?:_|$)", phenotype_name)
    h2_match = re.search(r"(?:^|_)h2_([0-9]+p[0-9]+|[0-9.]+)(?:_|$)",
                         phenotype_name)
    cov_match = re.search(
        r"(?:^|_)cov_([0-9]+p[0-9]+|[0-9.]+)(?:_|$)",
        phenotype_name
    )
    noise_match = re.search(
        r"(?:^|_)noise_([0-9]+p[0-9]+|[0-9.]+)(?:_|$)",
        phenotype_name
    )

    return {
        "n_causal": int(n_match.group(1)) if n_match else np.nan,
        "h2": parse_number_token(h2_match.group(1)) if h2_match else np.nan,
        "cov_variance": (
            parse_number_token(cov_match.group(1))
            if cov_match else np.nan
        ),
        "noise_variance": (
            parse_number_token(noise_match.group(1))
            if noise_match else np.nan
        ),
    }


def add_simulation_parameters(df, phenotype_column):
    """
    Parse phenotype names and add simulation parameter columns.
    """
    parsed = df[phenotype_column].apply(extract_simulation_parameters)
    parsed_df = pd.DataFrame(parsed.tolist(), index=df.index)

    return pd.concat([df, parsed_df], axis=1)


def calculate_se(values):
    """
    Calculate standard error across repetitions.

    For one available repetition, SE is returned as NaN because it cannot
    be estimated from one observation.
    """
    values = pd.Series(values).dropna()

    if len(values) <= 1:
        return np.nan

    return values.std(ddof=1) / np.sqrt(len(values))


def approximately_equal(series, value, tolerance=1e-8):
    """
    Floating-point-safe comparison.
    """
    return np.isclose(
        pd.to_numeric(series, errors="coerce"),
        value,
        atol=tolerance,
        rtol=0
    )


# =============================================================================
# Read and prepare DRIP results
# =============================================================================

def load_drip_results():
    files = sorted(glob.glob(DRIP_RESULTS_PATTERN))

    if not files:
        raise FileNotFoundError(
            "No updated DRIP LR metric files were found using:\n"
            f"{DRIP_RESULTS_PATTERN}"
        )

    print(f"Number of DRIP metric files found: {len(files)}")

    drip_parts = []

    for path in files:
        current = pd.read_csv(path)

        # Keep only the newest appended result within this file.
        # CSV row order is treated as chronological order, so the last
        # row for each repetition and phenotype is retained.
        current["_row_order"] = np.arange(len(current))

        required_columns = {
            "pheno",
            "rep",
            "ROC_AUC",
            "n_test",
            "n_cases_test",
            "n_controls_test",
        }

        missing = required_columns.difference(current.columns)

        if missing:
            raise ValueError(
                f"Missing columns in {path}: {sorted(missing)}"
            )

        rows_before = len(current)

        current = (
            current.sort_values("_row_order")
            .drop_duplicates(
                subset=["rep", "pheno"],
                keep="last"
            )
            .drop(columns="_row_order")
        )

        rows_removed = rows_before - len(current)

        if rows_removed > 0:
            print(
                f"Using only the latest row per repetition and phenotype "
                f"in {path}; removed {rows_removed} older row(s)."
            )

        current["source_file"] = path
        drip_parts.append(current)

    drip = pd.concat(drip_parts, ignore_index=True)

    numeric_columns = [
        "rep",
        "ROC_AUC",
        "n_test",
        "n_cases_test",
        "n_controls_test",
    ]

    for column in numeric_columns:
        drip[column] = pd.to_numeric(
            drip[column],
            errors="coerce"
        )

    # Check that test counts are internally consistent.
    inconsistent_counts = (
        drip["n_cases_test"] + drip["n_controls_test"]
        != drip["n_test"]
    )

    if inconsistent_counts.any():
        print(
            "\nWarning: some DRIP files have inconsistent test counts:"
        )
        print(
            drip.loc[
                inconsistent_counts,
                [
                    "pheno",
                    "rep",
                    "n_test",
                    "n_cases_test",
                    "n_controls_test",
                    "source_file",
                ],
            ].to_string(index=False)
        )

    # Check for phenotypes with only one observed class.
    one_class = (
        (drip["n_cases_test"] == 0)
        | (drip["n_controls_test"] == 0)
    )

    if one_class.any():
        print(
            "\nWarning: some DRIP results have only one class "
            "in the test set and will be excluded:"
        )
        print(
            drip.loc[
                one_class,
                [
                    "pheno",
                    "rep",
                    "n_cases_test",
                    "n_controls_test",
                    "source_file",
                ],
            ].to_string(index=False)
        )

        drip = drip.loc[~one_class].copy()

    drip = add_simulation_parameters(
        drip,
        phenotype_column="pheno"
    )

    drip["method"] = "DRIP"

    # Use repetitions currently available for PRS-CS.
    drip = drip[
        drip["rep"].isin(REPETITIONS_TO_USE)
    ].copy()

    # Keep only valid ROC-AUC values.
    invalid_auc = (
        drip["ROC_AUC"].isna()
        | (drip["ROC_AUC"] < 0)
        | (drip["ROC_AUC"] > 1)
    )

    if invalid_auc.any():
        print(
            "\nWarning: invalid DRIP ROC-AUC values will be excluded:"
        )
        print(
            drip.loc[
                invalid_auc,
                ["pheno", "rep", "ROC_AUC", "source_file"],
            ].to_string(index=False)
        )

        drip = drip.loc[~invalid_auc].copy()

    # Check for repeated results for the same repetition and phenotype.
    duplicate_mask = drip.duplicated(
        subset=["rep", "pheno"],
        keep=False
    )

    if duplicate_mask.any():
        duplicate_rows = drip.loc[
            duplicate_mask,
            [
                "rep",
                "pheno",
                "ROC_AUC",
                "source_file",
            ],
        ].sort_values(
            ["rep", "pheno", "source_file"]
        )

        print(
            "\nWarning: duplicate updated DRIP result files were found:"
        )
        print(duplicate_rows.to_string(index=False))

        # Prefer the most recently modified file.
        drip["file_modification_time"] = drip[
            "source_file"
        ].apply(os.path.getmtime)

        drip = (
            drip.sort_values("file_modification_time")
            .drop_duplicates(
                subset=["rep", "pheno"],
                keep="last"
            )
            .drop(columns="file_modification_time")
        )

    print("\nUpdated DRIP results by repetition:")
    print(
        drip.groupby("rep")
        .agg(
            n_phenotypes=("pheno", "nunique"),
            mean_auc=("ROC_AUC", "mean"),
            minimum_auc=("ROC_AUC", "min"),
            maximum_auc=("ROC_AUC", "max"),
        )
        .to_string()
    )

    return drip

def load_prscs_results():
    """
    Load the combined PRS-CS evaluation results and prepare them for
    comparison with DRIP.
    """
    if not os.path.exists(PRSCS_RESULTS_FILE):
        raise FileNotFoundError(
            "PRS-CS result file was not found:\n"
            f"{PRSCS_RESULTS_FILE}"
        )

    prscs = pd.read_csv(PRSCS_RESULTS_FILE)

    # Preserve the original CSV order. When a result was appended more than
    # once, the final occurrence is considered the newest result.
    prscs["_row_order"] = np.arange(len(prscs))

    required_columns = {
        "rep",
        "phenotype",
        "model",
        "roc_auc",
    }

    missing_columns = required_columns.difference(prscs.columns)

    if missing_columns:
        raise ValueError(
            "Missing required columns in the PRS-CS summary file: "
            f"{sorted(missing_columns)}"
        )

    # Convert relevant columns to numeric.
    prscs["rep"] = pd.to_numeric(
        prscs["rep"],
        errors="coerce"
    )

    prscs["roc_auc"] = pd.to_numeric(
        prscs["roc_auc"],
        errors="coerce"
    )

    # Retain only the repetitions currently available for PRS-CS.
    prscs = prscs[
        prscs["rep"].isin(REPETITIONS_TO_USE)
    ].copy()

    # Retain the two PRS-CS models that should appear in the figure.
    model_name_map = {
        "PRS_plus_covariates": "PRS-CS",
    }

    prscs = prscs[
        prscs["model"].isin(model_name_map.keys())
    ].copy()

    prscs["method"] = prscs["model"].map(model_name_map)

    # Rename the metric to match the DRIP column name.
    prscs["ROC_AUC"] = prscs["roc_auc"]

    # Parse N, h2, covariate variance and noise variance from phenotype.
    prscs = add_simulation_parameters(
        prscs,
        phenotype_column="phenotype"
    )

    # Exclude missing or invalid ROC-AUC values.
    invalid_auc = (
        prscs["ROC_AUC"].isna()
        | (prscs["ROC_AUC"] < 0)
        | (prscs["ROC_AUC"] > 1)
    )

    if invalid_auc.any():
        print(
            "\nWarning: invalid PRS-CS ROC-AUC values will be excluded:"
        )

        print(
            prscs.loc[
                invalid_auc,
                [
                    "rep",
                    "phenotype",
                    "model",
                    "ROC_AUC",
                ],
            ].to_string(index=False)
        )

        prscs = prscs.loc[~invalid_auc].copy()

    # Keep only the latest appended row for every repetition,
    # phenotype and model.
    duplicate_key = [
        "rep",
        "phenotype",
        "method",
    ]

    duplicate_mask = prscs.duplicated(
        subset=duplicate_key,
        keep=False
    )

    if duplicate_mask.any():
        print(
            "\nMultiple PRS-CS rows were found for the same repetition, "
            "phenotype and model. Only the latest row will be used:"
        )

        print(
            prscs.loc[
                duplicate_mask,
                duplicate_key + ["ROC_AUC", "_row_order"],
            ]
            .sort_values("_row_order")
            .to_string(index=False)
        )

    rows_before = len(prscs)

    prscs = (
        prscs.sort_values("_row_order")
        .drop_duplicates(
            subset=duplicate_key,
            keep="last"
        )
        .drop(columns="_row_order")
    )

    rows_removed = rows_before - len(prscs)

    if rows_removed > 0:
        print(
            f"Removed {rows_removed} older PRS-CS result row(s)."
        )

    print("\nPRS-CS results by repetition and method:")

    print(
        prscs.groupby(
            ["rep", "method"]
        )
        .agg(
            n_phenotypes=("phenotype", "nunique"),
            mean_auc=("ROC_AUC", "mean"),
            minimum_auc=("ROC_AUC", "min"),
            maximum_auc=("ROC_AUC", "max"),
        )
        .to_string()
    )

    return prscs

def combine_results(drip, prscs):
    """
    Combine DRIP and PRS-CS results into one standardized dataframe.

    Both datasets must contain:
        rep
        method
        ROC_AUC
        n_causal
        h2
        cov_variance
        noise_variance
    """

    required_columns = [
        "rep",
        "method",
        "ROC_AUC",
        "n_causal",
        "h2",
        "cov_variance",
        "noise_variance",
    ]

    # Check that all required columns exist in DRIP.
    missing_drip = [
        column
        for column in required_columns
        if column not in drip.columns
    ]

    if missing_drip:
        raise ValueError(
            "The following required columns are missing from "
            f"the DRIP results: {missing_drip}\n"
            f"Available DRIP columns: {list(drip.columns)}"
        )

    # Check that all required columns exist in PRS-CS.
    missing_prscs = [
        column
        for column in required_columns
        if column not in prscs.columns
    ]

    if missing_prscs:
        raise ValueError(
            "The following required columns are missing from "
            f"the PRS-CS results: {missing_prscs}\n"
            f"Available PRS-CS columns: {list(prscs.columns)}"
        )

    drip_standardized = drip[
        required_columns
    ].copy()

    prscs_standardized = prscs[
        required_columns
    ].copy()

    # Add an optional source column for easier debugging.
    drip_standardized["source"] = "DRIP"
    prscs_standardized["source"] = "PRS-CS"

    combined = pd.concat(
        [
            drip_standardized,
            prscs_standardized,
        ],
        ignore_index=True
    )

    # Ensure that numerical columns are numeric.
    numeric_columns = [
        "rep",
        "ROC_AUC",
        "n_causal",
        "h2",
        "cov_variance",
        "noise_variance",
    ]

    for column in numeric_columns:
        combined[column] = pd.to_numeric(
            combined[column],
            errors="coerce"
        )

    # Remove rows without the essential plotting values.
    missing_essential_values = combined[
        [
            "rep",
            "method",
            "ROC_AUC",
            "n_causal",
            "h2",
        ]
    ].isna().any(axis=1)

    if missing_essential_values.any():
        print(
            "\nWarning: rows with missing essential values "
            "will be excluded:"
        )

        print(
            combined.loc[
                missing_essential_values
            ].to_string(index=False)
        )

        combined = combined.loc[
            ~missing_essential_values
        ].copy()

    # Check that AUC values are valid.
    invalid_auc = (
        (combined["ROC_AUC"] < 0)
        | (combined["ROC_AUC"] > 1)
    )

    if invalid_auc.any():
        print(
            "\nWarning: rows with invalid ROC-AUC values "
            "will be excluded:"
        )

        print(
            combined.loc[
                invalid_auc,
                [
                    "rep",
                    "method",
                    "ROC_AUC",
                    "n_causal",
                    "h2",
                    "source",
                ],
            ].to_string(index=False)
        )

        combined = combined.loc[
            ~invalid_auc
        ].copy()

    print("\nCombined results by method:")

    print(
        combined.groupby("method")
        .agg(
            n_rows=("ROC_AUC", "size"),
            n_repetitions=("rep", "nunique"),
            n_simulation_settings=("n_causal", "size"),
            mean_roc_auc=("ROC_AUC", "mean"),
            minimum_roc_auc=("ROC_AUC", "min"),
            maximum_roc_auc=("ROC_AUC", "max"),
        )
        .to_string()
    )

    return combined

# =============================================================================
# Match repetitions across methods
# =============================================================================

def retain_complete_repetitions(
    data,
    scale_column,
    expected_methods
):
    """
    For each point on a scale, retain only repetitions for which all methods
    are available.

    This ensures that the method comparison is based on matched simulation
    repetitions rather than different subsets of repetitions.
    """
    kept_parts = []

    for scale_value, group in data.groupby(scale_column, dropna=False):
        method_sets_by_rep = (
            group.groupby("rep")["method"]
            .apply(set)
        )

        valid_reps = [
            rep
            for rep, method_set in method_sets_by_rep.items()
            if set(expected_methods).issubset(method_set)
        ]

        if not valid_reps:
            print(
                f"Warning: no complete repetitions for "
                f"{scale_column}={scale_value}"
            )
            continue

        incomplete_reps = sorted(
            set(group["rep"].dropna().astype(int)) -
            set(int(rep) for rep in valid_reps)
        )

        if incomplete_reps:
            print(
                f"Excluding incomplete repetitions for "
                f"{scale_column}={scale_value}: {incomplete_reps}"
            )

        kept_parts.append(
            group[group["rep"].isin(valid_reps)].copy()
        )

    if not kept_parts:
        return data.iloc[0:0].copy()

    return pd.concat(kept_parts, ignore_index=True)


# =============================================================================
# Summarize across repetitions
# =============================================================================

def summarize_results(data, scale_column):
    """
    Calculate mean, SD and SE across repetitions.
    """
    summary = (
        data.groupby(
            [scale_column, "method"],
            as_index=False
        )
        .agg(
            mean_roc_auc=("ROC_AUC", "mean"),
            sd_roc_auc=("ROC_AUC", "std"),
            se_roc_auc=("ROC_AUC", calculate_se),
            n_repetitions=("ROC_AUC", "count"),
        )
    )

    return summary


# =============================================================================
# Plot grouped bars
# =============================================================================

def create_grouped_bar_plot(
    summary,
    scale_column,
    scale_order,
    x_tick_labels,
    x_axis_label,
    title,
    output_prefix,
):
    method_order = [
        "DRIP",
        "PRS-CS",
    ]

    # Keep only methods and scale values that are meant to be plotted.
    summary = summary[
        summary["method"].isin(method_order)
        & summary[scale_column].isin(scale_order)
    ].copy()

    if summary.empty:
        print(f"No data available for: {title}")
        return

    x = np.arange(len(scale_order))
    number_of_methods = len(method_order)

    total_group_width = 0.78
    bar_width = total_group_width / number_of_methods

    fig, ax = plt.subplots(figsize=(10, 6.5))

    method_colors = {
        "DRIP": "#4CAF50",      # green
        "PRS-CS": "#87CEFA",    # light blue
    }

    for method_index, method in enumerate(method_order):
        method_data = (
            summary[summary["method"] == method]
            .set_index(scale_column)
            .reindex(scale_order)
        )

        means = method_data["mean_roc_auc"].to_numpy(dtype=float)
        errors = method_data["se_roc_auc"].to_numpy(dtype=float)

        # Matplotlib handles zero error bars more reliably than NaN.
        errors_for_plotting = np.nan_to_num(
            errors,
            nan=0.0
        )

        offset = (
            method_index - (number_of_methods - 1) / 2
        ) * bar_width

        bars = ax.bar(
            x + offset,
            means,
            width=bar_width,
            label=method,
            color=method_colors[method],
            yerr=errors_for_plotting,
            capsize=4,
            edgecolor="black",
            linewidth=0.7,
        )

        # Add number of repetitions above every available bar.
        for bar_index, bar in enumerate(bars):
            if np.isnan(means[bar_index]):
                continue

            n_reps = method_data.iloc[
                bar_index
            ]["n_repetitions"]

            label_height = (
                means[bar_index]
                + errors_for_plotting[bar_index]
                + 0.008
            )

            ax.text(
                bar.get_x() + bar.get_width() / 2,
                label_height,
                f"n={int(n_reps)}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90,
            )

    ax.axhline(
        0.5,
        linestyle="--",
        linewidth=1,
        label="Random classifier"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(x_tick_labels)

    ax.set_xlabel(x_axis_label, fontsize=12)
    ax.set_ylabel("ROC-AUC", fontsize=12)
    ax.set_title(title, fontsize=14)

    # Start somewhat below 0.5 so the random-classifier line remains visible.
    plotted_values = summary["mean_roc_auc"].dropna()

    if len(plotted_values) > 0:
        plotted_errors = summary["se_roc_auc"].fillna(0)
        upper_limit = min(
            1.0,
            max(0.85, (plotted_values + plotted_errors).max() + 0.08)
        )
    else:
        upper_limit = 1.0

    ax.set_ylim(0.45, upper_limit)

    ax.grid(
        axis="y",
        linestyle=":",
        alpha=0.5
    )

    ax.legend(
        frameon=False,
        loc="best"
    )

    fig.tight_layout()

    png_path = os.path.join(
        OUTPUT_DIR,
        f"{output_prefix}.png"
    )
    pdf_path = os.path.join(
        OUTPUT_DIR,
        f"{output_prefix}.pdf"
    )

    fig.savefig(
        png_path,
        dpi=FIGURE_DPI,
        bbox_inches="tight"
    )
    fig.savefig(
        pdf_path,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


# =============================================================================
# Heritability-scale analysis
# =============================================================================

def create_heritability_figure(combined):
    """
    Create the DRIP versus PRS-CS ROC-AUC figure for the heritability scale.

    Heritability experiment:
        - Number of causal SNPs is fixed at N = 100.
        - Heritability varies from 0.0 to 0.7.
        - Covariate and noise variances change with heritability.
        - Only repetitions available for all compared methods are retained.
        - Duplicate representations of the same simulation setting are removed.
    """

    n_causal_required = 100

    expected_settings = pd.DataFrame(
        {
            "h2": [
                0.0,
                0.1,
                0.2,
                0.3,
                0.4,
                0.5,
                0.6,
                0.7,
            ],
            "expected_cov_variance": [
                0.286,
                0.257,
                0.229,
                0.200,
                0.171,
                0.143,
                0.114,
                0.086,
            ],
            "expected_noise_variance": [
                0.714,
                0.643,
                0.571,
                0.500,
                0.429,
                0.357,
                0.286,
                0.214,
            ],
        }
    )

    method_order = [
        "DRIP",
        "PRS-CS",
    ]

    required_columns = {
        "rep",
        "method",
        "ROC_AUC",
        "n_causal",
        "h2",
        "cov_variance",
        "noise_variance",
    }

    missing_columns = required_columns.difference(combined.columns)

    if missing_columns:
        raise ValueError(
            "The combined dataframe is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    # -------------------------------------------------------------------------
    # 1. Keep only N = 100 and the requested methods
    # -------------------------------------------------------------------------

    subset = combined[
        (pd.to_numeric(combined["n_causal"], errors="coerce")
         == n_causal_required)
        & combined["method"].isin(method_order)
    ].copy()

    if subset.empty:
        raise ValueError(
            f"No results were found for N = {n_causal_required}."
        )

    # Make sure all relevant columns are numeric.
    numeric_columns = [
        "rep",
        "ROC_AUC",
        "n_causal",
        "h2",
        "cov_variance",
        "noise_variance",
    ]

    for column in numeric_columns:
        subset[column] = pd.to_numeric(
            subset[column],
            errors="coerce"
        )

    subset = subset.dropna(
        subset=[
            "rep",
            "method",
            "ROC_AUC",
            "h2",
            "cov_variance",
            "noise_variance",
        ]
    )

    # -------------------------------------------------------------------------
    # 2. Keep only the exact settings belonging to the heritability experiment
    # -------------------------------------------------------------------------

    matched_setting_parts = []

    for setting in expected_settings.itertuples(index=False):
        current = subset[
            np.isclose(
                subset["h2"],
                setting.h2,
                atol=1e-8,
                rtol=0
            )
            & np.isclose(
                subset["cov_variance"],
                setting.expected_cov_variance,
                atol=1e-6,
                rtol=0
            )
            & np.isclose(
                subset["noise_variance"],
                setting.expected_noise_variance,
                atol=1e-6,
                rtol=0
            )
        ].copy()

        matched_setting_parts.append(current)

    subset = pd.concat(
        matched_setting_parts,
        ignore_index=True
    )

    if subset.empty:
        raise ValueError(
            "No results matched the expected heritability-scale settings."
        )

    # -------------------------------------------------------------------------
    # 3. Restrict to the requested repetitions
    # -------------------------------------------------------------------------

    subset = subset[
        subset["rep"].isin(REPETITIONS_TO_USE)
    ].copy()

    if subset.empty:
        raise ValueError(
            "No heritability-scale results remained after filtering by "
            f"REPETITIONS_TO_USE={REPETITIONS_TO_USE}."
        )

    # -------------------------------------------------------------------------
    # 4. Remove duplicate representations of the same result
    #
    # Example:
    #   h2_0p3_cov_0p200_noise_0p500
    #   h2_0p3_cov_0p2_noise_0p5
    #
    # These are numerically the same phenotype and should count only once.
    # -------------------------------------------------------------------------

    duplicate_key = [
        "rep",
        "method",
        "n_causal",
        "h2",
        "cov_variance",
        "noise_variance",
    ]

    duplicate_mask = subset.duplicated(
        subset=duplicate_key,
        keep=False
    )

    if duplicate_mask.any():
        print(
            "\nDuplicate heritability results detected and reduced to "
            "one row per repetition, method and setting:"
        )

        columns_to_print = [
            "rep",
            "method",
            "n_causal",
            "h2",
            "cov_variance",
            "noise_variance",
            "ROC_AUC",
        ]

        optional_name_columns = [
            column
            for column in ["pheno", "phenotype", "source_file"]
            if column in subset.columns
        ]

        print(
            subset.loc[
                duplicate_mask,
                columns_to_print + optional_name_columns,
            ]
            .sort_values(
                [
                    "h2",
                    "method",
                    "rep",
                ]
            )
            .to_string(index=False)
        )

    subset = subset.drop_duplicates(
        subset=duplicate_key,
        keep="first"
    )

    # -------------------------------------------------------------------------
    # 5. Retain only repetitions available for every method at each h2
    # -------------------------------------------------------------------------

    matched_parts = []

    for h2_value in expected_settings["h2"]:
        current_h2 = subset[
            np.isclose(
                subset["h2"],
                h2_value,
                atol=1e-8,
                rtol=0
            )
        ].copy()

        if current_h2.empty:
            print(
                f"Warning: no results were found for h2={h2_value:.1f}."
            )
            continue

        method_sets_by_rep = (
            current_h2.groupby("rep")["method"]
            .apply(set)
        )

        valid_repetitions = [
            rep
            for rep, available_methods in method_sets_by_rep.items()
            if set(method_order).issubset(available_methods)
        ]

        if not valid_repetitions:
            print(
                f"Warning: no repetition had all three methods for "
                f"h2={h2_value:.1f}."
            )
            continue

        print(
            f"h2={h2_value:.1f}: matched repetitions "
            f"{sorted(int(rep) for rep in valid_repetitions)}"
        )

        matched_parts.append(
            current_h2[
                current_h2["rep"].isin(valid_repetitions)
            ].copy()
        )

    if not matched_parts:
        raise ValueError(
            "No complete matched repetitions remained for the "
            "heritability-scale figure."
        )

    subset = pd.concat(
        matched_parts,
        ignore_index=True
    )

    # -------------------------------------------------------------------------
    # 6. Verify one row per repetition, method and heritability level
    # -------------------------------------------------------------------------

    remaining_duplicates = subset.duplicated(
        subset=["rep", "method", "h2"],
        keep=False
    )

    if remaining_duplicates.any():
        print(
            subset.loc[
                remaining_duplicates,
                [
                    "rep",
                    "method",
                    "h2",
                    "cov_variance",
                    "noise_variance",
                    "ROC_AUC",
                ],
            ]
            .sort_values(["h2", "method", "rep"])
            .to_string(index=False)
        )

        raise ValueError(
            "More than one row remains for the same repetition, method "
            "and heritability level."
        )

    # -------------------------------------------------------------------------
    # 7. Calculate mean, SD and SE across repetitions
    # -------------------------------------------------------------------------

    summary = (
        subset.groupby(
            ["h2", "method"],
            as_index=False
        )
        .agg(
            mean_roc_auc=("ROC_AUC", "mean"),
            sd_roc_auc=("ROC_AUC", "std"),
            n_repetitions=("rep", "nunique"),
        )
    )

    summary["se_roc_auc"] = (
        summary["sd_roc_auc"]
        / np.sqrt(summary["n_repetitions"])
    )

    # SE cannot be estimated from only one repetition.
    summary.loc[
        summary["n_repetitions"] < 2,
        "se_roc_auc",
    ] = np.nan

    # -------------------------------------------------------------------------
    # 8. Keep the intended h2 order
    # -------------------------------------------------------------------------

    scale_order = expected_settings["h2"].tolist()

    available_h2 = summary["h2"].dropna().to_numpy(dtype=float)

    scale_order = [
        h2_value
        for h2_value in scale_order
        if np.isclose(
            available_h2,
            h2_value,
            atol=1e-8,
            rtol=0
        ).any()
    ]

    x_tick_labels = [
        f"{h2_value:.1f}"
        for h2_value in scale_order
    ]

    # -------------------------------------------------------------------------
    # 9. Save the data used in the figure
    # -------------------------------------------------------------------------

    raw_output = os.path.join(
        OUTPUT_DIR,
        "heritability_scale_matched_repetition_values.csv"
    )

    summary_output = os.path.join(
        OUTPUT_DIR,
        "heritability_scale_summary.csv"
    )

    subset.sort_values(
        [
            "h2",
            "method",
            "rep",
        ]
    ).to_csv(
        raw_output,
        index=False
    )

    summary.sort_values(
        [
            "h2",
            "method",
        ]
    ).to_csv(
        summary_output,
        index=False
    )

    print("\nHeritability-scale summary:")
    print(
        summary.sort_values(
            [
                "h2",
                "method",
            ]
        ).to_string(index=False)
    )

    print(f"\nSaved raw matched values to:\n{raw_output}")
    print(f"Saved summarized values to:\n{summary_output}")

    # -------------------------------------------------------------------------
    # 10. Create the grouped bar plot
    # -------------------------------------------------------------------------

    create_grouped_bar_plot(
        summary=summary,
        scale_column="h2",
        scale_order=scale_order,
        x_tick_labels=x_tick_labels,
        x_axis_label="Simulated SNP heritability",
        title=(
            "Prediction performance across heritability levels\n"
            f"N causal SNPs = {n_causal_required}"
        ),
        output_prefix=(
            "ROC_AUC_heritability_scale_N100_DRIP_vs_PRSCS_new"
        ),
    )

# =============================================================================
# Causal-SNP-scale analysis
# =============================================================================

def create_causal_snp_figure(combined):
    """
    Compare methods across causal-SNP counts while fixing heritability,
    covariate variance and noise variance.
    """
    subset = combined[
        approximately_equal(
            combined["h2"],
            CAUSAL_SNP_SCALE_H2
        )
        & approximately_equal(
            combined["cov_variance"],
            FIXED_COVARIATE_VARIANCE
        )
        & approximately_equal(
            combined["noise_variance"],
            FIXED_NOISE_VARIANCE
        )
    ].copy()

    if subset.empty:
        print(
            "\nNo rows were found for the causal-SNP scale with:"
        )
        print(
            f"  h2 = {CAUSAL_SNP_SCALE_H2}"
        )
        print(
            f"  covariate variance = {FIXED_COVARIATE_VARIANCE}"
        )
        print(
            f"  noise variance = {FIXED_NOISE_VARIANCE}"
        )
        return

    method_order = [
        "DRIP",
        "PRS-CS",
    ]

    subset = retain_complete_repetitions(
        subset,
        scale_column="n_causal",
        expected_methods=method_order
    )

    # Ensure one observation per method, SNP count and repetition.
    # This removes duplicate representations of the same simulation setting
    # and guarantees that repetitions 1-5 contribute at most five values.
    duplicate_key = ["method", "n_causal", "rep"]
    duplicate_mask = subset.duplicated(
        subset=duplicate_key,
        keep=False
    )

    if duplicate_mask.any():
        print(
            "\nDuplicate causal-SNP-scale results detected. "
            "Keeping only the first row for each method, SNP count and repetition:"
        )
        print(
            subset.loc[
                duplicate_mask,
                duplicate_key + ["ROC_AUC", "h2", "cov_variance", "noise_variance"],
            ]
            .sort_values(duplicate_key)
            .to_string(index=False)
        )

    subset = (
        subset.sort_values(["method", "n_causal", "rep"])
        .drop_duplicates(
            subset=duplicate_key,
            keep="first"
        )
        .copy()
    )

    # Keep only repetitions 1-5 explicitly.
    subset = subset[
        subset["rep"].isin(REPETITIONS_TO_USE)
    ].copy()

    if subset.empty:
        print(
            "No matched repetitions remained for the causal-SNP figure."
        )
        return

    summary = summarize_results(
        subset,
        scale_column="n_causal"
    )

    scale_order = sorted(
        summary["n_causal"].dropna().astype(int).unique()
    )

    x_tick_labels = [
        f"{value:,}"
        for value in scale_order
    ]

    raw_output = os.path.join(
        OUTPUT_DIR,
        "causal_snp_scale_matched_repetition_values.csv"
    )
    summary_output = os.path.join(
        OUTPUT_DIR,
        "causal_snp_scale_summary.csv"
    )

    subset.sort_values(
        ["n_causal", "method", "rep"]
    ).to_csv(raw_output, index=False)

    summary.sort_values(
        ["n_causal", "method"]
    ).to_csv(summary_output, index=False)

    print("\nCausal-SNP-scale summary:")
    print(
        summary.sort_values(
            ["n_causal", "method"]
        ).to_string(index=False)
    )

    create_grouped_bar_plot(
        summary=summary,
        scale_column="n_causal",
        scale_order=scale_order,
        x_tick_labels=x_tick_labels,
        x_axis_label="Number of causal SNPs",
        title=(
            "Prediction performance across causal-SNP counts\n"
            f"SNP heritability = {CAUSAL_SNP_SCALE_H2:g}"
        ),
        output_prefix="ROC_AUC_causal_SNP_scale_DRIP_vs_PRSCS_new",
    )


# =============================================================================
# Main
# =============================================================================

def main():
    drip = load_drip_results()
    prscs = load_prscs_results()

    print("\nAvailable DRIP repetitions:")
    print(sorted(drip["rep"].dropna().astype(int).unique()))

    print("Available PRS-CS repetitions:")
    print(sorted(prscs["rep"].dropna().astype(int).unique()))

    print("\nParsed DRIP simulation settings:")
    print(
        drip[
            [
                "n_causal",
                "h2",
                "cov_variance",
                "noise_variance",
            ]
        ]
        .drop_duplicates()
        .sort_values(["h2", "n_causal"])
        .to_string(index=False)
    )

    print("\nParsed PRS-CS simulation settings:")
    print(
        prscs[
            [
                "n_causal",
                "h2",
                "cov_variance",
                "noise_variance",
            ]
        ]
        .drop_duplicates()
        .sort_values(["h2", "n_causal"])
        .to_string(index=False)
    )

    combined = combine_results(
        drip=drip,
        prscs=prscs
    )

    combined_output = os.path.join(
        OUTPUT_DIR,
        "combined_DRIP_PRSCS_ROC_AUC_values.csv"
    )
    combined.to_csv(combined_output, index=False)

    print(f"\nSaved combined values: {combined_output}")

    create_heritability_figure(combined)
    create_causal_snp_figure(combined)


if __name__ == "__main__":
    main()