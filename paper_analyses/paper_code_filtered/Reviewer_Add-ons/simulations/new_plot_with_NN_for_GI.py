#!/usr/bin/env python3

"""
Compare prediction performance across G×G interaction levels.

Methods:
    1. DRIP + Logistic Regression
    2. DRIP + Neural Network
    3. PRS-CS + covariates

Metric:
    ROC-AUC

Error bars:
    Standard error across matched simulation repetitions.
"""

import glob
import os
import re
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =============================================================================
# Paths
# =============================================================================

DRIP_LR_PATTERN = (
    "/path/to/your/project/simulations/LR_results_chr1_only/"
    "metrics_by_pheno/*.csv"
)

DRIP_NN_PATTERN = (
    "/path/to/your/project/simulations/GxG_NN_results_UKB_settings/"
    "metrics_by_pheno/*.csv"
)

PRSCS_FILE = (
    "/path/to/your/project/simulations/PRScs/evaluation/"
    "PRSCS_metrics_summary.csv"
)

OUTPUT_DIR = (
    "/path/to/your/project/simulations/comparison_figures/"
    "GxG_DRIP_LR_NN_PRSCS"
)


# =============================================================================
# Analysis settings
# =============================================================================

# Keep only the G×G experiment with 100 causal SNPs.
N_CAUSAL_REQUIRED = 100

# PRS-CS is currently available for repetitions 1–3.
# Set to None to use all repetitions that are available for all three methods.
REPETITIONS_TO_USE = [1, 2, 3]

# Expected G×G levels in the current UKB-settings NN experiment.
GI_LEVEL_ORDER = [0, 10, 20, 30]

# Labels shown on the x-axis.
GI_LABELS = {
    0: "0.00",
    10: "0.10",
    20: "0.20",
    30: "0.30",
}

METHOD_ORDER = [
    "DRIP + LR",
    "DRIP + NN",
    "PRS-CS",
]

FIGURE_DPI = 300


# =============================================================================
# General helpers
# =============================================================================

def ensure_output_directory():
    """Create the output directory if it does not already exist."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def validate_required_columns(data, required_columns, source_name):
    """Raise an informative error when an input file lacks required columns."""
    missing = sorted(set(required_columns).difference(data.columns))

    if missing:
        raise ValueError(
            f"{source_name} is missing required columns: {missing}\n"
            f"Available columns: {list(data.columns)}"
        )


def parse_numeric_token(token):
    """
    Convert simulation filename tokens to floats.

    Examples:
        0p15 -> 0.15
        0.15 -> 0.15
        0p200 -> 0.200
    """
    if token is None or pd.isna(token):
        return np.nan

    try:
        return float(str(token).replace("p", "."))
    except ValueError:
        return np.nan


def extract_first_match(text, pattern, converter=None):
    """Extract the first regular-expression match from a phenotype name."""
    match = re.search(pattern, str(text))

    if match is None:
        return np.nan

    value = match.group(1)

    if converter is not None:
        try:
            return converter(value)
        except (TypeError, ValueError):
            return np.nan

    return value


def parse_gxg_phenotype(phenotype_name):
    """
    Parse G×G simulation parameters from a phenotype name.

    Example:
        rep1_chr1_N100_GI_15_h2add_0p15_h2gxg_0p15_cov_0p200_noise_0p500
    """
    phenotype_name = str(phenotype_name)

    return {
        "n_causal": extract_first_match(
            phenotype_name,
            r"(?:^|_)N(\d+)(?:_|$)",
            int,
        ),
        "gi_level": extract_first_match(
            phenotype_name,
            r"(?:^|_)GI_(\d+)(?:_|$)",
            int,
        ),
        "h2_additive": extract_first_match(
            phenotype_name,
            r"(?:^|_)h2add_([0-9]+p[0-9]+|[0-9.]+)(?:_|$)",
            parse_numeric_token,
        ),
        "h2_gxg": extract_first_match(
            phenotype_name,
            r"(?:^|_)h2gxg_([0-9]+p[0-9]+|[0-9.]+)(?:_|$)",
            parse_numeric_token,
        ),
        "cov_variance": extract_first_match(
            phenotype_name,
            r"(?:^|_)cov_([0-9]+p[0-9]+|[0-9.]+)(?:_|$)",
            parse_numeric_token,
        ),
        "noise_variance": extract_first_match(
            phenotype_name,
            r"(?:^|_)noise_([0-9]+p[0-9]+|[0-9.]+)(?:_|$)",
            parse_numeric_token,
        ),
    }


def add_gxg_parameters(data, phenotype_column):
    """Add parsed G×G simulation parameters to a dataframe."""
    parsed = data[phenotype_column].apply(parse_gxg_phenotype)
    parsed = pd.DataFrame(parsed.tolist(), index=data.index)

    # Avoid collisions if this function is called more than once.
    overlapping = [
        column for column in parsed.columns
        if column in data.columns
    ]

    if overlapping:
        data = data.drop(columns=overlapping)

    return pd.concat([data, parsed], axis=1)


def standardize_rep_column(data):
    """Convert repetition identifiers to numeric values."""
    data = data.copy()
    data["rep"] = pd.to_numeric(data["rep"], errors="coerce")
    return data


def standardize_auc_column(data, auc_column):
    """Create a standardized ROC_AUC column."""
    data = data.copy()
    data["ROC_AUC"] = pd.to_numeric(
        data[auc_column],
        errors="coerce",
    )
    return data


def filter_valid_auc(data, source_name):
    """Remove missing or invalid ROC-AUC values."""
    data = data.copy()

    invalid = (
        data["ROC_AUC"].isna()
        | (data["ROC_AUC"] < 0)
        | (data["ROC_AUC"] > 1)
    )

    if invalid.any():
        print(
            f"\nWarning: excluding {invalid.sum()} invalid "
            f"ROC-AUC rows from {source_name}."
        )

    return data.loc[~invalid].copy()


def apply_repetition_filter(data):
    """Optionally retain only explicitly requested repetitions."""
    if REPETITIONS_TO_USE is None:
        return data.copy()

    return data[
        data["rep"].isin(REPETITIONS_TO_USE)
    ].copy()


# =============================================================================
# Load DRIP + LR
# =============================================================================

def load_drip_lr_results():
    """Read the per-phenotype DRIP logistic-regression metric files."""
    files = sorted(glob.glob(DRIP_LR_PATTERN))

    if not files:
        raise FileNotFoundError(
            "No DRIP LR metric files were found using:\n"
            f"{DRIP_LR_PATTERN}"
        )

    rows = []

    for file_path in files:
        try:
            current = pd.read_csv(file_path)
        except Exception as error:
            warnings.warn(
                f"Could not read {file_path}: {error}"
            )
            continue

        validate_required_columns(
            current,
            required_columns=["rep", "pheno", "ROC_AUC"],
            source_name=file_path,
        )

        # Only G×G phenotypes are relevant here.
        current = current[
            current["pheno"].astype(str).str.contains(
                r"(?:^|_)GI_\d+",
                regex=True,
                na=False,
            )
        ].copy()

        if current.empty:
            continue

        current["source_file"] = file_path
        rows.append(current)

    if not rows:
        raise ValueError(
            "DRIP LR files were found, but none contained G×G phenotypes."
        )

    data = pd.concat(rows, ignore_index=True)

    data = standardize_rep_column(data)
    data = standardize_auc_column(data, "ROC_AUC")
    data = add_gxg_parameters(data, "pheno")

    data["method"] = "DRIP + LR"
    data["phenotype_name"] = data["pheno"].astype(str)

    data = filter_valid_auc(data, "DRIP + LR")
    data = apply_repetition_filter(data)

    return retain_requested_gxg_settings(
        data,
        source_name="DRIP + LR",
    )


# =============================================================================
# Load DRIP + NN
# =============================================================================

def load_drip_nn_results():
    """Read and combine the per-phenotype neural-network metric files."""
    files = sorted(glob.glob(DRIP_NN_PATTERN))

    if not files:
        raise FileNotFoundError(
            "No DRIP NN metric files were found using:\n"
            f"{DRIP_NN_PATTERN}"
        )

    rows = []

    for file_path in files:
        try:
            current = pd.read_csv(file_path)
        except Exception as error:
            warnings.warn(
                f"Could not read {file_path}: {error}"
            )
            continue

        if current.empty:
            warnings.warn(f"Skipping empty NN metrics file: {file_path}")
            continue

        filename_stem = os.path.splitext(
            os.path.basename(file_path)
        )[0]

        # Support either ROC_AUC or roc_auc.
        if "ROC_AUC" in current.columns:
            auc_column = "ROC_AUC"
        elif "roc_auc" in current.columns:
            auc_column = "roc_auc"
        else:
            warnings.warn(
                f"Skipping {file_path}: no ROC_AUC or roc_auc column. "
                f"Available columns: {list(current.columns)}"
            )
            continue

        # Infer repetition from the filename when it is absent in the CSV.
        if "rep" not in current.columns:
            rep_match = re.search(r"(?:^|_)rep(\d+)(?:_|$)", filename_stem)
            if rep_match is None:
                warnings.warn(
                    f"Skipping {file_path}: could not infer repetition."
                )
                continue
            current["rep"] = int(rep_match.group(1))

        # Infer phenotype from the filename when it is absent in the CSV.
        if "pheno" not in current.columns:
            if "phenotype" in current.columns:
                current["pheno"] = current["phenotype"].astype(str)
            else:
                current["pheno"] = filename_stem

        # Only G×G phenotypes are relevant here.
        current = current[
            current["pheno"].astype(str).str.contains(
                r"(?:^|_)GI_\d+",
                regex=True,
                na=False,
            )
        ].copy()

        if current.empty:
            continue

        current = standardize_rep_column(current)
        current = standardize_auc_column(current, auc_column)
        current = add_gxg_parameters(current, "pheno")

        # The UKB-settings filenames do not necessarily contain an N token.
        # This directory represents the N=100 experiment used in this figure.
        current["n_causal"] = current["n_causal"].fillna(
            N_CAUSAL_REQUIRED
        )

        current["method"] = "DRIP + NN"
        current["phenotype_name"] = current["pheno"].astype(str)
        current["source_file"] = file_path

        rows.append(current)

    if not rows:
        raise ValueError(
            "NN metric files were found, but none contained usable "
            "G×G ROC-AUC results."
        )

    data = pd.concat(rows, ignore_index=True)

    data = filter_valid_auc(data, "DRIP + NN")
    data = apply_repetition_filter(data)

    return retain_requested_gxg_settings(
        data,
        source_name="DRIP + NN",
    )


# =============================================================================
# Load PRS-CS
# =============================================================================

def load_prscs_results():
    """
    Read PRS-CS results.

    Only the PRS_plus_covariates model is retained.
    """
    if not os.path.exists(PRSCS_FILE):
        raise FileNotFoundError(
            f"PRS-CS file was not found:\n{PRSCS_FILE}"
        )

    data = pd.read_csv(PRSCS_FILE)

    validate_required_columns(
        data,
        required_columns=[
            "rep",
            "phenotype",
            "model",
            "roc_auc",
        ],
        source_name=PRSCS_FILE,
    )

    data = data[
        data["model"].eq("PRS_plus_covariates")
    ].copy()

    data = data[
        data["phenotype"].astype(str).str.contains(
            r"(?:^|_)GI_\d+",
            regex=True,
            na=False,
        )
    ].copy()

    data = standardize_rep_column(data)
    data = standardize_auc_column(data, "roc_auc")
    data = add_gxg_parameters(data, "phenotype")

    data["method"] = "PRS-CS"
    data["phenotype_name"] = data["phenotype"].astype(str)
    data["source_file"] = PRSCS_FILE

    data = filter_valid_auc(data, "PRS-CS")
    data = apply_repetition_filter(data)

    return retain_requested_gxg_settings(
        data,
        source_name="PRS-CS",
    )


# =============================================================================
# G×G filtering and duplicate handling
# =============================================================================

def retain_requested_gxg_settings(data, source_name):
    """
    Retain only the intended G×G experiment.

    Conditions:
        N = 100
        GI in the levels listed in GI_LEVEL_ORDER
    """
    data = data.copy()

    data = data[
        data["n_causal"].eq(N_CAUSAL_REQUIRED)
        & data["gi_level"].isin(GI_LEVEL_ORDER)
    ].copy()

    if data.empty:
        raise ValueError(
            f"No requested G×G results remained for {source_name}.\n"
            f"Required N={N_CAUSAL_REQUIRED} and "
            f"GI levels={GI_LEVEL_ORDER}."
        )

    # Check that the parsed interaction heritability agrees with GI.
    expected_h2_gxg = data["gi_level"] / 100.0

    inconsistent_gi = (
        data["h2_gxg"].notna()
        & ~np.isclose(
            data["h2_gxg"],
            expected_h2_gxg,
            atol=1e-8,
            rtol=0,
        )
    )

    if inconsistent_gi.any():
        print(
            f"\nWarning: {source_name} contains rows where GI level "
            "and h2gxg do not agree:"
        )

        print(
            data.loc[
                inconsistent_gi,
                [
                    "rep",
                    "phenotype_name",
                    "gi_level",
                    "h2_gxg",
                    "ROC_AUC",
                ],
            ].to_string(index=False)
        )

    # There should be only one result per repetition and GI level.
    duplicate_key = [
        "rep",
        "method",
        "gi_level",
    ]

    duplicate_mask = data.duplicated(
        subset=duplicate_key,
        keep=False,
    )

    if duplicate_mask.any():
        print(
            f"\nWarning: duplicate {source_name} results were found."
        )

        print(
            data.loc[
                duplicate_mask,
                [
                    "rep",
                    "method",
                    "gi_level",
                    "phenotype_name",
                    "ROC_AUC",
                    "source_file",
                ],
            ]
            .sort_values(duplicate_key)
            .to_string(index=False)
        )

        # Keep the final occurrence after sorting by source file.
        data = (
            data.sort_values(
                [
                    "rep",
                    "gi_level",
                    "source_file",
                ]
            )
            .drop_duplicates(
                subset=duplicate_key,
                keep="last",
            )
        )

    return data


# =============================================================================
# Combine and match repetitions
# =============================================================================

def combine_results(drip_lr, drip_nn, prscs):
    """Combine all methods into one standardized dataframe."""
    columns = [
        "rep",
        "phenotype_name",
        "method",
        "ROC_AUC",
        "n_causal",
        "gi_level",
        "h2_additive",
        "h2_gxg",
        "cov_variance",
        "noise_variance",
        "source_file",
    ]

    combined = pd.concat(
        [
            drip_lr[columns],
            drip_nn[columns],
            prscs[columns],
        ],
        ignore_index=True,
    )

    combined = combined.sort_values(
        [
            "gi_level",
            "method",
            "rep",
        ]
    ).reset_index(drop=True)

    return combined


def report_available_results(combined):
    """Print repetition coverage before matching."""
    print("\nAvailable repetitions before matching:")

    coverage = (
        combined.groupby(
            ["gi_level", "method"]
        )["rep"]
        .apply(
            lambda values: ", ".join(
                str(int(value))
                for value in sorted(values.dropna().unique())
            )
        )
        .reset_index(name="repetitions")
    )

    print(coverage.to_string(index=False))


def retain_matched_repetitions(combined):
    """
    At each GI level, retain only repetitions available for all methods.
    """
    matched_parts = []

    for gi_level in GI_LEVEL_ORDER:
        current = combined[
            combined["gi_level"].eq(gi_level)
        ].copy()

        if current.empty:
            print(
                f"\nWarning: no results found for GI_{gi_level:02d}."
            )
            continue

        methods_by_rep = (
            current.groupby("rep")["method"]
            .apply(set)
        )

        valid_repetitions = [
            rep
            for rep, available_methods in methods_by_rep.items()
            if set(METHOD_ORDER).issubset(available_methods)
        ]

        if not valid_repetitions:
            print(
                f"\nWarning: no complete repetitions were available for "
                f"GI_{gi_level:02d}."
            )
            continue

        valid_repetitions = sorted(valid_repetitions)

        print(
            f"GI_{gi_level:02d}: using matched repetitions "
            f"{[int(rep) for rep in valid_repetitions]}"
        )

        matched_parts.append(
            current[
                current["rep"].isin(valid_repetitions)
            ].copy()
        )

    if not matched_parts:
        raise ValueError(
            "No GI level had at least one repetition available "
            "for all three methods."
        )

    matched = pd.concat(
        matched_parts,
        ignore_index=True,
    )

    # Final safety check.
    duplicate_mask = matched.duplicated(
        subset=["rep", "method", "gi_level"],
        keep=False,
    )

    if duplicate_mask.any():
        raise ValueError(
            "Duplicate rows remain after matching:\n"
            + matched.loc[
                duplicate_mask,
                [
                    "rep",
                    "method",
                    "gi_level",
                    "phenotype_name",
                ],
            ].to_string(index=False)
        )

    return matched


# =============================================================================
# Summary statistics
# =============================================================================

def summarize_across_repetitions(matched):
    """Calculate mean ROC-AUC and SE across unique repetitions."""
    summary = (
        matched.groupby(
            ["gi_level", "method"],
            as_index=False,
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

    # A standard error cannot be estimated from one repetition.
    summary.loc[
        summary["n_repetitions"] < 2,
        "se_roc_auc",
    ] = np.nan

    return summary


# =============================================================================
# Plotting
# =============================================================================

def create_gxg_bar_plot(summary):
    """Create grouped ROC-AUC bars with SE error bars."""
    available_levels = set(
        summary["gi_level"]
        .dropna()
        .astype(int)
        .tolist()
    )

    scale_order = [
        level
        for level in GI_LEVEL_ORDER
        if level in available_levels
    ]

    if not scale_order:
        raise ValueError(
            "No expected GI levels were available for plotting."
        )

    x = np.arange(len(scale_order))

    number_of_methods = len(METHOD_ORDER)
    total_group_width = 0.78
    bar_width = total_group_width / number_of_methods

    fig, ax = plt.subplots(
        figsize=(10.5, 6.5)
    )

    method_colors = {
        "DRIP + LR": "#4CAF50",
        "DRIP + NN": "#2E7D32",
        "PRS-CS": "#87CEFA",
    }

    for method_index, method in enumerate(METHOD_ORDER):
        method_summary = (
            summary[
                summary["method"].eq(method)
            ]
            .set_index("gi_level")
            .reindex(scale_order)
        )

        means = method_summary[
            "mean_roc_auc"
        ].to_numpy(dtype=float)

        standard_errors = method_summary[
            "se_roc_auc"
        ].to_numpy(dtype=float)

        errors_for_plot = np.nan_to_num(
            standard_errors,
            nan=0.0,
        )

        offset = (
            method_index
            - (number_of_methods - 1) / 2
        ) * bar_width

        bars = ax.bar(
            x + offset,
            means,
            width=bar_width,
            yerr=errors_for_plot,
            capsize=4,
            label=method,
            color=method_colors[method],
            edgecolor="black",
            linewidth=0.7,
        )

        # Show the number of matched repetitions above each bar.
        for row_index, bar in enumerate(bars):
            if np.isnan(means[row_index]):
                continue

            n_repetitions = method_summary.iloc[
                row_index
            ]["n_repetitions"]

            if pd.isna(n_repetitions):
                continue

            label_height = (
                means[row_index]
                + errors_for_plot[row_index]
                + 0.008
            )

            ax.text(
                bar.get_x() + bar.get_width() / 2,
                label_height,
                f"n={int(n_repetitions)}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90,
            )

    ax.axhline(
        0.5,
        linestyle="--",
        linewidth=1,
        label="Random classifier",
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        [
            GI_LABELS[level]
            for level in scale_order
        ]
    )

    ax.set_xlabel(
        "Epistatic genetic variance (G×G)",
        fontsize=12,
    )

    ax.set_ylabel(
        "ROC-AUC",
        fontsize=12,
    )

    ax.set_title(
        "Prediction performance across G×G interaction levels\n"
        f"N causal SNPs = {N_CAUSAL_REQUIRED}",
        fontsize=14,
    )

    plotted_upper_values = (
        summary["mean_roc_auc"]
        + summary["se_roc_auc"].fillna(0)
    )

    if plotted_upper_values.notna().any():
        upper_limit = min(
            1.0,
            max(
                0.85,
                plotted_upper_values.max() + 0.08,
            ),
        )
    else:
        upper_limit = 1.0

    ax.set_ylim(0.45, upper_limit)

    ax.grid(
        axis="y",
        linestyle=":",
        alpha=0.5,
    )

    ax.legend(
        frameon=False,
        loc="best",
    )

    fig.tight_layout()

    png_path = os.path.join(
        OUTPUT_DIR,
        "ROC_AUC_GxG_DRIP_LR_NN_PRSCS.png",
    )

    pdf_path = os.path.join(
        OUTPUT_DIR,
        "ROC_AUC_GxG_DRIP_LR_NN_PRSCS.pdf",
    )

    fig.savefig(
        png_path,
        dpi=FIGURE_DPI,
        bbox_inches="tight",
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"\nSaved PNG figure:\n{png_path}")
    print(f"Saved PDF figure:\n{pdf_path}")


# =============================================================================
# Save tables
# =============================================================================

def save_results(matched, summary):
    """Save raw matched values and summarized statistics."""
    raw_path = os.path.join(
        OUTPUT_DIR,
        "GxG_matched_repetition_ROC_AUC_values.csv",
    )

    summary_path = os.path.join(
        OUTPUT_DIR,
        "GxG_ROC_AUC_summary.csv",
    )

    matched.sort_values(
        [
            "gi_level",
            "method",
            "rep",
        ]
    ).to_csv(
        raw_path,
        index=False,
    )

    summary.sort_values(
        [
            "gi_level",
            "method",
        ]
    ).to_csv(
        summary_path,
        index=False,
    )

    print(f"\nSaved matched values:\n{raw_path}")
    print(f"Saved summary:\n{summary_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    ensure_output_directory()

    print("Loading DRIP + LR results...")
    drip_lr = load_drip_lr_results()

    print("Loading DRIP + NN results...")
    drip_nn = load_drip_nn_results()

    print("Loading PRS-CS results...")
    prscs = load_prscs_results()

    print("\nRows loaded:")
    print(f"  DRIP + LR: {len(drip_lr)}")
    print(f"  DRIP + NN: {len(drip_nn)}")
    print(f"  PRS-CS:    {len(prscs)}")

    combined = combine_results(
        drip_lr=drip_lr,
        drip_nn=drip_nn,
        prscs=prscs,
    )

    report_available_results(combined)

    matched = retain_matched_repetitions(combined)

    summary = summarize_across_repetitions(matched)

    print("\nFinal G×G summary:")
    print(
        summary.sort_values(
            [
                "gi_level",
                "method",
            ]
        ).to_string(index=False)
    )

    save_results(
        matched=matched,
        summary=summary,
    )

    create_gxg_bar_plot(summary)


if __name__ == "__main__":
    main()