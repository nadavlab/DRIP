#!/usr/bin/env python3
"""
Calculate correlations among PRSice-2, PRS-CS, and DRIP scores across repetitions.

For each repetition, the script:
1. Reads the PRSice-2 score file.
2. Reads the PRS-CS score file.
3. Reads the DRIP prediction file.
4. Merges/alines the three scores.
5. Calculates the three possible pairwise Pearson correlations.
6. Saves the merged data.

Across repetitions, it:
7. Calculates the mean correlation and standard error.
8. Creates a bar plot showing mean ± SE.

Important
---------
The DRIP input must be predictions_with_ids.csv and must contain FID or IID.
DRIP is always merged by participant ID. Row-order alignment is not allowed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PAIR_DEFINITIONS = [
    ("PRSice-2 vs PRS-CS", "PRSice_score", "PRScs_score"),
    ("PRSice-2 vs DRIP", "PRSice_score", "DRIP_score"),
    ("PRS-CS vs DRIP", "PRScs_score", "DRIP_score"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate pairwise correlations among three PRS methods."
    )
    parser.add_argument(
        "--data-base",
        type=Path,
        default=Path(
            "/path/to/your/project//"
            "impoving_PRS/data"
        ),
        help="Base data directory.",
    )
    parser.add_argument(
        "--phenotype",
        default="Hypertension",
        help="Phenotype directory/name. Default: Hypertension",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5],
        help="Repetition numbers. Default: 1 2 3 4 5",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory. Default: "
            "<data-base>/PRS_score_correlations/<phenotype>"
        ),
    )
    parser.add_argument(
        "--no-save-merged",
        dest="save_merged",
        action="store_false",
        help="Do not save the merged score table for each repetition.",
    )
    parser.set_defaults(save_merged=True)
    return parser.parse_args()


def remove_unnamed_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove index columns such as 'Unnamed: 0'."""
    return df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed:")].copy()


def normalize_id(series: pd.Series) -> pd.Series:
    """
    Convert IDs to consistent strings.

    The regex removes a trailing '.0' that can appear when integer IDs are read
    as floating-point values.
    """
    return (
        series.astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )


def choose_id_column(
    df: pd.DataFrame,
    candidates: list[str],
    file_label: str,
    required: bool = True,
) -> Optional[str]:
    """Return the first available ID column from a list of candidates."""
    for column in candidates:
        if column in df.columns:
            return column

    if required:
        raise ValueError(
            f"Could not find an ID column in {file_label}. "
            f"Tried: {candidates}. Available columns: {list(df.columns)}"
        )
    return None


def validate_unique_ids(df: pd.DataFrame, file_label: str) -> None:
    """Stop if participant IDs are duplicated."""
    duplicated = df["IID_merge"].duplicated(keep=False)
    if duplicated.any():
        examples = df.loc[duplicated, "IID_merge"].head(10).tolist()
        raise ValueError(
            f"{file_label} contains duplicated participant IDs. "
            f"Examples: {examples}"
        )


def find_phenotype_column(
    df: pd.DataFrame,
    phenotype: str,
) -> Optional[str]:
    """Find a phenotype column without relying on exact capitalization."""
    phenotype_lower = phenotype.lower()
    for column in df.columns:
        if str(column).lower() == phenotype_lower:
            return column
    return None


def read_prsice(path: Path, phenotype: str) -> pd.DataFrame:
    """Read and standardize the PRSice-2 score file."""
    if not path.exists():
        raise FileNotFoundError(f"PRSice-2 file not found: {path}")

    df = remove_unnamed_columns(pd.read_csv(path))

    id_column = choose_id_column(
        df,
        candidates=["IID", "IID_x", "IID_y", "FID"],
        file_label=str(path),
    )

    if "PRS" not in df.columns:
        raise ValueError(
            f"Column 'PRS' was not found in {path}. "
            f"Available columns: {list(df.columns)}"
        )

    # Keep only samples used in the PRSice regression, when this information
    # is available.
    if "In_Regression" in df.columns:
        in_regression = (
            df["In_Regression"]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin(["yes", "true", "1"])
        )
        if in_regression.any():
            df = df.loc[in_regression].copy()

    phenotype_column = find_phenotype_column(df, phenotype)

    keep_columns = [id_column, "PRS"]
    if phenotype_column is not None:
        keep_columns.append(phenotype_column)

    result = df[keep_columns].copy()
    result["IID_merge"] = normalize_id(result[id_column])
    result["PRSice_score"] = pd.to_numeric(result["PRS"], errors="coerce")

    output_columns = ["IID_merge", "PRSice_score"]
    if phenotype_column is not None:
        result["PRSice_y"] = pd.to_numeric(
            result[phenotype_column], errors="coerce"
        )
        output_columns.append("PRSice_y")

    result = result[output_columns]
    validate_unique_ids(result, f"PRSice-2 file {path}")
    return result


def read_prscs(path: Path, phenotype: str) -> pd.DataFrame:
    """Read and standardize the PRS-CS score file."""
    if not path.exists():
        raise FileNotFoundError(f"PRS-CS file not found: {path}")

    df = remove_unnamed_columns(pd.read_csv(path))

    id_column = choose_id_column(
        df,
        candidates=["IID", "IID_x", "IID_y", "FID"],
        file_label=str(path),
    )

    if "SUM_SCORESUM" not in df.columns:
        raise ValueError(
            f"Column 'SUM_SCORESUM' was not found in {path}. "
            f"Available columns: {list(df.columns)}"
        )

    phenotype_column = find_phenotype_column(df, phenotype)

    keep_columns = [id_column, "SUM_SCORESUM"]
    if phenotype_column is not None:
        keep_columns.append(phenotype_column)

    result = df[keep_columns].copy()
    result["IID_merge"] = normalize_id(result[id_column])
    result["PRScs_score"] = pd.to_numeric(
        result["SUM_SCORESUM"], errors="coerce"
    )

    output_columns = ["IID_merge", "PRScs_score"]
    if phenotype_column is not None:
        result["PRScs_y"] = pd.to_numeric(
            result[phenotype_column], errors="coerce"
        )
        output_columns.append("PRScs_y")

    result = result[output_columns]
    result["_PRScs_row_order"] = np.arange(len(result))

    validate_unique_ids(result, f"PRS-CS file {path}")
    return result


def read_drip(path: Path) -> tuple[pd.DataFrame, Optional[str]]:
    """Read and standardize the DRIP prediction file."""
    if not path.exists():
        raise FileNotFoundError(f"DRIP file not found: {path}")

    df = remove_unnamed_columns(pd.read_csv(path))

    if "prediction" not in df.columns:
        raise ValueError(
            f"Column 'prediction' was not found in {path}. "
            f"Available columns: {list(df.columns)}"
        )

    id_column = choose_id_column(
        df,
        candidates=["IID", "IID_x", "IID_y", "FID"],
        file_label=str(path),
        required=False,
    )

    result = pd.DataFrame()
    if id_column is not None:
        result["IID_merge"] = normalize_id(df[id_column])

    result["DRIP_score"] = pd.to_numeric(df["prediction"], errors="coerce")

    if "y_true" in df.columns:
        result["DRIP_y"] = pd.to_numeric(df["y_true"], errors="coerce")

    if id_column is not None:
        validate_unique_ids(result, f"DRIP file {path}")

    return result, id_column


def report_binary_agreement(
    merged: pd.DataFrame,
    reference_column: str,
    other_column: str,
    repetition: int,
) -> None:
    """
    Report phenotype agreement as a diagnostic.

    This is only a warning/check; it is not used to reorder participants.
    """
    if reference_column not in merged.columns or other_column not in merged.columns:
        return

    complete = merged[[reference_column, other_column]].dropna()
    if complete.empty:
        return

    agreement = (
        complete[reference_column].astype(float)
        == complete[other_column].astype(float)
    ).mean()

    message = (
        f"rep{repetition}: phenotype agreement between "
        f"{reference_column} and {other_column}: "
        f"{agreement:.2%} ({len(complete):,} samples)"
    )

    if agreement < 0.99:
        print(f"WARNING: {message}", file=sys.stderr)
    else:
        print(message)


def merge_scores_for_repetition(
    repetition: int,
    prsice_path: Path,
    prscs_path: Path,
    drip_path: Path,
    phenotype: str,
) -> tuple[pd.DataFrame, str]:
    """Read and merge all three score files strictly by participant ID."""
    prsice = read_prsice(prsice_path, phenotype)
    prscs = read_prscs(prscs_path, phenotype)
    drip, drip_id_column = read_drip(drip_path)

    if drip_id_column is None:
        raise ValueError(
            f"rep{repetition}: DRIP file has no FID/IID column: {drip_path}. "
            "This script requires predictions_with_ids.csv and does not allow "
            "row-order alignment."
        )

    n_prsice_before = len(prsice)
    n_prscs_before = len(prscs)
    n_drip_before = len(drip)

    merged = (
        prscs
        .merge(
            prsice,
            on="IID_merge",
            how="inner",
            validate="one_to_one",
        )
        .merge(
            drip,
            on="IID_merge",
            how="inner",
            validate="one_to_one",
        )
        .reset_index(drop=True)
    )

    if merged.empty:
        raise ValueError(
            f"rep{repetition}: no overlapping participant IDs across "
            "PRSice-2, PRS-CS, and DRIP."
        )

    alignment_note = "All three methods merged strictly by participant ID"

    print(
        f"rep{repetition}: PRSice-2={n_prsice_before:,}, "
        f"PRS-CS={n_prscs_before:,}, DRIP={n_drip_before:,}, "
        f"common participants={len(merged):,}"
    )
    print(f"rep{repetition}: {alignment_note}")

    report_binary_agreement(
        merged, "PRScs_y", "DRIP_y", repetition
    )
    report_binary_agreement(
        merged, "PRSice_y", "DRIP_y", repetition
    )
    report_binary_agreement(
        merged, "PRSice_y", "PRScs_y", repetition
    )

    columns_to_keep = [
        "IID_merge",
        "PRSice_score",
        "PRScs_score",
        "DRIP_score",
    ]

    for optional_column in ["PRSice_y", "PRScs_y", "DRIP_y"]:
        if optional_column in merged.columns:
            columns_to_keep.append(optional_column)

    print(
        merged[
            ["IID_merge", "PRSice_score", "PRScs_score", "DRIP_score"]
        ].head().to_string(index=False)
    )

    return merged[columns_to_keep], alignment_note

def calculate_correlations(
    merged: pd.DataFrame,
    repetition: int,
    alignment_note: str,
) -> list[dict]:
    """Calculate all three pairwise Pearson correlations."""
    records: list[dict] = []

    for comparison, score_1, score_2 in PAIR_DEFINITIONS:
        pair_data = merged[[score_1, score_2]].dropna()

        if len(pair_data) < 2:
            correlation = np.nan
        elif (
            pair_data[score_1].nunique(dropna=True) < 2
            or pair_data[score_2].nunique(dropna=True) < 2
        ):
            correlation = np.nan
        else:
            correlation = pair_data[score_1].corr(
                pair_data[score_2],
                method="pearson",
            )

        records.append(
            {
                "repetition": repetition,
                "comparison": comparison,
                "score_1": score_1,
                "score_2": score_2,
                "pearson_correlation": correlation,
                "n_samples": len(pair_data),
                "DRIP_alignment": alignment_note,
            }
        )

    return records


def summarize_correlations(per_rep: pd.DataFrame) -> pd.DataFrame:
    """Calculate mean, SD, and SE of correlations across repetitions."""
    summary = (
        per_rep.groupby("comparison", sort=False)["pearson_correlation"]
        .agg(
            mean_correlation="mean",
            sd_correlation="std",
            n_repetitions="count",
        )
        .reset_index()
    )

    summary["se_correlation"] = (
        summary["sd_correlation"]
        / np.sqrt(summary["n_repetitions"])
    )

    # Preserve the intended pair order.
    pair_order = [pair[0] for pair in PAIR_DEFINITIONS]
    summary["comparison"] = pd.Categorical(
        summary["comparison"],
        categories=pair_order,
        ordered=True,
    )
    summary = summary.sort_values("comparison").reset_index(drop=True)
    summary["comparison"] = summary["comparison"].astype(str)

    return summary[
        [
            "comparison",
            "mean_correlation",
            "sd_correlation",
            "se_correlation",
            "n_repetitions",
        ]
    ]


def create_bar_plot(summary: pd.DataFrame, output_dir: Path) -> None:
    """Create a bar plot of mean Pearson correlation ± standard error."""
    plot_data = summary.dropna(
        subset=["mean_correlation", "se_correlation"]
    ).copy()

    if plot_data.empty:
        raise ValueError("No valid correlations were available for plotting.")

    x = np.arange(len(plot_data))

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.bar(
        x,
        plot_data["mean_correlation"],
        yerr=plot_data["se_correlation"],
        capsize=6,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        plot_data["comparison"],
        rotation=20,
        ha="right",
    )
    ax.set_ylabel("Pearson correlation")
    ax.set_title(
        "Correlation between PRS scores across repetitions\n"
        "Mean ± standard error"
    )
    ax.axhline(0, linewidth=0.8)
    ax.set_ylim(-1.05, 1.05)

    for bar, value in zip(bars, plot_data["mean_correlation"]):
        if pd.notna(value):
            y_position = value + 0.04 if value >= 0 else value - 0.08
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y_position,
                f"{value:.3f}",
                ha="center",
                va="center",
            )

    fig.tight_layout()

    png_path = output_dir / "PRS_pairwise_correlations_mean_SE.png"
    pdf_path = output_dir / "PRS_pairwise_correlations_mean_SE.pdf"

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved plot: {png_path}")
    print(f"Saved plot: {pdf_path}")


def main() -> None:
    args = parse_args()

    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else args.data_base / "PRS_score_correlations" / args.phenotype
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    all_records: list[dict] = []

    for repetition in args.repetitions:
        prsice_path = (
            args.data_base
            / "PRSice_results"
            / args.phenotype
            / f"rep{repetition}"
            / f"{args.phenotype}_merge_PRS_score_with_pheno.csv"
        )

        prscs_path = (
            args.data_base
            / "PRScs"
            / "PRScs_results"
            / args.phenotype
            / f"rep{repetition}"
            / "final_PRS.final"
        )

        drip_path = (
            args.data_base
            / "our_model"
            / "PCA"
            / "logistic_regression"
            / args.phenotype
            / f"rep{repetition}"
            / "reviewer_feature_set_comparison"
            / "04_our_new_pcs_only"
            / "predictions_with_ids.csv"
        )

        try:
            merged, alignment_note = merge_scores_for_repetition(
                repetition=repetition,
                prsice_path=prsice_path,
                prscs_path=prscs_path,
                drip_path=drip_path,
                phenotype=args.phenotype,
            )

            if args.save_merged:
                merged_path = (
                    output_dir
                    / f"merged_PRS_scores_rep{repetition}.csv"
                )
                merged.to_csv(merged_path, index=False)
                print(f"Saved merged data: {merged_path}")

            all_records.extend(
                calculate_correlations(
                    merged=merged,
                    repetition=repetition,
                    alignment_note=alignment_note,
                )
            )

        except Exception as error:
            raise RuntimeError(
                f"Failed while processing repetition {repetition}: {error}"
            ) from error

    per_rep = pd.DataFrame(all_records)
    if per_rep.empty:
        raise RuntimeError("No correlations were calculated.")

    per_rep_path = output_dir / "correlations_by_repetition.csv"
    per_rep.to_csv(per_rep_path, index=False)

    summary = summarize_correlations(per_rep)
    summary_path = output_dir / "correlations_summary_mean_SE.csv"
    summary.to_csv(summary_path, index=False)

    create_bar_plot(summary, output_dir)

    print("\nPairwise correlations by repetition:")
    print(
        per_rep[
            [
                "repetition",
                "comparison",
                "pearson_correlation",
                "n_samples",
            ]
        ].to_string(index=False)
    )

    print("\nCorrelation summary:")
    print(summary.to_string(index=False))

    print(f"\nSaved per-repetition results: {per_rep_path}")
    print(f"Saved summary results: {summary_path}")


if __name__ == "__main__":
    main()
