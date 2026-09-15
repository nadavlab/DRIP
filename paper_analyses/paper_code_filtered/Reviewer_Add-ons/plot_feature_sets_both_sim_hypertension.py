#!/usr/bin/env python3
"""
Create one combined publication figure comparing LR feature sets.

Figure structure
----------------
A. Simulation results
   - Individual feature components
   - Combined feature sets

B. Hypertension results
   - Individual feature components
   - Combined feature sets

Only the panel letters A/B and the two subplot titles are shown above the plots.

Example
-------
python3 plot_feature_sets_both_sim_hypertension.py \
  --simulation-base \
  /path/to/your/project//simulations/LR_basic_sim_feature_sets \
  --hypertension-metrics \
  /path/to/your/project//our_model/PCA/logistic_regression/Hypertension/reviewer_feature_set_comparison_metrics.csv \
  --outdir \
  /path/to/your/project//combined_feature_set_figure
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =============================================================================
# Feature-set definitions
# =============================================================================

SIMULATION_FEATURE_LABELS = {
    "01_created_PCs_only": "DRIP PCs\nonly",
    "02_cov_only_no_created_PCs": "Covariates +\nUKB PCs",
    "03_created_PCs_plus_non_UKB_PC_cov": "DRIP PCs +\ncovariates",
    "04_UKB_PCs_only": "UKB PCs\nonly",
    "05_non_PC_cov_only": "Covariates\nonly",
}


SIMULATION_COMPARISON_PANELS = [
    (
        "Individual feature components",
        [
            "01_created_PCs_only",
            "04_UKB_PCs_only",
            "05_non_PC_cov_only",
        ],
    ),
    (
        "Combined feature sets",
        [
            "03_created_PCs_plus_non_UKB_PC_cov",
            "02_cov_only_no_created_PCs",
        ],
    ),
]


HYPERTENSION_FEATURE_LABELS = {
    "01_cov_only": "Covariates\nonly",
    "02_ukb_40_pcs_only": "UKB 40 PCs\nonly",
    "03_cov_plus_ukb_40_pcs": "Covariates +\nUKB 40 PCs",
    "04_our_new_pcs_only": "DRIP PCs\nonly",
    "05_our_new_pcs_plus_cov": "DRIP PCs +\ncovariates",
}


HYPERTENSION_COMPARISON_PANELS = [
    (
        "Individual feature components",
        [
            "04_our_new_pcs_only",
            "02_ukb_40_pcs_only",
            "01_cov_only",
        ],
    ),
    (
        "Combined feature sets",
        [
            "05_our_new_pcs_plus_cov",
            "03_cov_plus_ukb_40_pcs",
        ],
    ),
]


# Same colors are used for equivalent feature sets in both panels.
FEATURE_COLORS = {
    "created_pcs_only": "#2b6cb0",
    "ukb_pcs_only": "#2f855a",
    "covariates_only": "#b7791f",
    "created_pcs_plus_covariates": "#805ad5",
    "ukb_pcs_plus_covariates": "#c53030",
}


# =============================================================================
# Arguments
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a combined A/B figure comparing LR feature sets in "
            "simulation and Hypertension data."
        )
    )

    parser.add_argument(
        "--simulation-base",
        required=True,
        type=Path,
        help="Directory containing rep1, rep2, ... simulation folders.",
    )

    parser.add_argument(
        "--hypertension-metrics",
        required=True,
        type=Path,
        help=(
            "Hypertension metrics CSV containing pheno, rep, DRM, "
            "feature_set, and metric columns."
        ),
    )

    parser.add_argument(
        "--simulation",
        default=None,
        help=(
            "Simulation value from the simulation column. "
            "May be omitted when only one simulation is available."
        ),
    )

    parser.add_argument(
        "--pheno",
        default="Hypertension",
        help="Phenotype to select from the Hypertension metrics file.",
    )

    parser.add_argument(
        "--drm",
        default=None,
        help=(
            "DRM value to select from the Hypertension metrics file. "
            "May be omitted when only one DRM is available."
        ),
    )

    parser.add_argument(
        "--metric",
        default="ROC_AUC",
        help="Metric column to summarize and plot. Default: ROC_AUC.",
    )

    parser.add_argument(
        "--outdir",
        default=Path("combined_feature_set_figure"),
        type=Path,
        help="Output directory.",
    )

    parser.add_argument(
        "--output-name",
        default="combined_simulation_hypertension_feature_sets",
        help="Output filename without extension.",
    )

    parser.add_argument(
        "--ymin",
        type=float,
        default=None,
        help="Optional fixed lower y-axis limit.",
    )

    parser.add_argument(
        "--ymax",
        type=float,
        default=None,
        help="Optional fixed upper y-axis limit.",
    )

    return parser.parse_args()


# =============================================================================
# Simulation data
# =============================================================================

def read_simulation_metrics(
    base: Path,
    metric: str,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []

    for metrics_path in sorted(base.glob("rep*/**/metrics.csv")):
        feature_set = metrics_path.parent.name

        if feature_set not in SIMULATION_FEATURE_LABELS:
            continue

        df = pd.read_csv(metrics_path)

        required_columns = {"simulation", metric}
        missing_columns = required_columns - set(df.columns)

        if missing_columns:
            missing_text = ", ".join(sorted(missing_columns))
            raise ValueError(
                f"{metrics_path} is missing required columns: {missing_text}"
            )

        df = df.copy()
        df["feature_set"] = feature_set
        df["metrics_path"] = str(metrics_path)

        rows.append(df)

    if not rows:
        raise FileNotFoundError(
            f"No recognized metrics.csv files were found under {base}"
        )

    return pd.concat(rows, ignore_index=True)


def summarize_simulation_metrics(
    metrics: pd.DataFrame,
    metric: str,
) -> pd.DataFrame:
    summary = (
        metrics.groupby(
            ["simulation", "feature_set"],
            as_index=False,
        )
        .agg(
            mean=(metric, "mean"),
            sd=(metric, "std"),
            n_reps=(metric, "count"),
        )
        .sort_values(["simulation", "feature_set"])
    )

    summary["sd"] = summary["sd"].fillna(0.0)
    summary["se"] = summary["sd"] / np.sqrt(summary["n_reps"])

    summary["feature_label"] = summary["feature_set"].map(
        SIMULATION_FEATURE_LABELS
    )

    return summary


def select_simulation(
    summary: pd.DataFrame,
    requested_simulation: str | None,
) -> pd.DataFrame:
    available_simulations = sorted(
        summary["simulation"].dropna().astype(str).unique()
    )

    if not available_simulations:
        raise ValueError("No simulation values were found.")

    if requested_simulation is None:
        if len(available_simulations) > 1:
            available_text = "\n  - ".join(available_simulations)

            raise ValueError(
                "Multiple simulations were found. Select one using "
                "--simulation.\n\n"
                f"Available simulations:\n  - {available_text}"
            )

        requested_simulation = available_simulations[0]

    subset = summary[
        summary["simulation"].astype(str) == str(requested_simulation)
    ].copy()

    if subset.empty:
        available_text = "\n  - ".join(available_simulations)

        raise ValueError(
            f"Simulation {requested_simulation!r} was not found.\n\n"
            f"Available simulations:\n  - {available_text}"
        )

    return subset


# =============================================================================
# Hypertension data
# =============================================================================

def read_and_summarize_hypertension(
    metrics_path: Path,
    metric: str,
) -> pd.DataFrame:
    df = pd.read_csv(metrics_path)

    required_columns = {
        "pheno",
        "rep",
        "DRM",
        "feature_set",
        metric,
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))

        raise ValueError(
            f"{metrics_path} is missing required columns: {missing_text}"
        )

    df = df[
        df["feature_set"].isin(HYPERTENSION_FEATURE_LABELS)
    ].copy()

    if df.empty:
        raise ValueError(
            f"No recognized feature-set values were found in {metrics_path}"
        )

    summary = (
        df.groupby(
            ["pheno", "DRM", "feature_set"],
            as_index=False,
        )
        .agg(
            mean=(metric, "mean"),
            sd=(metric, "std"),
            n_reps=(metric, "count"),
        )
        .sort_values(["pheno", "DRM", "feature_set"])
    )

    summary["sd"] = summary["sd"].fillna(0.0)
    summary["se"] = summary["sd"] / np.sqrt(summary["n_reps"])

    summary["feature_label"] = summary["feature_set"].map(
        HYPERTENSION_FEATURE_LABELS
    )

    return summary


def select_hypertension_data(
    summary: pd.DataFrame,
    pheno: str,
    requested_drm: str | None,
) -> pd.DataFrame:
    pheno_subset = summary[
        summary["pheno"].astype(str).str.lower() == str(pheno).lower()
    ].copy()

    if pheno_subset.empty:
        available_phenotypes = sorted(
            summary["pheno"].dropna().astype(str).unique()
        )

        raise ValueError(
            f"Phenotype {pheno!r} was not found. "
            f"Available phenotypes: {', '.join(available_phenotypes)}"
        )

    available_drms = sorted(
        pheno_subset["DRM"].dropna().astype(str).unique()
    )

    if requested_drm is None:
        if len(available_drms) > 1:
            available_text = "\n  - ".join(available_drms)

            raise ValueError(
                "Multiple DRM values were found. Select one using --drm.\n\n"
                f"Available DRM values:\n  - {available_text}"
            )

        requested_drm = available_drms[0]

    subset = pheno_subset[
        pheno_subset["DRM"].astype(str) == str(requested_drm)
    ].copy()

    if subset.empty:
        available_text = "\n  - ".join(available_drms)

        raise ValueError(
            f"DRM {requested_drm!r} was not found.\n\n"
            f"Available DRM values:\n  - {available_text}"
        )

    return subset


# =============================================================================
# Plot preparation
# =============================================================================

def ordered_panel_data(
    subset: pd.DataFrame,
    feature_sets: list[str],
    feature_labels: dict[str, str],
    panel_name: str,
    dataset_name: str,
) -> pd.DataFrame:
    available_features = set(subset["feature_set"])

    missing_features = [
        feature_set
        for feature_set in feature_sets
        if feature_set not in available_features
    ]

    if missing_features:
        missing_labels = ", ".join(
            feature_labels[feature_set].replace("\n", " ")
            for feature_set in missing_features
        )

        print(
            f"WARNING: missing feature sets for {dataset_name}, "
            f"{panel_name}: {missing_labels}"
        )

    ordered = (
        subset.set_index("feature_set")
        .reindex(feature_sets)
        .dropna(subset=["mean"])
        .reset_index()
    )

    if ordered.empty:
        print(
            f"WARNING: no feature sets available for "
            f"{dataset_name}, {panel_name}"
        )

    return ordered


def get_feature_color(
    feature_set: str,
    dataset_type: str,
) -> str:
    if dataset_type == "simulation":
        color_mapping = {
            "01_created_PCs_only": FEATURE_COLORS["created_pcs_only"],
            "04_UKB_PCs_only": FEATURE_COLORS["ukb_pcs_only"],
            "05_non_PC_cov_only": FEATURE_COLORS["covariates_only"],
            "03_created_PCs_plus_non_UKB_PC_cov": (
                FEATURE_COLORS["created_pcs_plus_covariates"]
            ),
            "02_cov_only_no_created_PCs": (
                FEATURE_COLORS["ukb_pcs_plus_covariates"]
            ),
        }

    elif dataset_type == "hypertension":
        color_mapping = {
            "04_our_new_pcs_only": FEATURE_COLORS["created_pcs_only"],
            "02_ukb_40_pcs_only": FEATURE_COLORS["ukb_pcs_only"],
            "01_cov_only": FEATURE_COLORS["covariates_only"],
            "05_our_new_pcs_plus_cov": (
                FEATURE_COLORS["created_pcs_plus_covariates"]
            ),
            "03_cov_plus_ukb_40_pcs": (
                FEATURE_COLORS["ukb_pcs_plus_covariates"]
            ),
        }

    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    return color_mapping.get(feature_set, "#808080")


def calculate_global_y_limits(
    all_panels: list[pd.DataFrame],
    requested_ymin: float | None,
    requested_ymax: float | None,
) -> tuple[float, float]:
    non_empty = [
        panel
        for panel in all_panels
        if not panel.empty
    ]

    if not non_empty:
        raise ValueError("No data are available to plot.")

    combined = pd.concat(non_empty, ignore_index=True)

    observed_lower = float(
        (combined["mean"] - combined["se"]).min()
    )

    observed_upper = float(
        (combined["mean"] + combined["se"]).max()
    )

    if requested_ymin is None:
        lower = max(0.0, observed_lower - 0.03)
    else:
        lower = requested_ymin

    if requested_ymax is None:
        upper = min(1.0, observed_upper + 0.05)
    else:
        upper = requested_ymax

    if lower >= upper:
        raise ValueError(
            f"Invalid y-axis limits: ymin={lower}, ymax={upper}"
        )

    if upper - lower < 0.08:
        center = (upper + lower) / 2
        lower = max(0.0, center - 0.04)
        upper = min(1.0, center + 0.04)

    return lower, upper


# =============================================================================
# Plotting
# =============================================================================

def plot_panel(
    ax,
    ordered: pd.DataFrame,
    panel_title: str,
    dataset_type: str,
    y_limits: tuple[float, float],
    show_y_axis: bool,
) -> None:
    if ordered.empty:
        ax.set_title(
            panel_title,
            fontsize=12,
            fontweight="bold",
        )

        ax.text(
            0.5,
            0.5,
            "No data",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

        ax.set_xticks([])
        ax.set_ylim(*y_limits)
        return

    x_positions = np.arange(len(ordered))

    colors = [
        get_feature_color(feature_set, dataset_type)
        for feature_set in ordered["feature_set"]
    ]

    ax.bar(
        x_positions,
        ordered["mean"],
        yerr=ordered["se"],
        capsize=5,
        color=colors,
        edgecolor="#222222",
        linewidth=0.8,
        width=0.72,
        zorder=3,
    )

    annotation_offset = max(
        0.002,
        (y_limits[1] - y_limits[0]) * 0.018,
    )

    for x_position, (_, row) in zip(
        x_positions,
        ordered.iterrows(),
    ):
        ax.text(
            x_position,
            row["mean"] + row["se"] + annotation_offset,
            f"{row['mean']:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_xticks(x_positions)

    ax.set_xticklabels(
        ordered["feature_label"],
        fontsize=9,
    )

    ax.set_title(
        panel_title,
        fontsize=12,
        fontweight="bold",
        pad=10,
    )

    ax.set_ylim(*y_limits)

    ax.grid(
        axis="y",
        alpha=0.25,
        linewidth=0.8,
        zorder=0,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if not show_y_axis:
        ax.spines["left"].set_visible(False)

        ax.tick_params(
            axis="y",
            left=False,
            labelleft=False,
        )


def create_combined_figure(
    simulation_subset: pd.DataFrame,
    hypertension_subset: pd.DataFrame,
    metric: str,
    output_prefix: Path,
    requested_ymin: float | None,
    requested_ymax: float | None,
) -> None:
    simulation_panels = [
        ordered_panel_data(
            subset=simulation_subset,
            feature_sets=feature_sets,
            feature_labels=SIMULATION_FEATURE_LABELS,
            panel_name=panel_name,
            dataset_name="simulation",
        )
        for panel_name, feature_sets in SIMULATION_COMPARISON_PANELS
    ]

    hypertension_panels = [
        ordered_panel_data(
            subset=hypertension_subset,
            feature_sets=feature_sets,
            feature_labels=HYPERTENSION_FEATURE_LABELS,
            panel_name=panel_name,
            dataset_name="Hypertension",
        )
        for panel_name, feature_sets in HYPERTENSION_COMPARISON_PANELS
    ]

    y_limits = calculate_global_y_limits(
        all_panels=simulation_panels + hypertension_panels,
        requested_ymin=requested_ymin,
        requested_ymax=requested_ymax,
    )

    fig, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(12.5, 8.4),
        sharey=True,
        gridspec_kw={
            "width_ratios": [3, 2],
            "hspace": 0.42,
            "wspace": 0.12,
        },
    )

    # -------------------------------------------------------------------------
    # Panel A: simulation
    # -------------------------------------------------------------------------

    for column_index, (
        (panel_title, _),
        ordered,
    ) in enumerate(
        zip(
            SIMULATION_COMPARISON_PANELS,
            simulation_panels,
        )
    ):
        plot_panel(
            ax=axes[0, column_index],
            ordered=ordered,
            panel_title=panel_title,
            dataset_type="simulation",
            y_limits=y_limits,
            show_y_axis=(column_index == 0),
        )

    # -------------------------------------------------------------------------
    # Panel B: Hypertension
    # -------------------------------------------------------------------------

    for column_index, (
        (panel_title, _),
        ordered,
    ) in enumerate(
        zip(
            HYPERTENSION_COMPARISON_PANELS,
            hypertension_panels,
        )
    ):
        plot_panel(
            ax=axes[1, column_index],
            ordered=ordered,
            panel_title=panel_title,
            dataset_type="hypertension",
            y_limits=y_limits,
            show_y_axis=(column_index == 0),
        )

    # Y-axis labels.
    axes[0, 0].set_ylabel(
        f"{metric} mean ± SE",
        fontsize=11,
    )

    axes[1, 0].set_ylabel(
        f"{metric} mean ± SE",
        fontsize=11,
    )

    # Panel letters only.
    axes[0, 0].text(
        -0.16,
        1.07,
        "A",
        transform=axes[0, 0].transAxes,
        fontsize=18,
        fontweight="bold",
        ha="left",
        va="top",
    )

    axes[1, 0].text(
        -0.16,
        1.07,
        "B",
        transform=axes[1, 0].transAxes,
        fontsize=18,
        fontweight="bold",
        ha="left",
        va="top",
    )

    fig.subplots_adjust(
        left=0.11,
        right=0.98,
        top=0.96,
        bottom=0.10,
    )

    png_path = output_prefix.with_suffix(".png")
    pdf_path = output_prefix.with_suffix(".pdf")
    svg_path = output_prefix.with_suffix(".svg")

    fig.savefig(
        png_path,
        dpi=600,
        bbox_inches="tight",
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    fig.savefig(
        svg_path,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"WROTE: {png_path}")
    print(f"WROTE: {pdf_path}")
    print(f"WROTE: {svg_path}")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()

    args.outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------------------
    # Read and summarize simulation results
    # -------------------------------------------------------------------------

    simulation_metrics = read_simulation_metrics(
        base=args.simulation_base,
        metric=args.metric,
    )

    simulation_summary = summarize_simulation_metrics(
        metrics=simulation_metrics,
        metric=args.metric,
    )

    simulation_subset = select_simulation(
        summary=simulation_summary,
        requested_simulation=args.simulation,
    )

    # -------------------------------------------------------------------------
    # Read and summarize Hypertension results
    # -------------------------------------------------------------------------

    hypertension_summary = read_and_summarize_hypertension(
        metrics_path=args.hypertension_metrics,
        metric=args.metric,
    )

    hypertension_subset = select_hypertension_data(
        summary=hypertension_summary,
        pheno=args.pheno,
        requested_drm=args.drm,
    )

    # -------------------------------------------------------------------------
    # Save the values used in the figure
    # -------------------------------------------------------------------------

    simulation_output = simulation_subset.copy()
    simulation_output.insert(0, "figure_panel", "A")
    simulation_output.insert(1, "dataset", "Simulation")

    hypertension_output = hypertension_subset.copy()
    hypertension_output.insert(0, "figure_panel", "B")
    hypertension_output.insert(1, "dataset", "Hypertension")

    combined_summary = pd.concat(
        [
            simulation_output,
            hypertension_output,
        ],
        ignore_index=True,
        sort=False,
    )

    summary_path = (
        args.outdir
        / f"{args.output_name}_summary.csv"
    )

    combined_summary.to_csv(
        summary_path,
        index=False,
    )

    print(f"WROTE: {summary_path}")

    # -------------------------------------------------------------------------
    # Create combined figure
    # -------------------------------------------------------------------------

    output_prefix = (
        args.outdir
        / args.output_name
    )

    create_combined_figure(
        simulation_subset=simulation_subset,
        hypertension_subset=hypertension_subset,
        metric=args.metric,
        output_prefix=output_prefix,
        requested_ymin=args.ymin,
        requested_ymax=args.ymax,
    )

    print(
        f"Finished. Outputs were written to: "
        f"{args.outdir.resolve()}"
    )


if __name__ == "__main__":
    main()