#!/usr/bin/env python3
"""
Plot Hypertension ROC_AUC feature-set comparisons across repetitions.

Example:
  python3 plot_hypertension_feature_set_comparison.py \
    --metrics /path/to/your/project//our_model/PCA/logistic_regression/Hypertension/reviewer_feature_set_comparison_metrics.csv \
    --outdir /path/to/your/project//our_model/PCA/logistic_regression/Hypertension/plots
"""

from __future__ import annotations

import argparse
from pathlib import Path


FEATURE_LABELS = {
    "01_cov_only": "Covariates only",
    "02_ukb_40_pcs_only": "UKB 40 PCs only",
    "03_cov_plus_ukb_40_pcs": "Covariates + UKB 40 PCs",
    "04_our_new_pcs_only": "Created PCs only",
    "05_our_new_pcs_plus_cov": "Created PCs + covariates",
}

COMPARISON_PANELS = [
    (
        "Individual feature components",
        [
            "04_our_new_pcs_only",
            "02_ukb_40_pcs_only",
            "01_cov_only",
        ],
    ),
    (
        "Combined covariate sets",
        [
            "05_our_new_pcs_plus_cov",
            "03_cov_plus_ukb_40_pcs",
        ],
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create Hypertension feature-set ROC_AUC mean +/- SE plot."
    )
    parser.add_argument(
        "--metrics",
        required=True,
        type=Path,
        help="CSV file with columns pheno, rep, DRM, feature_set, and metric columns.",
    )
    parser.add_argument(
        "--outdir",
        default=Path("hypertension_feature_set_plots"),
        type=Path,
        help="Output directory for PNG/PDF plot and summary CSV.",
    )
    parser.add_argument(
        "--metric",
        default="ROC_AUC",
        help="Metric column to summarize and plot. Default: ROC_AUC.",
    )
    return parser.parse_args()


def safe_name(value: str) -> str:
    keep = []
    for char in value:
        keep.append(char if char.isalnum() or char in ("-", "_", ".") else "_")
    return "".join(keep).strip("_")[:180]


def summarize(metrics_path: Path, metric: str):
    import pandas as pd

    df = pd.read_csv(metrics_path)
    required_cols = {"pheno", "rep", "DRM", "feature_set", metric}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        missing = ", ".join(sorted(missing_cols))
        raise ValueError(f"{metrics_path} is missing required columns: {missing}")

    df = df[df["feature_set"].isin(FEATURE_LABELS)].copy()
    if df.empty:
        raise ValueError(f"No known feature_set values found in {metrics_path}")

    group_cols = ["pheno", "DRM", "feature_set"]
    summary = (
        df.groupby(group_cols, as_index=False)
        .agg(
            mean=(metric, "mean"),
            sd=(metric, "std"),
            n_reps=(metric, "count"),
        )
        .sort_values(group_cols)
    )
    summary["se"] = summary["sd"].fillna(0) / summary["n_reps"].pow(0.5)
    summary["feature_label"] = summary["feature_set"].map(FEATURE_LABELS)
    return summary


def ordered_panel_data(subset, feature_sets: list[str], panel_name: str):
    available_features = set(subset["feature_set"])
    missing_features = [
        feature_set for feature_set in feature_sets if feature_set not in available_features
    ]
    if missing_features:
        missing_labels = ", ".join(FEATURE_LABELS[feature_set] for feature_set in missing_features)
        print(f"WARNING: missing feature sets for {panel_name}: {missing_labels}")

    ordered = (
        subset.set_index("feature_set")
        .reindex(feature_sets)
        .dropna(subset=["mean"])
        .reset_index()
    )
    if ordered.empty:
        print(f"SKIP: no requested feature sets found for {panel_name}")
    return ordered


def plot_panel(ax, ordered, panel_title: str, colors: list[str]) -> None:
    if ordered.empty:
        ax.set_title(panel_title)
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
        return

    ax.bar(
        ordered["feature_label"],
        ordered["mean"],
        yerr=ordered["se"],
        capsize=6,
        color=colors,
        edgecolor="#222222",
        linewidth=0.8,
    )

    for index, row in ordered.iterrows():
        ax.text(
            index,
            row["mean"] + row["se"] + 0.002,
            f"{row['mean']:.3f}\nSE {row['se']:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_title(panel_title)
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", rotation=25)


def plot_combined(subset, metric: str, output_prefix: Path) -> bool:
    import matplotlib.pyplot as plt
    import pandas as pd

    pheno = str(subset["pheno"].iloc[0])
    drm = str(subset["DRM"].iloc[0])
    n_reps = int(subset["n_reps"].max())

    panels = [
        (panel_name, ordered_panel_data(subset, feature_sets, panel_name))
        for panel_name, feature_sets in COMPARISON_PANELS
    ]
    non_empty = [ordered for _, ordered in panels if not ordered.empty]
    if not non_empty:
        print(f"SKIP: no requested feature sets found for {output_prefix.name}")
        return False

    all_ordered = pd.concat(non_empty, ignore_index=True)
    lower = max(0.0, all_ordered["mean"].min() - all_ordered["se"].max() - 0.03)
    upper = min(1.0, all_ordered["mean"].max() + all_ordered["se"].max() + 0.05)
    if upper - lower < 0.08:
        center = (upper + lower) / 2
        lower = max(0.0, center - 0.04)
        upper = min(1.0, center + 0.04)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5.2),
        sharey=True,
        gridspec_kw={"width_ratios": [3, 2], "wspace": 0.16},
    )
    panel_colors = [
        ["#2b6cb0", "#2f855a", "#b7791f"],
        ["#805ad5", "#c53030"],
    ]

    for ax, (panel_name, ordered), colors in zip(axes, panels, panel_colors):
        plot_panel(ax, ordered, panel_name, colors[: len(ordered)])
        ax.set_ylim(lower, upper)

    axes[0].set_ylabel(f"{metric} mean +/- SE")
    axes[1].spines["left"].set_visible(False)
    axes[1].tick_params(axis="y", left=False)
    fig.suptitle(f"{pheno} {metric} mean across {n_reps} repetitions\n{drm}", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.9))

    png_path = output_prefix.parent / f"{output_prefix.name}.png"
    pdf_path = output_prefix.parent / f"{output_prefix.name}.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"WROTE: {png_path}")
    print(f"WROTE: {pdf_path}")
    return True


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    summary = summarize(args.metrics, args.metric)
    summary_path = args.outdir / f"{args.metric}_hypertension_feature_set_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"WROTE: {summary_path}")

    for (pheno, drm), subset in summary.groupby(["pheno", "DRM"]):
        output_prefix = args.outdir / f"{safe_name(str(pheno))}__{safe_name(str(drm))}__combined_feature_set_comparisons"
        plot_combined(subset, args.metric, output_prefix)

    print(f"Wrote summary and plots to: {args.outdir.resolve()}")


if __name__ == "__main__":
    main()
