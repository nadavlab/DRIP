#!/usr/bin/env python3
"""
Plot ROC_AUC comparisons across simulation repetitions for LR feature sets.

Example:
  python plot_feature_set_roc.py \
    --base /path/to/your/project//simulations/LR_basic_sim_feature_sets \
    --outdir roc_feature_set_plots
"""

from __future__ import annotations

import argparse
from pathlib import Path


FEATURE_LABELS = {
    "01_created_PCs_only": "Created PCs only",
    "02_cov_only_no_created_PCs": "Covariates incl. UKB PCs",
    "03_created_PCs_plus_non_UKB_PC_cov": "Created PCs + non-PC covariates",
    "04_UKB_PCs_only": "UKB PCs only",
    "05_non_PC_cov_only": "Non-PC covariates only",
}

COMPARISON_PANELS = [
    (
        "Individual feature components",
        [
            "01_created_PCs_only",
            "04_UKB_PCs_only",
            "05_non_PC_cov_only",
        ],
    ),
    (
        "Combined covariate sets",
        [
            "03_created_PCs_plus_non_UKB_PC_cov",
            "02_cov_only_no_created_PCs",
        ],
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create ROC_AUC mean +/- SE plots for LR feature-set simulations."
    )
    parser.add_argument(
        "--base",
        required=True,
        type=Path,
        help="Base directory containing rep1, rep2, ... folders.",
    )
    parser.add_argument(
        "--outdir",
        default=Path("roc_feature_set_plots"),
        type=Path,
        help="Output directory for PNG/PDF plots and summary CSV.",
    )
    parser.add_argument(
        "--metric",
        default="ROC_AUC",
        help="Metric column to summarize and plot. Default: ROC_AUC.",
    )
    return parser.parse_args()


def read_metrics(base: Path, metric: str) -> pd.DataFrame:
    import pandas as pd

    rows = []
    for metrics_path in sorted(base.glob("rep*/**/metrics.csv")):
        feature_set = metrics_path.parent.name
        if feature_set not in FEATURE_LABELS:
            continue

        df = pd.read_csv(metrics_path)
        if metric not in df.columns:
            raise ValueError(f"{metrics_path} does not contain column {metric!r}")

        df = df.copy()
        df["metrics_path"] = str(metrics_path)
        df["feature_set"] = feature_set
        rows.append(df)

    if not rows:
        raise FileNotFoundError(
            f"No metrics.csv files found under {base} with known feature-set folders."
        )

    return pd.concat(rows, ignore_index=True)


def summarize(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    group_cols = ["simulation", "feature_set"]
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


def safe_name(value: str) -> str:
    keep = []
    for char in value:
        keep.append(char if char.isalnum() or char in ("-", "_", ".") else "_")
    return "".join(keep).strip("_")[:180]


def ordered_panel_data(
    subset: pd.DataFrame,
    feature_sets: list[str],
    panel_name: str,
) -> pd.DataFrame:
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


def plot_panel(ax, ordered: pd.DataFrame, panel_title: str, colors: list[str]) -> None:
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


def plot_combined_comparisons(
    subset: pd.DataFrame,
    title: str,
    ylabel: str,
    output_prefix: Path,
) -> bool:
    import matplotlib.pyplot as plt
    import pandas as pd

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

    axes[0].set_ylabel(ylabel)
    axes[1].spines["left"].set_visible(False)
    axes[1].tick_params(axis="y", left=False)
    fig.suptitle(title, y=0.98)
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

    metrics = read_metrics(args.base, args.metric)
    summary = summarize(metrics, args.metric)
    summary.to_csv(args.outdir / f"{args.metric}_feature_set_summary.csv", index=False)
    print(f"WROTE: {args.outdir / f'{args.metric}_feature_set_summary.csv'}")

    for simulation, sim_summary in summary.groupby("simulation"):
        stem = safe_name(simulation)
        n_reps = int(sim_summary["n_reps"].max())
        plot_combined_comparisons(
            sim_summary,
            title=f"{args.metric} mean across {n_reps} repetitions\n{simulation}",
            ylabel=f"{args.metric} mean +/- SE",
            output_prefix=args.outdir / f"{stem}__combined_feature_set_comparisons",
        )

    print(f"Wrote summary and plots to: {args.outdir.resolve()}")


if __name__ == "__main__":
    main()
