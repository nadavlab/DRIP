#!/usr/bin/env python3

from datetime import datetime
from pathlib import Path
import re

import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional

###############################################################################
# Paths
###############################################################################

LR_LOG_DIR = Path("logs")

GWAS_DIR = Path(
    "/path/to/your/project/simulations/GWAS_results_DRIP_train"
)

NN_FILE = Path(
    "/path/to/your/project/simulations/GxG_NN_results/NN_results.csv"
)

OUTDIR = Path(
    "/path/to/your/project/simulations/runtime_comparison"
)

OUTDIR.mkdir(exist_ok=True, parents=True)


###############################################################################
# Helper functions
###############################################################################

def normalize_phenotype_name(name: str) -> str:
    """
    Normalize phenotype names across LR, GWAS, and NN outputs.

    Example:
        rep3_chr1_N100_GI_30_... -> chr1_N100_GI_30_...
    """
    name = str(name).strip()
    return re.sub(r"^rep\d+_", "", name)


def extract_rep(name: str) -> Optional[int]:
    """Extract repetition number from a name such as rep3_chr1_..."""
    match = re.search(r"(?:^|_)rep(\d+)(?:_|$)", str(name))
    return int(match.group(1)) if match else None


###############################################################################
# Collect running times
###############################################################################

rows = []


###############################################################################
# DRIP-LR
###############################################################################

for log in LR_LOG_DIR.glob("LR_chr1_*.out"):
    text = log.read_text(errors="ignore")

    if "All done in" not in text:
        continue

    phenotype_match = re.search(r"Done ([^:]+):", text)
    runtime_match = re.search(
        r"All done in ([0-9.]+) minutes",
        text,
    )

    if phenotype_match is None or runtime_match is None:
        continue

    original_name = phenotype_match.group(1).strip()

    rows.append(
        {
            "method": "DRIP-LR",
            "run": original_name,
            "rep": extract_rep(original_name),
            "phenotype": normalize_phenotype_name(original_name),
            "time_minutes": float(runtime_match.group(1)),
            "source_file": str(log),
        }
    )


###############################################################################
# GWAS
###############################################################################

time_format = "%a %b %d %H:%M:%S %Y"

for log in GWAS_DIR.rglob("*.log"):
    text = log.read_text(errors="ignore")

    if "End time:" not in text:
        continue

    start_match = re.search(r"Start time:\s*(.+)", text)
    end_match = re.search(r"End time:\s*(.+)", text)
    phenotype_match = re.search(
        r"--pheno-name\s+([^\n\r ]+)",
        text,
    )

    if (
        start_match is None
        or end_match is None
        or phenotype_match is None
    ):
        continue

    try:
        start_time = datetime.strptime(
            start_match.group(1).strip(),
            time_format,
        )
        end_time = datetime.strptime(
            end_match.group(1).strip(),
            time_format,
        )
    except ValueError as error:
        print(f"WARNING: could not parse time in {log}: {error}")
        continue

    runtime_minutes = (
        end_time - start_time
    ).total_seconds() / 60

    original_name = phenotype_match.group(1).strip()

    rows.append(
        {
            "method": "GWAS",
            "run": original_name,
            "rep": extract_rep(original_name),
            "phenotype": normalize_phenotype_name(original_name),
            "time_minutes": runtime_minutes,
            "source_file": str(log),
        }
    )


###############################################################################
# DRIP-NN
###############################################################################

if not NN_FILE.is_file():
    raise FileNotFoundError(
        f"NN results file was not found: {NN_FILE}"
    )

nn = pd.read_csv(NN_FILE)

required_nn_columns = {"pheno", "time_minutes"}
missing_nn_columns = required_nn_columns.difference(nn.columns)

if missing_nn_columns:
    raise ValueError(
        "The NN results file is missing columns: "
        f"{sorted(missing_nn_columns)}"
    )

for _, row in nn.iterrows():
    original_name = str(row["pheno"]).strip()

    runtime_minutes = pd.to_numeric(
        row["time_minutes"],
        errors="coerce",
    )

    if pd.isna(runtime_minutes):
        continue

    rep = None

    if "rep" in nn.columns:
        rep_value = pd.to_numeric(
            row["rep"],
            errors="coerce",
        )

        if pd.notna(rep_value):
            rep = int(rep_value)

    if rep is None:
        rep = extract_rep(original_name)

    rows.append(
        {
            "method": "DRIP-NN",
            "run": original_name,
            "rep": rep,
            "phenotype": normalize_phenotype_name(original_name),
            "time_minutes": float(runtime_minutes),
            "source_file": str(NN_FILE),
        }
    )


###############################################################################
# Build complete runtime table
###############################################################################

runtime_all_df = pd.DataFrame(rows)

if runtime_all_df.empty:
    raise ValueError("No valid runtime records were found.")

runtime_all_df["time_minutes"] = pd.to_numeric(
    runtime_all_df["time_minutes"],
    errors="coerce",
)

runtime_all_df = runtime_all_df.dropna(
    subset=["method", "phenotype", "time_minutes"]
)

runtime_all_df.to_csv(
    OUTDIR / "running_times_all_records.csv",
    index=False,
)


###############################################################################
# Keep only phenotypes present in all three methods
###############################################################################

required_methods = {"DRIP-LR", "GWAS", "DRIP-NN"}

methods_per_phenotype = (
    runtime_all_df
    .groupby("phenotype")["method"]
    .agg(lambda values: set(values))
)

common_phenotypes = methods_per_phenotype[
    methods_per_phenotype.apply(
        lambda methods: required_methods.issubset(methods)
    )
].index

runtime_df = runtime_all_df.loc[
    runtime_all_df["phenotype"].isin(common_phenotypes)
].copy()

if runtime_df.empty:
    print("No phenotype was found in all three methods.")
    print()
    print("Unique phenotypes by method:")
    print(
        runtime_all_df
        .groupby("method")["phenotype"]
        .nunique()
    )
    raise SystemExit(1)

runtime_df = runtime_df.sort_values(
    ["phenotype", "rep", "method"],
    na_position="last",
)

runtime_df.to_csv(
    OUTDIR / "running_times_common_phenotypes.csv",
    index=False,
)

pd.DataFrame(
    {"phenotype": sorted(common_phenotypes)}
).to_csv(
    OUTDIR / "common_phenotypes.csv",
    index=False,
)


###############################################################################
# Summary
###############################################################################

summary = (
    runtime_df
    .groupby("method")["time_minutes"]
    .agg(
        mean="mean",
        std="std",
        count="count",
    )
    .reset_index()
)

summary["SE"] = summary["std"] / summary["count"].pow(0.5)

summary.to_csv(
    OUTDIR / "running_times_summary_common_phenotypes.csv",
    index=False,
)


###############################################################################
# Plot
###############################################################################

plt.figure(figsize=(6, 6))

method_colors = {
    "DRIP-LR": "#4CAF50",   # green
    "DRIP-NN": "#2E7D32",   # darker green
    "GWAS": "#87CEFA",      # same light blue as PRS-CS
}

colors = [method_colors[m] for m in summary["method"]]

plt.bar(
    summary["method"],
    summary["mean"],
    color=colors,
    yerr=summary["SE"],
    capsize=8,
    edgecolor="black",
    linewidth=0.7,
)

plt.ylabel("Running time (minutes)")
plt.title(
    "Mean runtime for phenotypes available\n"
    "in all three methods"
)

plt.tight_layout()

plt.savefig(
    OUTDIR / "runtime_comparison_common_phenotypes.png",
    dpi=300,
)

plt.close()


###############################################################################
# Report
###############################################################################

print()
print("Phenotypes found in all three methods:")
print("----------------------------------------")

for phenotype in sorted(common_phenotypes):
    print(phenotype)

print()
print(f"Number of common phenotypes: {len(common_phenotypes)}")
print(f"Number of included runtime records: {len(runtime_df)}")

print()
print("Number of records by method:")
print(runtime_df["method"].value_counts().sort_index())

print()
print("Runtime summary:")
print(summary)
