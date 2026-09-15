#!/usr/bin/env python3
"""
Attach true participant IDs to existing DRIP prediction files using y_test chunks.

This script does NOT rerun PCA or logistic regression.

For each repetition, it:
1. Reads the original y_test pickle chunks in their saved order.
2. Extracts FID from each chunk.
3. Uses IID from the chunk when available; otherwise sets IID = FID.
4. Reads the existing predictions.csv.
5. Verifies that the saved y_true values exactly match the original y_test values.
6. Writes predictions_with_ids.csv.
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Attach FID/IID values to existing DRIP predictions using "
            "the original y_test pickle chunks."
        )
    )
    parser.add_argument(
        "--base",
        type=Path,
        default=Path(
            "/path/to/your/project/our_model/PCA"
        ),
    )
    parser.add_argument("--phenotype", default="Hypertension")
    parser.add_argument(
        "--repetitions", type=int, nargs="+", default=[1, 2, 3, 4, 5]
    )
    parser.add_argument("--feature-set", default="04_our_new_pcs_only")
    parser.add_argument("--output-name", default="predictions_with_ids.csv")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def normalize_id(series):
    return (
        series.astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )


def detect_phenotype_column(y_chunk):
    phenotype_columns = [
        column for column in y_chunk.columns if str(column) not in {"FID", "IID"}
    ]
    if len(phenotype_columns) != 1:
        raise ValueError(
            "Expected exactly one phenotype column after excluding FID/IID, "
            f"but found {len(phenotype_columns)}: {phenotype_columns}"
        )
    return phenotype_columns[0]


def load_ids_and_y_from_chunks(y_test_file):
    id_chunks = []
    y_chunks = []
    chunk_number = 0

    with open(y_test_file, "rb") as handle:
        while True:
            try:
                y_chunk = pickle.load(handle)
            except EOFError:
                break

            chunk_number += 1

            if isinstance(y_chunk, pd.Series):
                raise TypeError(
                    f"Chunk {chunk_number} is a Series and has no FID column."
                )
            if not isinstance(y_chunk, pd.DataFrame):
                y_chunk = pd.DataFrame(y_chunk)

            y_chunk = y_chunk.copy()
            y_chunk.columns = y_chunk.columns.astype(str)

            if "FID" not in y_chunk.columns:
                raise ValueError(
                    f"FID was not found in y_test chunk {chunk_number}. "
                    f"Available columns: {list(y_chunk.columns)}"
                )

            fid = normalize_id(y_chunk["FID"])
            iid = normalize_id(y_chunk["IID"]) if "IID" in y_chunk.columns else fid.copy()
            phenotype_column = detect_phenotype_column(y_chunk)
            y_values = pd.to_numeric(y_chunk[phenotype_column], errors="coerce")

            if y_values.isna().any():
                raise ValueError(
                    f"Chunk {chunk_number} contains missing/non-numeric phenotype values."
                )

            id_chunks.append(pd.DataFrame({"FID": fid.to_numpy(), "IID": iid.to_numpy()}))
            y_chunks.append(y_values.astype(int).to_numpy())

    if not id_chunks:
        raise ValueError(f"No y_test chunks were loaded from: {y_test_file}")

    ids = pd.concat(id_chunks, ignore_index=True)
    y_test = np.concatenate(y_chunks)
    print(f"  Loaded {len(ids):,} IDs from {chunk_number} y_test chunks.")
    return ids, y_test


def validate_ids(ids, repetition):
    for column in ["FID", "IID"]:
        missing = ids[column].isna() | (ids[column].astype(str).str.len() == 0)
        if missing.any():
            raise ValueError(
                f"rep{repetition}: found {int(missing.sum())} missing {column} values."
            )

    duplicated_fid = ids["FID"].duplicated(keep=False)
    if duplicated_fid.any():
        examples = ids.loc[duplicated_fid, "FID"].head(10).tolist()
        raise ValueError(
            f"rep{repetition}: duplicated FIDs were found. Examples: {examples}"
        )


def validate_y_true(predictions, original_y_test, repetition):
    if "y_true" not in predictions.columns:
        raise ValueError(
            f"rep{repetition}: predictions.csv has no y_true column; "
            "safe row-order validation is impossible."
        )

    saved_y_true = pd.to_numeric(predictions["y_true"], errors="coerce")
    if saved_y_true.isna().any():
        raise ValueError(
            f"rep{repetition}: predictions.csv contains invalid y_true values."
        )

    saved_y_true = saved_y_true.astype(int).to_numpy()
    original_y_test = np.asarray(original_y_test).astype(int)

    if len(saved_y_true) != len(original_y_test):
        raise ValueError(
            f"rep{repetition}: phenotype length mismatch: "
            f"saved={len(saved_y_true):,}, original={len(original_y_test):,}"
        )

    matches = saved_y_true == original_y_test
    print(
        f"  y_true agreement: {matches.mean():.2%} "
        f"({int(matches.sum()):,}/{len(matches):,})"
    )

    if not matches.all():
        first_mismatches = np.flatnonzero(~matches)[:10].tolist()
        raise ValueError(
            f"rep{repetition}: y_true does not match original y_test order. "
            f"First mismatches: {first_mismatches}"
        )


def process_repetition(base, phenotype, repetition, feature_set, output_name, overwrite):
    print("\n" + "=" * 80)
    print(f"Processing {phenotype}, rep{repetition}")
    print("=" * 80)

    y_test_file = (
        base / "Y_files" / f"rep{repetition}" /
        f"{phenotype}_Y_test_1k_chunks_no_missing.pkl"
    )
    prediction_dir = (
        base / "logistic_regression" / phenotype / f"rep{repetition}" /
        "reviewer_feature_set_comparison" / feature_set
    )
    predictions_file = prediction_dir / "predictions.csv"
    output_file = prediction_dir / output_name

    if not y_test_file.exists():
        raise FileNotFoundError(y_test_file)
    if not predictions_file.exists():
        raise FileNotFoundError(predictions_file)
    if output_file.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_file}\nUse --overwrite to replace it."
        )

    ids, original_y_test = load_ids_and_y_from_chunks(y_test_file)
    predictions = pd.read_csv(predictions_file)
    print(f"  Loaded {len(predictions):,} predictions.")

    if "prediction" not in predictions.columns:
        raise ValueError("predictions.csv has no 'prediction' column.")
    if len(ids) != len(predictions):
        raise ValueError(
            f"rep{repetition}: ID/prediction mismatch: "
            f"IDs={len(ids):,}, predictions={len(predictions):,}"
        )

    validate_ids(ids, repetition)
    validate_y_true(predictions, original_y_test, repetition)

    predictions = predictions.drop(columns=["FID", "IID"], errors="ignore")
    result = pd.concat(
        [ids.reset_index(drop=True), predictions.reset_index(drop=True)], axis=1
    )

    ordered = [c for c in ["FID", "IID", "y_true", "prediction"] if c in result.columns]
    remaining = [c for c in result.columns if c not in ordered]
    result = result[ordered + remaining]

    result.to_csv(output_file, index=False)
    print(f"  Saved: {output_file}")
    print(result.head().to_string(index=False))

    return {
        "repetition": repetition,
        "n_rows": len(result),
        "output_file": str(output_file),
        "status": "success",
    }


def main():
    args = parse_args()
    results = []

    for repetition in args.repetitions:
        results.append(
            process_repetition(
                args.base,
                args.phenotype,
                repetition,
                args.feature_set,
                args.output_name,
                args.overwrite,
            )
        )

    summary = pd.DataFrame(results)
    summary_file = (
        args.base / "logistic_regression" / args.phenotype /
        "predictions_with_ids_creation_summary.csv"
    )
    summary.to_csv(summary_file, index=False)
    print("\nAll requested repetitions completed successfully.")
    print(summary.to_string(index=False))
    print(f"\nSummary saved to: {summary_file}")


if __name__ == "__main__":
    main()
