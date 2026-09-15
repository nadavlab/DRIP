#!/usr/bin/env python3

import re
import os
import sys
import time
import pickle
import shutil
import tempfile

import numpy as np
import pandas as pd

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss, roc_auc_score, average_precision_score


INPUT_KEYS = ["X_train", "X_test", "y_train", "y_test"]


def prepare_y(y_df):
    y_df = y_df.drop(columns=["FID", "IID"], errors="ignore")
    return y_df.iloc[:, 0].astype(int).to_numpy().ravel()


def clean_col_name(c):
    c = str(c)
    return c.replace("('", "").replace("',)", "")


def get_feature_columns(X, feature_config):
    X = X.copy()
    X.columns = X.columns.astype(str)

    cols = list(X.columns)
    id_cols = ["FID", "IID"]

    ukb_pc_cols = [
        c for c in cols
        if c.startswith("genetic_principal_components_f22009_0_")
    ]

    cov_cols = []
    for c in cols:
        clean_c = clean_col_name(c)

        if (
            clean_c.startswith("uk_biobank_assessment_centre_f54_0_0_")
            or clean_c.startswith("genetic_sex_f22001_0_0_")
            or clean_c.startswith("genotype_measurement_batch_f22000_0_0_")
            or clean_c == "age_when_attended_assessment_centre_f21003_0_0"
        ):
            cov_cols.append(c)


    our_pc_cols = [
        c for c in cols
        if (
            c not in id_cols
            and c not in cov_cols
            and c not in ukb_pc_cols
            and (
                re.match(r"^\d+$", str(c))
                or re.match(r"^\d+_chr\d+$", str(c))
                or re.match(r"^PC\d+$", str(c), re.IGNORECASE)
                or re.match(r"^PC\d+_chr\d+$", str(c), re.IGNORECASE)
            )
        )]

    selected = []

    if feature_config["cov"]:
        selected += cov_cols

    if feature_config["ukb_pcs"]:
        selected += ukb_pc_cols

    if feature_config["our_pcs"]:
        selected += our_pc_cols

    selected = list(dict.fromkeys(selected))

    print("Feature set:")
    print("  use cov:", feature_config["cov"])
    print("  use UKB PCs:", feature_config["ukb_pcs"])
    print("  use our PCs:", feature_config["our_pcs"])
    print("  n cov cols:", len(cov_cols))
    print("  n UKB PC cols:", len(ukb_pc_cols))
    print("  n our PC cols:", len(our_pc_cols))
    print("  n selected cols:", len(selected))

    if len(selected) == 0:
        raise ValueError("No feature columns selected.")

    return selected


def prepare_X(X_df, feature_cols):
    X_df = X_df.copy()
    X_df.columns = X_df.columns.astype(str)
    return X_df[feature_cols].astype(float)


def copy_with_retry(src, dst, retries=5, delay=20):
    last_error = None

    for attempt in range(1, retries + 1):
        try:
            tmp_dst = dst + ".tmp"
            shutil.copy2(src, tmp_dst)
            os.replace(tmp_dst, dst)
            return dst
        except OSError as e:
            last_error = e
            print(f"Copy failed attempt {attempt}/{retries}: {src}")
            print(f"  error: {e}")
            time.sleep(delay)

    raise RuntimeError(f"Failed to copy after {retries} attempts: {src}") from last_error


def localize_input_files(paths, pheno_name, rep):
    scratch_base = (
        os.environ.get("SLURM_TMPDIR")
        or os.environ.get("TMPDIR")
        or tempfile.gettempdir()
    )

    local_dir = os.path.join(
        scratch_base,
        f"lr_reviewer_{pheno_name}_rep{rep}_{os.getpid()}",
    )
    os.makedirs(local_dir, exist_ok=True)

    local_paths = paths.copy()
    local_paths["_local_dir"] = local_dir

    print("Local scratch directory:")
    print(local_dir)

    for key in INPUT_KEYS:
        src = paths[key]
        dst = os.path.join(local_dir, os.path.basename(src))

        print(f"Copying {key} to local scratch:")
        print(f"  from: {src}")
        print(f"  to:   {dst}")

        local_paths[key] = copy_with_retry(src, dst)

    return local_paths


def load_first_pickle_object(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_y_chunks(y_chunks_file):
    ys = []

    with open(y_chunks_file, "rb") as fy:
        while True:
            try:
                y_raw = pickle.load(fy)
                ys.append(prepare_y(y_raw))
            except EOFError:
                break

    if not ys:
        raise ValueError(f"No y chunks were loaded from {y_chunks_file}")

    return np.concatenate(ys)


def create_validation_set(X_train_chunks_file, y_train_chunks_file, feature_config):
    X_val_raw = load_first_pickle_object(X_train_chunks_file)
    X_val_raw.columns = X_val_raw.columns.astype(str)
    feature_cols = get_feature_columns(X_val_raw, feature_config)
    X_val = prepare_X(X_val_raw, feature_cols)

    y_val_raw = load_first_pickle_object(y_train_chunks_file)
    y_val = prepare_y(y_val_raw)

    return X_val, y_val, feature_cols


def train_incremental(
    X_train_chunks_file,
    y_train_chunks_file,
    X_val,
    y_val,
    feature_cols,
    output_path,
    n_epochs=40,
):
    model = SGDClassifier(
        loss="log_loss",
        learning_rate="constant",
        eta0=1e-5,
        penalty="l2",
        random_state=42,
    )

    is_initialized = False
    train_losses = []
    val_losses = []

    for epoch in range(1, n_epochs + 1):
        print(f"Epoch {epoch}")
        batch_losses = []

        with open(X_train_chunks_file, "rb") as fx, open(y_train_chunks_file, "rb") as fy:
            batch_num = 1

            while True:
                try:
                    X_batch_raw = pickle.load(fx)
                    y_batch_raw = pickle.load(fy)

                    # The first train chunk is used as validation.
                    if batch_num == 1:
                        batch_num += 1
                        continue

                    X_batch = prepare_X(X_batch_raw, feature_cols)
                    y_batch = prepare_y(y_batch_raw)

                    if not is_initialized:
                        model.partial_fit(X_batch, y_batch, classes=np.array([0, 1]))
                        is_initialized = True
                    else:
                        model.partial_fit(X_batch, y_batch)

                    probs = model.predict_proba(X_batch)
                    batch_losses.append(log_loss(y_batch, probs, labels=[0, 1]))

                    batch_num += 1

                except EOFError:
                    break

        if not batch_losses:
            raise ValueError("No training batches were loaded after skipping validation chunk.")

        train_loss = float(np.mean(batch_losses))
        val_probs = model.predict_proba(X_val)
        val_loss = log_loss(y_val, val_probs, labels=[0, 1])

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch}: train loss={train_loss:.4f}, val loss={val_loss:.4f}")

    with open(os.path.join(output_path, "model.pkl"), "wb") as f:
        pickle.dump(model, f)

    pd.DataFrame({
        "epoch": list(range(1, n_epochs + 1)),
        "train_loss": train_losses,
        "val_loss": val_losses,
    }).to_csv(os.path.join(output_path, "losses.csv"), index=False)

    return model


def predict_pheno(model, X_test_chunks_file, feature_cols):
    predictions = []

    with open(X_test_chunks_file, "rb") as fx:
        while True:
            try:
                X_batch_raw = pickle.load(fx)
                X_batch = prepare_X(X_batch_raw, feature_cols)

                probs = model.predict_proba(X_batch)[:, 1]
                predictions.extend(probs)

            except EOFError:
                break

    return np.array(predictions)


def get_paths(pheno_name, rep, drm):
    if drm == "PCA":
        base = "/path/to/your/project//our_model/PCA"

        return {
            "X_train": f"{base}/X_train_1k_chunks_PCA_dim_remove_no_missing/rep{rep}/{pheno_name}_X_train_match_to_pheno_MinMax_cov_MinMax.pkl",
            "X_test": f"{base}/X_test_1k_chunks_PCA_dim_remove_no_missing/rep{rep}/{pheno_name}_X_test_match_to_pheno_MinMax_cov_MinMax.pkl",
            "y_train": f"{base}/Y_files/rep{rep}/{pheno_name}_Y_train_1k_chunks_no_missing.pkl",
            "y_test": f"{base}/Y_files/rep{rep}/{pheno_name}_Y_test_1k_chunks_no_missing.pkl",
            "out_base": f"{base}/logistic_regression/{pheno_name}/rep{rep}/reviewer_feature_set_comparison",
            "summary": f"{base}/logistic_regression/{pheno_name}/reviewer_feature_set_comparison_metrics.csv",
        }

    raise ValueError("Currently this script supports DRM='PCA' only.")


def main():
    if len(sys.argv) != 4:
        print("Usage:")
        print("python3 run_LR_reviewer_feature_sets.py <pheno_name> <rep> <DRM>")
        print("Example:")
        print("python3 run_LR_reviewer_feature_sets.py I10 1 PCA")
        sys.exit(1)

    pheno_name = sys.argv[1]
    rep = str(sys.argv[2])
    drm = sys.argv[3]

    feature_sets = {
        "01_cov_only": {
            "cov": True,
            "ukb_pcs": False,
            "our_pcs": False,
        },
        "02_ukb_40_pcs_only": {
            "cov": False,
            "ukb_pcs": True,
            "our_pcs": False,
        },
        "03_cov_plus_ukb_40_pcs": {
            "cov": True,
            "ukb_pcs": True,
            "our_pcs": False,
        },
        "04_our_new_pcs_only": {
            "cov": False,
            "ukb_pcs": False,
            "our_pcs": True,
        },
        "05_our_new_pcs_plus_cov": {
            "cov": True,
            "ukb_pcs": False,
            "our_pcs": True,
        },
        "06_our_new_pcs_plus_cov_plus_ukb_40_pcs": {
            "cov": True,
            "ukb_pcs": True,
            "our_pcs": True,
        },
    }

    original_paths = get_paths(pheno_name, rep, drm)

    for p in INPUT_KEYS:
        if not os.path.exists(original_paths[p]):
            raise FileNotFoundError(original_paths[p])

    paths = localize_input_files(original_paths, pheno_name, rep)

    all_metrics = []

    for feature_set_name, feature_config in feature_sets.items():
        start = time.time()

        print("\n" + "=" * 80)
        print(f"Running {pheno_name}, rep{rep}, {feature_set_name}")
        print("=" * 80)

        output_path = os.path.join(paths["out_base"], feature_set_name)
        os.makedirs(output_path, exist_ok=True)

        X_val, y_val, feature_cols = create_validation_set(
            paths["X_train"],
            paths["y_train"],
            feature_config,
        )

        pd.Series(feature_cols).to_csv(
            os.path.join(output_path, "feature_columns_used.csv"),
            index=False,
            header=False,
        )

        model = train_incremental(
            paths["X_train"],
            paths["y_train"],
            X_val,
            y_val,
            feature_cols,
            output_path,
            n_epochs=40,
        )

        predictions = predict_pheno(
            model,
            paths["X_test"],
            feature_cols,
        )

        y_test = load_y_chunks(paths["y_test"])

        if len(y_test) != len(predictions):
            raise ValueError(
                "Prediction/label length mismatch: "
                f"len(y_test)={len(y_test)}, len(predictions)={len(predictions)}"
            )

        pred_df = pd.DataFrame({
            "y_true": y_test,
            "prediction": predictions,
        })
        pred_df.to_csv(os.path.join(output_path, "predictions.csv"), index=False)

        metrics = {
            "pheno": pheno_name,
            "rep": rep,
            "DRM": drm,
            "feature_set": feature_set_name,
            "n_features": len(feature_cols),
            "ROC_AUC": roc_auc_score(y_test, predictions),
            "log_loss": log_loss(y_test, predictions, labels=[0, 1]),
            "average_precision_score": average_precision_score(y_test, predictions),
            "time_minutes": (time.time() - start) / 60,
        }

        pd.DataFrame([metrics]).to_csv(
            os.path.join(output_path, "metrics.csv"),
            index=False,
        )

        all_metrics.append(metrics)

        print(metrics)

    summary_df = pd.DataFrame(all_metrics)

    if os.path.exists(paths["summary"]):
        summary_df.to_csv(paths["summary"], mode="a", header=False, index=False)
    else:
        summary_df.to_csv(paths["summary"], index=False)

    print("Done.")
    print("Summary saved to:", paths["summary"])
    print("Local scratch files were kept here:", paths["_local_dir"])


if __name__ == "__main__":
    main()
