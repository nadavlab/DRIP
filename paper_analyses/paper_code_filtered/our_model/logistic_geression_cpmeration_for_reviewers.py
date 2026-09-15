#!/usr/bin/env python3

import os
import sys
import time
import pickle
import re
import numpy as np
import pandas as pd

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss, roc_auc_score, average_precision_score


# =========================
# CONFIG
# =========================

PHENO_NAME = sys.argv[1]
DRM = sys.argv[2]  # PCA / Autoencoder

REPS = ["1", "2", "3"]

BASE = "/path/to/your/project/our_model"

# ---- Adapt these to your column names ----
COV_COLS = [
    "Sex",
    "Age",
    "Genotype_Measurement_Batch",
    "UK_Biobank_assessment_centre",
]

# UKB PCs usually look like 22009-0.1, 22009-0.2 ...
UKB_PC_PATTERN = r"^22009-0\.\d+$"

# Adapt to your created PC column names.
# Examples: PC1, PC2... or PCA_1... or 0,1,2...
OUR_PC_PATTERN = r"^(PC|PCA|our_PC|dim)_?\d+$"

N_EPOCHS = 40
ETA0 = 1e-5


FEATURE_SETS = {
    "cov_only": {
        "cov": True,
        "ukb_pcs": False,
        "our_pcs": False,
    },
    "ukb_pcs_only": {
        "cov": False,
        "ukb_pcs": True,
        "our_pcs": False,
    },
    "cov_plus_ukb_pcs": {
        "cov": True,
        "ukb_pcs": True,
        "our_pcs": False,
    },
    "our_pcs_only": {
        "cov": False,
        "ukb_pcs": False,
        "our_pcs": True,
    },
    "our_pcs_plus_cov": {
        "cov": True,
        "ukb_pcs": False,
        "our_pcs": True,
    },
    "our_pcs_plus_ukb_pcs": {
        "cov": False,
        "ukb_pcs": True,
        "our_pcs": True,
    },
    "our_pcs_plus_cov_plus_ukb_pcs": {
        "cov": True,
        "ukb_pcs": True,
        "our_pcs": True,
    },
}


# =========================
# PATHS
# =========================

def get_paths(rep):
    if DRM == "PCA":
        prefix = f"{BASE}/PCA"
        return {
            "X_train": f"{prefix}/X_train_1k_chunks_PCA_dim_remove_no_missing/rep{rep}/{PHENO_NAME}_X_train_match_to_pheno_MinMax_cov_MinMax.pkl",
            "X_test": f"{prefix}/X_test_1k_chunks_PCA_dim_remove_no_missing/rep{rep}/{PHENO_NAME}_X_test_match_to_pheno_MinMax_cov_MinMax.pkl",
            "y_train": f"{prefix}/Y_files/rep{rep}/{PHENO_NAME}_Y_train_1k_chunks_no_missing.pkl",
            "y_test": f"{prefix}/Y_files/rep{rep}/{PHENO_NAME}_Y_test_1k_chunks_no_missing.pkl",
            "out": f"{prefix}/logistic_regression/{PHENO_NAME}/rep{rep}/reviewer_feature_set_comparison",
        }

    elif DRM == "Autoencoder":
        prefix = f"{BASE}/Autoencoder"
        return {
            "X_train": f"{prefix}/X_train_1k_chunks_dim_remove_no_missing_500_epochs/rep{rep}/{PHENO_NAME}_X_train_match_to_pheno_MinMax_cov_MinMax.pkl",
            "X_test": f"{prefix}/X_test_1k_chunks_dim_remove_no_missing_500_epochs/rep{rep}/{PHENO_NAME}_X_test_match_to_pheno_MinMax_cov_MinMax.pkl",
            "y_train": f"{prefix}/Y_files/rep{rep}/{PHENO_NAME}_Y_train_1k_chunks_no_missing.pkl",
            "y_test": f"{prefix}/Y_files/rep{rep}/{PHENO_NAME}_Y_test_1k_chunks_no_missing.pkl",
            "out": f"{BASE}/NN_with_Autoencoder/{PHENO_NAME}/rep{rep}/reviewer_feature_set_comparison",
        }

    else:
        raise ValueError("DRM must be PCA or Autoencoder")


# =========================
# FEATURE SELECTION
# =========================

def get_feature_columns(X, feature_config):
    cols = list(X.columns)

    if "FID" in cols:
        cols.remove("FID")

    cov_cols = [c for c in COV_COLS if c in X.columns]

    ukb_pc_cols = [
        c for c in cols
        if re.match(UKB_PC_PATTERN, str(c))
    ]

    our_pc_cols = [
        c for c in cols
        if re.match(OUR_PC_PATTERN, str(c))
    ]

    selected = []

    if feature_config["cov"]:
        selected += cov_cols

    if feature_config["ukb_pcs"]:
        selected += ukb_pc_cols

    if feature_config["our_pcs"]:
        selected += our_pc_cols

    selected = list(dict.fromkeys(selected))

    if len(selected) == 0:
        raise ValueError(
            "No feature columns selected. Check COV_COLS / UKB_PC_PATTERN / OUR_PC_PATTERN."
        )

    return selected


def prepare_X(X, feature_cols):
    X = X.copy()
    X.columns = X.columns.astype(str)
    return X[feature_cols].astype(float)


def prepare_y(y):
    y = y.drop(columns=["FID"], errors="ignore")
    return y.iloc[:, 0].astype(int).to_numpy().ravel()


# =========================
# DATA LOADERS
# =========================

def load_first_validation_chunk(X_train_file, y_train_file, feature_config):
    with open(X_train_file, "rb") as fx:
        X_val_raw = pickle.load(fx)
        X_val_raw.columns = X_val_raw.columns.astype(str)
        feature_cols = get_feature_columns(X_val_raw, feature_config)
        X_val = prepare_X(X_val_raw, feature_cols)

    with open(y_train_file, "rb") as fy:
        y_val_raw = pickle.load(fy)
        y_val = prepare_y(y_val_raw)

    return X_val, y_val, feature_cols


# =========================
# TRAIN
# =========================

def train_incremental(X_train_file, y_train_file, X_val, y_val, feature_cols, out_dir):
    model = SGDClassifier(
        loss="log_loss",
        learning_rate="constant",
        eta0=ETA0,
        penalty="l2",
        random_state=42
    )

    is_initialized = False
    train_losses = []
    val_losses = []

    for epoch in range(1, N_EPOCHS + 1):
        batch_losses = []

        with open(X_train_file, "rb") as fx, open(y_train_file, "rb") as fy:
            batch_num = 1

            while True:
                try:
                    X_batch_raw = pickle.load(fx)
                    y_batch_raw = pickle.load(fy)

                    # first chunk is validation
                    if batch_num == 1:
                        batch_num += 1
                        continue

                    X_batch_raw.columns = X_batch_raw.columns.astype(str)
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

        train_loss = np.mean(batch_losses)
        val_prob = model.predict_proba(X_val)
        val_loss = log_loss(y_val, val_prob, labels=[0, 1])

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch}: train loss={train_loss:.4f}, val loss={val_loss:.4f}")

    with open(os.path.join(out_dir, "model.pkl"), "wb") as f:
        pickle.dump(model, f)

    pd.DataFrame({
        "epoch": range(1, N_EPOCHS + 1),
        "train_loss": train_losses,
        "val_loss": val_losses,
    }).to_csv(os.path.join(out_dir, "losses.csv"), index=False)

    return model


# =========================
# PREDICT
# =========================

def predict_chunks(model, X_test_file, feature_cols):
    preds = []

    with open(X_test_file, "rb") as fx:
        while True:
            try:
                X_raw = pickle.load(fx)
                X_raw.columns = X_raw.columns.astype(str)
                X = prepare_X(X_raw, feature_cols)

                probs = model.predict_proba(X)[:, 1]
                preds.extend(probs)

            except EOFError:
                break

    return np.array(preds)


# =========================
# MAIN
# =========================

all_metrics = []
start_all = time.time()

for rep in REPS:
    paths = get_paths(rep)

    for feature_set_name, feature_config in FEATURE_SETS.items():
        print("\n" + "=" * 80)
        print(f"Running rep={rep}, feature_set={feature_set_name}")
        print("=" * 80)

        out_dir = os.path.join(paths["out"], feature_set_name)
        os.makedirs(out_dir, exist_ok=True)

        start = time.time()

        X_val, y_val, feature_cols = load_first_validation_chunk(
            paths["X_train"],
            paths["y_train"],
            feature_config
        )

        print(f"Number of features: {len(feature_cols)}")
        pd.Series(feature_cols).to_csv(
            os.path.join(out_dir, "feature_columns_used.csv"),
            index=False,
            header=False
        )

        model = train_incremental(
            paths["X_train"],
            paths["y_train"],
            X_val,
            y_val,
            feature_cols,
            out_dir
        )

        preds = predict_chunks(model, paths["X_test"], feature_cols)

        y_test_raw = pd.read_pickle(paths["y_test"])
        y_test = prepare_y(y_test_raw)

        pred_df = pd.DataFrame({
            "prediction": preds,
            "y_true": y_test
        })
        pred_df.to_csv(os.path.join(out_dir, "predictions.csv"), index=False)

        metrics = {
            "pheno": PHENO_NAME,
            "rep": rep,
            "DRM": DRM,
            "feature_set": feature_set_name,
            "n_features": len(feature_cols),
            "ROC_AUC": roc_auc_score(y_test, preds),
            "log_loss": log_loss(y_test, preds, labels=[0, 1]),
            "average_precision_score": average_precision_score(y_test, preds),
            "time_minutes": (time.time() - start) / 60,
        }

        all_metrics.append(metrics)

        pd.DataFrame([metrics]).to_csv(
            os.path.join(out_dir, "metrics.csv"),
            index=False
        )

        print(metrics)


summary_df = pd.DataFrame(all_metrics)

summary_out = f"{BASE}/PCA/logistic_regression/{PHENO_NAME}/reviewer_feature_set_comparison_summary_{DRM}.csv"
summary_df.to_csv(summary_out, index=False)

print(f"\nSaved summary to: {summary_out}")
print(f"Total time: {(time.time() - start_all) / 60:.2f} minutes")
