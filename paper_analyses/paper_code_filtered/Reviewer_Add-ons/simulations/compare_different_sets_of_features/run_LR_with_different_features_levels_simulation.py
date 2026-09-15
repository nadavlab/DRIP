#!/usr/bin/env python3

import argparse
import os
import time
import pickle
import warnings

import numpy as np
import pandas as pd

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss, roc_auc_score, average_precision_score


def iter_pickle_chunks(path):
    with open(path, "rb") as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break


def normalize_label_vector(y, pheno_name="phenotype"):
    """
    Supports both 0/1 and 1/2 labels.
    Converts 1=control, 2=case to 0/1.
    """
    y = pd.to_numeric(pd.Series(y), errors="coerce")
    mask = y.notna().to_numpy()
    y = y.loc[mask].astype(int).to_numpy()

    unique_labels = sorted(np.unique(y).tolist())
    print(f"{pheno_name} labels after removing NA:", unique_labels)

    if set(unique_labels).issubset({1, 2}):
        y = np.where(y == 2, 1, 0).astype(int)
    elif set(unique_labels).issubset({0, 1}):
        y = y.astype(int)
    else:
        raise ValueError(f"Unexpected labels for {pheno_name}: {unique_labels}")

    return y, mask


def get_y_for_pheno(y_df, pheno_name):
    if pheno_name not in y_df.columns:
        raise ValueError(
            f"Phenotype '{pheno_name}' not found. Available columns: {list(y_df.columns)}"
        )
    y, mask = normalize_label_vector(y_df[pheno_name], pheno_name=pheno_name)
    return y, mask


def clean_col_name(c):
    c = str(c)
    return c.replace("('", "").replace("',)", "")


def read_col_file(path):
    if path is None:
        return None
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]

def detect_feature_groups(X, created_pc_cols_file=None, ukb_pc_cols_file=None, cov_cols_file=None):
    import re

    X = X.copy()
    X.columns = X.columns.astype(str)
    cols = list(X.columns)
    id_cols = {"FID", "IID", "#FID"}

    explicit_created = read_col_file(created_pc_cols_file)
    explicit_ukb = read_col_file(ukb_pc_cols_file)
    explicit_cov = read_col_file(cov_cols_file)

    if explicit_ukb is not None:
        ukb_pc_cols = [c for c in explicit_ukb if c in cols]
    else:
        ukb_pc_cols = [
            c for c in cols
            if str(c).startswith("genetic_principal_components_f22009_0_")
        ]

    if explicit_created is not None:
        created_pc_cols = [c for c in explicit_created if c in cols]
    else:
        created_pc_cols = []
        for c in cols:
            s = str(c)
            clean_s = clean_col_name(s)

            if c in id_cols:
                continue
            if c in ukb_pc_cols:
                continue

            # Created PCs in the simulations:
            # 0_chr1, 1_chr1, ..., 732_chr22
            # Also keep compatibility with older numeric / PC / PCA names.
            if (
                re.match(r"^\d+_chr\d+$", clean_s)
                or clean_s.isdigit()
                or clean_s.startswith("PC")
                or clean_s.startswith("PCA")
                or clean_s.startswith("pc")
            ):
                created_pc_cols.append(c)

    if explicit_cov is not None:
        cov_cols = [c for c in explicit_cov if c in cols]
    else:
        cov_cols = []
        for c in cols:
            if c in id_cols:
                continue
            if c in created_pc_cols:
                continue
            if c in ukb_pc_cols:
                continue

            # Everything that is not ID, not created PCs, and not UKB PCs
            # is treated as covariate.
            cov_cols.append(c)

    non_pc_cov_cols = [
        c for c in cov_cols
        if c not in ukb_pc_cols and c not in created_pc_cols
    ]

    return {
        "created_pc_cols": list(dict.fromkeys(created_pc_cols)),
        "ukb_pc_cols": list(dict.fromkeys(ukb_pc_cols)),
        "cov_cols": list(dict.fromkeys(cov_cols)),
        "non_pc_cov_cols": list(dict.fromkeys(non_pc_cov_cols)),
    }


def select_feature_cols(groups, feature_set_name):
    created = groups["created_pc_cols"]
    ukb = groups["ukb_pc_cols"]
    cov = groups["cov_cols"]
    non_pc_cov = groups["non_pc_cov_cols"]

    feature_sets = {
        "01_created_PCs_only": created,
        "02_cov_only_no_created_PCs": cov,
        "03_created_PCs_plus_non_UKB_PC_cov": created + non_pc_cov,
        "04_UKB_PCs_only": ukb,
        "05_non_PC_cov_only": non_pc_cov,
    }

    if feature_set_name not in feature_sets:
        raise ValueError(f"Unknown feature set: {feature_set_name}")

    selected = list(dict.fromkeys(feature_sets[feature_set_name]))
    if len(selected) == 0:
        raise ValueError(f"No columns selected for feature set {feature_set_name}")

    print("Feature set:", feature_set_name)
    print("  n created PCs:", len(created))
    print("  n UKB PCs:", len(ukb))
    print("  n cov cols:", len(cov))
    print("  n non-PC cov cols:", len(non_pc_cov))
    print("  n selected:", len(selected))
    print("  first selected columns:", selected[:10])

    return selected


def prepare_X(X_df, feature_cols):
    X_df = X_df.copy()
    X_df.columns = X_df.columns.astype(str)
    return X_df[feature_cols].astype(float)


def find_basic_pheno(y_train_file):
    """
    Best-effort automatic detection for the baseline simulation phenotype:
    N=100 SNPs, additive heritability 0.3, noise 0.5, cov weight 0.2, no interactions.
    If names differ, pass --pheno-col explicitly.
    """
    first_y = next(iter_pickle_chunks(y_train_file))
    id_cols = {"FID", "IID", "#FID"}
    phenos = [c for c in first_y.columns if c not in id_cols]

    def score(col):
        s = str(col).lower()
        points = 0
        # N=100
        if "n100" in s or "n_100" in s or "100" in s:
            points += 2
        # h2/additive heritability 0.3
        if "h2add_0p30" in s or "h2add_0.30" in s or "h2_0p30" in s or "h2_0.30" in s or "heritability_0p3" in s or "0p3" in s:
            points += 3
        # noise 0.5
        if "noise_0p500" in s or "noise_0.500" in s or "noise_0p5" in s or "noise_0.5" in s:
            points += 2
        # cov 0.2
        if "cov_0p200" in s or "cov_0.200" in s or "cov_0p2" in s or "cov_0.2" in s:
            points += 2
        # no interactions / no gxg
        if "int0" in s or "int_0" in s or "interaction_0" in s or "h2gxg_0p00" in s or "h2gxg_0.00" in s or "gxg_0" in s:
            points += 2
        if "int" in s and not ("int0" in s or "int_0" in s or "interaction_0" in s):
            points -= 2
        if "gxg" in s and not ("h2gxg_0p00" in s or "h2gxg_0.00" in s or "gxg_0" in s):
            points -= 2
        return points

    scored = sorted([(score(c), c) for c in phenos], reverse=True, key=lambda x: x[0])
    print("Top phenotype-name matches for baseline simulation:")
    for pts, c in scored[:10]:
        print(f"  score={pts}: {c}")

    if not scored or scored[0][0] <= 0:
        raise ValueError(
            "Could not automatically identify the baseline phenotype. "
            "Please pass --pheno-col with the exact column name."
        )

    return scored[0][1]


def create_validation_set(x_train_file, y_train_file, pheno_name, feature_cols):
    X_val_raw = next(iter_pickle_chunks(x_train_file))
    y_val_raw = next(iter_pickle_chunks(y_train_file))

    X_val = prepare_X(X_val_raw, feature_cols)
    y_val, mask = get_y_for_pheno(y_val_raw, pheno_name)
    X_val = X_val.loc[mask].reset_index(drop=True)

    return X_val, y_val


def train_incremental(x_train_file, y_train_file, pheno_name, X_val, y_val, feature_cols, output_path, n_epochs=40):
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
        print(f"Epoch {epoch}/{n_epochs}")
        batch_losses = []
        batch_num = 1

        for X_raw, y_raw in zip(iter_pickle_chunks(x_train_file), iter_pickle_chunks(y_train_file)):
            if batch_num == 1:
                batch_num += 1
                continue

            X_batch = prepare_X(X_raw, feature_cols)
            y_batch, mask = get_y_for_pheno(y_raw, pheno_name)
            X_batch = X_batch.loc[mask].reset_index(drop=True)

            if len(y_batch) == 0:
                batch_num += 1
                continue

            if not is_initialized:
                model.partial_fit(X_batch, y_batch, classes=np.array([0, 1]))
                is_initialized = True
            else:
                model.partial_fit(X_batch, y_batch)

            probs = model.predict_proba(X_batch)
            batch_losses.append(log_loss(y_batch, probs, labels=[0, 1]))
            batch_num += 1

        if len(batch_losses) == 0:
            raise ValueError("No training batches found after validation chunk.")

        train_loss = float(np.mean(batch_losses))
        val_probs = model.predict_proba(X_val)
        val_loss = float(log_loss(y_val, val_probs, labels=[0, 1]))

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"Epoch {epoch}: train loss={train_loss:.5f}, val loss={val_loss:.5f}")

    with open(os.path.join(output_path, "model.pkl"), "wb") as f:
        pickle.dump(model, f)

    pd.DataFrame({
        "epoch": list(range(1, len(train_losses) + 1)),
        "train_loss": train_losses,
        "val_loss": val_losses,
    }).to_csv(os.path.join(output_path, "losses.csv"), index=False)

    return model, train_losses, val_losses


def predict_pheno(model, x_test_file, feature_cols):
    predictions = []
    for X_raw in iter_pickle_chunks(x_test_file):
        X_batch = prepare_X(X_raw, feature_cols)
        probs = model.predict_proba(X_batch)[:, 1]
        predictions.extend(probs.tolist())
    return np.array(predictions, dtype=float)


def load_test_y(y_test_file, pheno_name):
    obj = pd.read_pickle(y_test_file)
    if isinstance(obj, pd.DataFrame):
        y_df = obj
    else:
        y_df = pd.concat(list(iter_pickle_chunks(y_test_file)), ignore_index=True)
    y, mask = get_y_for_pheno(y_df, pheno_name)
    return y, mask


def safe_metrics(y_true, predictions):
    out = {
        "n_test": int(len(y_true)),
        "n_cases_test": int(np.sum(y_true == 1)),
        "n_controls_test": int(np.sum(y_true == 0)),
        "ROC_AUC": np.nan,
        "log_loss": np.nan,
        "average_precision_score": np.nan,
    }
    try:
        out["ROC_AUC"] = float(roc_auc_score(y_true, predictions))
    except Exception as e:
        warnings.warn(f"ROC_AUC failed: {e}")
    try:
        out["log_loss"] = float(log_loss(y_true, predictions, labels=[0, 1]))
    except Exception as e:
        warnings.warn(f"log_loss failed: {e}")
    try:
        out["average_precision_score"] = float(average_precision_score(y_true, predictions))
    except Exception as e:
        warnings.warn(f"AP failed: {e}")
    return out


def main():
    parser = argparse.ArgumentParser(
        description="LR feature-set benchmark for the basic simulated phenotype only."
    )
    parser.add_argument("--rep", required=True)
    parser.add_argument("--pheno-col", default=None, help="Exact baseline phenotype column. If omitted, best-effort auto-detection is used.")
    parser.add_argument("--matched-root", default="/path/to/your/project//simulations/matched_data_ids")
    parser.add_argument("--output-root", default="/path/to/your/project//simulations/LR_basic_sim_feature_sets")
    parser.add_argument("--summary-file", default=None)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--created-pc-cols-file", default=None)
    parser.add_argument("--ukb-pc-cols-file", default=None)
    parser.add_argument("--cov-cols-file", default=None)

    args = parser.parse_args()
    matched_dir = args.matched_root
    x_train = os.path.join(matched_dir, f"X_train_match_to_pheno_rep_{args.rep}.pkl")
    y_train = os.path.join(matched_dir, f"Y_train_all_rep_{args.rep}.pkl")
    x_test = os.path.join(matched_dir, f"X_test_match_to_pheno_rep_{args.rep}.pkl")
    y_test = os.path.join(matched_dir, f"Y_test_all_rep_{args.rep}.pkl")

    for p in [x_train, y_train, x_test, y_test]:
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    pheno_name = args.pheno_col or find_basic_pheno(y_train)
    print("Using phenotype:", pheno_name)

    first_X = next(iter_pickle_chunks(x_train))
    groups = detect_feature_groups(
        first_X,
        created_pc_cols_file=args.created_pc_cols_file,
        ukb_pc_cols_file=args.ukb_pc_cols_file,
        cov_cols_file=args.cov_cols_file,
    )

    feature_set_names = [
        "01_created_PCs_only",
        "02_cov_only_no_created_PCs",
        "03_created_PCs_plus_non_UKB_PC_cov",
        "04_UKB_PCs_only",
        "05_non_PC_cov_only",
    ]

    summary_rows = []
    out_base = os.path.join(args.output_root, f"rep{args.rep}", str(pheno_name))
    os.makedirs(out_base, exist_ok=True)

    for feature_set_name in feature_set_names:
        start = time.time()
        print("\n" + "=" * 80)
        print(f"Running rep={args.rep} | pheno={pheno_name} | {feature_set_name}")
        print("=" * 80)

        feature_cols = select_feature_cols(groups, feature_set_name)
        output_path = os.path.join(out_base, feature_set_name)
        os.makedirs(output_path, exist_ok=True)

        pd.Series(feature_cols).to_csv(
            os.path.join(output_path, "feature_columns_used.csv"),
            index=False,
            header=False,
        )

        X_val, y_val = create_validation_set(x_train, y_train, pheno_name, feature_cols)
        model, train_losses, val_losses = train_incremental(
            x_train,
            y_train,
            pheno_name,
            X_val,
            y_val,
            feature_cols,
            output_path,
            n_epochs=args.epochs,
        )

        predictions = predict_pheno(model, x_test, feature_cols)
        y_true, mask = load_test_y(y_test, pheno_name)
        predictions = predictions[mask]

        pd.DataFrame({"y_true": y_true, "prediction": predictions}).to_csv(
            os.path.join(output_path, "predictions.csv"), index=False
        )

        metrics = {
            "rep": args.rep,
            "pheno": pheno_name,
            "simulation": "basic_N100_h2add0.3_noise0.5_cov0.2_no_interactions",
            "model": "LR_SGDClassifier",
            "feature_set": feature_set_name,
            "n_features": len(feature_cols),
            "epochs": args.epochs,
            "final_train_loss": train_losses[-1] if train_losses else np.nan,
            "final_val_loss": val_losses[-1] if val_losses else np.nan,
            "time_minutes": (time.time() - start) / 60,
            **safe_metrics(y_true, predictions),
        }

        pd.DataFrame([metrics]).to_csv(os.path.join(output_path, "metrics.csv"), index=False)
        summary_rows.append(metrics)
        print(metrics)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = args.summary_file or os.path.join(args.output_root, f"rep{args.rep}_basic_sim_feature_set_metrics.csv")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)

    if os.path.exists(summary_path):
        summary_df.to_csv(summary_path, mode="a", header=False, index=False)
    else:
        summary_df.to_csv(summary_path, index=False)

    print("Done.")
    print("Summary saved to:", summary_path)


if __name__ == "__main__":
    main()
