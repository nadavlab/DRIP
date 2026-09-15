#!/usr/bin/env python3

import argparse
import os
import pickle
import time
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    log_loss,
    roc_auc_score,
    roc_curve,
    average_precision_score,
)


ID_COLS = {"FID", "IID", "#FID"}


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_first_chunk(pkl_file: str) -> pd.DataFrame:
    with open(pkl_file, "rb") as f:
        return pickle.load(f)


def iter_pickle_chunks(pkl_file: str):
    with open(pkl_file, "rb") as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break


def prepare_x(X: pd.DataFrame, feature_start: Optional[int] = None) -> pd.DataFrame:
    """
    Keeps the same general logic as the original LR code.
    If feature_start is supplied, uses X.iloc[:, feature_start:].
    Otherwise, drops FID/IID/#FID and uses all remaining columns.
    """
    X = X.copy()

    if feature_start is not None:
        X = X.iloc[:, feature_start:]
    else:
        drop_cols = [c for c in X.columns if c in ID_COLS]
        X = X.drop(columns=drop_cols, errors="ignore")

    X.columns = X.columns.astype(str)
    return X


def get_pheno_columns(y_train_file: str, requested: Optional[str] = None) -> List[str]:
    y0 = load_first_chunk(y_train_file)
    y0 = y0.rename(columns={"#FID": "FID"})
    all_cols = [c for c in y0.columns if c not in ID_COLS]

    if requested is None or requested.strip() == "" or requested.lower() == "all":
        return all_cols

    requested_cols = [x.strip() for x in requested.split(",") if x.strip()]
    missing = [c for c in requested_cols if c not in all_cols]
    if missing:
        raise ValueError(f"Requested phenotype columns not found in Y file: {missing}")
    return requested_cols


def create_validation_set(
    x_train_file: str,
    y_train_file: str,
    pheno_name: str,
    feature_start: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    X_val = prepare_x(load_first_chunk(x_train_file), feature_start=feature_start)
    y_val_df = load_first_chunk(y_train_file).rename(columns={"#FID": "FID"})
    

    if pheno_name not in y_val_df.columns:
        raise ValueError(f"{pheno_name} not found in {y_train_file}")

    y_val = y_val_df[pheno_name].astype(int).reset_index(drop=True)
    y_val = normalize_binary_labels(y_val_df[pheno_name]).reset_index(drop=True)
    return X_val, y_val


def normalize_binary_labels(y):
    y = pd.to_numeric(y, errors="coerce")
    y = y.dropna().astype(int)

    vals = set(y.unique())
    if vals.issubset({1, 2}):
        y = y.map({1: 0, 2: 1})
    elif not vals.issubset({0, 1}):
        raise ValueError(f"Unexpected labels: {sorted(vals)}")

    return y.astype(int)

def plot_loss(
    train_loss: List[float],
    val_loss: List[float],
    pheno_name: str,
    output_path: str,
    epoch_from: int,
    epoch_to: int,
) -> None:
    fig = plt.figure()
    plt.plot(train_loss, label="Train Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.title("Logistic Regression Loss on " + pheno_name)
    plt.ylabel("Log Loss")
    plt.xlabel("Epochs")
    plt.legend(loc="upper left")
    plt.xlim(0, max(1, epoch_to - epoch_from + 1))

    finite_vals = [x for x in train_loss + val_loss if np.isfinite(x)]
    if finite_vals:
        plt.ylim(0, max(finite_vals))

    fig.savefig(os.path.join(output_path, f"{epoch_from}_{epoch_to}_logloss.png"))
    plt.close(fig)


def train_incremental_lr(
    x_train_file: str,
    y_train_file: str,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    pheno_name: str,
    output_path: str,
    epoch_from: int,
    epoch_to: int,
    eta0: float,
    penalty: str,
    learning_rate: str,
    alpha: float,
    random_state: int,
    feature_start: Optional[int] = None,
) -> SGDClassifier:

    train_loss = []
    val_loss = []

    model = SGDClassifier(
        loss="log_loss",
        learning_rate=learning_rate,
        eta0=eta0,
        penalty=penalty,
        alpha=alpha,
        random_state=random_state,
    )

    is_initialized = False

    for epoch in range(epoch_from, epoch_to + 1):
        print(f"[{pheno_name}] Epoch {epoch}", flush=True)
        batch_train_loss = []

        x_iter = iter_pickle_chunks(x_train_file)
        y_iter = iter_pickle_chunks(y_train_file)

        for batch_num, (X_batch_raw, y_batch_raw) in enumerate(zip(x_iter, y_iter), start=1):
            X_batch = prepare_x(X_batch_raw, feature_start=feature_start)
            y_batch_raw = y_batch_raw.rename(columns={"#FID": "FID"})

            # First chunk is validation, as in the original code.
            if batch_num == 1:
                print(f"[{pheno_name}] validation chunk size = {len(X_batch)}", flush=True)
                continue

            if pheno_name not in y_batch_raw.columns:
                raise ValueError(f"{pheno_name} not found in Y batch columns")

            y_batch = y_batch_raw[pheno_name].astype(int).to_numpy().ravel()

            # clean labels before partial_fit
            y_batch = np.asarray(y_batch)

            mask = pd.notna(y_batch)
            X_batch = X_batch.loc[mask].reset_index(drop=True)
            y_batch = y_batch[mask]

            y_batch = pd.to_numeric(y_batch, errors="coerce")
            mask = pd.notna(y_batch)

            X_batch = X_batch.loc[mask].reset_index(drop=True)
            y_batch = np.asarray(y_batch[mask]).astype(int)

            unique_labels = sorted(np.unique(y_batch))
            print("Unique labels:", unique_labels)

            # convert 1/2 labels to 0/1
            if set(unique_labels).issubset({1, 2}):
                y_batch = np.where(y_batch == 2, 1, 0)

            elif not set(unique_labels).issubset({0, 1}):
                raise ValueError(f"Unexpected labels in phenotype: {unique_labels}")

            if len(y_batch) == 0:
                continue

            # If a batch has only one class, partial_fit can still work after init,
            # but log_loss needs labels=[0,1].
            if not is_initialized:
                model.partial_fit(X_batch, y_batch, classes=np.array([0, 1]))
                is_initialized = True
            else:
                model.partial_fit(X_batch, y_batch)

            probs = model.predict_proba(X_batch)
            batch_train_loss.append(log_loss(y_batch, probs, labels=[0, 1]))
            
        if not is_initialized:
            raise RuntimeError(f"Model for {pheno_name} was not trained. Check number of chunks.")

        avg_train_loss = float(np.mean(batch_train_loss)) if batch_train_loss else np.nan
        train_loss.append(avg_train_loss)

        probs_val = model.predict_proba(X_val)
        current_val_loss = log_loss(y_val.to_numpy().ravel(), probs_val, labels=[0, 1])
        val_loss.append(float(current_val_loss))

        print(
            f"[{pheno_name}] Epoch {epoch}: Avg training loss = {avg_train_loss:.4f}, "
            f"Validation loss = {current_val_loss:.4f}",
            flush=True,
        )

        if epoch % 10 == 0:
            with open(os.path.join(output_path, f"model_epoch_{epoch}.pkl"), "wb") as f:
                pickle.dump(model, f, protocol=4)

    plot_loss(train_loss, val_loss, pheno_name, output_path, epoch_from, epoch_to)
    return model


def predict_all_chunks(
    model: SGDClassifier,
    x_test_file: str,
    feature_start: Optional[int] = None,
) -> np.ndarray:
    predictions = []

    for X_batch_raw in iter_pickle_chunks(x_test_file):
        X_batch = prepare_x(X_batch_raw, feature_start=feature_start)
        probs = model.predict_proba(X_batch)[:, 1]
        predictions.extend(probs.tolist())

    return np.array(predictions)


def load_y_test_all(y_test_file: str) -> pd.DataFrame:
    """
    Supports either a single dataframe pickle or chunked pickle.
    """
    chunks = []
    with open(y_test_file, "rb") as f:
        while True:
            try:
                obj = pickle.load(f)
                chunks.append(obj)
            except EOFError:
                break

    if len(chunks) == 0:
        raise ValueError(f"No objects found in {y_test_file}")

    y = pd.concat(chunks, ignore_index=True) if len(chunks) > 1 else chunks[0]
    y = y.rename(columns={"#FID": "FID"})
    return y


def safe_metrics(y_true: np.ndarray, pred: np.ndarray) -> dict:
    out = {}
    out["log_loss"] = log_loss(y_true, pred, labels=[0, 1])

    if len(np.unique(y_true)) < 2:
        out["ROC_AUC"] = np.nan
        out["average_precision_score"] = np.nan
    else:
        out["ROC_AUC"] = roc_auc_score(y_true, pred)
        out["average_precision_score"] = average_precision_score(y_true, pred)

    out["n_test"] = int(len(y_true))
    out["n_cases_test"] = int(np.sum(y_true == 1))
    out["n_controls_test"] = int(np.sum(y_true == 0))
    return out


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--rep", required=True, help="Repetition number, e.g. 1")
    parser.add_argument("--x-train", required=True)
    parser.add_argument("--y-train", required=True)
    parser.add_argument("--x-test", required=True)
    parser.add_argument("--y-test", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--metrics-file", required=True)

    parser.add_argument("--pheno-cols", default="all", help="Comma-separated phenotype columns, or all")
    parser.add_argument("--feature-start", type=int, default=None, help="Use X.iloc[:, feature_start:] as in old code")

    parser.add_argument("--epoch-from", type=int, default=1)
    parser.add_argument("--epoch-to", type=int, default=40)
    parser.add_argument("--eta0", type=float, default=1e-5)
    parser.add_argument("--penalty", default="l2")
    parser.add_argument("--learning-rate", default="constant")
    parser.add_argument("--alpha", type=float, default=0.0001)
    parser.add_argument("--random-state", type=int, default=42)

    parser.add_argument("--skip-existing", action="store_true")

    args = parser.parse_args()
    start_time_all = time.time()

    ensure_dir(args.output_root)
    ensure_dir(os.path.dirname(args.metrics_file) or ".")

    pheno_cols = get_pheno_columns(args.y_train, requested=args.pheno_cols)
    print(f"Phenotypes to run ({len(pheno_cols)}): {pheno_cols}", flush=True)

    y_test_all = load_y_test_all(args.y_test)

    all_metrics = []

    for pheno_name in pheno_cols:
        start_time_pheno = time.time()
        pheno_out = os.path.join(args.output_root, f"rep{args.rep}", pheno_name)
        ensure_dir(pheno_out)

        final_model_file = os.path.join(pheno_out, "LogisticRegression_model.pkl")
        pred_file = os.path.join(pheno_out, "predictions_df.csv")

        print(f"\n================ {pheno_name} | rep {args.rep} ================", flush=True)

        if args.skip_existing and os.path.exists(final_model_file) and os.path.exists(pred_file):
            print(f"Skipping existing {pheno_name}", flush=True)
            pred = pd.read_csv(pred_file)[pheno_name].to_numpy()
        else:
            X_val, y_val = create_validation_set(
                args.x_train,
                args.y_train,
                pheno_name,
                feature_start=args.feature_start,
            )

            model = train_incremental_lr(
                x_train_file=args.x_train,
                y_train_file=args.y_train,
                X_val=X_val,
                y_val=y_val,
                pheno_name=pheno_name,
                output_path=pheno_out,
                epoch_from=args.epoch_from,
                epoch_to=args.epoch_to,
                eta0=args.eta0,
                penalty=args.penalty,
                learning_rate=args.learning_rate,
                alpha=args.alpha,
                random_state=args.random_state,
                feature_start=args.feature_start,
            )

            with open(final_model_file, "wb") as f:
                pickle.dump(model, f, protocol=4)

            pred = predict_all_chunks(model, args.x_test, feature_start=args.feature_start)
            pd.DataFrame({pheno_name: pred}).to_csv(pred_file, index=False)

        if pheno_name not in y_test_all.columns:
            raise ValueError(f"{pheno_name} not found in y-test file")

        y_true = y_test_all[pheno_name].astype(int).to_numpy().ravel()
        y_true = normalize_binary_labels(y_test_all[pheno_name]).to_numpy().ravel()
        
        if len(y_true) != len(pred):
            raise ValueError(
                f"Length mismatch for {pheno_name}: y_test={len(y_true)}, predictions={len(pred)}"
            )

        m = safe_metrics(y_true, pred)
        m.update(
            {
                "pheno": pheno_name,
                "rep": args.rep,
                "model": "SGDClassifier_log_loss",
                "eta0": args.eta0,
                "penalty": args.penalty,
                "learning_rate": args.learning_rate,
                "alpha": args.alpha,
                "epoch_from": args.epoch_from,
                "epoch_to": args.epoch_to,
                "feature_start": args.feature_start,
                "output_path": pheno_out,
                "time_minutes": (time.time() - start_time_pheno) / 60.0,
            }
        )
        all_metrics.append(m)

        # Append after each phenotype so partial results are saved even if a later phenotype fails.
        metrics_df = pd.DataFrame([m])
        if os.path.exists(args.metrics_file):
            metrics_df.to_csv(args.metrics_file, mode="a", header=False, index=False)
        else:
            metrics_df.to_csv(args.metrics_file, index=False)

        print(
            f"Done {pheno_name}: AUC={m['ROC_AUC']}, log_loss={m['log_loss']}, "
            f"AP={m['average_precision_score']}",
            flush=True,
        )

    print(f"\nAll done in {(time.time() - start_time_all) / 60.0:.2f} minutes.", flush=True)


if __name__ == "__main__":
    main()
