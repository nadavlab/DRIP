#!/usr/bin/env python3
import argparse
import os
import time
import pickle
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import log_loss, roc_auc_score, average_precision_score

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


def setup_gpu_memory_growth():
    try:
        gpus = tf.config.experimental.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        print(f"Could not set GPU memory growth: {e}")

def normalize_column_name(col):
    """
    Convert unusual column labels into ordinary strings.

    Examples:
        ('0_chr1',)       -> '0_chr1'
        ['0_chr1']        -> '0_chr1'
        np.array(['BMI']) -> 'BMI'
        'age'             -> 'age'
    """
    if isinstance(col, tuple):
        if len(col) == 1:
            return str(col[0])
        return "_".join(str(x) for x in col)

    if isinstance(col, (list, np.ndarray)):
        values = np.asarray(col).reshape(-1).tolist()

        if len(values) == 1:
            return str(values[0])

        return "_".join(str(x) for x in values)

    return str(col)

def iter_pickle_chunks(path):
    with open(path, "rb") as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break


def clean_x_chunk(X, feature_start=None, drop_cols=("FID", "IID", "#FID")):
    """
    Feature selection for the chr1-PC experiment.

    Keeps:
    - all regular covariates
    - all UKB PCs
    - only our created PCs from chromosome 1

    Removes:
    - our created PCs from chromosomes 2-22

    Expected created-PC column names:
        0_chr1, 1_chr1, ...
        0_chr2, 1_chr2, ...
    """
    X = X.copy()
    
    print("Original X column-index type:", type(X.columns), flush=True)
    print("First 10 original X columns:", list(X.columns[:10]), flush=True)

    X.columns = [normalize_column_name(col) for col in X.columns]

    print("First 10 normalized X columns:", list(X.columns[:10]), flush=True)

    if feature_start is not None:
        raise ValueError(
            "Do not use --feature-start for the chr1-PC experiment. "
            "Features are selected according to their column names."
        )

    # Remove participant identifier columns.
    id_cols_to_drop = [
        col for col in drop_cols
        if col in X.columns
    ]
    X = X.drop(columns=id_cols_to_drop, errors="ignore")

    created_pc_cols = []
    chr1_pc_cols = []
    created_pc_cols_to_remove = []

    for col in X.columns:
        col_str = str(col)

        # Created PC names are expected to end with _chr<number>.
        if "_chr" not in col_str:
            continue

        prefix, chromosome_suffix = col_str.rsplit("_chr", 1)

        # Require a numeric PC prefix and a numeric chromosome suffix.
        # This avoids accidentally removing unrelated covariates containing "_chr".
        if not prefix.isdigit() or not chromosome_suffix.isdigit():
            continue

        chromosome = int(chromosome_suffix)

        if 1 <= chromosome <= 22:
            created_pc_cols.append(col)

            if chromosome == 1:
                chr1_pc_cols.append(col)
            else:
                created_pc_cols_to_remove.append(col)

    if len(created_pc_cols) == 0:
        raise ValueError(
            "No created chromosome-specific PCs were detected. "
            "Expected columns such as 0_chr1, 1_chr1, 0_chr2, etc."
        )

    if len(chr1_pc_cols) == 0:
        raise ValueError(
            "Created PCs were detected, but no chromosome 1 PCs were found. "
            "Expected names such as 0_chr1, 1_chr1, etc."
        )

    X = X.drop(columns=created_pc_cols_to_remove, errors="ignore")

    print(
        "Feature selection summary: "
        f"total retained={X.shape[1]}, "
        f"chr1 created PCs retained={len(chr1_pc_cols)}, "
        f"created PCs from chr2-22 removed={len(created_pc_cols_to_remove)}, "
        f"other covariates/features retained="
        f"{X.shape[1] - len(chr1_pc_cols)}",
        flush=True,
    )

    return X.astype(np.float32)


def get_y_for_pheno(y_batch, pheno_name, print_labels=False):
    if pheno_name not in y_batch.columns:
        raise ValueError(
            f"Phenotype '{pheno_name}' was not found in Y batch. "
            f"Available columns: {list(y_batch.columns)}"
        )

    y = pd.to_numeric(y_batch[pheno_name], errors="raise").astype(int)

    # Convert 1/2 labels to 0/1
    unique_labels = sorted(y.unique())
    if print_labels:
        print(f"{pheno_name} labels: {unique_labels}")

    if set(unique_labels).issubset({1, 2}):
        y = y.replace({1: 0, 2: 1})
    elif not set(unique_labels).issubset({0, 1}):
        raise ValueError(
            f"Unexpected labels for {pheno_name}: {unique_labels}"
        )

    return y.to_numpy().reshape(-1, 1)

def get_pheno_columns(y_train_file, requested=None):
    first_y = next(iter_pickle_chunks(y_train_file))
    id_cols = {"FID", "IID", "#FID"}

    if requested:
        phenos = [p.strip() for p in requested.split(",") if p.strip()]
    else:
        phenos = [c for c in first_y.columns if c not in id_cols]

    if not phenos:
        raise ValueError("No phenotype columns found.")

    missing = [p for p in phenos if p not in first_y.columns]
    if missing:
        raise ValueError(f"Requested phenotype columns not found: {missing}")

    return phenos


def create_validation_set(x_train_file, y_train_file, pheno_name, feature_start=None):
    """
    Uses the first chunk as validation, same logic as the original code.
    """
    X_val_raw = next(iter_pickle_chunks(x_train_file))
    y_val_raw = next(iter_pickle_chunks(y_train_file))
    print("##########y_val_raw.columns###########")
    print(y_val_raw.columns)
    X_val = clean_x_chunk(X_val_raw, feature_start=feature_start)
    y_val = get_y_for_pheno(y_val_raw, pheno_name,True)

    return X_val, y_val


def build_model(number_features, learning_rate=1e-4, dropout=0.1):
    """
    Same architecture idea as original:
    Dense(number_features*0.2) -> Dropout -> Dense(number_features*0.1) -> Dropout -> sigmoid
    """
    units1 = max(1, int(number_features * 0.2))
    units2 = max(1, int(number_features * 0.1))

    model = models.Sequential(name="NN")
    model.add(layers.Dense(units=units1, input_shape=[number_features]))
    model.add(layers.PReLU())
    model.add(layers.Dropout(dropout))

    model.add(layers.Dense(units=units2))
    model.add(layers.PReLU())
    model.add(layers.Dropout(dropout))

    model.add(layers.Dense(units=1, activation="sigmoid"))

    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        optimizer=Adam(learning_rate=learning_rate),
    )

    return model


def plot_loss(train_loss, val_loss, pheno_name, output_path):
    if not train_loss or not val_loss:
        return

    fig = plt.figure()
    plt.plot(train_loss, label="Train Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.title("NN Loss for " + pheno_name)
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(loc="upper left")
    plt.ylim(0, max(1, max(train_loss + val_loss)))
    fig.savefig(os.path.join(output_path, "loss.png"), bbox_inches="tight")
    plt.close(fig)


def fit_nn(
    x_train_file,
    y_train_file,
    pheno_name,
    output_path,
    num_epochs=80,
    batch_size=50,
    learning_rate=1e-7,
    dropout=0.1,
    patience=5,
    min_delta=1e-4,
    feature_start=None,
):
    X_val, y_val = create_validation_set(
        x_train_file=x_train_file,
        y_train_file=y_train_file,
        pheno_name=pheno_name,
        feature_start=feature_start,
    )

    model = build_model(
        number_features=X_val.shape[1],
        learning_rate=learning_rate,
        dropout=dropout,
    )
    model.summary()

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1,
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
        verbose=1,
    )

    train_loss_per_epoch = []
    val_loss_per_epoch = []
    best_val_loss = np.inf
    epochs_since_improvement = 0

    for epoch in range(1, num_epochs + 1):
        print(f"\nPhenotype {pheno_name} | Epoch {epoch}/{num_epochs}")

        batch_losses = []
        batch_num = 1

        x_iter = iter_pickle_chunks(x_train_file)
        y_iter = iter_pickle_chunks(y_train_file)

        for X_raw, y_raw in zip(x_iter, y_iter):
            X_batch = clean_x_chunk(X_raw, feature_start=feature_start)
            y_batch = get_y_for_pheno(y_raw, pheno_name,False)

            # Keep first chunk as validation, as in the original code.
            if batch_num == 1:
                batch_num += 1
                continue

            history = model.fit(
                X_batch,
                y_batch,
                epochs=1,
                batch_size=batch_size,
                verbose=0,
            )

            batch_losses.append(float(history.history["loss"][-1]))
            batch_num += 1

        if len(batch_losses) == 0:
            raise ValueError(
                f"No training batches found for {pheno_name}. "
                "Only validation chunk was found."
            )

        epoch_train_loss = float(np.mean(batch_losses))
        epoch_val_loss = float(model.evaluate(X_val, y_val, verbose=0))

        train_loss_per_epoch.append(epoch_train_loss)
        val_loss_per_epoch.append(epoch_val_loss)

        print(f"Train loss: {epoch_train_loss:.5f} | Val loss: {epoch_val_loss:.5f}")

        #reduce_lr.on_epoch_end(epoch, logs={"val_loss": epoch_val_loss})

        if epoch_val_loss < best_val_loss - min_delta:
            best_val_loss = epoch_val_loss
            epochs_since_improvement = 0
            model.save(os.path.join(output_path, "best_model.keras"))
        else:
            epochs_since_improvement += 1

        if epochs_since_improvement >= patience:
            print(f"Early stopping triggered at epoch {epoch}")
            break

        if epoch % 10 == 0:
            model.save(os.path.join(output_path, f"model_epoch_{epoch}.keras"))

    plot_loss(train_loss_per_epoch, val_loss_per_epoch, pheno_name, output_path)
    return model, train_loss_per_epoch, val_loss_per_epoch


def predict_pheno(model, x_test_file, feature_start=None):
    predictions = []

    for X_raw in iter_pickle_chunks(x_test_file):
        X_batch = clean_x_chunk(X_raw, feature_start=feature_start)
        pred = model.predict(X_batch, verbose=0).flatten()
        predictions.extend(pred.tolist())

    return np.array(predictions, dtype=float)


def load_test_y(y_test_file, pheno_name):
    """
    Load test labels and normalize binary labels from 1/2 to 0/1.
    Supports both a single dataframe pickle and a chunked pickle.
    """
    chunks = list(iter_pickle_chunks(y_test_file))

    if not chunks:
        raise ValueError(f"No objects found in {y_test_file}")

    y_df = (
        pd.concat(chunks, ignore_index=True)
        if len(chunks) > 1
        else chunks[0]
    )

    y_df = y_df.rename(columns={"#FID": "FID"})

    if pheno_name not in y_df.columns:
        raise ValueError(
            f"Phenotype '{pheno_name}' not found in {y_test_file}"
        )

    y = pd.to_numeric(y_df[pheno_name], errors="coerce")

    if y.isna().any():
        raise ValueError(
            f"Missing or non-numeric test labels found for {pheno_name}"
        )

    y = y.astype(int)
    unique_labels = set(y.unique())

    if unique_labels.issubset({1, 2}):
        y = y.replace({1: 0, 2: 1})
    elif not unique_labels.issubset({0, 1}):
        raise ValueError(
            f"Unexpected test labels for {pheno_name}: "
            f"{sorted(unique_labels)}"
        )

    return y.to_numpy()


def evaluate_predictions(y_true, predictions):
    unique = np.unique(y_true)

    metrics = {
        "n_test": int(len(y_true)),
        "n_cases_test": int(np.sum(y_true == 1)),
        "n_controls_test": int(np.sum(y_true == 0)),
        "log_loss": np.nan,
        "ROC_AUC": np.nan,
        "average_precision_score": np.nan,
    }

    if len(unique) < 2:
        warnings.warn("Only one class in y_true; ROC_AUC/AP/log_loss may be undefined.")

    try:
        metrics["log_loss"] = float(log_loss(y_true, predictions, labels=[0, 1]))
    except Exception as e:
        print(f"log_loss failed: {e}")

    try:
        metrics["ROC_AUC"] = float(roc_auc_score(y_true, predictions))
    except Exception as e:
        print(f"ROC_AUC failed: {e}")

    try:
        metrics["average_precision_score"] = float(average_precision_score(y_true, predictions))
    except Exception as e:
        print(f"average_precision_score failed: {e}")

    return metrics


def append_metrics(metrics_file, row):
    os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
    df = pd.DataFrame([row])
    if os.path.exists(metrics_file):
        df.to_csv(metrics_file, mode="a", header=False, index=False)
    else:
        df.to_csv(metrics_file, index=False)


def main():
    parser = argparse.ArgumentParser(
        description="Train the original-style NN on all simulated phenotypes for one repetition."
    )

    parser.add_argument("--rep", required=True)
    parser.add_argument("--matched-root", default="/path/to/your/project//simulations/matched_data_ids/", help="Root directory containing rep*/X_train_match_to_pheno.pkl etc. PCA-only matched files.")
    parser.add_argument("--x-train", default=None, help="Optional override for X_train_match_to_pheno.pkl")
    parser.add_argument("--y-train", default=None, help="Optional override for Y_train_all.pkl")
    parser.add_argument("--x-test", default=None, help="Optional override for X_test_match_to_pheno.pkl")
    parser.add_argument("--y-test", default=None, help="Optional override for Y_test_all.pkl")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--metrics-file", required=True)

    parser.add_argument(
        "--pheno-cols",
        default=None,
        help="Optional comma-separated phenotype columns. If omitted, all non-FID/IID columns are used.",
    )
    parser.add_argument(
        "--feature-start",
        type=int,
        default=None,
        help="Optional column index to mimic old X.iloc[:, feature_start:]. If omitted, drops FID/IID and uses all remaining columns.",
    )

    parser.add_argument("--max-epochs", "--epochs", dest="epochs", type=int, default=1000, help="Maximum epochs. Training stops earlier after convergence by validation loss.")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=5, help="Stop after this many epochs without sufficient validation-loss improvement.")
    parser.add_argument("--min-delta", type=float, default=1e-4, help="Minimum validation-loss improvement required to reset patience.")

    args = parser.parse_args()

    # PCA-only default paths, based on the matched files created for the simulations.
    matched_dir = args.matched_root
    args.x_train = args.x_train or os.path.join(matched_dir, f"X_train_match_to_pheno_rep_{args.rep}.pkl")
    args.y_train = args.y_train or os.path.join(matched_dir, f"Y_train_all_rep_{args.rep}.pkl")
    args.x_test = args.x_test or os.path.join(matched_dir, f"X_test_match_to_pheno_rep_{args.rep}.pkl")
    args.y_test = args.y_test or os.path.join(matched_dir, f"Y_test_all_rep_{args.rep}.pkl")

    setup_gpu_memory_growth()

    print("=" * 80)
    print("Requested phenotype:")
    print(args.pheno_cols)
    print("=" * 80)

    first_y = next(iter_pickle_chunks(args.y_train))

    print("Available phenotype columns:")
    for col in first_y.columns:
        print(col)

    print("=" * 80)

    phenos = get_pheno_columns(args.y_train, requested=args.pheno_cols)
    print("Using PCA matched simulation files:")
    print("X train:", args.x_train)
    print("Y train:", args.y_train)
    print("X test :", args.x_test)
    print("Y test :", args.y_test)
    print(f"Phenotypes to run ({len(phenos)}): {phenos}")

    for pheno_name in phenos:
        start_time = time.time()

        pheno_output = os.path.join(args.output_root, f"rep{args.rep}", pheno_name)
        os.makedirs(pheno_output, exist_ok=True)

        print("\n" + "=" * 80)
        print(f"Running NN | rep={args.rep} | pheno={pheno_name}")
        print("=" * 80)

        model, train_losses, val_losses = fit_nn(
            x_train_file=args.x_train,
            y_train_file=args.y_train,
            pheno_name=pheno_name,
            output_path=pheno_output,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            dropout=args.dropout,
            patience=args.patience,
            min_delta=args.min_delta,
            feature_start=args.feature_start,
        )

        final_model_path = os.path.join(pheno_output, "final_model.keras")
        model.save(final_model_path)

        predictions = predict_pheno(
            model=model,
            x_test_file=args.x_test,
            feature_start=args.feature_start,
        )

        pred_df = pd.DataFrame({"prediction": predictions})
        pred_df.to_csv(os.path.join(pheno_output, "predictions.csv"), index=False)

        y_true = load_test_y(args.y_test, pheno_name)
        metrics = evaluate_predictions(y_true, predictions)

        row = {
            "rep": args.rep,
            "pheno": pheno_name,
            "model": "NN",
            "epochs_requested": args.epochs,
            "epochs_trained": len(train_losses),
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "dropout": args.dropout,
            "patience": args.patience,
            "min_delta": args.min_delta,
            "feature_start": args.feature_start,
            "final_train_loss": train_losses[-1] if train_losses else np.nan,
            "final_val_loss": val_losses[-1] if val_losses else np.nan,
            "best_val_loss": min(val_losses) if val_losses else np.nan,
            "time_minutes": (time.time() - start_time) / 60.0,
            **metrics,
        }

        append_metrics(args.metrics_file, row)

        # Clear memory between phenotypes
        tf.keras.backend.clear_session()

    print("Done.")


if __name__ == "__main__":
    main()
