import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score, log_loss, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from collections import Counter
import numpy as np

import os
import sys

def calculate_metrics_for_final_scores(data_paths, pheno_path, pheno, rep , cov_path):
    final_scores = pd.DataFrame()

    for idx, data_path in enumerate(data_paths):
        data = pd.read_csv(data_path, delim_whitespace=True)
        data.columns = data.columns.str.strip()  # Clean column names early
        #print(data.columns)
        #print(data.shape)

        temp = data[['FID', 'IID', 'SCORESUM']].copy()
        temp.rename(columns={'SCORESUM': f'SCORESUM_{idx}'}, inplace=True)

        if final_scores.empty:
            final_scores = temp
        else:
            final_scores = pd.merge(final_scores, temp, on=['FID', 'IID'], how='outer')

    # Replace NaNs with 0 and sum across all SCORESUM_* columns
    score_cols = [col for col in final_scores.columns if col.startswith('SCORESUM_')]
    final_scores[score_cols] = final_scores[score_cols].fillna(0)
    final_scores['SUM_SCORESUM'] = final_scores[score_cols].sum(axis=1)

    # Merge with phenotype
    pheno_df = pd.read_csv(pheno_path, delim_whitespace=True)
    ###################################to be change#############################
    cov_df = pd.read_csv(cov_path, delim_whitespace=True)
    pheno_df.columns = ["FID", "IID",pheno.lower()]
    #print(pheno_df.shape)
    #################################################################
    merged_data = pd.merge(final_scores, cov_df, on=["FID", "IID"])
    #merged_data = final_scores
    print('merged_data')
    print(merged_data)
    merged_data = pd.merge(merged_data, pheno_df, on=["FID", "IID"])
    print('merged_data')
    print(merged_data)
    #print(merged_data.head())

    # Optional: Save final PRS
    merged_data.to_csv(f"/path/to/your/project/PRScs/PRScs_results/{pheno}/rep{rep}/final_PRS.final", index=False)
    #print("Available columns:", merged_data.columns.tolist())
    print(merged_data.head())
    phenotype = merged_data[pheno.lower()]

    metrics = {'pheno': pheno.lower(), 'rep': rep}

    if phenotype.nunique() == 2:

        #print(Counter(phenotype))

        X = merged_data.iloc[:, 2:-1]
        y = (phenotype > 1).astype(int)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        print('X_train.head()')
        print(X_train.head())
        print('X_test.head()')
        print(X_test.head())
        #print(Counter(y_train))

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        clf = LogisticRegression(random_state=42, max_iter=1000)
        #print(y_train)
        clf.fit(X_train_scaled, y_train)

        y_pred_prob = clf.predict_proba(X_test_scaled)[:, 1]

        metrics['roc_auc'] = roc_auc_score(y_test, y_pred_prob)
        metrics['log_loss'] = log_loss(y_test, y_pred_prob)
        metrics['average_precision'] = average_precision_score(y_test, y_pred_prob)

        print("Binary phenotype metrics (Logistic Regression):")
        print(metrics)

        print("Binary phenotype metrics:")
        print(metrics)

    else:
        X = merged_data.iloc[:, 2:-1]
        y = phenotype

        mask = ~np.isnan(y)

        X = X[mask]
        y = y[mask]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)


    
        # Now fit the model
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)
        print("Number of NaNs in y_train:", np.isnan(y_train).sum())

        y_pred = model.predict(X_test_scaled)

        metrics['r2'] = r2_score(y_test, y_pred)
        metrics['mse'] = mean_squared_error(y_test, y_pred)

        print("Continuous phenotype metrics:")
        print(metrics)

    # Save metrics to a CSV file (append mode)
    output_csv = f"/path/to/your/project/PRScs/PRScs_results/metrics_summary.csv"
    file_exists = os.path.exists(output_csv)

    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(output_csv, mode='a', index=False, header=not file_exists)

# Entry point for the script
if __name__ == "__main__":
    pheno = sys.argv[1]
    rep = sys.argv[2]

    data_paths = [
        f"/path/to/your/project/PRScs/PRScs_results/{pheno}/rep{rep}/PRS_results_chr{chro}_rep{rep}.profile"
        for chro in range(1, 23)
    ]

    pheno_path = f"/path/to/your/project/phenotypes/{pheno.lower()}"
    cov_path = f"/path/to/your/project/cov_matrix/rep{rep}/cov_matrix_MinMax_scaled_no_missing.txt"
    calculate_metrics_for_final_scores(data_paths, pheno_path, pheno, rep, cov_path)
