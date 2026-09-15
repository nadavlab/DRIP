from sklearn import datasets
from sklearn.metrics import r2_score,log_loss, roc_auc_score, roc_curve, average_precision_score,mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import os
from sklearn.linear_model import LinearRegression , LogisticRegression
from sklearn.preprocessing import StandardScaler
from collections import Counter


pheno=sys.argv[1]
pheno_low=pheno.lower()
rep=sys.argv[2]
y_score = pd.read_csv("/path/to/your/project/PRSice_results/"+pheno+"/rep"+str(rep)+"/PRS_"+pheno+"_no-clump_cov_MinMax_scaled.best",sep=" ")
y_test = pd.read_csv("/path/to/your/project/phenotypes/"+pheno_low,sep=' ')
#cov = pd.read_csv("/path/to/your/project/cov_matrix/rep"+str(rep)+"/cov_matrix_MinMax_scaled_no_missing.txt",sep='\t')
print(y_score.head(10))
print(y_test.head(10))

#print(cov.head(10))

y_test.iloc[:, 2] = y_test.iloc[:, 2].replace([1], 0)
y_test.iloc[:, 2] = y_test.iloc[:, 2].replace([2],1)
print(y_test.head(10))

y_score['FID'] = y_score['FID'].astype(int)
y_test['FID'] = y_test['FID'].astype(int)

#Y_score_with_cov = pd.merge(y_score, cov, on=['FID', 'IID'])

print(y_score.shape)
#print(Y_score_with_cov.shape)


print(y_score.head(10))
print(y_test.head(10))
print(y_score.columns.tolist())
print(y_test.columns.tolist())
print(len(y_score.index))
print(len(y_test.index))

#merge_data=pd.merge(Y_score_with_cov,y_test,on=["FID","IID"],how="inner")
merge_data=pd.merge(y_score,y_test,on=["FID","IID"],how="inner")

print(merge_data.head(10))
print(len(merge_data.index))

merge_data.replace({'Yes': 1, 'No': 2}, inplace=True)

#merge_data.to_csv("/path/to/your/project/PRSice_results/"+pheno+"/rep"+str(rep)+"/"+pheno+"_merge_PRS_score_and_cov_with_pheno.csv")
phenotype = merge_data.iloc[:,-1]

metrics = {'pheno': pheno.lower(), 'rep': rep}

if merge_data.iloc[:,-1].nunique() == 2:

    print(Counter(phenotype))

    X = merge_data.iloc[:, 3:-1]
    y = (phenotype > 0).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    print('X_train.head()')
    print(X_train.head())
    print('X_test.head()')
    print(X_test.head())
    # print(Counter(y_train))

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = LogisticRegression(random_state=42, max_iter=1000)
    print(Counter(y_train))
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
    X = merge_data.iloc[:, 3:-1]
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

#output_csv = f"/path/to/your/project/PRSice_results/metrics_summary.csv"
output_csv = f"/path/to/your/project/PRSice_results/metrics_no_cov_summary.csv"

file_exists = os.path.exists(output_csv)

metrics_df = pd.DataFrame([metrics])
metrics_df.to_csv(output_csv, mode='a', index=False, header=not file_exists)




'''
false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(merge_data.iloc[:,-1],merge_data['PRS'])

print('log_loss=', log_loss(merge_data.iloc[:,-1], merge_data['PRS']))
fpr, tpr, thresholds = roc_curve(merge_data.iloc[:,-1], merge_data['PRS'])
print('roc_auc_score=',roc_auc_score(merge_data.iloc[:,-1], merge_data['PRS']))
print('average_precision_score =',average_precision_score(merge_data.iloc[:,-1], merge_data['PRS']))
#plot_roc_curve(fpr, tpr)


plt.subplots(1, figsize=(10,10))
plt.title('Receiver Operating Characteristic -'+pheno)
plt.plot(false_positive_rate1, true_positive_rate1)
plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig("/path/to/your/project/PRSice_results/"+pheno+"/rep"+str(rep)+"/_ROC.jpg")
#plt.show()
'''
print("finished")


