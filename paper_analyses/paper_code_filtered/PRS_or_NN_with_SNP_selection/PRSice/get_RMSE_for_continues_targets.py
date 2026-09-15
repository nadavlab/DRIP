from sklearn import datasets
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import math


from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
from statistics import mean


pheno=sys.argv[1]
pheno_low=pheno.lower()
rep=sys.argv[2]
y_score = pd.read_csv("/path/to/your/project/PRSice_results/"+pheno+"/rep"+str(rep)+"/PRS_"+pheno+"_no-clump_cov_MinMax_scaled.best",sep=" ")
y_test = pd.read_csv("/path/to/your/project/phenotypes/"+pheno_low,sep='\t')

print(y_test.head(10))

merge_data=pd.merge(y_score,y_test,right_on="FID",left_on="FID",how="inner")

print(merge_data.head(10))
print(len(merge_data.index))

merge_data=merge_data.dropna()
print(len(merge_data.index))

merge_data.to_csv("/path/to/your/project/PRSice_results/"+pheno+"/rep"+str(rep)+"/"+pheno+"_merge_PRS_score_with_pheno.csv")

# Replace these arrays with your actual data
prs_scores = np.array(merge_data.iloc[:,3])

observed_phenotypes = np.array(merge_data.iloc[:,5])
print("observed_phenotypes")
print(observed_phenotypes)

# Res8hape the data if needed
prs_scores = prs_scores.reshape(-1, 1)

# Standardize PRS scores
#scaler = StandardScaler()
#prs_scores_scaled = scaler.fit_transform(prs_scores)

#print("prs_scores_scaled")
#print(prs_scores_scaled)
# Define the model
model = LinearRegression()

# Fit the model with scaled PRS scores
#model.fit(prs_scores_scaled, observed_phenotypes)
model.fit(prs_scores, observed_phenotypes)

# Predict phenotypes using scaled PRS scores
#predicted_phenotypes_scaled = model.predict(prs_scores_scaled)
predicted_phenotypes = model.predict(prs_scores)

print("predicted_phenotypes")
print(predicted_phenotypes)

print("mean(predicted_phenotypes)")
print(mean(predicted_phenotypes))

print("mean(observed_phenotypes)")
print(mean(observed_phenotypes))

# Calculate residuals
#residuals = observed_phenotypes - predicted_phenotypes_scaled
residuals = observed_phenotypes - predicted_phenotypes

# Square residuals
squared_residuals = residuals**2
my_rmse=math.sqrt(sum(squared_residuals)/len(squared_residuals))
print("my_rmse:",my_rmse)
# Calculate Mean Squared Error
rmse = math.sqrt(mean_squared_error(observed_phenotypes, predicted_phenotypes))

print("Root Mean Squared Error:", rmse)

my_R2 = 1 - (sum((observed_phenotypes-predicted_phenotypes)**2)/sum((observed_phenotypes-mean(observed_phenotypes))**2))
print("my_R2: ", my_R2)
r2 = r2_score(observed_phenotypes, predicted_phenotypes)
print("R2: ", r2)


