import pandas as pd
import numpy as np
import sys
import os
import subprocess



pheno=sys.argv[1]
type=sys.argv[2]

if type=="l":
    file_name_end=".glm.linear"
elif type=="b":
    file_name_end = ".glm.logistic.hybrid"

for rep in range(1,6):

    #read dataframe
    data = open('/path/to/your/project/merge_GWAS_results/'+pheno+'/rep'+str(rep)+'/all_chr_train_cov_MinMax_scaled_with_values_A2.'+pheno+file_name_end, "r")
    sagnificant_data = open('/path/to/your/project/NN_with_SNP_selection/sagnificant_for_SNP_selection/'+pheno+'/rep'+str(rep)+'/sagnificant_all_chr_train_cov_MinMax_scaled_with_values_A2.'+pheno+file_name_end, "w")
    #print(len(data.columns.values.tolist()[0].split('\t')))
    #data.columns= data.columns.values.tolist()[0].split('\t')

    #write relevant rows to the new file
    first_line = data.readline()
    sagnificant_data.write(first_line)
    lines = data.readlines()
    print(first_line)

    #creat vectors for SNPs and their Pvalues
    vec_pvalue=[0 for x in range(len(lines))]
    vec_snpname=["_" for x in range(len(lines))]

    x=0
    for line in lines:
        sentence = line.split()
        vec_pvalue[x]=float(sentence[len(sentence)-3])
        vec_snpname[x]=sentence[2]
        x=x+1

    #colculate the 10 % threshold
    print("threshold")
    threshold=np.sort(vec_pvalue)[round(len(vec_pvalue)*1/10)]
    print(threshold)
    print(vec_snpname[1:10])
