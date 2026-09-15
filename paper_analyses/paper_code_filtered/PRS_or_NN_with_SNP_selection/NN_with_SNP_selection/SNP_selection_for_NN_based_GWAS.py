import pandas as pd
import numpy as np
import sys
import os
import subprocess


#vec=[1,2,3,4]
#vec=np.array(vec)
#print(vec[vec<4])

pheno=sys.argv[1]
type=sys.argv[2]

if type=="l":
    file_name_end=".glm.linear"
elif type=="b":
    file_name_end = ".glm.logistic.hybrid"

    #os.makedirs("/path/to/your/project/sagnificant_for_SNP_selection/")

if not os.path.exists("/path/to/your/project/NN_with_SNP_selection/sagnificant_for_SNP_selection/"+pheno):
    os.makedirs("/path/to/your/project/NN_with_SNP_selection/sagnificant_for_SNP_selection/"+pheno)
    print("1created")
else:
    print("/path/to/your/project/sagnificant_for_SNP_selection/"+pheno+" exist")


if not os.path.exists("/path/to/your/project/NN_with_SNP_selection/training_bed_files/"+pheno):
    os.makedirs("/path/to/your/project/NN_with_SNP_selection/training_bed_files/"+pheno)
    print("2created")

else:
    print("/path/to/your/project/NN_with_SNP_selection/training_bed_files/"+pheno+" exist")
    
if not os.path.exists("/path/to/your/project/NN_with_SNP_selection/test_bed_files/"+pheno):
    os.makedirs("/path/to/your/project/NN_with_SNP_selection/test_bed_files/"+pheno)
    print("3created")
else:
    print("/path/to/your/project/NN_with_SNP_selection/test_bed_files/"+pheno+" exist ")

for rep in range(1,6):
    if not os.path.exists("/path/to/your/project/NN_with_SNP_selection/sagnificant_for_SNP_selection/"+pheno+"/rep"+str(rep)):
    	os.makedirs("/path/to/your/project/NN_with_SNP_selection/sagnificant_for_SNP_selection/"+pheno+"/rep"+str(rep))
       # print("4created")
    else:
        print("/path/to/your/project/NN_with_SNP_selection/NN_with_SNP_selection/sagnificant_for_SNP_selection/"+pheno+"/rep"+str(rep)+" exist ")

    if not os.path.exists("/path/to/your/project/NN_with_SNP_selection/training_bed_files/"+pheno+"/rep"+str(rep)):
        os.makedirs("/path/to/your/project/NN_with_SNP_selection/training_bed_files/"+pheno+"/rep"+str(rep))
        print("5created")
    else:
        print("/path/to/your/project/NN_with_SNP_selection/training_bed_files/"+pheno+"/rep"+str(rep)+ " exist ")

    if not os.path.exists("/path/to/your/project/NN_with_SNP_selection/test_bed_files/"+pheno+"/rep"+str(rep)):
        os.makedirs("/path/to/your/project/NN_with_SNP_selection/test_bed_files/"+pheno+"/rep"+str(rep))
        print("6created")
    else:
        print("/path/to/your/project/NN_with_SNP_selection/test_bed_files/"+pheno+"/rep"+str(rep)+ " exist ")


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

    #write the data to new dataset
    for line in lines:
        sentence = line.split()

        if float(sentence[len(sentence)-3]) <= threshold:
            sagnificant_data.write(' '.join(sentence) + "\n")

    #creat SNPs list with for the selested SNPs
    print(np.array(vec_pvalue)<=threshold)
    sagnificant_snp=np.array(vec_snpname)[np.array(vec_pvalue)<=threshold]

    #write the list to file
    file_snp = open('/path/to/your/project/NN_with_SNP_selection/sagnificant_for_SNP_selection/'+pheno+'/rep'+str(rep)+'/'+pheno+"_filtered_snp_list.snplist", "w")
    sagnificant_snp=np.array(sagnificant_snp)

    for snp in sagnificant_snp:
        #content = str(sagnificant_snp)
        file_snp.write(snp+"\n")
    file_snp.close()


    data.close()
    sagnificant_data.close()



    bash_script = """
#!/bin/bash
## $1 is the pheno
## $2 is the rep

pheno="$1"
rep="$2"

echo
echo $2
echo
echo "$2"
echo
echo $pheno
echo
echo ${pheno}
echo
echo ${1}
echo
echo ${{pheno}}
echo

for x in {1..22}
do
    sbatch --mem=100g --time=5:00:00 -c 48 --job-name=filt_snpc$3r$2 -o /path/to/your/project/NN_with_SNP_selection/training_bed_files/$1/rep$2/chr${x}.out --wrap="/path/to/your/project/plink2 --bed /path/to/your/project/training_bed_files/rep$rep/chr${x}_X_train_no_cov_no_missing.bed --fam /path/to/your/project/training_bed_files/rep$rep/chr${x}_X_train_no_cov_no_missing.fam --bim /path/to/your/project/training_bed_files/rep$rep/chr${x}_X_train_no_cov_no_missing.bim --extract /path/to/your/project/NN_with_SNP_selection/sagnificant_for_SNP_selection/$1/rep$2/$1_filtered_snp_list.snplist --make-bed --out /path/to/your/project/NN_with_SNP_selection/training_bed_files/$1/rep$2/chr${x}_training_filtered_snp_10_precent"
    sbatch --mem=100g --time=5:00:00 -c 48 --job-name=filt_snpc$3r$2 -o /path/to/your/project/NN_with_SNP_selection/test_bed_files/$pheno/rep$rep/chr${x}.out --wrap="/path/to/your/project/plink2 --bed /path/to/your/project/test_bed_files/rep$rep/chr${x}_X_test_no_cov_no_missing.bed --fam /path/to/your/project/test_bed_files/rep$rep/chr${x}_X_test_no_cov_no_missing.fam --bim /path/to/your/project/test_bed_files/rep$rep/chr${x}_X_test_no_cov_no_missing.bim --extract /path/to/your/project/NN_with_SNP_selection/sagnificant_for_SNP_selection/$pheno/rep$rep/$1_filtered_snp_list.snplist --make-bed --out /path/to/your/project/NN_with_SNP_selection/test_bed_files/$1/rep$2/chr${x}_test_filtered_snp_10_precent"
done
"""
	

    result = subprocess.run(['bash', '-c', bash_script, '_', pheno, str(rep)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Check the result
    if result.returncode == 0:
        print("Script executed successfully.")
        print("Output:")
        print(result.stdout)
    else:
        print("Script execution failed.")
        print("Error:")
        print(result.stderr)


