
# Load necessary library
library(data.table)

split_tsv_by_chromosome <- function(input_path, output_dir,type) {
  
  # Read the TSV file
  data <- fread(input_path, sep = " ", header = TRUE)
  
  print(head(data))

  # Loop through chromosomes 1 to 22
  for (chr_num in 1:22) {
    # Filter the data for the current chromosome
    #print(data$CHROM[1:10])
    chr_data <- data[data$CHROM == chr_num, ]
    #print(nrow(chr_data))
    
    if (type=="b")
    {
    PRScs_format_data<-chr_data[,c("ID","A1","A2","OR","LOG(OR)_SE")]
    names(PRScs_format_data)<-c("SNP","A1","A2" ,"OR","SE")
    }
    if (type=="l")
    {
      PRScs_format_data<-chr_data[,c("ID","A1","A2","BETA","SE")]
      names(PRScs_format_data)<-c("SNP","A1","A2" ,"BETA","SE")
    }
    # Define output file path
    output_file <- file.path(output_dir, paste0("chr", chr_num, ".tsv"))
    
    # Write the subset to a new TSV file
    fwrite(PRScs_format_data, output_file, sep = "\t", quote = FALSE, row.names = FALSE)
  }
}

binary_phenotypes <-c("Hypertension","Type_2_Diabetes","Multiple_Sclerosis","Parkinson","Alzheimer","Schizophrenia")
continues_phenotypes <-c("Height","Platelet_Count","BMI","Systolic_Blood_Pressure")
binary_phenotypes <-c("Hypertension","Type_2_Diabetes")
#continues_phenotypes <-c("Height")

for (pheno in binary_phenotypes)
  {
    for (rep in 1:5)
    {
      input_path<-paste0("/path/to/your/project/merge_GWAS_results/",pheno,"/rep",rep,"/all_chr_train_cov_MinMax_scaled_with_values_A2.",pheno,".glm.logistic.hybrid.spaces_fixed")
      output_path1<-paste0("/path/to/your/project/PRScs/SS_formated_data/",pheno)
      if (!file.exists(output_path1)){
        dir.create(output_path1, showWarnings = FALSE)
      }
      output_path2<-paste0(output_path1,"/rep",rep)
      print(output_path2)
      if (!file.exists(output_path2)){
        dir.create(output_path2, showWarnings = FALSE)
      }
      output_path3<-paste0(output_path2,"/GWAS_results_PRScs_formatted")
      print(output_path3)
      if (!file.exists(output_path3)){
        dir.create(output_path3, showWarnings = FALSE)
      }
      split_tsv_by_chromosome(input_path, output_path3,"b")
    }
}

for (pheno in continues_phenotypes)
{
  for (rep in 1:5)
  {
    input_path<-paste0("/path/to/your/project/merge_GWAS_results/",pheno,"/rep",rep,"/all_chr_train_cov_MinMax_scaled_with_values_A2.",pheno,".glm.linear")
    output_path1<-paste0("/path/to/your/project/PRScs/SS_formated_data/",pheno)
    if (!file.exists(output_path1)){
      dir.create(output_path1, showWarnings = FALSE)
    }
    output_path2<-paste0(output_path1,"/rep",rep)
    print(output_path2)
    if (!file.exists(output_path2)){
      dir.create(output_path2, showWarnings = FALSE)
    }
    output_path3<-paste0(output_path2,"/GWAS_results_PRScs_formatted")
    print(output_path3)
    if (!file.exists(output_path3)){
      dir.create(output_path3, showWarnings = FALSE)
    }
    split_tsv_by_chromosome(input_path, output_path3,"l")
  }
}

