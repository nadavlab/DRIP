#!/usr/bin/env Rscript

#install.packages("lassosum")
library(lassosum)
library(data.table)
library(methods)
library(magrittr)
library(parallel)
library(stringr)
library(data.table)

#pheno = commandArgs(trailingOnly=TRUE)[1]
#type = commandArgs(trailingOnly=TRUE)[2]
#rep= commandArgs(trailingOnly=TRUE)[3]

pheno = "Hypertension"
type = "b"
rep= "1"
print(type)

if (type == "l")
{
  print(type)
  
  file_end=".glm.linear.spaces_fixed"
}
if (type=="b")
{
  print(type)
  
  file_end=".glm.logistic.hybrid.spaces_fixed"
}


print(pheno)

pheno_file<- fread(paste("/path/to/your/project/phenotypes/",tolower(pheno),sep=""))
pheno_file<-na.omit(pheno_file)

head(pheno_file)
table(pheno_file[,3])
pheno_file[,3]<-ifelse(pheno_file[,3]==1,0,1)
table(pheno_file[,3])

pheno_name=names(pheno_file)[3]
#names(pheno)<-c("FID","IID",pheno_name)

### Read summary statistics file ###
ss <- fread(paste("/path/to/your/project/merge_GWAS_results/",pheno,"/rep",rep,"/all_chr_train_cov_MinMax_scaled_with_values_A2.",pheno,file_end,sep=""))
head(ss)
cov <- fread(paste("/path/to/your/project/cov_matrix/rep",rep,"/cov_matrix_MinMax_scaled_no_missing_1_11_23_rep",rep,".txt",sep=""))

#names(ss)[1:3]<-c("chr","pos","snp")
#head(ss)

out_al=1

keep<-rep(F,290688)
keep[sample(1:290688, 5000)]<-T
#subset_samples <- sample(1:290688, 5000) 

results_list <- list()

for (chr in 1:22)
{
  print(nrow(ss))
  ss.chr <- ss[ss$CHROM == chr]
  print(nrow(ss.chr))
  
  ### Specify the PLINK file stub of the reference panel ###
  ref.bfile <- paste("/path/to/your/project/training_bed_files/rep",rep,"/chr",chr,"_X_train_no_cov_no_missing",sep="")
  ### Specify the PLINK file stub of the test data ###
  test.bfile <- paste("/path/to/your/project/test_bed_files/rep",rep,"/chr",chr,"_X_test_no_cov_no_missing",sep="")
  
  
  ### Read LD region file ###
  LDblocks <- "EUR.hg19" # This will use LD regions as defined in Berisa and Pickrell (2015) for the European population and the hg19 genome.
  # Other alternatives available. Type ?lassosum.pipeline for more details.
  
  print(type)
  if (type=="b")
  {
    cor <- p2cor(p = ss$P, n = 332689, sign=log(ss$OR))
    
  } else if (type=="l")
  {
    apply(ss,2,class)
    cor <- p2cor(p = ss$P, n = 332689, sign=ss$BETA)
  }
  # n is the sample size 
  
  cl <- makeCluster(2, type="FORK") # Parallel over 2 nodes
  out <- lassosum.pipeline(cor=cor, chr=ss$CHROM, pos=ss$POS, 
                           A1=ss$A1, A2=ss$A2, # A2 is not required but advised
                           s=c(0.2, 0.5, 1),
                           ref.bfile=ref.bfile, test.bfile=test.bfile, 
                           LDblocks = LDblocks, cluster=cl,
                          # keep.ref	= keep)
                           sample = 5000,max.ref.bfile.n = 5000)
  # max.ref.bfile.n=5000)
  saveRDS(out,paste("/path/to/your/project/lassosum/",pheno,"/rep",rep,"/lassosum_results_chr",chr,".rds",sep=""))
  
  #v <- validate(out,test.bfile=test.bfile,covar=cov,pheno=pheno_file)
  #saveRDS(v,paste("/path/to/your/project/lassosum/",pheno,"/rep",rep,"/lassosum_results_v_chr",chr,".rds",sep=""))
  
  #out2 <- subset(out, s=v$best.s, lambda=v$best.lambda)
  
  results_list[[chr]] <- out
  
  #if (chr==1)
  #  out_al=out
  #else 
  #  out_al=merge(out_al,out)
  
}

out_al=merge(results_list)
#v <- validate(out,test.bfile=test.bfile,)

#pheno_name = commandArgs(trailingOnly=TRUE)[1]
#(pheno_name)
#out_list<-vector(mode='list', length=22)




target.res <- validate(out_al,covar=cov,pheno=pheno_file)

print("validated lassosum")
r2 <- max(target.res$validation.table$value)^2
print('r2')
print(r2) 
saveRDS(out_al,paste("/path/to/your/project/lassosum/",pheno,"/rep",rep,"/lassosum_results_out_all_coon_bin_code.rds",sep=""))
#saveRDS(out_list,paste("/path/to/your/project/lassosum/",pheno,"/rep",rep,"/lassosum_results_out_list.rds",sep=""))
saveRDS(target.res,paste("/path/to/your/project/lassosum/",pheno,"/rep",rep,"/lassosum_results_target.res_con_bin_file.rds",sep=""))


