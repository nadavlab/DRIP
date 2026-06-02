# DRIP: Dimensionality Reduction Informs Polygenic Scoring

DRIP is a scalable framework for polygenic prediction that combines:

1. **Phenotype-agnostic dimensionality reduction**
2. **Incremental machine learning prediction models**

The framework was designed for large-scale genomic datasets and was evaluated using UK Biobank genotype and phenotype data.

Unlike conventional PRS pipelines based on GWAS-derived SNP selection, DRIP learns compressed genomic representations directly from genotype matrices using unsupervised dimensionality reduction methods such as PCA and autoencoders.

---

# Paper

**DRIP: Dimensionality Reduction Informs Polygenic Scoring**  
Hadasa Kaufman, Yarden Hochenberg, Michal Linial, Nadav Rappoport

Preprint / publication link: TBD

---

# Overview

The DRIP pipeline consists of several stages:

## 1. Chromosome-wise dimensionality reduction

For each chromosome independently:

- SNP matrices are split into manageable chunks
- PCA or autoencoders are trained on training individuals
- Learned representations are projected onto train and test sets

This design allows DRIP to scale to hundreds of thousands of individuals and hundreds of thousands of SNPs.

---

## 2. Genome-wide representation assembly

Chromosome-specific representations are:
- concatenated,
- merged with covariates,
- matched to phenotype-specific cohorts.

---

## 3. Incremental phenotype prediction

Prediction models are trained incrementally using chunked genomic representations:

### Binary phenotypes
- Incremental Logistic Regression (`SGDClassifier`)

### Continuous phenotypes
- Incremental Linear Regression (`SGDRegressor`)

Supported dimensionality reduction methods:
- PCA
- Autoencoder embeddings

---

# Main Components

## PCA Dimensionality Reduction

Main functionality:
- chromosome-wise PCA fitting,
- incremental transformation of chunked genotype matrices,
- reconstruction quality evaluation using variance explained and R².

Key features:
- scalable chunk-based processing,
- train/test separation,
- reusable pretrained PCA transformers.

---

## Incremental Logistic Regression

Binary phenotype prediction using:

```python
SGDClassifier(loss='log_loss')
```

Features:
- chunk-wise training using `partial_fit`,
- scalable to very large cohorts,
- validation monitoring using log loss,
- ROC-AUC and average precision evaluation.

---

## Incremental Linear Regression

Continuous phenotype prediction using:

```python
SGDRegressor(loss='squared_error')
```

Features:
- chunk-wise incremental learning,
- mean squared error tracking,
- R² evaluation.

---

## Phenotype Matching Pipeline

The repository includes utilities for:
- matching genotype representations to phenotype cohorts,
- generating chunked target files,
- maintaining consistent individual ordering across train/test sets.

---

## Covariate Integration

Chromosome-specific reduced representations are merged with:
- demographic covariates,
- principal covariates,
- additional phenotype metadata.

---

# Data Availability

This project uses genotype and phenotype data from the UK Biobank.

Due to UK Biobank restrictions, raw genotype and phenotype files are not distributed in this repository.

UK Biobank: https://www.ukbiobank.ac.uk/

Application ID used in this study: 26664.

---

# Pretrained PCA Transformers

Pretrained chromosome-specific PCA transformers used in the manuscript are available at:

https://drive.google.com/drive/folders/1oukhU_B4nM5kH9z2BxC81Kfn4kp05JAm?usp=drive_link

Example structure:

```text
rep1/
  PCA_transformer_1.pkl
  PCA_transformer_2.pkl
  ...
```

These transformers can be directly used to project new genotype matrices into the DRIP latent representation space.

---

# Installation

Clone the repository:

```bash
git clone https://github.com/nadavlab/DRIP.git
cd DRIP
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Required Python Packages

Main dependencies include:

```text
numpy
pandas
scikit-learn
matplotlib
pickle5
pandas-plink
scipy
xgboost
tensorflow
```

---

# Example Pipeline

## Step 1 — Train PCA transformers
(In the case of using exactly our SNPs and population filters, you can skip this step and download the PCA provided by us : .  https://drive.google.com/drive/folders/1oukhU_B4nM5kH9z2BxC81Kfn4kp05JAm?usp=drive_link )

```bash
python PCA_train.py <chromosome> <rep>
```

---

## Step 2 — Transform genotype chunks

```bash
python PCA_transform.py <chromosome> <rep>
```

---

## Step 3 — Merge chromosome representations

```bash
python join_chromosomes.py PCA <rep>
```

---

## Step 4 — Match phenotypes

```bash
python match_phenotype.py <phenotype> <type> PCA
```

Where:
- `type = b` for binary phenotype
- `type = c` for continuous phenotype

---

## Step 5 — Train prediction model

### Logistic regression

```bash
python LogisticRegression_incremental.py <phenotype> <rep> PCA
```

### Linear regression

```bash
python LinearRegression_incremental.py <phenotype> <rep> PCA
```

---

# Scalability

DRIP was specifically designed for large-scale genomic datasets.

The framework supports:
- chunk-wise processing,
- incremental learning,
- chromosome-wise decomposition,
- memory-efficient model fitting.

This allows analyses on:
- hundreds of thousands of individuals,
- hundreds of thousands of SNPs,
- limited-memory compute environments.

---

# Hardware Used

Experiments were conducted using:
- UK Biobank genotype data
- ~342k individuals
- ~467k SNPs
- High-memory CPU nodes
- RTX6000 GPU (for autoencoder experiments)

---

# Main Findings

- DRIP achieved predictive performance comparable to standard PRS approaches
- PCA-based representations consistently performed strongly
- Incremental learning enabled scalable training on UK Biobank-scale data
- Phenotype-agnostic genomic representations generalized across multiple traits

---

# Citation

If you use this repository, please cite:

```bibtex
@article{kaufman2026drip,
  title={DRIP: Dimensionality Reduction Informs Polygenic Scoring},
  author={Kaufman, Hadasa and Hochenberg, Yarden and Linial, Michal and Rappoport, Nadav},
  year={2026}
}
```

---

# Contact

Hadasa Kaufman  
The Hebrew University of Jerusalem

For questions or collaborations:

hadasa.kaufman@mail.huji.ac.il
