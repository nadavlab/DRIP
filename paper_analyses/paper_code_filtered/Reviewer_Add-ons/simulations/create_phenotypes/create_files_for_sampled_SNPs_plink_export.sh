#!/bin/bash

# --------------------
# Paths
# --------------------
PLINK2=/path/to/your/project//plink2
GENETICS_DIR=/path/to/your/project//ukbb_filtered_maf_0.01_hwe_1e-6_geno_0.0_ex_mismtach
SNPS_BASE_DIR=/path/to/your/project//simulations/SNPs_lists
OUT_BASE_DIR=/path/to/your/project//simulations/random_SNPs_samples

# Create output dir if missing
mkdir -p "$OUT_BASE_DIR"

# --------------------
# Parameters
# --------------------
RUNS=(1 2 3 4 5)
N_SNPs=(10 50 100 500 1000 5000 10000)       # size of each SNP list
X_VALUES=(200)

# --------------------
# Loop over repetitions
# --------------------
for RUN in "${RUNS[@]}"; do
    echo "=== Processing repetition $RUN ==="

    for X in "${N_SNPs[@]}"; do
        SNP_FILE_Y2="$SNPS_BASE_DIR/rep_${RUN}_snps_list_chro1_N_${X}.txt"
        OUT_DIR_Y2="$OUT_BASE_DIR/rep_${RUN}_chro1_N_${X}"
        mkdir -p "$OUT_DIR_Y2"

        echo "Processing $X SNPs..."
        for CHR in {1..1}; do
            echo "  Chromosome $CHR ..."
            $PLINK2 \
                --bed "$GENETICS_DIR/chr${CHR}_imputed_snp_id_filtered.bed" \
                --fam "$GENETICS_DIR/chr${CHR}_imputed_snp_id_filtered.fam" \
                --bim "$GENETICS_DIR/chr${CHR}_imputed_snp_id_filtered.bim" \
                --extract "$SNP_FILE_Y2" \
                --export A \
                --freq counts \
                --out "$OUT_DIR_Y2"
        done
    done

done

#                #--out "$OUT_DIR_Y2/chr${CHR}_Y2_overlap_${X}"
