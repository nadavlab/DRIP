import random

# --------------------
# Parameters
# --------------------
N_SNPs = [10,50,100,500,1000,5000,10000]        # size of each SNP list
RANDOM_SEED = 42
N_RUNS = 5

random.seed(RANDOM_SEED)

# --------------------
# Functions
# --------------------
def save_snps(filename, row_set):
    """Save SNP names (column 2 only) to a file, sorted"""
    with open(filename, "w") as f:
        for row in sorted(row_set):
            snp_id = row.split()[1]   # <-- column 2 from BIM
            f.write(snp_id + "\n")

# --------------------
# Read SNPs from BIM
# --------------------
with open("/path/to/your/project//ukbb_filtered_maf_0.01_hwe_1e-6_geno_0.0_ex_mismtach/chr1_imputed_snp_id_filtered.bim", "r") as f:
    snps = [line.strip() for line in f]


# --------------------
# Main loop over repetitions
# --------------------
for run in range(1, N_RUNS + 1):
    print(f"\n=== Rep {run} ===")

    for N in N_SNPs:

        if len(snps) < 3 * N:
            raise ValueError("Not enough SNPs in all_snps.txt")

        # --------------------
    # Sample Y1 once per repetition
    # --------------------
        list_Y1 = set(random.sample(snps, N))
        save_snps(f"/path/to/your/project//simulations/SNPs_lists/rep_{run}_snps_list_chro1_N_{N}.txt", list_Y1)
        print("List Y1 size:", len(list_Y1))
