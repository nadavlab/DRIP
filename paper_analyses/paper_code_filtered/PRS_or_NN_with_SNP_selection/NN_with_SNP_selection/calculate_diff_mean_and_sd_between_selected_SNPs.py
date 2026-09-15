import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ===== Phenotypes to run over =====
pheno_list = [
    "Hypertension",
    "Type_2_Diabetes",
    "Multiple_Sclerosis",
    "Parkinson",
    "Schizophrenia",
    "Alzheimer",
    "Height",
    "BMI",
    "Platelet_count"
]

# ===== 1. Load all BIM files into a dictionary: { "1": bim_df_1, "2": bim_df_2, ... } =====
bim_dir = "/path/to/your/project/GEN/"
bim_pattern = os.path.join(bim_dir, "ukb{chrom}.bim")

bim_dict = {}
for chrom in range(1, 23):  # 1-22; extend if you have X/Y
    bim_path = bim_pattern.format(chrom=chrom)
    if not os.path.exists(bim_path):
        print(f"Warning: BIM for chr{chrom} not found at {bim_path}")
        continue

    bim = pd.read_csv(
        bim_path,
        delim_whitespace=True,
        header=None,
        names=["CHR", "SNP", "CM", "BP", "A1", "A2"]
    )
    bim_dict[str(chrom)] = bim[["SNP", "BP"]]

# 🔹 collect summary over *all* phenotypes & reps
summary_all = []

# ===== 2. Loop over phenotypes =====
for pheno_name in pheno_list:
    print("\n" + "=" * 80)
    print(f"Processing phenotype: {pheno_name}")
    print("=" * 80)

    summary = []  # summary over reps for this phenotype

    # ===== 3. Loop over repetitions =====
    for i in range(1, 6):
        rep = str(i)
        print(f"\n=== {pheno_name} – Rep {rep} ===")

        # ---- Load X matrix ----
        x_path = (
            f"/path/to/your/project/"
            f"impoving_PRS/data/NN_with_SNP_selection/X_files/rep{rep}/"
            f"{pheno_name}_X_train_all_chr_MinMax_cov_MinMax.pkl"
        )
        if not os.path.exists(x_path):
            print(f"  X file not found: {x_path}  --> skipping rep {rep}")
            continue

        df = pd.read_pickle(x_path)

        # ---- Get SNP column names (from column 198 onward) ----
        snp_cols = df.columns[198:]

        # Dataframe with one row per SNP
        new_df = pd.DataFrame({"SNP_full": snp_cols})

        print("  N SNPs (before BP merge):", len(new_df.index))

        # "rs114860644_chr22" → snp="rs114860644", chr="22"
        new_df[["snp", "chr"]] = new_df["SNP_full"].str.split("_chr", n=1, expand=True)

        # ---- Add position (BP) using BIM files ----
        new_df["BP"] = np.nan

        for chrom, idx in new_df.groupby("chr").groups.items():
            if chrom not in bim_dict:
                print(f"    Warning: no BIM loaded for chr{chrom} in rep {rep}")
                continue

            bim_chr = bim_dict[chrom]  # SNP, BP
            sub = new_df.loc[idx, ["snp"]]

            merged = sub.merge(
                bim_chr,
                left_on="snp",
                right_on="SNP",
                how="left"
            )

            new_df.loc[idx, "BP"] = merged["BP"].values

        # Drop SNPs without BP (optional)
        missing = new_df["BP"].isna().sum()
        if missing > 0:
            print(f"    {missing} SNPs in rep {rep} have no position in BIM and will be dropped.")
        new_df = new_df.dropna(subset=["BP"])
        new_df["BP"] = new_df["BP"].astype(int)

        # ---- Sort by chromosome and position ----
        new_df = new_df.sort_values(["chr", "BP"]).reset_index(drop=True)

        # ---- Difference from previous SNP per chromosome ----
        new_df["diff"] = new_df.groupby("chr")["BP"].diff()
        new_df["diff"] = new_df["diff"].fillna(new_df["BP"])

        # ---- Summary stats & outliers ----
        diff = new_df["diff"]
        mean_diff = diff.mean()
        sd_diff = diff.std()

        is_outlier = (diff > mean_diff + 3 * sd_diff) | (diff < mean_diff - 3 * sd_diff)
        new_df["is_outlier"] = is_outlier

        count_gt_300k = (diff > 300000).sum()
        count_lt_10k = (diff < 10000).sum()
        count_outliers = is_outlier.sum()

        print(f"    mean diff: {mean_diff:.2f}, sd diff: {sd_diff:.2f}")
        print(f"    >300k: {count_gt_300k}, <10k: {count_lt_10k}, outliers: {count_outliers}")

        # ======== 250kb WINDOWS: COUNT SNPs + LIST SNPs PER WINDOW ========
        window_size = 250_000
        windows_rows = []

        for chrom, grp in new_df.groupby("chr"):
            min_bp = grp["BP"].min()
            max_bp = grp["BP"].max()

            # define windows (aligned to window_size, starting near min_bp)
            start_aligned = (min_bp // window_size) * window_size + 1

            grp = grp.copy()
            grp["bin_index"] = ((grp["BP"] - start_aligned) // window_size).astype(int)

            # counts for windows that actually have SNPs
            counts = grp.groupby("bin_index").size().reset_index(name="n_snps")

            # list of SNPs per window (for windows that have SNPs)
            snp_lists = grp.groupby("bin_index")["snp"].apply(list).reset_index(name="snp_list")

            # create all windows, including 0-SNP ones
            max_bin = grp["bin_index"].max()
            all_bins = pd.DataFrame({"bin_index": np.arange(0, max_bin + 1, dtype=int)})

            win_info = (
                all_bins
                .merge(counts, on="bin_index", how="left")
                .merge(snp_lists, on="bin_index", how="left")
            )

            # fill windows with no SNPs
            win_info["n_snps"] = win_info["n_snps"].fillna(0).astype(int)

            # for snp_list, replace NaN with empty list
            win_info["snp_list"] = win_info["snp_list"].apply(
                lambda x: x if isinstance(x, list) else []
            )

            for _, row in win_info.iterrows():
                idx_win = int(row["bin_index"])
                n_snps = int(row["n_snps"])
                snp_list = row["snp_list"]  # list of SNP IDs (possibly empty)

                window_start = start_aligned + idx_win * window_size
                window_end = window_start + window_size - 1

                windows_rows.append({
                    "rep": i,
                    "pheno": pheno_name,
                    "chr": chrom,
                    "window_index": idx_win,
                    "window_start_bp": window_start,
                    "window_end_bp": window_end,
                    "n_snps": n_snps,
                    "snp_list": snp_list,
                })

        windows_df = pd.DataFrame(windows_rows)

        # ---- window stats and bins ----
        n = windows_df["n_snps"]

        mean_snps_per_window = n.mean()
        sd_snps_per_window = n.std()

        win_n0 = (n == 0).sum()
        win_n1 = (n == 1).sum()
        win_n2 = (n == 2).sum()
        win_n3 = (n == 3).sum()
        win_n4 = (n == 4).sum()
        win_n5 = (n == 5).sum()
        win_n_6_10 = ((n >= 6) & (n <= 10)).sum()
        win_n_11_30 = ((n >= 11) & (n <= 30)).sum()
        win_n_31_100 = ((n >= 31) & (n <= 100)).sum()
        win_n_gt100 = (n > 100).sum()

        # ---- Histogram WITHOUT outliers ----
        diff_no_outliers = diff[~is_outlier]

        plt.figure(figsize=(10, 6))
        plt.hist(diff_no_outliers, bins=100)
        plt.title(f"Distribution of SNP Position Differences (no outliers) – {pheno_name}, Rep {rep}")
        plt.xlabel("Difference (bp)")
        plt.ylabel("Count")
        plt.axvline(mean_diff, linestyle="--")

        plot_path = (
            "/path/to/your/project/"
            f"impoving_PRS/data/NN_with_SNP_selection/SNPs_list/"
            f"rep{rep}_{pheno_name}_diff_hist_no_outliers.png"
        )
        plt.savefig(plot_path, dpi=200, bbox_inches="tight")
        plt.close()

        # ---- Save per-rep SNP table ----
        out_snps_path = (
            "/path/to/your/project/"
            f"impoving_PRS/data/NN_with_SNP_selection/SNPs_list/"
            f"rep{rep}_{pheno_name}_SNP_with_pos_diff_outliers.csv"
        )
        new_df.to_csv(out_snps_path, index=False)

        # ---- Append to summary for this phenotype ----
        summary.append({
            "pheno": pheno_name,
            "rep": i,
            "mean_diff_bp": mean_diff,
            "sd_diff_bp": sd_diff,
            "N_snps_after_BP": len(new_df.index),
            "N_missing_BP": int(missing),
            "count_diff_gt_300k": int(count_gt_300k),
            "count_diff_lt_10k": int(count_lt_10k),
            "count_outliers": int(count_outliers),

            "mean_snps_per_window": float(mean_snps_per_window),
            "sd_snps_per_window": float(sd_snps_per_window),

            "win_n0": int(win_n0),
            "win_n1": int(win_n1),
            "win_n2": int(win_n2),
            "win_n3": int(win_n3),
            "win_n4": int(win_n4),
            "win_n5": int(win_n5),
            "win_n_6_10": int(win_n_6_10),
            "win_n_11_30": int(win_n_11_30),
            "win_n_31_100": int(win_n_31_100),
            "win_n_gt100": int(win_n_gt100),
        })

    # ===== 4. Save summary of all repetitions for this phenotype =====
    if len(summary) == 0:
        print(f"\nNo reps successfully processed for phenotype {pheno_name}, skipping summary CSV.")
        continue

    summary_df = pd.DataFrame(summary)
    summary_out = (
        "/path/to/your/project/"
        f"impoving_PRS/data/NN_with_SNP_selection/SNPs_list/"
        f"{pheno_name}_diff_and_window_summary_over_reps.csv"
    )
    summary_df.to_csv(summary_out, index=False)
    print(f"\nSummary over repetitions for {pheno_name}:")
    print(summary_df)

    # 🔹 add this phenotype's rows to the global summary
    summary_all.extend(summary_df.to_dict("records"))

# ===== 5. GLOBAL summary over ALL phenotypes & reps =====
if len(summary_all) > 0:
    summary_all_df = pd.DataFrame(summary_all)
    summary_all_out = (
        "/path/to/your/project/"
        "impoving_PRS/data/NN_with_SNP_selection/SNPs_list/"
        "ALL_PHENOS_diff_and_window_summary_over_reps.csv"
    )
    summary_all_df.to_csv(summary_all_out, index=False)
    print("\nGlobal summary over ALL phenotypes & reps:")
    print(summary_all_df)
else:
    print("\nNo phenotype had any successful reps; global summary not created.")
