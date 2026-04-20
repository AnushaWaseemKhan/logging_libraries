import re
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import friedmanchisquare, wilcoxon
from statsmodels.stats.multitest import multipletests


# =========================================================
# 0. SETTINGS
# =========================================================
BASELINE_J = 1465.73
ALPHA = 0.05

project_root = Path(__file__).resolve().parents[2]
data_path = project_root / "main" / "results" / "emission_per_run_3libs_allrates"
output_dir = project_root / "main" / "results" / "stats_output_allrates"
output_dir.mkdir(parents=True, exist_ok=True)

raw_plot_dir = output_dir / "raw_plots"
raw_plot_dir.mkdir(exist_ok=True)

clean_plot_dir = output_dir / "cleaned_plots"
clean_plot_dir.mkdir(exist_ok=True)

files = list(data_path.glob("*.csv"))
if not files:
    raise FileNotFoundError(f"No CSV files found in {data_path.resolve()}")


# =========================================================
# 1. LOAD CSV FILES
# =========================================================
df_list = []
for file in files:
    temp = pd.read_csv(file)
    temp["source_file"] = file.name
    df_list.append(temp)

df = pd.concat(df_list, ignore_index=True)

if "energy_consumed" not in df.columns:
    raise ValueError("Column 'energy_consumed' not found in CSV files.")

if "project_name" not in df.columns:
    raise ValueError("Column 'project_name' not found in CSV files.")


# =========================================================
# 2. CONVERT TO JOULES + BASELINE SUBTRACTION
# =========================================================
df["energy_j"] = df["energy_consumed"] * 3_600_000
df["net_energy_j"] = df["energy_j"] - BASELINE_J


# =========================================================
# 3. EXTRACT METADATA FROM project_name
# =========================================================
def extract_lib(name):
    m = re.match(r"^([^_]+)", str(name))
    return m.group(1) if m else None


def extract_rate(name):
    m = re.search(r"_rate([0-9.]+)", str(name))
    return float(m.group(1)) if m else None


def extract_rep(name):
    m = re.search(r"_rep(\d+)", str(name))
    return int(m.group(1)) if m else None


df["lib"] = df["project_name"].apply(extract_lib)
df["rate"] = df["project_name"].apply(extract_rate)
df["rep"] = df["project_name"].apply(extract_rep)

df = df.dropna(subset=["lib", "rate", "rep", "energy_j", "net_energy_j"]).copy()
df["rate"] = df["rate"].astype(float)
df["rep"] = df["rep"].astype(int)
df = df.sort_values(["lib", "rate", "rep"]).copy()

rate_values_all = sorted(df["rate"].unique())
rate_labels_all = [str(r) for r in rate_values_all]


# =========================================================
# 4. RAW-STYLE PLOTS (NOW BASELINE REMOVED TOO)
# =========================================================
print("\n========== GENERATING NET ENERGY PLOTS (BASELINE REMOVED) ==========\n")

# ---- (A) SEPARATE CLEAN PLOTS PER LIBRARY ----
for lib, group in df.groupby("lib"):
    group = group.sort_values(["rate", "rep"]).copy()

    summary = (
        group.groupby("rate", as_index=False)
        .agg(
            mean_net_energy_j=("net_energy_j", "mean"),
            std_net_energy_j=("net_energy_j", "std"),
            n=("net_energy_j", "count"),
        )
        .sort_values("rate")
    )

    plt.figure(figsize=(9, 5.5))

    plt.plot(
        summary["rate"],
        summary["mean_net_energy_j"],
        marker="o",
        linewidth=3,
        markersize=7,
        label=lib,
    )

    plt.xscale("log")
    plt.xticks(rate_values_all, rate_labels_all)
    plt.xlabel("Rate (logs/sec)", fontsize=11)
    plt.ylabel("Net Energy (J)", fontsize=11)
    plt.title(f"Energy vs Rate — {lib}", fontsize=13, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(frameon=True)
    plt.tight_layout()

    save_path = raw_plot_dir / f"{lib}_net_energy.png"
    plt.savefig(save_path, dpi=400, bbox_inches="tight")
    plt.close()

    print(f"Saved: {save_path}")

# ---- (B) COMBINED PLOT ----
plt.figure(figsize=(10, 6))

for lib, group in df.groupby("lib"):
    summary = (
        group.groupby("rate", as_index=False)
        .agg(mean_net_energy_j=("net_energy_j", "mean"))
        .sort_values("rate")
    )

    plt.plot(
        summary["rate"],
        summary["mean_net_energy_j"],
        marker="o",
        linewidth=2.2,
        markersize=6,
        label=lib,
    )

plt.xscale("log")
plt.xticks(rate_values_all, rate_labels_all)
plt.xlabel("Rate (logs/sec)", fontsize=11)
plt.ylabel("Net Energy (J)", fontsize=11)
plt.title("Energy vs Rate — All Libraries\n(Baseline Removed)", fontsize=13, fontweight="bold")
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend(frameon=True)
plt.tight_layout()

combined_raw_path = raw_plot_dir / "combined_net_energy.png"
plt.savefig(combined_raw_path, dpi=400, bbox_inches="tight")
plt.close()

print(f"Saved: {combined_raw_path}")


# =========================================================
# 5. REMOVE FIRST AND LAST REP PER (lib, rate)
# =========================================================
df["rep_rank_asc"] = df.groupby(["lib", "rate"]).cumcount() + 1
df["rep_rank_desc"] = df.groupby(["lib", "rate"]).cumcount(ascending=False) + 1

filtered_df = df[(df["rep_rank_asc"] > 1) & (df["rep_rank_desc"] > 1)].copy()

if filtered_df.empty:
    raise ValueError("No data left after removing first and last reps.")


# =========================================================
# 6. PRINT CLEANED DATA
# =========================================================
print("\n========== DATA AFTER REMOVING FIRST & LAST REP ==========\n")
for (lib, rate), group in filtered_df.groupby(["lib", "rate"]):
    group = group.sort_values("rep")
    print(f"\nLibrary = {lib} | Rate = {rate}")
    for _, row in group.iterrows():
        print(
            f"  Rep {row['rep']}: "
            f"Raw = {row['energy_j']:.4f} J | "
            f"Net = {row['net_energy_j']:.4f} J"
        )


# =========================================================
# 7. AVERAGE NET ENERGY PER LIBRARY AND RATE
# =========================================================
avg_df = (
    filtered_df.groupby(["lib", "rate"], as_index=False)
    .agg(
        avg_net_energy_j=("net_energy_j", "mean"),
        std_net_energy_j=("net_energy_j", "std"),
        n=("net_energy_j", "count"),
    )
    .sort_values(["rate", "avg_net_energy_j", "lib"])
)

print("\n========== AVERAGE NET ENERGY BY LIBRARY AND RATE ==========\n")
for _, row in avg_df.iterrows():
    print(
        f"Library = {row['lib']} | "
        f"Rate = {row['rate']} | "
        f"Avg Net Energy = {row['avg_net_energy_j']:.4f} J | "
        f"Std = {row['std_net_energy_j']:.4f} | "
        f"N = {row['n']}"
    )

avg_csv_path = output_dir / "average_net_energy_by_library_and_rate_no_first_last.csv"
avg_df.to_csv(avg_csv_path, index=False)


# =========================================================
# 8. CLEANED COMBINED PLOT
# =========================================================
plt.figure(figsize=(10, 6))

for lib, group in avg_df.groupby("lib"):
    group = group.sort_values("rate")

    plt.plot(
        group["rate"],
        group["avg_net_energy_j"],
        marker="o",
        linewidth=2.2,
        markersize=6,
        label=lib,
    )

plt.xscale("log")
plt.xticks(rate_values_all, rate_labels_all)
plt.xlabel("Rate (logs/sec)", fontsize=11)
plt.ylabel("Average Net Energy (J)", fontsize=11)
plt.title(
    "Cleaned Energy vs Rate\n(Baseline Removed, First/Last Reps Excluded)",
    fontsize=13,
    fontweight="bold",
)
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend(frameon=True)
plt.tight_layout()

clean_path = clean_plot_dir / "cleaned_energy_vs_rate.png"
plt.savefig(clean_path, dpi=400, bbox_inches="tight")
plt.close()

print(f"Saved: {clean_path}")


# =========================================================
# 9. HELPER FUNCTIONS FOR STATISTICS
# =========================================================
def kendalls_w_from_friedman(chi2_stat, n_blocks, k_groups):
    if n_blocks <= 0 or k_groups <= 1:
        return np.nan
    return chi2_stat / (n_blocks * (k_groups - 1))


def build_rate_wide_table(data, rate_value):
    sub = data[data["rate"] == rate_value].copy()
    wide = sub.pivot_table(index="rep", columns="lib", values="net_energy_j", aggfunc="mean")
    wide = wide.dropna(axis=0, how="any")
    wide = wide.sort_index(axis=0).sort_index(axis=1)
    return wide


def friedman_for_wide_table(wide_df):
    if wide_df.shape[0] < 2 or wide_df.shape[1] < 2:
        return None

    samples = [wide_df[col].values for col in wide_df.columns]
    stat, p = friedmanchisquare(*samples)
    w = kendalls_w_from_friedman(stat, n_blocks=wide_df.shape[0], k_groups=wide_df.shape[1])

    return {
        "statistic": stat,
        "p_value": p,
        "kendalls_w": w,
        "n_blocks": wide_df.shape[0],
        "k_groups": wide_df.shape[1],
    }


def pairwise_wilcoxon_holm(wide_df, alpha=0.05):
    libs = list(wide_df.columns)
    pairs = list(combinations(libs, 2))

    records = []
    raw_pvals = []

    for lib1, lib2 in pairs:
        x = wide_df[lib1].values
        y = wide_df[lib2].values

        if np.allclose(x - y, 0):
            stat = 0.0
            p = 1.0
        else:
            stat, p = wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")

        median_x = float(np.median(x))
        median_y = float(np.median(y))
        median_diff = float(np.median(x - y))

        if median_x < median_y:
            lower_median_lib = lib1
        elif median_y < median_x:
            lower_median_lib = lib2
        else:
            lower_median_lib = "tie"

        raw_pvals.append(p)
        records.append({
            "lib1": lib1,
            "lib2": lib2,
            "wilcoxon_stat": stat,
            "p_raw": p,
            "median_lib1": median_x,
            "median_lib2": median_y,
            "median_diff_lib1_minus_lib2": median_diff,
            "lower_median_lib": lower_median_lib,
        })

    reject, p_holm, _, _ = multipletests(raw_pvals, alpha=alpha, method="holm")

    for i in range(len(records)):
        records[i]["p_holm"] = p_holm[i]
        records[i]["significant"] = bool(reject[i])

    results_df = pd.DataFrame(records).sort_values(
        ["p_holm", "p_raw", "lib1", "lib2"]
    ).reset_index(drop=True)
    return results_df


# =========================================================
# 10. PER-RATE FRIEDMAN + POST-HOC
# =========================================================
print("\n========== PER-RATE FRIEDMAN ANALYSIS ==========\n")

per_rate_summary = []
per_rate_ranks = []

rates_sorted = sorted(filtered_df["rate"].unique())

for rate in rates_sorted:
    wide = build_rate_wide_table(filtered_df, rate)

    if wide.empty or wide.shape[1] < 2 or wide.shape[0] < 2:
        print(f"Rate = {rate}: not enough matched data for Friedman test.")
        continue

    friedman_result = friedman_for_wide_table(wide)
    if friedman_result is None:
        print(f"Rate = {rate}: Friedman test could not be computed.")
        continue

    means_this_rate = wide.mean(axis=0).sort_values()
    best_mean_lib = means_this_rate.index[0]
    best_mean_energy = means_this_rate.iloc[0]

    rank_means = wide.rank(axis=1, method="average", ascending=True).mean(axis=0).sort_values()

    print(f"Rate = {rate}")
    print(f"  Libraries: {list(wide.columns)}")
    print(f"  Matched reps used: {wide.shape[0]}")
    print(f"  Friedman chi-square = {friedman_result['statistic']:.4f}")
    print(f"  p-value = {friedman_result['p_value']:.6f}")
    print(f"  Kendall's W = {friedman_result['kendalls_w']:.4f}")
    print(f"  Lowest mean energy = {best_mean_lib} ({best_mean_energy:.4f} J)")

    interpretation = (
        "Significant difference exists"
        if friedman_result["p_value"] < ALPHA
        else "No significant difference"
    )

    per_rate_summary.append({
        "rate": rate,
        "n_blocks": friedman_result["n_blocks"],
        "friedman_chi2": friedman_result["statistic"],
        "p_value": friedman_result["p_value"],
        "kendalls_w": friedman_result["kendalls_w"],
        "lowest_mean_lib": best_mean_lib,
        "lowest_mean_energy_j": best_mean_energy,
        "best_avg_rank_lib": rank_means.index[0],
        "best_avg_rank": rank_means.iloc[0],
        "interpretation": interpretation,
    })

    for rank_pos, (lib_name, avg_rank) in enumerate(rank_means.items(), start=1):
        per_rate_ranks.append({
            "rate": rate,
            "rank_position": rank_pos,
            "lib": lib_name,
            "avg_rank": avg_rank,
        })

    if friedman_result["p_value"] < ALPHA:
        posthoc = pairwise_wilcoxon_holm(wide, alpha=ALPHA)
        posthoc_path = output_dir / f"posthoc_wilcoxon_holm_rate_{str(rate).replace('.', '_')}.csv"
        posthoc.to_csv(posthoc_path, index=False)

        print("  Post-hoc Wilcoxon + Holm:")
        print(posthoc[["lib1", "lib2", "p_raw", "p_holm", "significant", "lower_median_lib"]].round(6))
        print(f"  Saved post-hoc table: {posthoc_path}")
    else:
        print("  Post-hoc skipped (Friedman not significant).")

    print()

per_rate_summary_df = pd.DataFrame(per_rate_summary)
per_rate_ranks_df = pd.DataFrame(per_rate_ranks)

per_rate_summary_path = output_dir / "per_rate_friedman_summary.csv"
per_rate_ranks_path = output_dir / "per_rate_average_ranks.csv"

per_rate_summary_df.to_csv(per_rate_summary_path, index=False)
per_rate_ranks_df.to_csv(per_rate_ranks_path, index=False)

print(f"Saved per-rate summary: {per_rate_summary_path}")
print(f"Saved per-rate average ranks: {per_rate_ranks_path}")


# =========================================================
# 11. OVERALL ANALYSIS ACROSS ALL RATES
# =========================================================
print("\n========== OVERALL ANALYSIS ACROSS ALL RATES ==========\n")

overall_wide = filtered_df.pivot_table(
    index=["rate", "rep"],
    columns="lib",
    values="net_energy_j",
    aggfunc="mean",
).dropna(axis=0, how="any")

overall_wide = overall_wide.sort_index(axis=0).sort_index(axis=1)

if overall_wide.empty or overall_wide.shape[0] < 2 or overall_wide.shape[1] < 2:
    raise ValueError("Not enough matched data for overall Friedman analysis.")

overall_friedman = friedman_for_wide_table(overall_wide)
if overall_friedman is None:
    raise ValueError("Overall Friedman test could not be computed.")

print(f"Matched blocks used (rate, rep): {overall_wide.shape[0]}")
print(f"Libraries compared: {list(overall_wide.columns)}")
print(f"Overall Friedman chi-square = {overall_friedman['statistic']:.4f}")
print(f"Overall p-value = {overall_friedman['p_value']:.6f}")
print(f"Overall Kendall's W = {overall_friedman['kendalls_w']:.4f}")


# =========================================================
# 12. OVERALL RANKING (BEST OVERALL)
# =========================================================
overall_ranks = overall_wide.rank(axis=1, method="average", ascending=True)

average_rank_per_lib = overall_ranks.mean(axis=0).sort_values()
overall_mean_energy = overall_wide.mean(axis=0)

overall_rank_df = pd.DataFrame({
    "lib": average_rank_per_lib.index,
    "average_rank": average_rank_per_lib.values,
    "overall_mean_net_energy_j": [overall_mean_energy[lib] for lib in average_rank_per_lib.index],
}).sort_values(["average_rank", "overall_mean_net_energy_j"]).reset_index(drop=True)

overall_rank_df["overall_position"] = np.arange(1, len(overall_rank_df) + 1)

print("\nOverall average rank per library (LOWER = BETTER):")
for _, row in overall_rank_df.iterrows():
    print(
        f"  Rank {int(row['overall_position'])}: "
        f"{row['lib']} | Avg Rank = {row['average_rank']:.4f} | "
        f"Mean Net Energy = {row['overall_mean_net_energy_j']:.4f} J"
    )

best_overall_lib = overall_rank_df.iloc[0]["lib"]
print(f"\nBEST OVERALL LIBRARY BY AVERAGE RANK: {best_overall_lib}")

overall_rank_path = output_dir / "overall_average_ranks.csv"
overall_rank_df.to_csv(overall_rank_path, index=False)
print(f"Saved overall ranking: {overall_rank_path}")


# =========================================================
# 13. OVERALL POST-HOC
# =========================================================
if overall_friedman["p_value"] < ALPHA:
    overall_posthoc = pairwise_wilcoxon_holm(overall_wide, alpha=ALPHA)
    overall_posthoc_path = output_dir / "overall_posthoc_wilcoxon_holm.csv"
    overall_posthoc.to_csv(overall_posthoc_path, index=False)

    print("\nOverall post-hoc Wilcoxon + Holm:")
    print(overall_posthoc[["lib1", "lib2", "p_raw", "p_holm", "significant", "lower_median_lib"]].round(6))
    print(f"Saved overall post-hoc table: {overall_posthoc_path}")
else:
    overall_posthoc = None
    print("\nOverall post-hoc skipped because overall Friedman is not significant.")


# =========================================================
# 14. PLOT OVERALL AVERAGE RANKS
# =========================================================
plt.figure(figsize=(9, 5))
plt.bar(overall_rank_df["lib"], overall_rank_df["average_rank"])
plt.ylabel("Average Rank (Lower = Better)")
plt.xlabel("Library")
plt.title("Overall Library Ranking Across All Rates\n(Friedman-style Rank Aggregation)")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()

overall_rank_plot_path = output_dir / "overall_average_rank_barplot.png"
plt.savefig(overall_rank_plot_path, dpi=400, bbox_inches="tight")
plt.close()

print(f"Saved overall rank plot: {overall_rank_plot_path}")


# =========================================================
# 15. FINAL SUMMARY
# =========================================================
summary_lines = []
summary_lines.append("FINAL STATISTICAL SUMMARY")
summary_lines.append("=" * 60)
summary_lines.append("")
summary_lines.append("PER-RATE ANALYSIS:")

for _, row in per_rate_summary_df.iterrows():
    summary_lines.append(
        f"Rate={row['rate']}: "
        f"p={row['p_value']:.6f}, "
        f"Kendall_W={row['kendalls_w']:.4f}, "
        f"LowestMean={row['lowest_mean_lib']} ({row['lowest_mean_energy_j']:.4f} J), "
        f"BestAvgRank={row['best_avg_rank_lib']} ({row['best_avg_rank']:.4f}), "
        f"Interpretation={row['interpretation']}"
    )

summary_lines.append("")
summary_lines.append("OVERALL ANALYSIS ACROSS ALL RATES:")
summary_lines.append(f"Overall Friedman chi-square = {overall_friedman['statistic']:.4f}")
summary_lines.append(f"Overall p-value = {overall_friedman['p_value']:.6f}")
summary_lines.append(f"Overall Kendall's W = {overall_friedman['kendalls_w']:.4f}")
summary_lines.append("")
summary_lines.append("OVERALL RANKING (LOWER AVG RANK = BETTER):")

for _, row in overall_rank_df.iterrows():
    summary_lines.append(
        f"{int(row['overall_position'])}. {row['lib']} | "
        f"Avg Rank = {row['average_rank']:.4f} | "
        f"Overall Mean Net Energy = {row['overall_mean_net_energy_j']:.4f} J"
    )

summary_lines.append("")
summary_lines.append(f"BEST OVERALL LIBRARY: {best_overall_lib}")

if overall_posthoc is not None:
    summary_lines.append("")
    summary_lines.append("OVERALL POST-HOC (WILCOXON + HOLM):")
    for _, row in overall_posthoc.iterrows():
        summary_lines.append(
            f"{row['lib1']} vs {row['lib2']} | "
            f"p_raw = {row['p_raw']:.6f} | "
            f"p_holm = {row['p_holm']:.6f} | "
            f"significant = {row['significant']} | "
            f"lower_median_lib = {row['lower_median_lib']}"
        )

summary_txt_path = output_dir / "final_statistical_summary.txt"
with open(summary_txt_path, "w", encoding="utf-8") as f:
    f.write("\n".join(summary_lines))

print(f"Saved final summary: {summary_txt_path}")
print("\nDone.")
