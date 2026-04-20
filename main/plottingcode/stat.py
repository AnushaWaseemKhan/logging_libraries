import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from pathlib import Path
from scipy.stats import friedmanchisquare, wilcoxon
from statsmodels.stats.multitest import multipletests

# =========================================================
# SETTINGS
# =========================================================
BASELINE_J = 1465.73
ALPHA = 0.05

project_root = Path(__file__).resolve().parents[2]
data_path = project_root / "main" / "results" / "emission_per_run_3libs_allrates"

files = list(data_path.glob("*.csv"))
if not files:
    raise FileNotFoundError(f"No CSV files found in {data_path.resolve()}")

# =========================================================
# 1. LOAD CSVs
# =========================================================
df_list = []
for file in files:
    temp = pd.read_csv(file)
    temp["source_file"] = file.name
    df_list.append(temp)

df = pd.concat(df_list, ignore_index=True)

# =========================================================
# 2. REQUIRED COLUMNS
# =========================================================
if "energy_consumed" not in df.columns:
    raise ValueError("Column 'energy_consumed' not found in the CSV files.")

if "project_name" not in df.columns:
    raise ValueError("Column 'project_name' not found in the CSV files.")

# =========================================================
# 3. CONVERT TO JOULES
# =========================================================
df["energy_j"] = df["energy_consumed"] * 3_600_000
df["net_energy_j"] = df["energy_j"] - BASELINE_J

# =========================================================
# 4. EXTRACT METADATA
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

# =========================================================
# 5. CLEAN
# =========================================================
df = df.dropna(subset=["lib", "rate", "rep", "net_energy_j"]).copy()
df["rate"] = df["rate"].astype(float)
df["rep"] = df["rep"].astype(int)

# Optional:
# df["net_energy_j"] = df["net_energy_j"].clip(lower=0)

# =========================================================
# TABLE 1: WINNER BY RATE (MAIN JUSTIFICATION TABLE)
# This is the key justification:
# different libraries can appear best at different rates
# =========================================================
avg_by_rate = (
    df.groupby(["rate", "lib"])["net_energy_j"]
    .mean()
    .reset_index(name="avg_net_energy_j")
)

winner_by_rate = (
    avg_by_rate.sort_values(["rate", "avg_net_energy_j"])
    .groupby("rate", as_index=False)
    .first()
    .rename(columns={"lib": "best_library_at_this_rate"})
)

print("\n" + "=" * 80)
print("TABLE 1: BEST LIBRARY BY RATE")
print("=" * 80)
print(winner_by_rate.to_string(index=False))

winner_by_rate.to_csv("table1_best_library_by_rate.csv", index=False)

# Optional pivot version for easier reading in paper/debugging
avg_by_rate_pivot = avg_by_rate.pivot(index="rate", columns="lib", values="avg_net_energy_j")
print("\n" + "=" * 80)
print("SUPPORT TABLE: AVERAGE NET ENERGY BY RATE AND LIBRARY")
print("=" * 80)
print(avg_by_rate_pivot)

avg_by_rate_pivot.to_csv("support_avg_net_energy_by_rate_and_library.csv")

# =========================================================
# OPTIONAL SUPPORT TABLE: OVERALL DESCRIPTIVE SUMMARY
# Keep this only as descriptive, not as main justification
# =========================================================
overall_summary = (
    df.groupby("lib")["net_energy_j"]
    .agg(
        mean="mean",
        std="std",
        median="median",
        min="min",
        max="max",
        count="count"
    )
    .reset_index()
)

overall_summary["range_j"] = overall_summary["max"] - overall_summary["min"]
overall_summary["cv_percent"] = (overall_summary["std"] / overall_summary["mean"]) * 100
overall_summary = overall_summary.sort_values("mean").reset_index(drop=True)

print("\n" + "=" * 80)
print("SUPPORT TABLE: OVERALL DESCRIPTIVE SUMMARY")
print("=" * 80)
print(overall_summary.to_string(index=False))

overall_summary.to_csv("support_overall_descriptive_summary.csv", index=False)

# =========================================================
# MATCHED BLOCK TABLE FOR OVERALL STATISTICS
# rows = matched blocks (rate, rep)
# cols = libraries
# =========================================================
pivot_df = df.pivot_table(
    index=["rate", "rep"],
    columns="lib",
    values="net_energy_j",
    aggfunc="mean"
).dropna()

if pivot_df.shape[1] < 2:
    raise ValueError("Need at least two libraries for Friedman test.")

if pivot_df.shape[0] < 2:
    raise ValueError("Need at least two matched blocks for Friedman test.")

print("\n" + "=" * 80)
print("MATCHED BLOCK TABLE USED FOR OVERALL FRIEDMAN")
print("=" * 80)
print(pivot_df)

# =========================================================
# TABLE 2A: OVERALL FRIEDMAN TEST
# =========================================================
samples = [pivot_df[col].values for col in pivot_df.columns]
friedman_stat, friedman_p = friedmanchisquare(*samples)

table2a = pd.DataFrame([{
    "friedman_statistic": friedman_stat,
    "friedman_p_value": friedman_p,
    "overall_difference_significant": friedman_p < ALPHA
}])

print("\n" + "=" * 80)
print("TABLE 2A: OVERALL FRIEDMAN TEST")
print("=" * 80)
print(table2a.to_string(index=False))

table2a.to_csv("table2a_overall_friedman_test.csv", index=False)

# =========================================================
# TABLE 2B: OVERALL MEAN RANKS
# lower mean rank = better overall
# =========================================================
mean_ranks = pivot_df.rank(axis=1, method="average", ascending=True).mean().sort_values()
best_lib = mean_ranks.idxmin()
best_rank = mean_ranks.min()

table2b = mean_ranks.reset_index()
table2b.columns = ["lib", "mean_rank"]
table2b = table2b.sort_values("mean_rank").reset_index(drop=True)

print("\n" + "=" * 80)
print("TABLE 2B: OVERALL MEAN RANKS (LOWER = BETTER)")
print("=" * 80)
print(table2b.to_string(index=False))

table2b.to_csv("table2b_overall_mean_ranks.csv", index=False)

# =========================================================
# TABLE 3: PAIRWISE WILCOXON VS BEST LIBRARY + HOLM
# =========================================================
pairwise_rows = []
raw_pvals = []
comparisons = []

for other in pivot_df.columns:
    if other == best_lib:
        continue

    x = pivot_df[best_lib].values
    y = pivot_df[other].values

    try:
        w_stat, p_raw = wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
    except ValueError:
        w_stat, p_raw = np.nan, 1.0

    comparisons.append((best_lib, other, w_stat, p_raw))
    raw_pvals.append(p_raw)

reject, p_holm, _, _ = multipletests(raw_pvals, alpha=ALPHA, method="holm")

for idx, (lib1, lib2, w_stat, p_raw) in enumerate(comparisons):
    pairwise_rows.append({
        "best_lib": lib1,
        "other_lib": lib2,
        "wilcoxon_W": w_stat,
        "raw_p_value": p_raw,
        "holm_corrected_p_value": p_holm[idx],
        "significant_after_holm": reject[idx]
    })

table3 = pd.DataFrame(pairwise_rows).sort_values("holm_corrected_p_value").reset_index(drop=True)

print("\n" + "=" * 80)
print("TABLE 3: PAIRWISE WILCOXON VS BEST LIBRARY")
print("=" * 80)
print(table3.to_string(index=False))

table3.to_csv("table3_pairwise_wilcoxon_vs_best.csv", index=False)

# =========================================================
# PLOT 1: Average net energy by rate
# =========================================================
plt.figure(figsize=(9, 5))
for lib, group in avg_by_rate.groupby("lib"):
    group = group.sort_values("rate")
    plt.plot(group["rate"], group["avg_net_energy_j"], marker="o", label=lib)

plt.xlabel("Rate (logs/sec)")
plt.ylabel("Average Net Energy (J)")
plt.title("Average Net Energy by Rate")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("plot_avg_energy_by_rate.png", dpi=300, bbox_inches="tight")
plt.show()

# =========================================================
# PLOT 2: Overall mean ranks
# =========================================================
plt.figure(figsize=(8, 5))
table2b.set_index("lib")["mean_rank"].plot(kind="bar")
plt.ylabel("Mean Rank (Lower = Better)")
plt.title("Overall Mean Rank by Library")
plt.tight_layout()
plt.savefig("plot_overall_mean_ranks.png", dpi=300, bbox_inches="tight")
plt.show()

# =========================================================
# FINAL INTERPRETATION
# =========================================================
print("\n" + "=" * 80)
print("FINAL INTERPRETATION")
print("=" * 80)

print("1. Table 1 shows that different libraries can appear best at different rates.")
print("2. Therefore, raw observation alone does not reveal a single obvious best overall library.")
print("3. Table 2A applies the overall Friedman test to check whether libraries differ significantly overall.")
print("4. Table 2B uses mean ranks to identify the best overall library.")
print("5. Table 3 confirms whether the best overall library significantly outperforms the others.")

print(f"\nBest overall library: {best_lib}")

if friedman_p < ALPHA:
    print("Overall Friedman result: statistically significant.")
else:
    print("Overall Friedman result: not statistically significant.")

print("\nLibraries ranked from best to worst:")
for i, row in table2b.iterrows():
    print(f"{i+1}. {row['lib']} -> mean rank = {row['mean_rank']:.4f}")
