import pandas as pd
import matplotlib.pyplot as plt
import re
from pathlib import Path
from scipy.stats import friedmanchisquare
import scikit_posthocs as sp

# =========================
# 0. Settings
# =========================
BASELINE_J = 1465.73

project_root = Path(__file__).resolve().parents[2]
data_path = project_root / "main" / "results" / "emission_per_run_3libs_allrates"
files = list(data_path.glob("*.csv"))

if not files:
    raise FileNotFoundError(f"No CSV files found in {data_path.resolve()}")

# =========================
# 1. Load CSVs
# =========================
df_list = []
for file in files:
    temp = pd.read_csv(file)
    temp["source_file"] = file.name
    df_list.append(temp)

df = pd.concat(df_list, ignore_index=True)

# =========================
# 2. Convert to Joules
# =========================
if "energy_consumed" not in df.columns:
    raise ValueError("Column 'energy_consumed' not found in the CSV files.")

df["energy_j"] = df["energy_consumed"] * 3_600_000

# =========================
# 3. Extract metadata from project_name
# =========================
if "project_name" not in df.columns:
    raise ValueError("Column 'project_name' not found in the CSV files.")

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

# =========================
# 4. Clean
# =========================
df = df.dropna(subset=["lib", "rate", "rep", "energy_j"]).copy()
df["rate"] = df["rate"].astype(float)
df["rep"] = df["rep"].astype(int)

# =========================
# 5. Compute net energy
# =========================
df["net_energy_j"] = df["energy_j"] - BASELINE_J

# Optional: clip tiny negative values if you want
# df["net_energy_j"] = df["net_energy_j"].clip(lower=0)

# =========================
# 6. Print raw data
# =========================
print("\n========== RAW DATA (ALL REPS) ==========\n")

for (lib, rate), group in df.sort_values(["lib", "rate", "rep"]).groupby(["lib", "rate"]):
    print(f"\nLibrary = {lib} | Rate = {rate}")
    for _, row in group.iterrows():
        print(
            f"  Rep {row['rep']}: "
            f"Raw = {row['energy_j']:.6f} J | "
            f"Net = {row['net_energy_j']:.6f} J"
        )

# =========================
# 7. Average by lib and rate
# =========================
avg_df = (
    df.groupby(["lib", "rate"])
    .agg(
        avg_energy_j=("energy_j", "mean"),
        avg_net_energy_j=("net_energy_j", "mean"),
        std_net_energy_j=("net_energy_j", "std"),
        median_net_energy_j=("net_energy_j", "median"),
        n=("net_energy_j", "count"),
    )
    .reset_index()
    .sort_values(["rate", "lib"])
)

print("\n========== FINAL AVERAGE BY LIBRARY AND RATE (ALL REPS) ==========\n")
for _, row in avg_df.iterrows():
    print(
        f"Library {row['lib']} | Rate {row['rate']}: "
        f"Avg Raw = {row['avg_energy_j']:.6f} J | "
        f"Avg Net = {row['avg_net_energy_j']:.6f} J | "
        f"Median Net = {row['median_net_energy_j']:.6f} J | "
        f"N = {row['n']}"
    )

avg_df.to_csv("average_energy_by_library_and_rate.csv", index=False)

# =========================
# 8. Overall summary by library
# =========================
overall_summary = (
    df.groupby("lib")["net_energy_j"]
    .agg(["mean", "std", "median", "count"])
    .reset_index()
    .sort_values(["mean", "median"])
)

print("\n========== OVERALL SUMMARY BY LIBRARY ==========\n")
print(overall_summary)

overall_summary.to_csv("overall_library_summary_all_reps.csv", index=False)

# =========================
# 9. Friedman test
# =========================
print("\n========== FRIEDMAN TEST (OVERALL) ==========\n")

# rows = matched blocks, columns = libraries
pivot_df = df.pivot_table(
    index=["rate", "rep"],
    columns="lib",
    values="net_energy_j",
    aggfunc="mean"
)

# Keep only complete matched blocks
pivot_df = pivot_df.dropna()

if pivot_df.shape[1] < 2:
    raise ValueError("Need at least two libraries for Friedman test.")

if pivot_df.shape[0] < 2:
    raise ValueError("Need at least two matched blocks for Friedman test.")

print("Matched block table used for Friedman:")
print(pivot_df)

samples = [pivot_df[col].values for col in pivot_df.columns]

stat, p = friedmanchisquare(*samples)

print(f"\nFriedman statistic = {stat:.4f}")
print(f"p-value = {p:.6f}")

if p < 0.05:
    print("✅ Significant difference between libraries")
else:
    print("❌ No significant difference between libraries")

# =========================
# 10. Mean ranks
# =========================
mean_ranks = pivot_df.rank(axis=1, method="average", ascending=True).mean().sort_values()

print("\n========== MEAN RANKS (LOWER = BETTER) ==========\n")
print(mean_ranks)

mean_ranks_df = mean_ranks.reset_index()
mean_ranks_df.columns = ["lib", "mean_rank"]
mean_ranks_df.to_csv("friedman_mean_ranks.csv", index=False)

best_lib = mean_ranks.idxmin()
best_rank = mean_ranks.min()

print(f"\n⭐ BEST LIBRARY BY FRIEDMAN MEAN RANK: {best_lib} (mean rank = {best_rank:.4f})")

# =========================
# 11. Nemenyi post hoc
# =========================
print("\n========== NEMENYI POST HOC TEST ==========\n")

# IMPORTANT: use pivot_df.values, then rebuild DataFrame labels
nemenyi_array = sp.posthoc_nemenyi_friedman(pivot_df.values)
nemenyi = pd.DataFrame(
    nemenyi_array,
    index=pivot_df.columns,
    columns=pivot_df.columns
)

print(nemenyi)
nemenyi.to_csv("nemenyi_posthoc_friedman.csv")

print(f"\nComparison vs best library: {best_lib}")
for other in nemenyi.columns:
    if other == best_lib:
        continue
    p_val = nemenyi.loc[best_lib, other]
    if p_val < 0.05:
        print(f"  {best_lib} vs {other}: significant (p = {p_val:.6f})")
    else:
        print(f"  {best_lib} vs {other}: not significant (p = {p_val:.6f})")

# =========================
# 12. Plot mean net energy vs rate
# =========================
plt.figure(figsize=(8, 5))

for lib, group in avg_df.groupby("lib"):
    group = group.sort_values("rate")
    plt.plot(group["rate"], group["avg_net_energy_j"], marker="o", label=lib)

plt.xlabel("Rate (logs/sec)")
plt.ylabel("Net Energy (Joules)")
plt.title("Average Net Energy vs Rate")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("energy_vs_rate_all_reps.png", dpi=300, bbox_inches="tight")
plt.show()

# =========================
# 13. Overall boxplot
# =========================
plt.figure(figsize=(8, 5))
df.boxplot(column="net_energy_j", by="lib", grid=False)
plt.title("Overall Net Energy Distribution by Library")
plt.suptitle("")
plt.xlabel("Library")
plt.ylabel("Net Energy (Joules)")
plt.tight_layout()
plt.savefig("overall_net_energy_boxplot_all_reps.png", dpi=300, bbox_inches="tight")
plt.show()

# =========================
# 14. Mean rank plot
# =========================
plt.figure(figsize=(8, 5))
mean_ranks.sort_values().plot(kind="bar")
plt.ylabel("Mean Rank (Lower = Better)")
plt.title("Friedman Mean Ranks by Library")
plt.tight_layout()
plt.savefig("friedman_mean_ranks.png", dpi=300, bbox_inches="tight")
plt.show()

# =========================
# 15. Final conclusion print
# =========================
print("\n========== FINAL CONCLUSION ==========\n")
print(f"Best overall library by Friedman mean rank: {best_lib}")

if p < 0.05:
    print("The overall difference between libraries is statistically significant.")
else:
    print("The overall difference between libraries is not statistically significant.")

print("\nLibraries ranked from best to worst:")
for i, (lib, rank_val) in enumerate(mean_ranks.items(), start=1):
    print(f"{i}. {lib} -> mean rank = {rank_val:.4f}")
