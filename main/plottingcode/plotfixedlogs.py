import pandas as pd
import matplotlib.pyplot as plt
import re
from pathlib import Path

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
# 2. Convert to Joules and subtract baseline
# =========================
if "energy_consumed" not in df.columns:
    raise ValueError("Column 'energy_consumed' not found in CSV files.")

if "project_name" not in df.columns:
    raise ValueError("Column 'project_name' not found in CSV files.")

df["energy_j"] = df["energy_consumed"] * 3_600_000
df["net_energy_j"] = df["energy_j"] - BASELINE_J

# =========================
# 3. Extract metadata from project_name
# =========================
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
# 4. Clean data
# =========================
df = df.dropna(subset=["lib", "rate", "rep", "energy_j", "net_energy_j"]).copy()
df["rate"] = df["rate"].astype(float)
df["rep"] = df["rep"].astype(int)

# =========================
# 5. Remove first and last rep for each (lib, rate)
# =========================
df = df.sort_values(["lib", "rate", "rep"]).copy()

df["rep_rank_asc"] = df.groupby(["lib", "rate"]).cumcount() + 1
df["rep_rank_desc"] = df.groupby(["lib", "rate"]).cumcount(ascending=False) + 1

filtered_df = df[(df["rep_rank_asc"] > 1) & (df["rep_rank_desc"] > 1)].copy()

if filtered_df.empty:
    raise ValueError("No data left after removing first and last reps.")

# =========================
# 6. Print remaining reps
# =========================
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

# =========================
# 7. Average net energy by library and rate
# =========================
avg_df = (
    filtered_df.groupby(["lib", "rate"], as_index=False)
    .agg(
        avg_net_energy_j=("net_energy_j", "mean"),
        n=("net_energy_j", "count"),
    )
    .sort_values(["lib", "rate"])
)

print("\n========== AVERAGE NET ENERGY BY LIBRARY AND RATE ==========\n")
for _, row in avg_df.iterrows():
    print(
        f"Library = {row['lib']} | "
        f"Rate = {row['rate']} | "
        f"Avg Net Energy = {row['avg_net_energy_j']:.4f} J | "
        f"N = {row['n']}"
    )

avg_df.to_csv("average_net_energy_by_library_and_rate_no_first_last.csv", index=False)

# =========================
# 8. Plot average net energy vs rate (NO STD, SAVE ONLY)
# =========================
plt.figure(figsize=(9, 5))

for lib, group in avg_df.groupby("lib"):
    group = group.sort_values("rate")
    plt.plot(
        group["rate"],
        group["avg_net_energy_j"],
        marker="o",
        linewidth=2,
        label=lib
    )

plt.xscale("log")
plt.xlabel("Rate (logs/sec)")
plt.ylabel("Average Net Energy (Joules)")
plt.title("Average Net Energy vs Rate (All Libraries)\nBaseline Removed, First and Last Reps Excluded")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()

save_path = "net_energy_vs_rate_all_libs_no_std_no_first_last.png"
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"\n✅ Graph saved at: {save_path}")
