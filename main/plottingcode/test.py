import pandas as pd
import numpy as np
from pathlib import Path
import re

# =========================================================
# SETTINGS
# =========================================================
project_root = Path(__file__).resolve().parents[2]
data_path = project_root / "main" / "results" / "emission_per_run"

LIMIT_RATE = 1.0
LIMIT_REPS_TO = 3
RATE_TOL = 1e-6

# IMPORTANT: replace with your exact baseline value in Joules
BASELINE_J = 1465.73

# =========================================================
# HELPERS
# =========================================================
def extract_energy_j(file_path):
    try:
        df = pd.read_csv(file_path)

        if "energy_consumed" not in df.columns:
            return None

        df = df.dropna(subset=["energy_consumed"])
        if df.empty:
            return None

        return df["energy_consumed"].iloc[-1] * 3_600_000

    except Exception as e:
        print(f"[WARN] Could not read {file_path.name}: {e}")
        return None


def extract_lib(name):
    m = re.match(r"^([^_]+)", name)
    return m.group(1) if m else None


def extract_size(name):
    m = re.search(r"_size(\d+)", name)
    return int(m.group(1)) if m else None


def extract_rate(name):
    m = re.search(r"_rate([0-9]+(?:\.[0-9]+)?)", name)
    return float(m.group(1)) if m else None


def extract_rep(name):
    m = re.search(r"_rep(\d+)", name)
    return int(m.group(1)) if m else None


# =========================================================
# LOAD DATA
# =========================================================
rows = []

for file in data_path.glob("*.csv"):
    lib = extract_lib(file.name)
    size = extract_size(file.name)
    rate = extract_rate(file.name)
    rep = extract_rep(file.name)

    if None in [lib, size, rate, rep]:
        continue

    energy = extract_energy_j(file)
    if energy is None:
        continue

    rows.append({
        "lib": lib,
        "size": size,
        "rate": rate,
        "rep": rep,
        "energy_j": energy
    })

df = pd.DataFrame(rows)

if df.empty:
    print("No valid data found.")
    raise SystemExit

# =========================================================
# SUBTRACT BASELINE ENERGY
# =========================================================
df["net_energy_j"] = df["energy_j"] - BASELINE_J

# =========================================================
# LIMIT RATE 1.0 TO ONLY 3 REPS
# =========================================================
df = df[
    (~np.isclose(df["rate"], LIMIT_RATE, atol=RATE_TOL)) |
    (df["rep"] <= LIMIT_REPS_TO)
].copy()

# =========================================================
# SHOW FILTERED DATA INFO
# =========================================================
print("\n========== FILTERED DATA SUMMARY ==========")
summary = (
    df.groupby(["rate", "lib", "size"])["rep"]
    .count()
    .reset_index(name="num_reps")
    .sort_values(["rate", "lib", "size"])
)
print(summary.to_string(index=False))

# =========================================================
# OPTIONAL: SHOW RAW + NET SAMPLE
# =========================================================
print("\n========== SAMPLE RAW VS NET ENERGY ==========")
print(
    df[["lib", "size", "rate", "rep", "energy_j", "net_energy_j"]]
    .sort_values(["rate", "lib", "size", "rep"])
    .head(20)
    .to_string(index=False)
)

# =========================================================
# AVG TABLE
# rows = size
# columns = (rate, lib)
# values = mean NET energy across reps
# =========================================================
print("\n========== AVG NET ENERGY TABLE (SIZE × RATE × LIB) ==========")

avg_df = (
    df.groupby(["size", "rate", "lib"], as_index=False)["net_energy_j"]
    .mean()
)

avg_table = avg_df.pivot(index="size", columns=["rate", "lib"], values="net_energy_j")
avg_table = avg_table.sort_index().sort_index(axis=1)

print(avg_table.round(4))

# =========================================================
# STD TABLE
# =========================================================
print("\n========== STD DEV NET ENERGY TABLE ==========")

std_df = (
    df.groupby(["size", "rate", "lib"], as_index=False)["net_energy_j"]
    .std()
)

std_table = std_df.pivot(index="size", columns=["rate", "lib"], values="net_energy_j")
std_table = std_table.sort_index().sort_index(axis=1)

print(std_table.round(4))

# =========================================================
# REP COUNT TABLE
# =========================================================
print("\n========== REP COUNT TABLE ==========")

count_df = (
    df.groupby(["size", "rate", "lib"], as_index=False)["rep"]
    .count()
    .rename(columns={"rep": "num_reps"})
)

count_table = count_df.pivot(index="size", columns=["rate", "lib"], values="num_reps")
count_table = count_table.sort_index().sort_index(axis=1)

print(count_table)

# =========================================================
# FINAL READABLE TABLE
# =========================================================
print("\n========== FINAL TABLE (NET AVG ± STD (n)) ==========")

final_rows = []

all_sizes = sorted(df["size"].unique())
all_rates = sorted(df["rate"].unique())
all_libs = sorted(df["lib"].unique())

for size in all_sizes:
    row = {"size": size}

    for rate in all_rates:
        for lib in all_libs:
            subset = df[
                (df["size"] == size) &
                (df["rate"] == rate) &
                (df["lib"] == lib)
            ]

            col_name = f"{lib}_r{rate}"

            if subset.empty:
                row[col_name] = "-"
            else:
                mean_val = subset["net_energy_j"].mean()
                std_val = subset["net_energy_j"].std()
                n_val = len(subset)

                if pd.isna(std_val):
                    row[col_name] = f"{mean_val:.4f} (n={n_val})"
                else:
                    row[col_name] = f"{mean_val:.4f} ± {std_val:.4f} (n={n_val})"

    final_rows.append(row)

final_table = pd.DataFrame(final_rows).set_index("size")
print(final_table)

# =========================================================
# SAVE
# =========================================================
output_dir = project_root / "main" / "results" / "summary_tables"
output_dir.mkdir(parents=True, exist_ok=True)

avg_table.to_csv(output_dir / "avg_net_size_rate_lib_filtered.csv")
std_table.to_csv(output_dir / "std_net_size_rate_lib_filtered.csv")
count_table.to_csv(output_dir / "count_size_rate_lib_filtered.csv")
final_table.to_csv(output_dir / "final_readable_net_size_rate_lib_filtered.csv")

print(f"\nSaved to: {output_dir}")
