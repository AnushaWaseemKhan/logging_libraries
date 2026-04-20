import pandas as pd
import numpy as np
from pathlib import Path
import re
from scipy.stats import friedmanchisquare, wilcoxon
from itertools import combinations

# =========================================================
# SETTINGS
# =========================================================
project_root = Path(__file__).resolve().parents[2]
data_path = project_root / "main" / "results" / "emission_per_run_finalfixedrate_1and1000_sizevary"

FIXED_RATE = 1.00
RATE_TOL = 1e-6

# =========================================================
# HELPERS
# =========================================================
def extract_energy_j(file):
    df = pd.read_csv(file)

    if "energy_consumed" not in df.columns:
        return None

    df = df.dropna(subset=["energy_consumed"])
    if df.empty:
        return None

    return df["energy_consumed"].iloc[-1] * 3_600_000


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


def holm_correction(pairs):
    m = len(pairs)
    pairs = sorted(pairs, key=lambda x: x[2])

    results = []
    prev = 0

    for i, (a, b, p) in enumerate(pairs):
        adj = min((m - i) * p, 1.0)
        adj = max(adj, prev)
        prev = adj

        results.append((a, b, p, adj, adj < 0.05))

    return results


# =========================================================
# LOAD DATA
# =========================================================
rows = []

for file in data_path.glob("*.csv"):
    lib = extract_lib(file.name)
    size = extract_size(file.name)
    rate = extract_rate(file.name)
    rep = extract_rep(file.name)

0    if None in [lib, size, rate, rep]:
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

# =========================================================
# FILTER FIXED RATE
# =========================================================
df = df[np.isclose(df["rate"], FIXED_RATE, atol=RATE_TOL)].copy()

print("\n========== SIZE-BASED ANALYSIS ==========")

all_libs = sorted(df["lib"].unique())

for size in sorted(df["size"].unique()):
    subset = df[df["size"] == size].copy()

    pivot = subset.pivot_table(
        index="rep",
        columns="lib",
        values="energy_j",
        aggfunc="mean"
    )

    pivot = pivot.dropna()

    libs = list(pivot.columns)

    print(f"\nSIZE = {size}")
    print(f"Libraries: {libs}")
    print(f"Matched reps: {len(pivot)}")

    if len(libs) < 3 or len(pivot) < 2:
        print("Not enough data → skipping")
        continue

   
