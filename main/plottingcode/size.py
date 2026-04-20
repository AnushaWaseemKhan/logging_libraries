import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re

# =========================================================
# SETTINGS
# =========================================================
project_root = Path(__file__).resolve().parents[2]
data_path = project_root / "main" / "results" / "emission_per_run_finalfixedrate_1and1000_sizevary"

FIXED_RATE = 1.00
RATE_TOL = 1e-6

BASELINE_J = 1465.73
USE_NET_ENERGY = True   # True = plot net energy, False = raw energy

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
    raise ValueError("No valid data found. Check filenames and CSV contents.")

# =========================================================
# FILTER FIXED RATE
# =========================================================
df = df[np.isclose(df["rate"], FIXED_RATE, atol=RATE_TOL)].copy()

if df.empty:
    raise ValueError(f"No data found for fixed rate = {FIXED_RATE}")

# =========================================================
# COMPUTE NET ENERGY
# =========================================================
df["net_energy_j"] = df["energy_j"] - BASELINE_J

# Uncomment if needed
# df["net_energy_j"] = df["net_energy_j"].clip(lower=0)

# =========================================================
# PRINT RAW DATA
# =========================================================
print("\n========== SIZE-BASED ANALYSIS ==========\n")

for (lib, size), group in df.sort_values(["lib", "size", "rep"]).groupby(["lib", "size"]):
    print(f"\nLibrary = {lib} | Size = {size}")
    for _, row in group.iterrows():
        print(
            f"  Rep {row['rep']}: "
            f"Raw = {row['energy_j']:.6f} J | "
            f"Net = {row['net_energy_j']:.6f} J"
        )

# =========================================================
# AVERAGE BY LIBRARY AND SIZE
# =========================================================
avg_df = (
    df.groupby(["lib", "size"])
    .agg(
        avg_energy_j=("energy_j", "mean"),
        avg_net_energy_j=("net_energy_j", "mean"),
        std_energy_j=("energy_j", "std"),
        std_net_energy_j=("net_energy_j", "std"),
        median_energy_j=("energy_j", "median"),
        median_net_energy_j=("net_energy_j", "median"),
        n=("energy_j", "count"),
    )
    .reset_index()
    .sort_values(["size", "lib"])
)

print("\n========== FINAL AVERAGE BY LIBRARY AND SIZE ==========\n")
for _, row in avg_df.iterrows():
    print(
        f"Library {row['lib']} | Size {row['size']}: "
        f"Avg Raw = {row['avg_energy_j']:.6f} J | "
        f"Avg Net = {row['avg_net_energy_j']:.6f} J | "
        f"Std Net = {row['std_net_energy_j']:.6f} J | "
        f"Median Net = {row['median_net_energy_j']:.6f} J | "
        f"N = {row['n']}"
    )

avg_df.to_csv(f"average_energy_by_library_and_size_rate_{FIXED_RATE}.csv", index=False)

# =========================================================
# OVERALL SUMMARY BY LIBRARY
# =========================================================
overall_summary = (
    df.groupby("lib")[["energy_j", "net_energy_j"]]
    .agg(["mean", "std", "median", "count"])
)

print("\n========== OVERALL SUMMARY BY LIBRARY ==========\n")
print(overall_summary)

# =========================================================
# PLOT SETTINGS
# =========================================================
output_dir = Path("plots_size")
output_dir.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 18,
    "axes.titlesize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 12,
    "axes.linewidth": 1.1,
    "lines.linewidth": 2.0,
    "savefig.dpi": 700
})

libs = sorted(avg_df["lib"].unique())
size_order = sorted(avg_df["size"].unique())

marker_list = ['o', 's', '^', 'D', 'v', 'P', 'X', '<', '>', 'h']
marker_map = {lib: marker_list[i % len(marker_list)] for i, lib in enumerate(libs)}

y_col = "avg_net_energy_j" if USE_NET_ENERGY else "avg_energy_j"
yerr_col = "std_net_energy_j" if USE_NET_ENERGY else "std_energy_j"
y_label = "Average Net Energy (J)" if USE_NET_ENERGY else "Average Energy (J)"

# =========================================================
# COMBINED SIZE PLOT
# =========================================================
fig, ax = plt.subplots(figsize=(11, 7))

for lib in libs:
    lib_data = avg_df[avg_df["lib"] == lib].sort_values("size")

    x = lib_data["size"].values
    y = lib_data[y_col].values
    yerr = lib_data[yerr_col].fillna(0).values

    ax.plot(
        x,
        y,
        linestyle='-',
        linewidth=2.0,
        marker=marker_map[lib],
        markersize=6.5,
        markerfacecolor='white',
        markeredgewidth=1.2,
        label=lib
    )

    ax.errorbar(
        x,
        y,
        yerr=yerr,
        fmt='none',
        elinewidth=1.0,
        capsize=3,
        capthick=1.0
    )

ax.set_xticks(size_order)
ax.set_xticklabels([str(s) for s in size_order])

ax.set_xlabel("Message Size (bytes)")
ax.set_ylabel(y_label)
ax.set_title(f"Energy Consumption vs Message Size (Rate = {FIXED_RATE:g} logs/sec)")

ax.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.6)
ax.grid(True, which="minor", linestyle=":", linewidth=0.3, alpha=0.35)

legend = ax.legend(title="Library", frameon=True, edgecolor="black", ncol=2)
legend.get_frame().set_linewidth(0.8)

plt.tight_layout()

png_path = output_dir / f"energy_vs_size_rate_{str(FIXED_RATE).replace('.', '_')}.png"
pdf_path = output_dir / f"energy_vs_size_rate_{str(FIXED_RATE).replace('.', '_')}.pdf"

plt.savefig(png_path, bbox_inches="tight")
plt.savefig(pdf_path, bbox_inches="tight")
plt.show()

print(f"\nSaved combined size plot to:\n{png_path.resolve()}\n{pdf_path.resolve()}")

# =========================================================
# INDIVIDUAL SIZE PLOTS PER LIBRARY
# =========================================================
individual_dir = output_dir / "individual_plots"
individual_dir.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 16,
    "axes.labelsize": 20,
    "axes.titlesize": 20,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "axes.linewidth": 1.2,
    "lines.linewidth": 2.5,
    "savefig.dpi": 700
})

for lib in libs:
    lib_data = avg_df[avg_df["lib"] == lib].sort_values("size")

    x = lib_data["size"].values
    y = lib_data[y_col].values
    yerr = lib_data[yerr_col].fillna(0).values

    fig, ax = plt.subplots(figsize=(9, 6.5))

    ax.plot(
        x,
        y,
        color='black',
        linestyle='-',
        linewidth=2.5,
        marker='o',
        markersize=8,
        markerfacecolor='white',
        markeredgecolor='black',
        markeredgewidth=1.5
    )

    ax.errorbar(
        x,
        y,
        yerr=yerr,
        fmt='none',
        ecolor='gray',
        elinewidth=1.2,
        capsize=4,
        capthick=1.2
    )

    ax.set_xticks(size_order)
    ax.set_xticklabels([str(s) for s in size_order])

    ax.set_xlabel("Message Size (bytes)")
    ax.set_ylabel(y_label)
    ax.set_title(f"{lib} (Rate = {FIXED_RATE:g} logs/sec)", pad=10)

    ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.6)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.4)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    safe_lib_name = lib.replace(" ", "_").replace("(", "").replace(")", "")
    png_lib_path = individual_dir / f"{safe_lib_name}_energy_vs_size.png"
    pdf_lib_path = individual_dir / f"{safe_lib_name}_energy_vs_size.pdf"

    plt.savefig(png_lib_path, bbox_inches="tight", dpi=700)
    plt.savefig(pdf_lib_path, bbox_inches="tight")
    plt.close()

print(f"\nSaved individual size plots in:\n{individual_dir.resolve()}")


# =========================================================
# COMBINED SIZE PLOT (NO STD / CLEAN VERSION)
# =========================================================
fig, ax = plt.subplots(figsize=(11, 7))

for lib in libs:
    lib_data = avg_df[avg_df["lib"] == lib].sort_values("size")

    x = lib_data["size"].values
    y = lib_data[y_col].values

    # Clean line only (NO error bars)
    ax.plot(
        x,
        y,
        linestyle='-',
        linewidth=2.2,
        marker=marker_map[lib],
        markersize=7,
        markerfacecolor='white',
        markeredgewidth=1.2,
        label=lib
    )

# =========================================================
# AXIS FORMATTING
# =========================================================
ax.set_xticks(size_order)
ax.set_xticklabels([str(s) for s in size_order])

ax.set_xlabel("Message Size (bytes)")
ax.set_ylabel(y_label)
ax.set_title(f"Energy Consumption vs Message Size (Rate = {FIXED_RATE:g} logs/sec)")

ax.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.6)
ax.grid(True, which="minor", linestyle=":", linewidth=0.3, alpha=0.35)

legend = ax.legend(title="Library", frameon=True, edgecolor="black", ncol=2)
legend.get_frame().set_linewidth(0.8)

plt.tight_layout()

png_path = output_dir / f"energy_vs_size_rate_{str(FIXED_RATE).replace('.', '_')}_clean.png"
pdf_path = output_dir / f"energy_vs_size_rate_{str(FIXED_RATE).replace('.', '_')}_clean.pdf"

plt.savefig(png_path, bbox_inches="tight")
plt.savefig(pdf_path, bbox_inches="tight")
plt.show()

print(f"\nSaved CLEAN combined size plot to:\n{png_path.resolve()}\n{pdf_path.resolve()}")
