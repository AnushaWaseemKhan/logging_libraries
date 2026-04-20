import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
# files
# ----------------------------
input_csv = "/home/cs/grad/waseema/Python_logging/size/results/emission_per_run/emissions_600duration.csv"

output_original = "avg_energy_by_rate.csv"
output_adjusted = "avg_energy_minus_baseline.csv"
plot_file = "/home/cs/grad/waseema/Python_logging/size/images/energy_vs_rate_all_libraries.png"

# baseline mean energy in joules
BASELINE_MEAN_J = 15187.580828195132

# ----------------------------
# load csv
# ----------------------------
df = pd.read_csv(input_csv)

# ----------------------------
# extract rate, library, msg_size
# example project_name:
# stdlib_fixed_size_file_debug_size2048_rate0.01_rep1
# ----------------------------
df["rate"] = df["project_name"].str.extract(r"_rate([0-9.]+)_rep\d+$")[0].astype(float)
df["library"] = df["project_name"].str.extract(r"^([^_]+)")[0]
df["msg_size"] = df["project_name"].str.extract(r"_size(\d+)_rate")[0].astype(int)

# ----------------------------
# average the 10 repetitions
# ----------------------------
result = (
    df.groupby(["library", "msg_size", "rate"], as_index=False)
      .agg(
          repetitions=("energy_consumed", "count"),
          avg_energy_kwh=("energy_consumed", "mean"),
          std_energy_kwh=("energy_consumed", "std"),
          avg_cpu_energy_kwh=("cpu_energy", "mean"),
          avg_gpu_energy_kwh=("gpu_energy", "mean"),
          avg_ram_energy_kwh=("ram_energy", "mean"),
      )
)

# ----------------------------
# convert to joules
# ----------------------------
result["avg_energy_joules"] = result["avg_energy_kwh"] * 3_600_000
result["std_energy_joules"] = result["std_energy_kwh"] * 3_600_000
result["avg_cpu_energy_joules"] = result["avg_cpu_energy_kwh"] * 3_600_000
result["avg_gpu_energy_joules"] = result["avg_gpu_energy_kwh"] * 3_600_000
result["avg_ram_energy_joules"] = result["avg_ram_energy_kwh"] * 3_600_000

# sort
result = result.sort_values(["library", "msg_size", "rate"]).reset_index(drop=True)

# ----------------------------
# save original averaged file
# ----------------------------
result.to_csv(output_original, index=False)
print("Saved original averaged file:", output_original)

# ----------------------------
# baseline-subtracted file
# ----------------------------
adjusted = result.copy()
adjusted["energy_minus_baseline_joules"] = adjusted["avg_energy_joules"] - BASELINE_MEAN_J

adjusted.to_csv(output_adjusted, index=False)
print("Saved baseline-subtracted file:", output_adjusted)

# ----------------------------
# plotting
# one graph for all libraries
# x-axis = rate
# y-axis = energy consumed in joules
# using baseline-subtracted energy
# ----------------------------
plt.figure(figsize=(10, 6))

for library in sorted(adjusted["library"].unique()):
    lib_df = adjusted[adjusted["library"] == library].copy()
    lib_df = lib_df.sort_values("rate")
    
    plt.plot(
        lib_df["rate"],
        lib_df["energy_minus_baseline_joules"],
        marker="o",
        label=library
    )

plt.xscale("log")  # useful because rates are like 0.01, 0.1, 1, 10, 100, 1000
plt.xlabel("Rate (messages per second)")
plt.ylabel("Energy Consumed (Joules, baseline-subtracted)")
plt.title("Energy Consumed vs Rate for All Libraries")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(plot_file, dpi=300)
plt.show()

print("Saved plot:", plot_file)