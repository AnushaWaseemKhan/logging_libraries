import pandas as pd

# input file
input_csv = "/home/cs/grad/waseema/Python_logging/baseline/emissions/baseline_energy_cluster_200duration.csv"

# output file
output_csv = "/home/cs/grad/waseema/Python_logging/baseline/emissions/baseline_energy_stats.csv"

# load data
df = pd.read_csv(input_csv)

# -----------------------
# calculate mean and std
# -----------------------
mean_kwh = df["energy_kwh"].mean()
std_kwh = df["energy_kwh"].std()

# -----------------------
# convert to joules
# -----------------------
mean_joules = mean_kwh * 3_600_000
std_joules = std_kwh * 3_600_000

# create results table
result = pd.DataFrame({
    "metric": ["mean", "std"],
    "energy_kwh": [mean_kwh, std_kwh],
    "energy_joules": [mean_joules, std_joules]
})

# save
result.to_csv(output_csv, index=False)

print(result)
print("\nSaved to:", output_csv)