from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Path to your CSV
csv_file = Path("baseline_energy_of_cluster.csv")

# Load data
df = pd.read_csv(csv_file)

# ---- Plot 1: Emissions vs Run ----
plt.figure()

plt.plot(df["run"], df["emissions_kg"], marker="o")

plt.xlabel("Run")
plt.ylabel("Emissions (kg CO2)")
plt.title("Baseline Energy Emissions per Run")

plt.grid(True)

plt.savefig("baseline_emissions_per_run.png")
plt.show()


# ---- Plot 2: Histogram of emissions ----
plt.figure()

plt.hist(df["emissions_kg"], bins=10)

plt.xlabel("Emissions (kg CO2)")
plt.ylabel("Frequency")
plt.title("Distribution of Baseline Emissions")

plt.grid(True)

plt.savefig("baseline_emissions_histogram.png")
plt.show()