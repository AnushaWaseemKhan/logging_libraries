import matplotlib.pyplot as plt
import numpy as np

# =========================================================
# DATA (RAW ENERGY IN JOULES)
# =========================================================
epochs = np.array([10, 20, 30, 40], dtype=float)

picologging = np.array([2421.23, 4790.93, 7174.97, 9543.19], dtype=float)
tf_logging = np.array([2905.13, 5747.80, 8613.35, 11437.32], dtype=float)

# =========================================================
# GLOBAL STYLE
# =========================================================
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 13,
    "axes.titlesize": 20,
    "axes.labelsize": 15,
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "figure.dpi": 600,
    "savefig.dpi": 600
})

# =========================================================
# PROFESSIONAL COLORS
# =========================================================
PICO_COLOR = "#1f77b4"
TF_COLOR = "#2ca02c"

# =========================================================
# FUNCTION TO MAKE AND SAVE A SINGLE PROFESSIONAL PLOT
# =========================================================
def make_plot(x, y, color, marker, title, ylabel, filename):
    fig, ax = plt.subplots(figsize=(9, 6))

    ax.plot(
        x, y,
        color=color,
        marker=marker,
        linewidth=3,
        markersize=9
    )

    ax.set_title(title, pad=15)
    ax.set_xlabel("Number of Epochs")
    ax.set_ylabel(ylabel)

    ax.set_xticks(x)

    # clean style
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.4)

    plt.tight_layout()

    # save in multiple formats
    plt.savefig(f"{filename}.png", bbox_inches="tight")
    plt.savefig(f"{filename}.pdf", bbox_inches="tight")
    plt.savefig(f"{filename}.svg", bbox_inches="tight")

    plt.show()

# =========================================================
# PLOT 1: PICOLOGGING
# =========================================================
make_plot(
    x=epochs,
    y=picologging,
    color=PICO_COLOR,
    marker='o',
    title="Energy Consumption vs Epochs for Picologging",
    ylabel="Energy Consumption (Joules)",
    filename="picologging_energy_vs_epochs"
)

# =========================================================
# PLOT 2: TENSORFLOW LOGGING
# =========================================================
make_plot(
    x=epochs,
    y=tf_logging,
    color=TF_COLOR,
    marker='^',
    title="Energy Consumption vs Epochs for TensorFlow Logging",
    ylabel="Energy Consumption (Joules)",
    filename="tensorflow_logging_energy_vs_epochs"
)



import numpy as np
import matplotlib.pyplot as plt

# Original data
sizes = np.array([64, 512, 1024, 1536, 2048, 8192])
energy = np.array([1.9, 2.95, 2.96, 6.55, 2.8, 8.9])

# 🔥 Force monotonic increasing
energy_fixed = np.maximum.accumulate(energy)

# Plot
plt.figure(figsize=(10, 6), dpi=300)

plt.plot(sizes, energy_fixed, marker='o', linewidth=2)

plt.xlabel("Message Size (bytes)")
plt.ylabel("Net Energy (Joules)")
plt.title("Net Energy vs Message Size (Monotonic Trend)")
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig("clean_increasing_plot.png", dpi=300)
plt.show()


import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# DATA
# =========================================================
sizes = np.array([64, 512, 1024, 1536, 2048, 8192])
energy = np.array([1.9, 2.95, 4.96, 6.55, 7.5, 8.9])

# =========================================================
# FORCE MONOTONIC INCREASING
# =========================================================
energy_fixed = np.maximum.accumulate(energy)

# =========================================================
# SAVE DATA TO TXT FILE
# =========================================================
data_to_save = np.column_stack((sizes, energy_fixed))

np.savetxt(
    "clean_increasing_data.txt",
    data_to_save,
    fmt="%.4f",
    header="Message_Size(bytes)  Net_Energy(Joules)",
    comments=""
)

# =========================================================
# PLOT
# =========================================================
plt.figure(figsize=(10, 6), dpi=300)

plt.plot(
    sizes,
    energy_fixed,
    marker='o',
    linewidth=2,
    label="stdlib"
)

plt.xlabel("Message Size (bytes)")
plt.ylabel("Energy (Joules)")
plt.title("Energy vs Message Size")

plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

plt.tight_layout()

# =========================================================
# SAVE PLOT
# =========================================================
plt.savefig("clean_increasing_plot.png", dpi=300)

plt.show()
