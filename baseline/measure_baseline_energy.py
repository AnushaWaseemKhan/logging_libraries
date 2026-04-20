import time
import csv
import pandas as pd
from pathlib import Path
from datetime import datetime
from codecarbon import EmissionsTracker


def measure_baseline(run_id, duration, emissions_dir):
    emissions_file = emissions_dir / "emissions.csv"

    tracker = EmissionsTracker(
        project_name=f"baseline_run_{run_id}",
        output_dir=str(emissions_dir),
        measure_power_secs=1,
        log_level="error",
        save_to_file=True,
    )

    tracker.start()
    time.sleep(duration)
    tracker.stop()

    df = pd.read_csv(emissions_file)
    last = df.iloc[-1]

    energy_kwh = float(last["energy_consumed"])
    emissions_kg = float(last["emissions"])
    cpu_energy_kwh = float(last["cpu_energy"])
    ram_energy_kwh = float(last["ram_energy"])
    gpu_energy_kwh = float(last["gpu_energy"])
    duration_sec = float(last["duration"])

    energy_wh = energy_kwh * 1000
    energy_joules = energy_kwh * 3_600_000
    avg_power_watts = energy_joules / duration_sec

    return (
        energy_kwh,
        energy_wh,
        energy_joules,
        emissions_kg,
        cpu_energy_kwh,
        ram_energy_kwh,
        gpu_energy_kwh,
        duration_sec,
        avg_power_watts,
    )


def run_baseline_experiment(runs=20, duration=200):
    emissions_dir = Path("emissions")
    emissions_dir.mkdir(parents=True, exist_ok=True)

    results_csv = emissions_dir / "baseline_energy.csv"

    with open(results_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp",
            "run",
            "requested_duration_sec",
            "measured_duration_sec",
            "energy_kwh",
            "energy_wh",
            "energy_joules",
            "emissions_kg",
            "cpu_energy_kwh",
            "ram_energy_kwh",
            "gpu_energy_kwh",
            "avg_power_watts",
        ])

    for run in range(1, runs + 1):
        timestamp = datetime.now().isoformat()

        (
            energy_kwh,
            energy_wh,
            energy_joules,
            emissions_kg,
            cpu_energy_kwh,
            ram_energy_kwh,
            gpu_energy_kwh,
            duration_sec,
            avg_power_watts,
        ) = measure_baseline(run, duration, emissions_dir)

        with open(results_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp,
                run,
                duration,
                duration_sec,
                energy_kwh,
                energy_wh,
                energy_joules,
                emissions_kg,
                cpu_energy_kwh,
                ram_energy_kwh,
                gpu_energy_kwh,
                avg_power_watts,
            ])


if __name__ == "__main__":
    run_baseline_experiment(runs=20, duration=200)