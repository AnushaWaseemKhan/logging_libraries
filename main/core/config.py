from pathlib import Path

#sizes = [64, 512, 1024, 1536, 2048, 8192]
#rates = [0.01, 0.10, 1.00, 10.00, 100.00, 1000.00]
reps = 1
pre_run_sleep_s = 5
duration_s = 200
num_logs=1000000

sizes = [64]
rates = [0.01, 0.10, 1.00, 10.00, 100.00, 1000.00]

#log_statements = ["debug", "info", "warning", "error", "critical"]
#outputs = ["file", "console"]
outputs=["console"]
log_statements=["debug"]
#outputs = ["file"]


codecarbon_kwargs = {
    "measure_power_secs": 1,
    "log_level": "error",
}

project_root = Path(__file__).resolve().parents[1]
logs_dir = project_root / "logs"
results_dir = project_root / "results"
