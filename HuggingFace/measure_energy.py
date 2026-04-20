import subprocess
from codecarbon import EmissionsTracker

tracker = EmissionsTracker(
    project_name="hf_picologging",
    output_dir="energy_results",
    output_file="picologging_hf_emissions.csv",
    save_to_file=True,
    log_level="error"
)

tracker.start()

subprocess.run(
    [
        "python",
        "run_glue_with_picologging.py",
        "--model_name_or_path", "distilbert-base-uncased",
        "--task_name", "sst2",
        "--max_length", "128",
        "--per_device_train_batch_size", "32",
        "--learning_rate", "2e-5",
        "--num_train_epochs", "5",
        "--output_dir", "./out_picologging",
    ],
    check=True,
)

emissions = tracker.stop()
print(f"Measured emissions: {emissions} kg CO2eq")
