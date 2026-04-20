from pathlib import Path
import pandas as pd
import os

from .config import (
    sizes, rates, reps, pre_run_sleep_s, duration_s,
    log_statements, outputs,
    logs_dir, results_dir, project_root,
    codecarbon_kwargs, num_logs
)
from .helpers import (
    safe_name, make_message, get_level,
    iter_sizes_rates, log_path_for, make_names, ensure_base_dirs,merge_experiment_csvs
)
from .libraries import LIBRARIES
from .energy_measure import measure


def map_index_to_experiment(mode, index):
    if index < 0:
        # Fallback: run all sequentially
        sizes_to_run, rates_to_run = iter_sizes_rates(mode, sizes, rates)
        reps_to_run = range(1, reps + 1)
        return sizes_to_run, rates_to_run, reps_to_run

    sizes_list = sizes
    rates_list = rates
    reps_list = list(range(1, reps + 1))

    total_experiments = len(sizes_list) * len(rates_list) * len(reps_list)
    if index >= total_experiments:
        raise ValueError(f"SLURM_ARRAY_TASK_ID={index} exceeds total experiments={total_experiments}")

    size_idx = index // (len(rates_list) * len(reps_list))
    rate_idx = (index // len(reps_list)) % len(rates_list)
    rep_idx = index % len(reps_list)

    sizes_to_run = [sizes_list[size_idx]]
    rates_to_run = [rates_list[rate_idx]]
    reps_to_run = [reps_list[rep_idx]]

    return sizes_to_run, rates_to_run, reps_to_run


def run_mode(mode):
    index = int(os.getenv("SLURM_ARRAY_TASK_ID", -1))
    sizes_to_run, rates_to_run, reps_to_run = map_index_to_experiment(mode, index)
    per_run_dir, emissions_dir = ensure_base_dirs(logs_dir, results_dir)

    rows = []

    for lib_name, logger_cls in LIBRARIES.items():
        for output in outputs:
            for statement in log_statements:
                for size in sizes_to_run:

                    rate_values = [None] if mode == "fixed_messages" else rates_to_run

                    for rate in rate_values:
                        for rep in reps_to_run:

                            msg = make_message(size)
                            level = get_level(statement)

                            if mode == "fixed_messages":
                                project_name = (
                                    f"{lib_name}_{mode}_{output}_{statement}_"
                                    f"size{size}_nmsg{num_logs}_rep{rep}"
                                )
                                log_tag = project_name
                            else:
                                _, log_tag, project_name = make_names(
                                    mode, lib_name, output, statement, size, rate, rep
                                )

                            log_subdir = logs_dir / safe_name(lib_name) / safe_name(mode) / safe_name(statement)
                            log_subdir.mkdir(parents=True, exist_ok=True)

                            per_run_subdir = per_run_dir / safe_name(lib_name) / safe_name(mode) / safe_name(statement)
                            per_run_subdir.mkdir(parents=True, exist_ok=True)

                            log_path = log_path_for(output, log_subdir, log_tag)

                            # Each job writes its own emissions file
                            emissions_path = emissions_dir / f"{safe_name(project_name)}.csv"

                            per_run_csv = per_run_subdir / f"{safe_name(project_name)}.csv"

                            # Skip if already exists
                            if per_run_csv.exists():
                                continue

                            measure_kwargs = dict(
                                project_name=project_name,
                                emissions_path=emissions_path,
                                logger_cls=logger_cls,
                                lib_name=lib_name,
                                msg=msg,
                                level=level,
                                output=output,
                                log_path=log_path,
                                pre_run_sleep_s=pre_run_sleep_s,
                                codecarbon_kwargs=codecarbon_kwargs,
                            )

                            if mode == "fixed_messages":
                                energy_kwh, runtime_s, log_count, actual_rate = measure(
                                    **measure_kwargs,
                                    num_logs=num_logs,
                                )
                            else:
                                energy_kwh, runtime_s, log_count, actual_rate = measure(
                                    **measure_kwargs,
                                    rate=rate,
                                    duration_s=duration_s,
                                )

                            row = {
                                "mode": mode,
                                "library": lib_name,
                                "output": output,
                                "statement": statement,
                                "msg_size_bytes": int(size),
                                "rep": int(rep),
                                "runtime_s": float(runtime_s),
                                "energy_kwh": float(energy_kwh),
                                "log_file": str(log_path) if log_path else "",
                                "project_name": project_name,
                                "emissions_file": str(emissions_path),
                                "num_logs": int(num_logs) if mode == "fixed_messages" else None,
                                "rate_msgs_per_s": float(rate) if mode != "fixed_messages" else None,
                                "duration_s": float(duration_s) if mode != "fixed_messages" else None,
                                "log_count": log_count,
                                "actual_rate": actual_rate
                            }

                            rows.append(row)

                            pd.DataFrame([row]).to_csv(per_run_csv, index=False)

    return None


def run():
    ensure_base_dirs(logs_dir, results_dir)
    outs = [
        # run_mode("fixed_rate"),
        # run_mode("fixed_messages"),
        run_mode("fixed_size"),
    ]
    return outs


if __name__ == "__main__":
    outs = run()
    print("All experiments finished")

    emissions_dir = results_dir / "emissions_per_run" 
    merged_file = merge_experiment_csvs(emissions_dir)
    print("Merged CSV:", merged_file)