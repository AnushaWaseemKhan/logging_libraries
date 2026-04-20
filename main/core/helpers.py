import logging
import re
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
from pathlib import Path


@dataclass(frozen=True)
class Level:
    name: str
    num: int


levels = {
    "debug": Level("debug", logging.DEBUG),
    "info": Level("info", logging.INFO),
    "warning": Level("warning", logging.WARNING),
    "error": Level("error", logging.ERROR),
    "critical": Level("critical", logging.CRITICAL),
}


def safe_name(s):
    s = str(s).strip().lower()
    s = re.sub(r"[^a-z0-9._-]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_")


def make_message(size):
    return "X" * int(size)


def get_level(statement):
    s = str(statement).lower()
    return levels.get(s, levels["info"])


def prepare_log(log_path):
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    return log_path


def iter_sizes_rates(mode, sizes, rates):
    if mode == "fixed_rate":
        return sizes,[rates[0]] 
    elif mode == "fixed_size":
        return [sizes[0]], rates
    elif mode == "fixed_messages":
        return [sizes[0]], [None]
    else:
        raise ValueError(f"Unknown mode: {mode}")


def log_path_for(output, lib_log_dir, tag):
    if output == "file":
        return lib_log_dir / f"{safe_name(tag)}.log"
    return None


def make_names(mode, lib_name, output, statement, size, rate, rep, num_logs=None):
    if mode == "fixed_messages":
        tag = f"{mode}_{output}_{statement}_size{size}_nmsg{num_logs}_rep{rep}"
    else:
        tag = f"{mode}_{output}_{statement}_size{size}_rate{rate}_rep{rep}"

    log_tag = tag
    project_name = f"{safe_name(lib_name)}_{safe_name(tag)}"

    return tag, log_tag, project_name


def ensure_base_dirs(logs_dir, results_dir):
    logs_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    per_run_dir = results_dir / "per_run"
    emissions_dir = results_dir / "emission_per_run"
    per_run_dir.mkdir(parents=True, exist_ok=True)
    emissions_dir.mkdir(parents=True, exist_ok=True)
    return per_run_dir, emissions_dir



def merge_experiment_csvs(folder: Path, merged_filename="emissions_merged.csv"):
    """
    Merge all per-experiment CSVs in a folder into one CSV.
    Deletes individual CSVs after merging.
    """
    all_csvs = list(folder.glob("*.csv"))
    if not all_csvs:
        print("No CSVs found to merge in", folder)
        return

    dfs = [pd.read_csv(f) for f in all_csvs]
    merged_df = pd.concat(dfs, ignore_index=True)

    merged_path = folder / merged_filename
    merged_df.to_csv(merged_path, index=False)

    # Optional: delete per-experiment CSVs
    for f in all_csvs:
        if f != merged_path:
            f.unlink()

    print(f"Merged {len(all_csvs)} CSVs into {merged_path}")
    return merged_path

