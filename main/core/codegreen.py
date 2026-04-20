import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

# =========================================================
# PROJECT ROOT FIX
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# =========================================================
# PER-LIBRARY AVG TIME PER LOG (console, 64-byte message)
# =========================================================
AVG_LOG_TIME_PER_LIB = {
    "stdlib": 0.000019,
    "loguru": 0.000036,
    "structlog": 0.000031,
    "logbook": 0.000028,
    "pythonjsonlogger": 0.000031,
    "picologging": 0.000009,
     "aiologger": 0.000041, 
}

# =========================================================
# IMPORT LOGGING ADAPTERS
# =========================================================
from main.core.libraries.stdlib import Stdlibrary
from main.core.libraries.loguru import LoguruAdapter
from main.core.libraries.structlog import StructlogAdapter
from main.core.libraries.logbook import LogbookAdapter
from main.core.libraries.pythonjsonlogger import PythonJsonLoggerAdapter
from main.core.libraries.picologging import PicoLoggingAdapter

LIBRARIES = {
    "stdlib": Stdlibrary,
    "loguru": LoguruAdapter,
    "structlog": StructlogAdapter,
    "logbook": LogbookAdapter,
    "pythonjsonlogger": PythonJsonLoggerAdapter,
    "picologging": PicoLoggingAdapter,
}

# Optional aiologger support
try:
    from main.core.libraries.aiologger import Aiologger
    LIBRARIES["aiologger"] = Aiologger
except Exception:
    pass


# =========================================================
# HELPERS
# =========================================================
def _get_codegreen_bin() -> str:
    venv_bin = PROJECT_ROOT / "venv" / "bin" / "codegreen"
    if venv_bin.exists():
        return str(venv_bin)
    return "codegreen"


def _extract_energy_j_from_json(output_file: Path):
    """
    Extract mean energy in joules from `codegreen run --json` output.
    """
    if not output_file.exists():
        return None

    try:
        with open(output_file, "r") as f:
            data = json.load(f)
    except Exception:
        return None

    if not isinstance(data, dict):
        return None

    # Primary format from `codegreen run --json`
    energy_joules = data.get("energy_joules")
    if isinstance(energy_joules, dict):
        mean_val = energy_joules.get("mean")
        if isinstance(mean_val, (int, float)):
            return float(mean_val)

    # fallback formats just in case
    for key in ["energy_j", "energy", "joules", "total_energy_j"]:
        val = data.get(key)
        if isinstance(val, (int, float)):
            return float(val)

    return None


def _serialize_level(level):
    """
    Convert incoming level object into subprocess-safe pieces.
    Supports objects with .num, .value, strings, or ints.
    """
    level_name = None
    level_value = None

    if hasattr(level, "name"):
        level_name = str(level.name)
    else:
        level_name = str(level)

    if hasattr(level, "num"):
        try:
            level_value = int(level.num)
        except Exception:
            level_value = None
    elif hasattr(level, "value"):
        try:
            level_value = int(level.value)
        except Exception:
            level_value = None
    else:
        try:
            level_value = int(level)
        except Exception:
            level_value = None

    return level_name, level_value


def _deserialize_level(level_name: str, level_value):
    name = str(level_name)

    try:
        value = int(level_value) if level_value is not None else None
    except Exception:
        value = None

    return SimpleNamespace(name=name, value=value, num=value)


# =========================================================
# INTERNAL WORKER
# =========================================================
def _worker_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cg-worker", action="store_true")
    parser.add_argument("--lib-name", required=True)
    parser.add_argument("--message", required=True)
    parser.add_argument("--level-name", required=True)
    parser.add_argument("--level-value", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--log-path", required=True)

    parser.add_argument("--mode", choices=["fixed", "rate"], required=True)
    parser.add_argument("--num-logs", type=int, default=None)
    parser.add_argument("--target-logs", type=int, default=None)
    parser.add_argument("--sleep-time", type=float, default=None)

    args = parser.parse_args()

    if args.lib_name not in LIBRARIES:
        raise ValueError(
            f"Unknown library '{args.lib_name}'. Available: {list(LIBRARIES.keys())}"
        )

    logger_cls = LIBRARIES[args.lib_name]
    level_obj = _deserialize_level(args.level_name, args.level_value)

    adapter = logger_cls(
        name=args.lib_name,
        level=level_obj,
        output=args.output,
        log_path=args.log_path,
    )

    try:
        if args.mode == "fixed":
            if args.num_logs is None or args.num_logs < 0:
                raise ValueError("num_logs must be provided and >= 0 in fixed mode")

            for _ in range(args.num_logs):
                adapter.log(level_obj, args.message)

        elif args.mode == "rate":
            if args.target_logs is None or args.target_logs <= 0:
                raise ValueError("target_logs must be > 0 in rate mode")
            if args.sleep_time is None or args.sleep_time < 0:
                raise ValueError("sleep_time must be >= 0 in rate mode")

            for _ in range(args.target_logs):
                adapter.log(level_obj, args.message)
                time.sleep(args.sleep_time)

    finally:
        if hasattr(adapter, "close"):
            adapter.close()


# =========================================================
# PUBLIC MEASURE FUNCTION
# =========================================================
def measure(
    *,
    project_name,
    emissions_path,
    logger_cls,   # compatibility only
    lib_name,
    msg,
    level,
    output,
    log_path,
    pre_run_sleep_s,
    codecarbon_kwargs=None,   # compatibility only
    rate=None,
    duration_s=None,
    num_logs=None,
):
    time.sleep(pre_run_sleep_s)

    using_fixed_logs = num_logs is not None
    using_rate_mode = rate is not None and duration_s is not None

    if using_fixed_logs and using_rate_mode:
        raise ValueError("Use either num_logs OR rate+duration_s, not both.")

    if not using_fixed_logs and not using_rate_mode:
        raise ValueError("Provide either num_logs, or rate and duration_s.")

    if lib_name not in LIBRARIES:
        raise ValueError(
            f"Unknown library '{lib_name}'. Available: {list(LIBRARIES.keys())}"
        )

    emissions_path = Path(emissions_path)
    emissions_path.parent.mkdir(parents=True, exist_ok=True)

    output_file = emissions_path.parent / f"{project_name}.json"

    CODEGREEN_BIN = _get_codegreen_bin()
    level_name, level_value = _serialize_level(level)

    cmd = [
    CODEGREEN_BIN,
    "run",
    "--json",
    "--repeat", "5",          # ✅ FIXED POSITION
    "--",
    "python",
    str(Path(__file__).resolve()),
    "--cg-worker",
    "--lib-name", lib_name,
    "--message", msg,
    "--level-name", str(level_name),
    "--output", output,
    "--log-path", str(log_path),
]

    if level_value is not None:
        cmd.extend(["--level-value", str(level_value)])

    if using_fixed_logs:
        if num_logs < 0:
            raise ValueError("num_logs must be >= 0")

        cmd.extend([
            "--mode", "fixed",
            "--num-logs", str(num_logs),
        ])

    else:
        if rate is None or rate <= 0:
            raise ValueError("rate must be > 0 in rate-based mode")
        if duration_s is None or duration_s <= 0:
            raise ValueError("duration_s must be > 0 in rate-based mode")

        if lib_name not in AVG_LOG_TIME_PER_LIB:
            raise ValueError(
                f"Library '{lib_name}' not found in AVG_LOG_TIME_PER_LIB. "
                f"Available: {list(AVG_LOG_TIME_PER_LIB.keys())}"
            )

        avg_log_time = AVG_LOG_TIME_PER_LIB[lib_name]
        interval = 1.0 / rate
        sleep_time = interval - avg_log_time

        if sleep_time < 0:
            raise ValueError(
                f"{lib_name} cannot sustain {rate} logs/sec "
                f"because avg_log_time={avg_log_time:.8f}s > interval={interval:.8f}s"
            )

        target_logs = int(rate * duration_s)
        if target_logs <= 0:
            raise ValueError("target_logs became 0. Check rate and duration_s.")

        cmd.extend([
            "--mode", "rate",
            "--target-logs", str(target_logs),
            "--sleep-time", str(sleep_time),
        ])

    env = dict(os.environ)
    existing_pythonpath = env.get("PYTHONPATH", "")
    if existing_pythonpath:
        env["PYTHONPATH"] = f"{PROJECT_ROOT}:{existing_pythonpath}"
    else:
        env["PYTHONPATH"] = str(PROJECT_ROOT)

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
        cwd=str(PROJECT_ROOT),
        env=env,
    )
    if result.returncode != 0:
        raise RuntimeError(
        "CodeGreen run failed.\n"
        f"COMMAND:\n{' '.join(cmd)}\n\n"
        f"STDOUT:\n{result.stdout}\n\n"
        f"STDERR:\n{result.stderr}"
    )


    output_file.write_text(result.stdout, encoding="utf-8")

    
    energy_j = _extract_energy_j_from_json(output_file)

    if energy_j is None:
        return str(output_file)

    return energy_j


# =========================================================
# ENTRY POINT
# =========================================================
if __name__ == "__main__":
    if "--cg-worker" in sys.argv:
        _worker_main()
