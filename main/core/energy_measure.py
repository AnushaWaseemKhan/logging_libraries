import time
from codecarbon import OfflineEmissionsTracker

AVG_LOG_TIME_PER_LIB = {
    "stdlib": 0.000019,
    "loguru": 0.000036,
    "structlog": 0.000031,
    "logbook": 0.000028,
    "pythonjsonlogger": 0.000031,
    "picologging": 0.000009,
    "aiologger":0.000041
}

def measure(
    *,
    project_name,
    emissions_path,
    logger_cls,
    lib_name,
    msg,
    level,
    output,
    log_path,
    pre_run_sleep_s,
    codecarbon_kwargs,
    rate,
    duration_s,
):
    # pre-run stabilization
    time.sleep(pre_run_sleep_s)

    # logger
    adapter = logger_cls(
        name=lib_name,
        level=level,
        output=output,
        log_path=log_path,
    )

    # checks
    if rate is None or rate <= 0:
        raise ValueError("rate must be > 0")
    if duration_s is None or duration_s <= 0:
        raise ValueError("duration_s must be > 0")

    lib_key = lib_name.lower()
    if lib_key not in AVG_LOG_TIME_PER_LIB:
        raise ValueError(
            f"No calibrated avg log time for '{lib_name}'. "
            f"Available: {list(AVG_LOG_TIME_PER_LIB.keys())}"
        )

    avg_log_time = AVG_LOG_TIME_PER_LIB[lib_key]
    target_logs = int(rate * duration_s)

    if target_logs <= 0:
        raise ValueError("target_logs must be > 0")

    sleep_time = (duration_s - (target_logs * avg_log_time)) / target_logs
    if sleep_time < 0:
        raise ValueError(
            f"Logging time exceeds duration for '{lib_name}'. "
            f"Increase duration_s or reduce rate."
        )

    # tracker
    tracker = OfflineEmissionsTracker(
        project_name=project_name,
        output_dir=str(emissions_path.parent),
        output_file=f"{project_name}.csv",
        save_to_file=True,
        **codecarbon_kwargs,
    )

    tracker.start()
    try:
        for _ in range(target_logs):
            adapter.log(level, msg)
            time.sleep(sleep_time)
    finally:
        emissions_kg = tracker.stop()
        if hasattr(adapter, "close"):
            adapter.close()

    return emissions_kg
