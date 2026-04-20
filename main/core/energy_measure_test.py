import time
from codecarbon import OfflineEmissionsTracker

def measure( *,project_name,emissions_path,logger_cls,lib_name,msg,level,output,log_path,pre_run_sleep_s,codecarbon_kwargs,rate=None,duration_s=None,num_logs=None,):
    #sleep time
    time.sleep(pre_run_sleep_s)
    
    #logger
    adapter = logger_cls(
        name=lib_name,
        level=level,
        output=output,
        log_path=log_path,
    )

    # Some checks
    using_fixed_logs = num_logs is not None
    using_rate_mode = rate is not None and duration_s is not None
    if using_fixed_logs and using_rate_mode:
        raise ValueError("Use either num_logs OR rate+duration_s, not both.")
    if not using_fixed_logs and not using_rate_mode:
        raise ValueError("Provide either num_logs, or rate and duration_s.")
    if using_fixed_logs:
        if num_logs < 0:
            raise ValueError("num_logs must be >= 0")
    else:
        if rate is None or rate <= 0:
            raise ValueError("rate must be > 0 in rate-based mode")
        if duration_s is None or duration_s <= 0:
            raise ValueError("duration_s must be > 0 in rate-based mode")

    # tracker
    tracker = OfflineEmissionsTracker(
        project_name=project_name,
        output_dir=str(emissions_path.parent),
        output_file=f"{project_name}.csv",
        save_to_file=True,
        **codecarbon_kwargs,
    )

    avg_log_time = 0.000003294396
    target_logs = int(rate * duration_s)

    sleep_time = (duration_s - (target_logs * avg_log_time)) / target_logs
    if sleep_time < 0:
        raise ValueError("Logging time exceeds total duration.")

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
