import pandas as pd
from .config import (sizes, rates, reps, pre_run_sleep_s, duration_s,log_statements, outputs,logs_dir, results_dir, project_root,codecarbon_kwargs, num_logs)
from .helpers import (
    safe_name, make_message, get_level,
    iter_sizes_rates, log_path_for, make_names, ensure_base_dirs,
)
from .libraries import LIBRARIES
from .codegreen import measure


def run_mode(mode):
    _, emissions_dir = ensure_base_dirs(logs_dir, results_dir)
    sizes_to_iterate, rates_to_iterate = iter_sizes_rates(mode, sizes, rates)
    rows = []

    #All the loops controlled from config file.
    for lib_name, logger_cls in LIBRARIES.items():
        for output in outputs:
            for statement in log_statements:
                for size in sizes_to_iterate:
                    for rate in rates_to_iterate:
                        for rep in range(1, reps + 1):
                            msg = make_message(size)
                            level = get_level(statement)

                            # Decide experiment mode arguments clearly
                            if mode == "fixed_messages":
                                c_num_logs = num_logs
                                c_rate = None
                                c_duration_s = None
                            else:
                                c_num_logs = None
                                c_rate = rate
                                c_duration_s = duration_s
                                
                            #function for file naming
                            _, log_tag, project_name = make_names(
                                mode=mode,
                                lib_name=lib_name,
                                output=output,
                                statement=statement,
                                size=size,
                                rate=c_rate,
                                rep=rep,
                                num_logs=c_num_logs,
                            )

                            #dir for logs
                            log_subdir = (
                                logs_dir
                                / safe_name(lib_name)
                                / safe_name(mode)
                                / safe_name(statement)
                            )
                            log_subdir.mkdir(parents=True, exist_ok=True)
                            log_path = log_path_for(output, log_subdir, log_tag)
                            emissions_path = emissions_dir / "emissions_all_experiments.csv"
                            
                            #Energy measure function for all libraries
                            emissions_kg= measure(
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
                                rate=c_rate,
                                duration_s=c_duration_s,
                                num_logs=c_num_logs,
                            )

                            row = {
                                "mode": mode,
                                "library": lib_name,
                                "output": output,
                                "statement": statement,
                                "msg_size_bytes": int(size),
                                "rep": int(rep),
                                "emissions_kg": emissions_kg,
                                "log_file": str(log_path) if log_path else "",
                                "project_name": project_name,
                                "emissions_file": str(emissions_path),
                                "num_logs": int(c_num_logs) if c_num_logs is not None else None,
                                "rate_msgs_per_s": float(c_rate) if c_rate is not None else None,
                                "duration_s": float(c_duration_s) if c_duration_s is not None else None,
                            }

                            rows.append(row)

    #final summary in file
    summary_csv = results_dir / f"summary_{mode}.csv"
    pd.DataFrame(rows).to_csv(summary_csv, index=False)
    return summary_csv


#Select mode of experiment
def run():
    outs = [
        #run_mode("fixed_rate"),
       #  run_mode("fixed_messages"),
        run_mode("fixed_size"),
    ]
    return outs


if __name__ == "__main__":
    outs = run()
    for p in outs:
        print("wrote:", p)
    print("project root:", project_root)
