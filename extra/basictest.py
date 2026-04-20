import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path


def setup_logger(log_path: str) -> logging.Logger:
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("perf_logger")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for h in list(logger.handlers):
        logger.removeHandler(h)
        h.close()

    handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


def run_workload(rate: float, duration_s: float) -> None:
    print(f"\n=== Running workload at rate = {rate} ===", flush=True)
    time.sleep(8.0)  # stabilization

    logger = setup_logger(f"logs/logs_rate_{rate}.log")

    interval = 1.0 / rate
    target_logs = int(rate * duration_s)

    print(f"Target logs   : {target_logs}", flush=True)
    print(f"Avg sleep     : {interval:.6f} s", flush=True)

    start_time = time.perf_counter()
    next_time = start_time

    try:
        for i in range(target_logs):
            logger.info(f"Iter {i + 1} | Rate={rate}")

            next_time += interval
            remaining = next_time - time.perf_counter()
            if remaining > 0:
                time.sleep(remaining)

        end_target = start_time + duration_s
        remaining_total = end_target - time.perf_counter()
        if remaining_total > 0:
            time.sleep(remaining_total)

    finally:
        end_time = time.perf_counter()
        for h in list(logger.handlers):
            logger.removeHandler(h)
            h.close()

    runtime = end_time - start_time
    print(f"Finished → Runtime: {runtime:.2f}s", flush=True)


def parse_perf_energy(stderr_text: str) -> dict:
    energy = {
        "pkg": None,
        "cores": None,
        "ram": None,
    }

    for line in stderr_text.splitlines():
        parts = line.split(";")
        if len(parts) < 3:
            continue

        value = parts[0].strip()
        event = parts[2].strip()

        try:
            val = float(value.replace(",", ""))
        except ValueError:
            continue

        if event == "power/energy-pkg/":
            energy["pkg"] = val
        elif event == "power/energy-cores/":
            energy["cores"] = val
        elif event == "power/energy-ram/":
            energy["ram"] = val

    return energy


def run_with_perf(rate: float, duration_s: float):
    cmd = [
        "perf",
        "stat",
        "-a",
        "-x",
        ";",
        "-e",
        "power/energy-pkg/,power/energy-cores/,power/energy-ram/",
        sys.executable,
        __file__,
        "--workload",
        "--rate",
        str(rate),
        "--duration",
        str(duration_s),
    ]

    print(f"\n=== Running perf stat for rate = {rate} ===")
    result = subprocess.run(
        cmd,
        text=True,
        capture_output=True,
        check=False,
    )

    if result.stdout:
        print(result.stdout)

    if result.returncode != 0:
        print("perf stderr:")
        print(result.stderr)
        raise RuntimeError(f"perf run failed for rate={rate}")

    energy = parse_perf_energy(result.stderr)

    if energy["pkg"] is None and energy["cores"] is None and energy["ram"] is None:
        print("Could not parse energy from perf output.")
        print(result.stderr)
        raise RuntimeError(f"Could not parse energy for rate={rate}")

    total_energy = sum(v for v in energy.values() if v is not None)

    return energy, total_energy, duration_s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", action="store_true")
    parser.add_argument("--rate", type=float, default=None)
    parser.add_argument("--duration", type=float, default=200.0)
    args = parser.parse_args()

    if args.workload:
        if args.rate is None or args.rate <= 0:
            raise ValueError("--rate must be > 0 in workload mode")
        run_workload(rate=args.rate, duration_s=args.duration)
        return

    duration_s = args.duration
    rates = [0.01, 0.1, 1.0, 10.0, 100.0]

    print("=== perf stat Energy Measurement ===\n")
    print("Using events: power/energy-pkg/, power/energy-cores/, power/energy-ram/\n")

    results = []
    for rate in rates:
        energy, total_energy, runtime = run_with_perf(rate=rate, duration_s=duration_s)
        results.append((rate, energy, total_energy, runtime))

    print("\n" + "=" * 120)
    print("FINAL SUMMARY - perf stat energy")
    print("=" * 120)
    print(
        f"{'Rate':<8} {'Logs':<8} {'Pkg(J)':<12} {'Cores(J)':<12} {'RAM(J)':<12} "
        f"{'Total(J)':<12} {'Runtime(s)':<12} {'Energy/log(J)':<15}"
    )
    print("-" * 120)

    for rate, energy, total_energy, runtime in results:
        logs = int(rate * duration_s)
        per_log = total_energy / logs if logs > 0 else 0.0

        pkg = energy['pkg'] if energy['pkg'] is not None else 0.0
        cores = energy['cores'] if energy['cores'] is not None else 0.0
        ram = energy['ram'] if energy['ram'] is not None else 0.0

        print(
            f"{rate:<8} {logs:<8} {pkg:<12.3f} {cores:<12.3f} {ram:<12.3f} "
            f"{total_energy:<12.3f} {runtime:<12.2f} {per_log:<15.2e}"
        )


if __name__ == "__main__":
    main()