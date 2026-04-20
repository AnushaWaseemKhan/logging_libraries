import time
import statistics


def measure_time_per_log(msg, repetitions=1000):
    times = []

    for _ in range(repetitions):
        t1 = time.perf_counter()

        print(msg, flush=True)   # console logging

        t2 = time.perf_counter()
        times.append(t2 - t1)

    avg = statistics.mean(times)
    std = statistics.stdev(times) if len(times) > 1 else 0.0

    return avg, std, min(times), max(times)


def main():
    msg = "A" * 64   
    repetitions = 2000000

    avg, std, min_t, max_t = measure_time_per_log(msg, repetitions)

    print("\n=== Console logging using print() ===")
    print(f"Repetitions: {repetitions}")
    print(f"Average time per log: {avg:.12f} s")
    print(f"Std deviation: {std:.12f} s")
    print(f"Min time: {min_t:.12f} s")
    print(f"Max time: {max_t:.12f} s")


if __name__ == "__main__":
    main()
