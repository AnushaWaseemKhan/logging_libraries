import time
import sys
from eliot import Message, add_destinations, remove_destination
from eliot.logwriter import FileDestination


def run_one(msg, level, rate, duration_s, output, log_path):
    f = None

    if output == "file":
        f = open(str(log_path), "a", encoding="utf-8")
        handler = FileDestination(file=f)
    else:
        handler = FileDestination(file=sys.stdout)

    add_destinations(handler)

    interval = 1.0 / float(rate)
    end_time = time.perf_counter() + float(duration_s)
    next_time = time.perf_counter()

    try:
        while time.perf_counter() < end_time:
            Message.log(message=msg)

            next_time += interval
            sleep_time = next_time - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)
    finally:
        remove_destination(handler)
        if f is not None:
            f.close()


