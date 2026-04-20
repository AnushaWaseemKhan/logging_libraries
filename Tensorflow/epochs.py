from __future__ import annotations

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import logging
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from codecarbon import OfflineEmissionsTracker


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

BASE_DIR = Path("training_realworld_tensorflow")
BASE_DIR.mkdir(parents=True, exist_ok=True)

EPOCH_LIST = [10, 20, 30, 40]
BATCH_SIZE = 128
MESSAGE_SIZE = 64
EVERY_N_BATCHES = 10
SLEEP_BETWEEN_RUNS = 60  # seconds


def load_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    return (x_train, y_train), (x_test, y_test)


def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(32, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def make_message(prefix: str, size: int) -> str:
    prefix = f"{prefix} | "
    return prefix + ("x" * max(0, size - len(prefix)))


def setup_console_logger(name: str = "tf_batch_logger"):
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    )
    logger.addHandler(handler)
    return logger


class BatchLoggingCallback(tf.keras.callbacks.Callback):
    def __init__(self, logger, message_size: int, every_n_batches: int = 10):
        super().__init__()
        self.logger = logger
        self.msg = make_message("batch_done", message_size)
        self.every_n_batches = every_n_batches

    def on_train_batch_end(self, batch, logs=None):
        if (batch + 1) % self.every_n_batches != 0:
            return

        logs = logs or {}
        loss = float(logs.get("loss", -1.0))
        acc = float(logs.get("accuracy", -1.0))

        self.logger.info(
            "%s | batch=%d | loss=%.6f | accuracy=%.6f",
            self.msg,
            batch + 1,
            loss,
            acc,
        )


def empty_metrics():
    return {
        "duration_s": None,
        "energy_consumed_kwh": None,
        "energy_j": None,
        "cpu_energy_kwh": None,
        "ram_energy_kwh": None,
        "gpu_energy_kwh": None,
        "cpu_power_w": None,
        "ram_power_w": None,
        "gpu_power_w": None,
    }


def read_emission_metrics(emissions_path: Path) -> dict:
    if not emissions_path.exists():
        return empty_metrics()

    try:
        df = pd.read_csv(emissions_path)
    except Exception:
        return empty_metrics()

    if df.empty:
        return empty_metrics()

    row = df.iloc[-1]
    energy_kwh = row.get("energy_consumed", None)

    return {
        "duration_s": row.get("duration", None),
        "energy_consumed_kwh": energy_kwh,
        "energy_j": energy_kwh * 3_600_000 if pd.notna(energy_kwh) else None,
        "cpu_energy_kwh": row.get("cpu_energy", None),
        "ram_energy_kwh": row.get("ram_energy", None),
        "gpu_energy_kwh": row.get("gpu_energy", None),
        "cpu_power_w": row.get("cpu_power", None),
        "ram_power_w": row.get("ram_power", None),
        "gpu_power_w": row.get("gpu_power", None),
    }


def run_with_codecarbon(
    experiment_name: str,
    emissions_file: str,
    train_fn,
    epochs: int,
    condition: str,
):
    tracker = OfflineEmissionsTracker(
        project_name=experiment_name,
        output_dir=str(BASE_DIR),
        output_file=emissions_file,
        country_iso_code="CAN",
        measure_power_secs=1,
        log_level="error",
    )

    history = None
    error_msg = None

    try:
        tracker.start()
        history = train_fn()
    except Exception as e:
        error_msg = str(e)
    finally:
        try:
            tracker.stop()
        except Exception as e:
            stop_error = str(e)
            if error_msg is None:
                error_msg = f"tracker.stop() failed: {stop_error}"
            else:
                error_msg = f"{error_msg} | tracker.stop() failed: {stop_error}"
            print(f"[WARNING] tracker.stop() failed for {experiment_name}: {e}")

    emissions_path = BASE_DIR / emissions_file
    metrics = read_emission_metrics(emissions_path)

    result = {
        "experiment_name": experiment_name,
        "condition": condition,
        "epochs": epochs,
        "emissions_file": str(emissions_path),
        "status": "success" if history is not None else "failed",
        "error": error_msg,
        "final_loss": None,
        "final_accuracy": None,
        **metrics,
    }

    if history is not None:
        result["final_loss"] = float(history.history["loss"][-1])
        result["final_accuracy"] = float(history.history["accuracy"][-1])

        print(f"\n{experiment_name} finished")
        print(f"Final loss: {result['final_loss']:.4f}")
        print(f"Final accuracy: {result['final_accuracy']:.4f}")
        print(
            f"Duration: {result['duration_s']:.4f} s"
            if result["duration_s"] is not None else "Duration: N/A"
        )
        print(
            f"Energy: {result['energy_j']:.4f} J"
            if result["energy_j"] is not None else "Energy: N/A"
        )
        print(f"Emissions file: {emissions_path}")
        print("-" * 60)
    else:
        print(f"\n{experiment_name} failed")
        print(f"Error: {error_msg}")
        print(f"Emissions file: {emissions_path}")
        print("-" * 60)

    return result


def train_no_logging(x_train, y_train, x_test, y_test, epochs):
    model = build_model()
    return model.fit(
        x_train,
        y_train,
        validation_data=(x_test, y_test),
        epochs=epochs,
        batch_size=BATCH_SIZE,
        verbose=0,
    )


def train_tf_batch_logging(x_train, y_train, x_test, y_test, epochs):
    model = build_model()
    logger = setup_console_logger()
    callback = BatchLoggingCallback(
        logger=logger,
        message_size=MESSAGE_SIZE,
        every_n_batches=EVERY_N_BATCHES,
    )

    return model.fit(
        x_train,
        y_train,
        validation_data=(x_test, y_test),
        epochs=epochs,
        batch_size=BATCH_SIZE,
        verbose=0,
        callbacks=[callback],
    )


def cooldown():
    print(f"\nCooling down for {SLEEP_BETWEEN_RUNS} seconds...")
    time.sleep(SLEEP_BETWEEN_RUNS)


def main():
    (x_train, y_train), (x_test, y_test) = load_data()
    all_results = []

    for epochs in EPOCH_LIST:
        print(f"\n{'=' * 25} RUNNING EPOCHS = {epochs} {'=' * 25}")

        result = run_with_codecarbon(
            experiment_name=f"train_no_logging_epochs{epochs}",
            emissions_file=f"no_logging_epochs{epochs}_emissions.csv",
            train_fn=lambda e=epochs: train_no_logging(x_train, y_train, x_test, y_test, e),
            epochs=epochs,
            condition="no_logging",
        )
        all_results.append(result)
        cooldown()

        result = run_with_codecarbon(
            experiment_name=f"train_tf_batch_logging_epochs{epochs}",
            emissions_file=f"tf_batch_logging_epochs{epochs}_emissions.csv",
            train_fn=lambda e=epochs: train_tf_batch_logging(x_train, y_train, x_test, y_test, e),
            epochs=epochs,
            condition="tf_batch_logging",
        )
        all_results.append(result)
        cooldown()

    summary_df = pd.DataFrame(all_results)
    summary_path = BASE_DIR / "training_summary_all_runs.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\nSummary saved to:")
    print(summary_path)
    print("\nSummary preview:")
    print(summary_df[
        [
            "condition",
            "epochs",
            "status",
            "duration_s",
            "energy_j",
            "energy_consumed_kwh",
            "final_loss",
            "final_accuracy",
            "error",
        ]
    ])


if __name__ == "__main__":
    main()
