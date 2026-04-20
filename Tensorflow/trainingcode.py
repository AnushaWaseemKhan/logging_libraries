from __future__ import annotations

import logging
import os
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
from codecarbon import OfflineEmissionsTracker

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

BASE_DIR = Path("training_realworld_tensorflow")
BASE_DIR.mkdir(exist_ok=True)

EPOCHS = 1
BATCH_SIZE = 128
MESSAGE_SIZE = 64
EVERY_N_BATCHES = 10


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


def setup_tf_logger():
    logger = tf.get_logger()

    for h in list(logger.handlers):
        logger.removeHandler(h)
        try:
            h.close()
        except Exception:
            pass

    logger.propagate = False
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)

    return logger


def setup_best_logger():
    """
    Placeholder for the winning logging library from your benchmark.

    Replace this function later with the setup for the most energy-efficient
    Python logging library once you confirm the benchmark winner.

    Example future options:
    - loguru
    - structlog
    - logbook
    - picologging
    """
    raise NotImplementedError(
        "Replace setup_best_logger() with the winning library after benchmarking."
    )


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

        # Works for loggers that expose .info(...)
        self.logger.info(
            "%s | batch=%d | loss=%.6f | accuracy=%.6f",
            self.msg,
            batch + 1,
            loss,
            acc,
        )


def run_with_codecarbon(experiment_name: str, emissions_file: str, train_fn):
    tracker = OfflineEmissionsTracker(
        project_name=experiment_name,
        output_dir=str(BASE_DIR),
        output_file=emissions_file,
        country_iso_code="CAN",
        measure_power_secs=1,
        log_level="error",
    )

    tracker.start()
    history = train_fn()
    tracker.stop()

    final_loss = float(history.history["loss"][-1])
    final_acc = float(history.history["accuracy"][-1])

    print(f"\n{experiment_name} finished")
    print(f"Final loss: {final_loss:.4f}")
    print(f"Final accuracy: {final_acc:.4f}")
    print(f"Emissions file: {BASE_DIR / emissions_file}")
    print("-" * 60)


def train_no_logging(x_train, y_train, x_test, y_test):
    model = build_model()
    return model.fit(
        x_train,
        y_train,
        validation_data=(x_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=0,
    )


def train_tf_batch_logging(x_train, y_train, x_test, y_test):
    model = build_model()
    logger = setup_tf_logger()
    callback = BatchLoggingCallback(
        logger=logger,
        message_size=MESSAGE_SIZE,
        every_n_batches=EVERY_N_BATCHES,
    )

    return model.fit(
        x_train,
        y_train,
        validation_data=(x_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=0,
        callbacks=[callback],
    )


def train_best_library_batch_logging(x_train, y_train, x_test, y_test):
    model = build_model()
    logger = setup_best_logger()
    callback = BatchLoggingCallback(
        logger=logger,
        message_size=MESSAGE_SIZE,
        every_n_batches=EVERY_N_BATCHES,
    )

    return model.fit(
        x_train,
        y_train,
        validation_data=(x_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=0,
        callbacks=[callback],
    )


def main():
    (x_train, y_train), (x_test, y_test) = load_data()

    # Case 1: No logging
    run_with_codecarbon(
        experiment_name="train_no_logging",
        emissions_file="no_logging_emissions.csv",
        train_fn=lambda: train_no_logging(x_train, y_train, x_test, y_test),
    )

    # Case 2: TensorFlow logging at batch level
    run_with_codecarbon(
        experiment_name="train_tf_batch_logging",
        emissions_file="tf_batch_logging_emissions.csv",
        train_fn=lambda: train_tf_batch_logging(x_train, y_train, x_test, y_test),
    )

    # Case 3: Placeholder for best-performing library
    try:
        run_with_codecarbon(
            experiment_name="train_best_library_batch_logging",
            emissions_file="best_library_batch_logging_emissions.csv",
            train_fn=lambda: train_best_library_batch_logging(x_train, y_train, x_test, y_test),
        )
    except NotImplementedError as e:
        print("\nSkipping best-library experiment for now.")
        print(str(e))
        print("-" * 60)


if __name__ == "__main__":
    main()
