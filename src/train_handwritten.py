"""Train a fresh CNN on the real handwritten Persian digit dataset (dataset_farsi).

Usage:
    python src/train_handwritten.py [--epochs 25] [--out models/digit_classifier_handwritten.keras]
"""
import argparse
import sys
from pathlib import Path

import tensorflow as tf

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from src.model import build_model  # noqa: E402

DATA_DIR = BASE / "data" / "dataset_farsi"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--out", default=str(BASE / "models" / "digit_classifier_handwritten.keras"))
    parser.add_argument("--batch", type=int, default=64)
    args = parser.parse_args()

    if not DATA_DIR.exists():
        raise SystemExit(f"Dataset not found: {DATA_DIR}")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        str(DATA_DIR),
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=(28, 28),
        batch_size=args.batch,
        color_mode="grayscale",
        label_mode="int",
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        str(DATA_DIR),
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=(28, 28),
        batch_size=args.batch,
        color_mode="grayscale",
        label_mode="int",
    )
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.map(lambda x, y: (x / 255.0, y)).prefetch(AUTOTUNE)
    val_ds = val_ds.map(lambda x, y: (x / 255.0, y)).prefetch(AUTOTUNE)

    model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=["accuracy"],
    )
    print(f"Training on {DATA_DIR.name} ({args.epochs} epochs)...")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=5, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, verbose=1
        ),
    ]
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        verbose=1,
        callbacks=callbacks,
    )
    model.save(args.out)
    print(f"\nSaved -> {args.out}")

    val_loss, val_acc = model.evaluate(val_ds, verbose=0)
    print(f"Validation accuracy: {val_acc:.1%}")


if __name__ == "__main__":
    main()
