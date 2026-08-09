"""Fine-tune the model on multi-font crop dataset.

Usage:
    python src/finetune_captcha.py [--data data/captcha_crops_multifont] [--epochs 20] [--out models/digit_classifier_multifont.keras]
"""
import argparse
import sys
from pathlib import Path

import tensorflow as tf

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(BASE / "data" / "captcha_crops_multifont"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--out", default=str(BASE / "models" / "digit_classifier_multifont.keras"))
    parser.add_argument("--base", default=str(BASE / "models" / "digit_classifier_captcha_v2.keras"))
    args = parser.parse_args()

    data_dir = Path(args.data)
    if not data_dir.exists():
        raise SystemExit(f"Dataset dir not found: {data_dir}")

    # Load data
    train_ds = tf.keras.utils.image_dataset_from_directory(
        str(data_dir),
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=(28, 28),
        batch_size=32,
        color_mode="grayscale",
        label_mode="int",
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        str(data_dir),
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=(28, 28),
        batch_size=32,
        color_mode="grayscale",
        label_mode="int",
    )
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.map(lambda x, y: (x / 255.0, y)).prefetch(AUTOTUNE)
    val_ds = val_ds.map(lambda x, y: (x / 255.0, y)).prefetch(AUTOTUNE)

    # Load base model (the BYekan-tuned model) and fine-tune further
    model = tf.keras.models.load_model(args.base)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=["accuracy"],
    )

    print(f"Fine-tuning {Path(args.base).name} on {data_dir.name} ({args.epochs} epochs)...")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        verbose=1,
    )

    model.save(args.out)
    print(f"\nSaved -> {args.out}")

    val_loss, val_acc = model.evaluate(val_ds, verbose=0)
    print(f"Validation accuracy: {val_acc:.1%}")


if __name__ == "__main__":
    main()
