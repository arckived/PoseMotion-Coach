import argparse
import json
from pathlib import Path

import pandas as pd
import tensorflow as tf

from .dataset import encode_labels, load_sequences, split_manifest
from .models import build_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-manifest", required=True)
    parser.add_argument("--model", choices=["lstm", "transformer"], default="lstm")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.processed_manifest)
    train_df, val_df, test_df = split_manifest(df, seed=args.seed)
    train_df.to_csv(out_dir / "train_manifest.csv", index=False)
    val_df.to_csv(out_dir / "val_manifest.csv", index=False)
    test_df.to_csv(out_dir / "test_manifest.csv", index=False)

    y_train, y_val, y_test, labels = encode_labels(train_df, val_df, test_df)
    with open(out_dir / "labels.json", "w") as f:
        json.dump(labels, f, indent=2)

    x_train = load_sequences(train_df)
    x_val = load_sequences(val_df)
    x_test = load_sequences(test_df)

    seq_len = x_train.shape[1]
    feature_dim = x_train.shape[2]
    num_classes = len(labels)

    model = build_model(args.model, seq_len, feature_dim, num_classes)
    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            out_dir / "best_model.keras",
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=8,
            restore_best_weights=True,
            mode="max",
        ),
        tf.keras.callbacks.CSVLogger(out_dir / "training_log.csv"),
    ]

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
    )

    model.save(out_dir / "final_model.keras")
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump({"test_loss": float(test_loss), "test_accuracy": float(test_acc)}, f, indent=2)
    print(f"Test accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()
