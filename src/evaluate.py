import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder

from .dataset import load_sequences
from .models import PositionalEmbedding


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--test-manifest", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.test_manifest)
    x = load_sequences(df)
    enc = LabelEncoder()
    y_true = enc.fit_transform(df["label"])

    model = tf.keras.models.load_model(args.model_path, compile=False, custom_objects={"PositionalEmbedding": PositionalEmbedding})
    probs = model.predict(x)
    y_pred = np.argmax(probs, axis=1)

    report = classification_report(y_true, y_pred, target_names=enc.classes_, output_dict=True, zero_division=0)
    with open(out_dir / "classification_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(classification_report(y_true, y_pred, target_names=enc.classes_, zero_division=0))

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=enc.classes_)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax, xticks_rotation=45)
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png", dpi=200)
    print(f"Saved confusion matrix to {out_dir / 'confusion_matrix.png'}")


if __name__ == "__main__":
    main()
