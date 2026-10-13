from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def load_sequences(df: pd.DataFrame) -> np.ndarray:
    arrays = [np.load(path).astype(np.float32) for path in df["sequence_path"]]
    return np.stack(arrays)


def split_manifest(
    df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if df["label"].nunique() < 2:
        raise ValueError("Need at least two classes to train.")

    stratify = df["label"] if df["label"].value_counts().min() >= 2 else None
    train_val, test = train_test_split(df, test_size=test_size, random_state=seed, stratify=stratify)
    stratify_tv = train_val["label"] if train_val["label"].value_counts().min() >= 2 else None
    train, val = train_test_split(train_val, test_size=val_size / (1 - test_size), random_state=seed, stratify=stratify_tv)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def encode_labels(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame):
    enc = LabelEncoder()
    y_train = enc.fit_transform(train["label"])
    y_val = enc.transform(val["label"])
    y_test = enc.transform(test["label"])
    labels = {int(i): label for i, label in enumerate(enc.classes_)}
    return y_train, y_val, y_test, labels
