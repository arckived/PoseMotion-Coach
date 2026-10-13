from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, Model


def build_lstm(seq_len: int, feature_dim: int, num_classes: int) -> Model:
    inputs = layers.Input(shape=(seq_len, feature_dim))
    x = layers.Masking(mask_value=0.0)(inputs)
    x = layers.Bidirectional(layers.LSTM(96, return_sequences=True))(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Bidirectional(layers.LSTM(64))(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = Model(inputs, outputs, name="pose_lstm")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


@tf.keras.utils.register_keras_serializable()
class PositionalEmbedding(layers.Layer):
    def __init__(self, seq_len: int, embed_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.pos_emb = layers.Embedding(input_dim=seq_len, output_dim=embed_dim)

    def call(self, x):
        positions = tf.range(start=0, limit=self.seq_len, delta=1)
        return x + self.pos_emb(positions)

    def get_config(self):
        config = super().get_config()
        config.update({
            "seq_len": self.seq_len,
            "embed_dim": self.embed_dim,
        })
        return config


def transformer_block(x, embed_dim: int, num_heads: int, ff_dim: int, dropout: float):
    attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads)(x, x)
    attn = layers.Dropout(dropout)(attn)
    x = layers.LayerNormalization(epsilon=1e-6)(x + attn)

    ffn = layers.Dense(ff_dim, activation="relu")(x)
    ffn = layers.Dense(embed_dim)(ffn)
    ffn = layers.Dropout(dropout)(ffn)
    return layers.LayerNormalization(epsilon=1e-6)(x + ffn)


def build_transformer(seq_len: int, feature_dim: int, num_classes: int) -> Model:
    inputs = layers.Input(shape=(seq_len, feature_dim))
    x = layers.Dense(128)(inputs)
    x = PositionalEmbedding(seq_len, 128)(x)
    x = transformer_block(x, embed_dim=128, num_heads=4, ff_dim=256, dropout=0.2)
    x = transformer_block(x, embed_dim=128, num_heads=4, ff_dim=256, dropout=0.2)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = Model(inputs, outputs, name="pose_transformer")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_model(model_name: str, seq_len: int, feature_dim: int, num_classes: int) -> Model:
    model_name = model_name.lower()
    if model_name == "lstm":
        return build_lstm(seq_len, feature_dim, num_classes)
    if model_name == "transformer":
        return build_transformer(seq_len, feature_dim, num_classes)
    raise ValueError(f"Unknown model: {model_name}. Use 'lstm' or 'transformer'.")
