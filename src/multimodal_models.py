# src/multimodal_models.py
import tensorflow as tf
from tensorflow.keras import layers as L
from models import build_backbone, TemporalBlock, TransformerEncoder, positional_encoding, _temporal_head_lstm

def audio_branch(input_shape=(800,128,1), emb_dim=256, dropout=0.2):
    a_in = L.Input(shape=input_shape, name="audio_in")
    x = a_in
    x = L.Conv2D(32, (5,5), padding="same", activation="relu")(x)
    x = L.BatchNormalization()(x)
    x = L.MaxPool2D((2,2))(x)

    x = L.Conv2D(64, (3,3), padding="same", activation="relu")(x)
    x = L.BatchNormalization()(x)
    x = L.MaxPool2D((2,2))(x)

    x = L.Conv2D(128, (3,3), padding="same", activation="relu")(x)
    x = L.BatchNormalization()(x)
    x = L.GlobalAveragePooling2D()(x)

    x = L.Dropout(dropout)(x)
    x = L.Dense(emb_dim, activation="relu")(x)
    return tf.keras.Model(a_in, x, name="audio_cnn")

def video_branch_tcn(seq_len=16, img_size=224, embed_dim=256, channels=256, num_blocks=4,
                     kernel_size=3, dilation_base=2, dropout=0.2, weights="imagenet"):
    backbone = build_backbone(img_size, weights)
    v_in = L.Input(shape=(seq_len, img_size, img_size, 3), name="video_in")
    x = L.TimeDistributed(backbone)(v_in)
    x = L.TimeDistributed(L.Dense(embed_dim))(x)
    x = L.LayerNormalization()(x)

    for b in range(num_blocks):
        x = TemporalBlock(channels=channels, kernel_size=kernel_size,
                          dilation_rate=(dilation_base ** b),
                          dropout=dropout, name=f"tblock_{b}")(x)

    x = L.GlobalAveragePooling1D()(x)
    x = L.Dense(256, activation="relu")(x)
    return tf.keras.Model(v_in, x, name="video_tcn_emb")

def video_branch_lstm(seq_len=16, img_size=224, dropout=0.3, weights="imagenet"):
    backbone = build_backbone(img_size, weights)
    v_in = L.Input(shape=(seq_len, img_size, img_size, 3), name="video_in")
    feats = L.TimeDistributed(backbone)(v_in)
    x = _temporal_head_lstm(feats, dropout=dropout)
    return tf.keras.Model(v_in, x, name="video_lstm_emb")

def video_branch_transformer(seq_len=16, img_size=224, embed_dim=256, depth=2, dropout=0.2, weights="imagenet"):
    backbone = build_backbone(img_size, weights)
    v_in = L.Input(shape=(seq_len, img_size, img_size, 3), name="video_in")
    x = L.TimeDistributed(backbone)(v_in)
    x = L.TimeDistributed(L.Dense(embed_dim))(x)
    pe = positional_encoding(seq_len, embed_dim)
    x = x + pe
    x = L.SpatialDropout1D(0.2)(x)
    for _ in range(depth):
        x = TransformerEncoder(embed_dim, num_heads=4, ff_dim=4*embed_dim, rate=dropout)(x)
    x = L.GlobalAveragePooling1D()(x)
    x = L.Dense(256, activation="relu")(x)
    return tf.keras.Model(v_in, x, name="video_transformer_emb")

def build_fusion_model(model_name: str, num_classes: int, seq_len: int=16, img_size: int=224,
                       weights="imagenet", dropout=0.3):
    # video embedding model
    if model_name == "tcn":
        v = video_branch_tcn(seq_len=seq_len, img_size=img_size, weights=weights)
    elif model_name == "lstm":
        v = video_branch_lstm(seq_len=seq_len, img_size=img_size, weights=weights)
    elif model_name == "transformer":
        v = video_branch_transformer(seq_len=seq_len, img_size=img_size, weights=weights)
    else:
        raise ValueError("Fusion supports: tcn | lstm | transformer")

    a = audio_branch(input_shape=(800,128,1), emb_dim=256, dropout=0.2)

    v_in = v.input
    a_in = a.input
    v_emb = v(v_in)
    a_emb = a(a_in)

    x = L.Concatenate()([v_emb, a_emb])
    x = L.LayerNormalization()(x)
    x = L.Dropout(dropout)(x)
    x = L.Dense(256, activation="relu")(x)
    x = L.Dropout(dropout)(x)
    out = L.Dense(num_classes, activation="softmax")(x)

    return tf.keras.Model(inputs=[v_in, a_in], outputs=out, name=f"fusion_{model_name}")
