# src/models.py
import tensorflow as tf
from tensorflow.keras import layers as L

def build_backbone(img_size: int = 224, weights: str = "imagenet") -> tf.keras.Model:
    # use weights="imagenet" for transfer, or weights=None for scratch backbone
    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights=weights,
        pooling="avg"
    )
    return backbone

def _temporal_head_lstm(x, dropout=0.3):
    x = L.TimeDistributed(L.BatchNormalization())(x)
    x = L.SpatialDropout1D(0.2)(x)
    x = L.Bidirectional(L.LSTM(256, return_sequences=True))(x)
    x = L.Bidirectional(L.LSTM(128))(x)
    x = L.Dropout(dropout)(x)
    x = L.Dense(256, activation="relu")(x)
    x = L.Dropout(dropout)(x)
    return x

def build_lstm_model(num_classes: int, seq_len: int = 16, img_size: int = 224,
                     dropout: float = 0.3, weights: str = "imagenet") -> tf.keras.Model:
    backbone = build_backbone(img_size, weights)
    video_in = L.Input(shape=(seq_len, img_size, img_size, 3))
    feats = L.TimeDistributed(backbone)(video_in)
    x = _temporal_head_lstm(feats, dropout=dropout)
    out = L.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(video_in, out, name="cnn_lstm")

@tf.keras.utils.register_keras_serializable(package='custom')
class TransformerEncoder(L.Layer):
    def __init__(self, embed_dim, num_heads=4, ff_dim=512, rate=0.2, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        self.attn = L.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([L.Dense(ff_dim, activation="relu"), L.Dense(embed_dim)])
        self.norm1 = L.LayerNormalization(epsilon=1e-6)
        self.norm2 = L.LayerNormalization(epsilon=1e-6)
        self.dropout1 = L.Dropout(rate)
        self.dropout2 = L.Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.attn(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.norm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.norm2(out1 + ffn_output)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"embed_dim": self.embed_dim, "num_heads": self.num_heads,
                    "ff_dim": self.ff_dim, "rate": self.rate})
        return cfg

def positional_encoding(maxlen: int, embed_dim: int):
    import numpy as np
    pos = np.arange(maxlen)[:, None]
    i = np.arange(embed_dim)[None, :]
    angle_rates = 1 / (10000 ** (2 * (i//2) / embed_dim))
    angle_rads = pos * angle_rates
    sines = np.sin(angle_rads[:, 0::2]); coses = np.cos(angle_rads[:, 1::2])
    pe = np.zeros((maxlen, embed_dim)); pe[:, 0::2] = sines; pe[:, 1::2] = coses
    return tf.convert_to_tensor(pe, dtype=tf.float32)

def build_transformer_model(num_classes: int, seq_len: int = 16, img_size: int = 224,
                            embed_dim: int = 256, depth: int = 2, dropout: float = 0.2,
                            weights: str = "imagenet") -> tf.keras.Model:
    backbone = build_backbone(img_size, weights)
    video_in = L.Input(shape=(seq_len, img_size, img_size, 3))
    x = L.TimeDistributed(backbone)(video_in)         # (B,T,D0)
    x = L.TimeDistributed(L.Dense(embed_dim))(x)      # (B,T,E)
    pe = positional_encoding(seq_len, embed_dim)
    x = x + pe
    x = L.SpatialDropout1D(0.2)(x)
    for _ in range(depth):
        x = TransformerEncoder(embed_dim, num_heads=4, ff_dim=4*embed_dim, rate=dropout)(x)
    x = L.GlobalAveragePooling1D()(x)
    x = L.Dropout(dropout)(x)
    x = L.Dense(256, activation="relu")(x)
    x = L.Dropout(dropout)(x)
    out = L.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(video_in, out, name="cnn_transformer")

@tf.keras.utils.register_keras_serializable(package='custom')
class TemporalBlock(L.Layer):
    def __init__(self, channels, kernel_size=3, dilation_rate=1, dropout=0.2, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.dropout = dropout

        self.conv1 = L.Conv1D(channels, kernel_size, padding="causal",
                              dilation_rate=dilation_rate)
        self.bn1 = L.BatchNormalization()
        self.act1 = L.ReLU()
        self.drop1 = L.Dropout(dropout)

        self.conv2 = L.Conv1D(channels, kernel_size, padding="causal",
                              dilation_rate=dilation_rate)
        self.bn2 = L.BatchNormalization()
        self.drop2 = L.Dropout(dropout)

        self.proj = None
        self.act_out = L.ReLU()

    def build(self, input_shape):
        in_ch = int(input_shape[-1])
        if in_ch != self.channels:
            self.proj = L.Conv1D(self.channels, 1, padding="same")
        super().build(input_shape)

    def call(self, x, training=False):
        y = self.conv1(x)
        y = self.bn1(y, training=training)
        y = self.act1(y)
        y = self.drop1(y, training=training)

        y = self.conv2(y)
        y = self.bn2(y, training=training)
        y = self.drop2(y, training=training)

        res = self.proj(x) if self.proj is not None else x
        return self.act_out(res + y)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "channels": self.channels,
            "kernel_size": self.kernel_size,
            "dilation_rate": self.dilation_rate,
            "dropout": self.dropout,
        })
        return cfg

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def build_tcn_model(num_classes: int, seq_len: int = 16, img_size: int = 224,
                    embed_dim: int = 256, channels: int = 256,
                    num_blocks: int = 4, kernel_size: int = 3,
                    dilation_base: int = 2, dropout: float = 0.2,weights=None) -> tf.keras.Model:
    # reuse your MobileNetV2 backbone
    backbone = build_backbone(img_size)
    video_in = L.Input(shape=(seq_len, img_size, img_size, 3))
    x = L.TimeDistributed(backbone)(video_in)                 # (B, T, feat)
    x = L.TimeDistributed(L.Dense(embed_dim))(x)              # (B, T, E)
    x = L.LayerNormalization()(x)

    # stack dilated temporal residual blocks
    for b in range(num_blocks):
        x = TemporalBlock(channels=channels,
                          kernel_size=kernel_size,
                          dilation_rate=(dilation_base ** b),
                          dropout=dropout,
                          name=f"tblock_{b}")(x)

    x = L.GlobalAveragePooling1D()(x)
    x = L.Dropout(dropout)(x)
    x = L.Dense(256, activation="relu")(x)
    x = L.Dropout(dropout)(x)
    out = L.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(video_in, out, name="cnn_tcn")

def tiny_frame_encoder(embed_dim=256):
    x_in = L.Input(shape=(None, None, 3))
    x = L.Conv2D(32, 3, padding="same", activation="relu")(x_in)
    x = L.BatchNormalization()(x); x = L.MaxPool2D()(x)
    x = L.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = L.BatchNormalization()(x); x = L.MaxPool2D()(x)
    x = L.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = L.BatchNormalization()(x); x = L.MaxPool2D()(x)
    x = L.Conv2D(128, 3, padding="same", activation="relu",
                 kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = L.GlobalAveragePooling2D()(x)
    x = L.Dense(embed_dim, activation="relu")(x)
    return tf.keras.Model(x_in, x, name="tiny_frame_encoder")

def build_scratch_2d_gru_model(num_classes: int, seq_len: int = 16, img_size: int = 224,
                               embed_dim: int = 256, dropout: float = 0.3) -> tf.keras.Model:
    enc = tiny_frame_encoder(embed_dim)
    video_in = L.Input(shape=(seq_len, img_size, img_size, 3))
    x = L.TimeDistributed(enc)(video_in)       # (B, T, D)
    x = L.LayerNormalization()(x)
    x = L.SpatialDropout1D(0.2)(x)             # temporal feature dropout
    x = L.Bidirectional(L.GRU(256, return_sequences=True,
                              dropout=0.2, recurrent_dropout=0.2))(x)
    x = L.Bidirectional(L.GRU(128, dropout=0.2, recurrent_dropout=0.2))(x)
    x = L.Dropout(dropout)(x)
    x = L.Dense(256, activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = L.Dropout(dropout)(x)
    out = L.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(video_in, out, name="scratch_2d_gru")
def _conv3d_block(x, filters, k=(3,3,3), s=(1,1,1), d=(1,1,1), name=None):
    y = L.Conv3D(filters, k, strides=s, padding="same",
                 dilation_rate=d, use_bias=False, name=(name and f"{name}_conv"))(x)
    y = L.LayerNormalization(epsilon=1e-6, name=(name and f"{name}_ln"))(y)
    y = L.Activation("gelu", name=(name and f"{name}_gelu"))(y)
    return y

def _res3d(x, filters, d_temporal=1, name=None, drop=0.2):
    y = _conv3d_block(x, filters, k=(3,3,3), d=(d_temporal,1,1), name=(name and f"{name}_a"))
    y = L.Dropout(drop)(y)
    y = L.Conv3D(filters, (3,3,3), padding="same", use_bias=False, name=(name and f"{name}_b_conv"))(y)
    y = L.LayerNormalization(epsilon=1e-6, name=(name and f"{name}_b_ln"))(y)
    # pre-activation style add
    if x.shape[-1] != filters:
        x = L.Conv3D(filters, (1,1,1), padding="same", use_bias=False, name=(name and f"{name}_proj"))(x)
    out = L.Add(name=(name and f"{name}_add"))([x, y])
    out = L.Activation("gelu", name=(name and f"{name}_out"))(out)
    return out

def build_scratch_3d_tcn_model(num_classes: int, seq_len: int = 16, img_size: int = 224) -> tf.keras.Model:
    inp = L.Input(shape=(seq_len, img_size, img_size, 3))

    # 3D stem: downsample spatially, keep full time
    x = _conv3d_block(inp, 32, k=(3,7,7), s=(1,2,2), name="stem1")
    x = _conv3d_block(x,   64, k=(3,3,3), s=(1,2,2), name="stem2")

    # residual 3D blocks with temporal dilation
    x = _res3d(x,  64, d_temporal=1, name="res1")   # T-dilation 1
    x = _res3d(x,  64, d_temporal=2, name="res2")   # T-dilation 2
    x = _res3d(x,  96, d_temporal=4, name="res3")   # T-dilation 4

    # per-frame spatial pooling -> (B, T, C)
    x = L.TimeDistributed(L.GlobalAveragePooling2D(), name="gap2d_per_frame")(x)

    # TCN head (dilated 1D convs over time)
    h = x
    for d in [1, 2, 4, 8]:
        y = L.Conv1D(256, 3, padding="causal", dilation_rate=d, use_bias=False)(h)
        y = L.LayerNormalization(epsilon=1e-6)(y)
        y = L.Activation("gelu")(y)
        y = L.Dropout(0.2)(y)
        y = L.Conv1D(256, 1, padding="same", use_bias=False)(y)
        if h.shape[-1] != 256:
            h = L.Conv1D(256, 1, padding="same", use_bias=False)(h)
        h = L.Add()([h, y])

    x = L.GlobalAveragePooling1D()(h)
    x = L.Dropout(0.3)(x)
    x = L.Dense(256, activation="relu")(x)
    x = L.Dropout(0.3)(x)
    out = L.Dense(num_classes, activation="softmax")(x)

    return tf.keras.Model(inp, out, name="scratch3d_tcn_v2")