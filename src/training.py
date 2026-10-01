# src/training.py
import argparse, os
import tensorflow as tf

from data_utils import build_dataset, CLASS_NAMES, set_seed
from models import (
    build_lstm_model,
    build_transformer_model,
    build_tcn_model,
    build_scratch_3d_tcn_model,
)

from multimodal_data_utils import build_multimodal_dataset
from multimodal_models import build_fusion_model


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, choices=["lstm", "transformer", "tcn", "scratch3d_tcn"], required=True)

    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq_len", type=int, default=16)
    ap.add_argument("--img_size", type=int, default=224)

    ap.add_argument("--fine_tune", action="store_true")
    ap.add_argument("--warmup_epochs", type=int, default=3)

    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--label_smoothing", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)

    # multimodal
    ap.add_argument("--multimodal", action="store_true")
    ap.add_argument("--fusion_base", type=str, default="tcn", choices=["tcn", "lstm", "transformer"])

    return ap.parse_args()


def build_video_only_model(name, num_classes, seq_len, img_size, use_pretrained=True):
    weights = "imagenet" if use_pretrained else None

    if name == "lstm":
        return build_lstm_model(num_classes, seq_len, img_size, weights=weights)
    if name == "transformer":
        return build_transformer_model(num_classes, seq_len, img_size, weights=weights)
    if name == "tcn":
        return build_tcn_model(num_classes, seq_len, img_size, weights=weights)
    if name == "scratch3d_tcn":
        return build_scratch_3d_tcn_model(num_classes, seq_len, img_size)

    raise ValueError(f"Unknown model: {name}")


def freeze_pretrained_backbone(model: tf.keras.Model):
    """
    Freeze MobileNet backbone and keep BN frozen.
    Works for both video-only and fusion models.
    """
    for l in model.layers:
        # freeze BN always when transfer learning (stability)
        if isinstance(l, tf.keras.layers.BatchNormalization):
            l.trainable = False

        # freeze the backbone if layer name suggests mobilenet
        # (covers the Functional backbone name too)
        if hasattr(l, "name") and ("mobilenet" in l.name.lower()):
            l.trainable = False


def unfreeze_top_conv_layers(model: tf.keras.Model, top_ratio: float = 0.30):
    """
    Unfreeze top portion of Conv2D/DepthwiseConv2D layers for fine-tuning.
    Keeps BN frozen.
    """
    conv_like = [l for l in model.layers if l.__class__.__name__ in ("Conv2D", "DepthwiseConv2D")]
    if len(conv_like) == 0:
        return

    cutoff = int((1.0 - top_ratio) * len(conv_like))
    for l in conv_like[cutoff:]:
        l.trainable = True

    for l in model.layers:
        if isinstance(l, tf.keras.layers.BatchNormalization):
            l.trainable = False


def main():
    args = get_args()
    set_seed(args.seed)

    os.makedirs("outputs/models", exist_ok=True)
    os.makedirs("outputs/logs", exist_ok=True)

    num_classes = len(CLASS_NAMES)

    # ---------------- datasets ----------------
    if args.multimodal:
        train_ds = build_multimodal_dataset(
            "data/splits/train.csv",
            args.seq_len,
            args.img_size,
            args.batch,
            shuffle=True,
            augment=True,
        )
        val_ds = build_multimodal_dataset(
            "data/splits/val.csv",
            args.seq_len,
            args.img_size,
            args.batch,
            shuffle=False,
            augment=False,
        )
    else:
        train_ds = build_dataset(
            "data/splits/train.csv",
            args.seq_len,
            args.img_size,
            args.batch,
            shuffle=True,
            augment=True,
        )
        val_ds = build_dataset(
            "data/splits/val.csv",
            args.seq_len,
            args.img_size,
            args.batch,
            shuffle=False,
            augment=False,
        )

    # ---------------- model ----------------
    if args.multimodal:
        tag = f"fusion_{args.fusion_base}"
        model = build_fusion_model(
            model_name=args.fusion_base,
            num_classes=num_classes,
            seq_len=args.seq_len,
            img_size=args.img_size,
            weights="imagenet",
            dropout=0.3,
        )
        use_pretrained = True  # fusion uses mobilenet video encoder
    else:
        tag = args.model
        use_pretrained = (args.model != "scratch3d_tcn")
        model = build_video_only_model(
            args.model,
            num_classes=num_classes,
            seq_len=args.seq_len,
            img_size=args.img_size,
            use_pretrained=use_pretrained,
        )

    # ---------------- freeze backbone (warmup) ----------------
    if use_pretrained:
        freeze_pretrained_backbone(model)

    # ---------------- compile ----------------
    opt = tf.keras.optimizers.AdamW(
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        clipnorm=1.0
    )
    loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=args.label_smoothing)

    model.compile(
        optimizer=opt,
        loss=loss,
        metrics=[
            "acc",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )

    # ---------------- checkpoints ----------------
    ckpt_best = tf.keras.callbacks.ModelCheckpoint(
        filepath=f"outputs/models/best_{tag}.keras",
        monitor="val_acc",
        mode="max",
        save_best_only=True,
        save_weights_only=False,
        verbose=1,
    )
    ckpt_last = tf.keras.callbacks.ModelCheckpoint(
        filepath=f"outputs/models/final_{tag}.keras",
        save_best_only=False,
        save_weights_only=False,
        verbose=1,
    )

    # ---------------- warmup phase ----------------
    early_warm = tf.keras.callbacks.EarlyStopping(
        monitor="val_acc",
        mode="max",
        patience=6,
        min_delta=1e-3,
        restore_best_weights=True,
        verbose=1,
    )
    reduce_warm = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_acc",
        mode="max",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1,
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.warmup_epochs,
        callbacks=[ckpt_best, reduce_warm, early_warm],
    )

    # ---------------- main / fine-tune phase ----------------
    if args.fine_tune and use_pretrained:
        # unfreeze top 30% conv-like layers; keep BN frozen
        unfreeze_top_conv_layers(model, top_ratio=0.30)

        # smaller LR for fine-tune
        tf.keras.backend.set_value(model.optimizer.learning_rate, args.lr * 0.1)

        early_ft = tf.keras.callbacks.EarlyStopping(
            monitor="val_acc",
            mode="max",
            patience=10,
            min_delta=1e-3,
            restore_best_weights=True,
            verbose=1,
        )
        reduce_ft = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_acc",
            mode="max",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        )

        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.epochs,
            callbacks=[ckpt_best, ckpt_last, reduce_ft, early_ft],
        )
    else:
        early_main = tf.keras.callbacks.EarlyStopping(
            monitor="val_acc",
            mode="max",
            patience=10,
            min_delta=1e-3,
            restore_best_weights=True,
            verbose=1,
        )
        reduce_main = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_acc",
            mode="max",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        )

        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.epochs,
            callbacks=[ckpt_best, ckpt_last, reduce_main, early_main],
        )


if __name__ == "__main__":
    main()
