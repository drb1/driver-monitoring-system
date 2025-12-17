# src/training.py 
import argparse, os, json
import tensorflow as tf
from data_utils import build_dataset, CLASS_NAMES, set_seed
from models import build_lstm_model, build_transformer_model, build_tcn_model, build_scratch_3d_tcn_model

def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, choices=["lstm","transformer","tcn","scratch3d_tcn"], required=True)
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
    return ap.parse_args()

def build_model(name, num_classes, seq_len, img_size, use_pretrained=True):
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
def main():
    args = get_args()
    set_seed(args.seed)
    os.makedirs("outputs/models", exist_ok=True)
    os.makedirs("outputs/logs", exist_ok=True)

    # datasets 
    train_ds = build_dataset("data/splits/train.csv", args.seq_len, args.img_size, args.batch,
                         shuffle=True, augment=True)
    val_ds   = build_dataset("data/splits/val.csv",   args.seq_len, args.img_size, args.batch,
                         shuffle=False, augment=False)


    use_pretrained = (args.model != "scratch3d_tcn")
    model = build_model(args.model, num_classes=len(CLASS_NAMES),
                        seq_len=args.seq_len, img_size=args.img_size,
                        use_pretrained=use_pretrained)

    # Freeze CNN if using pretrained transfer
    if use_pretrained:
        for l in model.layers:
            # leave BN frozen; freeze most convs initially
            if isinstance(l, tf.keras.layers.BatchNormalization):
                l.trainable = False
            elif hasattr(l, "name") and ("mobilenetv2" in l.name.lower() or
                                         l.__class__.__name__ in ("Conv2D","DepthwiseConv2D")):
                l.trainable = False

    # --- optimizer & loss (ONLY small tweak: add clipnorm) ---
    opt = tf.keras.optimizers.AdamW(learning_rate=args.lr,
                                    weight_decay=args.weight_decay,
                                    clipnorm=1.0)
    loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=args.label_smoothing)
    model.compile(optimizer=opt, loss=loss,
                  metrics=["acc",
                           tf.keras.metrics.Precision(name="precision"),
                           tf.keras.metrics.Recall(name="recall")])

    ckpt_best = tf.keras.callbacks.ModelCheckpoint(
        f"outputs/models/best_{args.model}.keras",
        monitor="val_acc", mode="max", save_best_only=True, save_weights_only=False, verbose=1
    )
    ckpt_last = tf.keras.callbacks.ModelCheckpoint(
        f"outputs/models/final_{args.model}.keras",
        save_best_only=False, save_weights_only=False, verbose=1
    )

    # -------- warmup phase (fresh callbacks) --------
    early_warm = tf.keras.callbacks.EarlyStopping(
        monitor="val_acc", mode="max", patience=6, min_delta=1e-3, restore_best_weights=True, verbose=1
    )
    reduce_warm = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_acc", mode="max", factor=0.5, patience=2, min_lr=1e-6, verbose=1
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.warmup_epochs,
        callbacks=[ckpt_best, reduce_warm, early_warm],
    )

    # -------- fine-tuning phase (fresh callbacks again) --------
    if args.fine_tune and use_pretrained:
        # unfreeze ~top 30% of conv-like layers, keep BN frozen
        conv_like = [l for l in model.layers if l.__class__.__name__ in ("Conv2D","DepthwiseConv2D")]
        cutoff = int(0.7 * len(conv_like))
        for l in conv_like[cutoff:]:
            l.trainable = True
        for l in model.layers:
            if isinstance(l, tf.keras.layers.BatchNormalization):
                l.trainable = False

        # smaller LR for FT
        tf.keras.backend.set_value(model.optimizer.learning_rate, args.lr * 0.1)

        early_ft = tf.keras.callbacks.EarlyStopping(
            monitor="val_acc", mode="max", patience=10, min_delta=1e-3,
            restore_best_weights=True, verbose=1
        )
        reduce_ft = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_acc", mode="max", factor=0.5, patience=3, min_lr=1e-6, verbose=1
        )

        history2 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.epochs,
            callbacks=[ckpt_best, ckpt_last, reduce_ft, early_ft],
        )
    else:
        # no FT: just train longer with slightly more patience
        early_main = tf.keras.callbacks.EarlyStopping(
            monitor="val_acc", mode="max", patience=10, min_delta=1e-3,
            restore_best_weights=True, verbose=1
        )
        reduce_main = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_acc", mode="max", factor=0.5, patience=3, min_lr=1e-6, verbose=1
        )
        history2 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.epochs,
            callbacks=[ckpt_best, ckpt_last, reduce_main, early_main],
        )

if __name__ == "__main__":
    main()
