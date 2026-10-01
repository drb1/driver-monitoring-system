# src/evaluation_fusion.py
import os, argparse, json, sys
# make project root importable when running as a script
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score, roc_curve, auc,
    precision_recall_curve, average_precision_score
)

from data_utils import CLASS_NAMES
from models import TransformerEncoder, TemporalBlock
from multimodal_data_utils import build_multimodal_dataset

# IMPORTANT:
# If your fusion model file contains any custom layers/classes, import them here.
# If build_fusion_model is just a function, it's not needed in custom_objects,
# but any custom Layers used by the fusion model ARE needed.
try:
    from multimodal_models import AudioConvBlock, FusionBlock  # change if your names differ
except Exception:
    AudioConvBlock, FusionBlock = None, None


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--test_csv", type=str, default="data/splits/test.csv")
    ap.add_argument("--seq_len", type=int, default=16)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out_prefix", type=str, default="")
    ap.add_argument("--tta", type=int, default=1, help="# temporal crops for TTA (>=1)")
    return ap.parse_args()


def _annotate_cm(ax, cm, fmt, classes):
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)

    thresh = cm.max() / 2.0 if cm.size > 0 else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], fmt),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "white"
            )
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")


def plot_confusion(cm, classes, normalize=False, outpng="cm.png"):
    if normalize:
        denom = (cm.sum(axis=1)[:, np.newaxis] + 1e-9)
        cm = cm.astype("float") / denom
    plt.figure()
    im = plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix (normalized)" if normalize else "Confusion Matrix")
    plt.colorbar(im)
    fmt = ".2f" if normalize else "d"
    _annotate_cm(plt.gca(), cm, fmt, classes)
    plt.tight_layout()
    plt.savefig(outpng, bbox_inches="tight")
    plt.close()


def plot_confusion_row(cm, classes, outpng="outputs/plots/confusion_matrices_row.png"):
    cm_norm = cm.astype(float) / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    im0 = axes[0].imshow(cm, interpolation="nearest", vmin=0, vmax=max(1, cm.max()))
    axes[0].set_title("Confusion Matrix")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    _annotate_cm(axes[0], cm, "d", classes)

    im1 = axes[1].imshow(cm_norm, interpolation="nearest", vmin=0.0, vmax=1.0)
    axes[1].set_title("Confusion Matrix (normalized)")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    _annotate_cm(axes[1], cm_norm, ".2f", classes)

    fig.suptitle("Confusion Matrices — Raw & Normalized", y=1.02, fontsize=12)
    fig.savefig(outpng, bbox_inches="tight")
    plt.close(fig)


def expected_calibration_error(y_true_idx, y_prob, n_bins=10):
    conf = y_prob.max(axis=1)
    pred = y_prob.argmax(axis=1)
    acc = (pred == y_true_idx).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.any():
            ece += np.abs(acc[m].mean() - conf[m].mean()) * (m.sum() / len(conf))
    return ece


def plot_reliability_diagram(y_true_idx, y_prob, outpng="outputs/plots/reliability.png", n_bins=10):
    conf = y_prob.max(axis=1)
    pred = y_prob.argmax(axis=1)
    acc = (pred == y_true_idx).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)

    accs, confs, weights = [], [], []
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.any():
            accs.append(acc[m].mean())
            confs.append(conf[m].mean())
            weights.append(m.sum())
        else:
            accs.append(0.0)
            confs.append((bins[i] + bins[i + 1]) / 2.0)
            weights.append(0)

    plt.figure()
    plt.plot([0, 1], [0, 1], linestyle="--")
    sizes = (np.array(weights, dtype=float) * 5.0) + 10.0
    plt.scatter(confs, accs, s=sizes)
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title("Reliability Diagram")
    plt.tight_layout()
    plt.savefig(outpng)
    plt.close()


def plot_roc_row(y_true_onehot, y_prob, class_names, outpng="outputs/plots/roc_row.png"):
    n_classes = y_prob.shape[1]
    fig, axes = plt.subplots(1, n_classes, figsize=(4.2 * n_classes, 4.2), sharex=True, sharey=True)
    if n_classes == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        fpr, tpr, _ = roc_curve(y_true_onehot[:, i], y_prob[:, i])
        rauc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"AUC={rauc:.3f}")
        ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
        ax.set_title(f"ROC – {class_names[i]}")
        ax.set_xlabel("FPR")
        if i == 0:
            ax.set_ylabel("TPR")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

    fig.suptitle("ROC Curves (One-vs-Rest) — All Classes", y=1.03, fontsize=12)
    plt.tight_layout()
    plt.savefig(outpng, bbox_inches="tight")
    plt.close(fig)


def plot_pr_row(y_true_onehot, y_prob, class_names, outpng="outputs/plots/pr_row.png"):
    n_classes = y_prob.shape[1]
    fig, axes = plt.subplots(1, n_classes, figsize=(4.2 * n_classes, 4.2), sharex=True, sharey=True)
    if n_classes == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        precision, recall, _ = precision_recall_curve(y_true_onehot[:, i], y_prob[:, i])
        ap = average_precision_score(y_true_onehot[:, i], y_prob[:, i])
        ax.plot(recall, precision, label=f"AP={ap:.3f}")
        ax.set_title(f"Precision–Recall – {class_names[i]}")
        ax.set_xlabel("Recall")
        if i == 0:
            ax.set_ylabel("Precision")
        ax.legend(loc="lower left")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Precision–Recall (One-vs-Rest) — All Classes", y=1.03, fontsize=12)
    plt.tight_layout()
    plt.savefig(outpng, bbox_inches="tight")
    plt.close(fig)


def predict_with_tta_fusion(model, videos, audios, tta=1):
    """
    videos: (B,T,H,W,3)
    audios: (B,800,128,1)  (kept unchanged across crops)
    """
    v_np = videos.numpy()
    a_np = audios.numpy()
    T = v_np.shape[1]

    if tta <= 1:
        return model.predict([v_np, a_np], verbose=0)

    probs = []
    for k in range(tta):
        start = int(k * T / tta)
        end = start + max(1, T // tta)
        v = v_np[:, start:end]
        if v.shape[1] < T:
            pad = np.repeat(v[:, -1:], T - v.shape[1], axis=1)
            v = np.concatenate([v, pad], axis=1)
        probs.append(model.predict([v, a_np], verbose=0))

    return np.mean(probs, axis=0)


def main():
    args = get_args()
    os.makedirs("outputs/metrics", exist_ok=True)
    os.makedirs("outputs/plots", exist_ok=True)

    # Build custom_objects for loading fusion model safely
    custom_objects = {
        "TransformerEncoder": TransformerEncoder,
        "TemporalBlock": TemporalBlock,
    }
    if AudioConvBlock is not None:
        custom_objects["AudioConvBlock"] = AudioConvBlock
    if FusionBlock is not None:
        custom_objects["FusionBlock"] = FusionBlock

    # Load fusion model
    model = tf.keras.models.load_model(
        args.model_path, compile=False, custom_objects=custom_objects
    )

    # Multimodal test dataset
    test_ds = build_multimodal_dataset(
        args.test_csv, args.seq_len, args.img_size, args.batch,
        shuffle=False, augment=False
    )

    y_true, y_prob = [], []
    for (videos, audios), labels in test_ds:
        probs = predict_with_tta_fusion(model, videos, audios, tta=args.tta)
        y_prob.append(probs)
        y_true.append(labels.numpy())

    y_prob = np.vstack(y_prob)           # (N, C)
    y_true = np.vstack(y_true)           # (N, C) one-hot

    y_pred = y_prob.argmax(axis=1)
    y_true_idx = y_true.argmax(axis=1)

    acc = (y_pred == y_true_idx).mean()
    print(f"Test Accuracy: {acc:.4f}")

    cm = confusion_matrix(y_true_idx, y_pred, labels=list(range(len(CLASS_NAMES))))
    np.savetxt("outputs/metrics/confusion_matrix.csv", cm, fmt="%d", delimiter=",")

    cm_norm = cm.astype(float) / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)
    np.savetxt("outputs/metrics/confusion_matrix_norm.csv", cm_norm, fmt="%.6f", delimiter=",")

    plot_confusion(cm, CLASS_NAMES, normalize=False, outpng="outputs/plots/confusion_matrix.png")
    plot_confusion(cm, CLASS_NAMES, normalize=True, outpng="outputs/plots/confusion_matrix_norm.png")
    plot_confusion_row(cm, CLASS_NAMES, outpng="outputs/plots/confusion_matrices_row.png")

    report = classification_report(y_true_idx, y_pred, target_names=CLASS_NAMES, output_dict=True, digits=4)
    pd.DataFrame(report).to_csv("outputs/metrics/classification_report.csv")
    print(pd.DataFrame(report).transpose())

    ovr_auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
    print(f"Macro ROC-AUC (OvR): {ovr_auc:.4f}")
    plot_roc_row(y_true, y_prob, CLASS_NAMES, outpng="outputs/plots/roc_row.png")

    aps = [average_precision_score(y_true[:, i], y_prob[:, i]) for i in range(len(CLASS_NAMES))]
    mAP = float(np.mean(aps)) if len(aps) > 0 else 0.0
    print(f"Macro Average Precision (mAP): {mAP:.4f}")
    plot_pr_row(y_true, y_prob, CLASS_NAMES, outpng="outputs/plots/pr_row.png")

    ece = float(expected_calibration_error(y_true_idx, y_prob, n_bins=10))
    print(f"Expected Calibration Error (ECE@10): {ece:.4f}")
    plot_reliability_diagram(y_true_idx, y_prob, outpng="outputs/plots/reliability.png", n_bins=10)

    prefix = args.out_prefix if args.out_prefix else "test_fusion"
    out_csv = f"outputs/metrics/{prefix}_preds.csv"
    df_out = pd.DataFrame({"y_true": y_true_idx, "y_pred": y_pred})
    for i, cls in enumerate(CLASS_NAMES):
        df_out[f"p_{cls}"] = y_prob[:, i]
    df_out.to_csv(out_csv, index=False)
    print(f"Saved per-sample predictions to: {out_csv}")

    metrics = {
        "accuracy": float(acc),
        "macro_roc_auc_ovr": float(ovr_auc),
        "macro_average_precision": float(mAP),
        "ece@10": ece
    }
    with open(f"outputs/metrics/{prefix}_summary.json", "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
