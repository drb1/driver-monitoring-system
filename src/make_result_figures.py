#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve, auc,
    precision_recall_curve, average_precision_score,
    classification_report, confusion_matrix
)

# -------- helpers --------

def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def _load_preds(pred_csv: str):
    df = pd.read_csv(pred_csv)
    # Try flexible column names
    # y_true / y_pred
    true_col = next((c for c in df.columns if c.lower() in {"y_true","true","label","target"}), None)
    pred_col = next((c for c in df.columns if c.lower() in {"y_pred","pred","pred_label"}), None)
    if true_col is None:
        raise ValueError("Could not find y_true column in the predictions CSV.")
    # probability columns: p_Class or prob_Class or just Class if rows sum ~1
    prob_cols = [c for c in df.columns if c.startswith("p_")] or \
                [c for c in df.columns if c.startswith("prob_")]
    if not prob_cols:
        # Fallback: any numeric cols whose rows sum to ~1.0
        numeric = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
        sums = df[numeric].sum(axis=1).to_numpy()
        if (np.isfinite(sums).mean() > 0.99) and (np.isclose(sums, 1.0, atol=1e-3).mean() > 0.7):
            prob_cols = numeric
        else:
            raise ValueError("Could not detect probability columns in the CSV.")
    # class names
    classes = [c.split("p_")[-1].split("prob_")[-1] for c in prob_cols]
    y_true = df[true_col].to_numpy()
    y_pred = df[pred_col].to_numpy() if pred_col in df.columns else np.argmax(df[prob_cols].to_numpy(), axis=1)
    probs  = df[prob_cols].to_numpy()
    return y_true.astype(int), y_pred.astype(int), probs.astype(float), classes

def _plot_cm(cm, classes, outpng, title="Confusion Matrix (row-normalised)", normalise=True):
    if normalise:
        with np.errstate(divide='ignore', invalid='ignore'):
            cm_plot = cm.astype(float) / cm.sum(axis=1, keepdims=True)
            cm_plot = np.nan_to_num(cm_plot)
    else:
        cm_plot = cm
    fig, ax = plt.subplots(figsize=(6,5), dpi=200)
    im = ax.imshow(cm_plot, cmap="Blues", vmin=0, vmax=(1.0 if normalise else None))
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Proportion" if normalise else "Count")
    ax.set_xticks(np.arange(len(classes))); ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, rotation=30, ha="right")
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted label"); ax.set_ylabel("True label"); ax.set_title(title)
    fmt = ".2f" if normalise else "d"
    thresh = cm_plot.max()/2. if cm_plot.size else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm_plot[i, j]
            ax.text(j, i, format(val, fmt), ha="center", va="center",
                    color=("white" if val > thresh else "black"), fontsize=9)
    fig.tight_layout()
    fig.savefig(outpng, bbox_inches="tight"); plt.close(fig)

def _plot_roc(y_true, probs, classes, outpng):
    # one-vs-rest ROC per class + macro AUC
    n = len(classes)
    y_ovr = np.eye(n)[y_true]
    fig, ax = plt.subplots(figsize=(6,5), dpi=200)
    aucs = []
    for i, cls in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_ovr[:, i], probs[:, i])
        roc_auc = auc(fpr, tpr); aucs.append(roc_auc)
        ax.plot(fpr, tpr, lw=1.5, label=f"{cls} (AUC={roc_auc:.3f})")
    ax.plot([0,1],[0,1], "--", lw=1, label="Chance", alpha=0.7)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curves (Macro AUC={np.mean(aucs):.3f})")
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(outpng, bbox_inches="tight"); plt.close(fig)

def _plot_pr(y_true, probs, classes, outpng):
    n = len(classes)
    y_ovr = np.eye(n)[y_true]
    fig, ax = plt.subplots(figsize=(6,5), dpi=200)
    aps = []
    for i, cls in enumerate(classes):
        p, r, _ = precision_recall_curve(y_ovr[:, i], probs[:, i])
        ap = average_precision_score(y_ovr[:, i], probs[:, i]); aps.append(ap)
        ax.plot(r, p, lw=1.5, label=f"{cls} (AP={ap:.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(f"Precision–Recall Curves (Macro mAP={np.mean(aps):.3f})")
    ax.legend(loc="lower left", fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(outpng, bbox_inches="tight"); plt.close(fig)

def _plot_prf_bars(y_true, y_pred, classes, outpng):
    rep = classification_report(y_true, y_pred, target_names=classes, output_dict=True, zero_division=0)
    prec = [rep[c]["precision"] for c in classes]
    rec  = [rep[c]["recall"] for c in classes]
    f1   = [rep[c]["f1-score"] for c in classes]
    x = np.arange(len(classes)); w = 0.25
    fig, ax = plt.subplots(figsize=(7,4), dpi=200)
    ax.bar(x - w, prec, width=w, label="Precision")
    ax.bar(x,     rec,  width=w, label="Recall")
    ax.bar(x + w, f1,   width=w, label="F1-score")
    ax.set_xticks(x); ax.set_xticklabels(classes, rotation=20, ha="right")
    ax.set_ylim(0, 1.05); ax.set_ylabel("Score"); ax.set_title("Per-class Precision, Recall, F1")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(outpng, bbox_inches="tight"); plt.close(fig)

def _plot_conf_hist(probs, outpng):
    fig, ax = plt.subplots(figsize=(6,4), dpi=200)
    ax.hist(probs.max(axis=1), bins=20, edgecolor="black", alpha=0.85)
    ax.set_xlabel("Max class probability"); ax.set_ylabel("Count")
    ax.set_title("Confidence Distribution")
    fig.tight_layout(); fig.savefig(outpng, bbox_inches="tight"); plt.close(fig)

def main():
    ap = argparse.ArgumentParser(description="Make clean figures for Results section from test_preds CSV.")
    ap.add_argument("--pred_csv", type=str, default="outputs/metrics/test_preds.csv",
                    help="Per-sample predictions written by evaluation.py")
    ap.add_argument("--outdir", type=str, default="outputs/figures", help="Directory for images")
    ap.add_argument("--cm_csv", type=str, default="outputs/metrics/confusion_matrix.csv",
                    help="Optional confusion matrix csv (counts). If missing, it will be recomputed.")
    args = ap.parse_args()

    _ensure_dir(args.outdir)

    y_true, y_pred, probs, classes = _load_preds(args.pred_csv)

    # confusion matrix (recompute to be safe)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    _plot_cm(cm, classes, os.path.join(args.outdir, "cm_norm.png"),
             title="Confusion Matrix (row-normalised)", normalise=True)

    # combined ROC / PR
    _plot_roc(y_true, probs, classes, os.path.join(args.outdir, "roc_multi.png"))
    _plot_pr(y_true, probs, classes, os.path.join(args.outdir, "pr_multi.png"))

    # per-class PR/F1 bars
    _plot_prf_bars(y_true, y_pred, classes, os.path.join(args.outdir, "prf_bars.png"))

    # confidence histogram
    _plot_conf_hist(probs, os.path.join(args.outdir, "confidence_hist.png"))

    print(f"Figures saved in: {args.outdir}")
    print("Suggested placements in Chapter 6:")
    print("  • cm_norm.png  – Section 6.x Confusion Matrix")
    print("  • roc_multi.png – Section 6.x ROC Curves and Macro AUC")
    print("  • pr_multi.png  – Section 6.x Precision–Recall and mAP")
    print("  • prf_bars.png  – Section 6.x Class-wise PRF")
    print("  • confidence_hist.png – Section 6.x Confidence distribution")
    
if __name__ == "__main__":
    main()
