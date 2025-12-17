import argparse, numpy as np, pandas as pd

def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds_a", type=str, required=True)
    ap.add_argument("--preds_b", type=str, required=True)
    ap.add_argument("--boot_iters", type=int, default=5000)
    return ap.parse_args()

def accuracy(y_true, y_pred):
    return (y_true == y_pred).mean()

def bootstrap_ci(y_true, y_pred, iters=5000, alpha=0.05, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    accs = []
    for _ in range(iters):
        idx = rng.integers(0, n, size=n)
        accs.append(accuracy(y_true[idx], y_pred[idx]))
    lo, hi = np.percentile(accs, [100*alpha/2, 100*(1-alpha/2)])
    return lo, hi

def permutation_test(y_true, y_pred_a, y_pred_b, iters=5000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    correct_a = (y_true == y_pred_a).astype(int)
    correct_b = (y_true == y_pred_b).astype(int)
    diff_obs = (correct_a - correct_b).mean()
    diffs = []
    for _ in range(iters):
        swap = rng.integers(0, 2, size=n).astype(bool)
        a_tmp = np.where(swap, correct_b, correct_a)
        b_tmp = np.where(swap, correct_a, correct_b)
        diffs.append((a_tmp - b_tmp).mean())
    p = (np.sum(np.abs(diffs) >= np.abs(diff_obs)) + 1) / (iters + 1)
    return diff_obs, p

def main():
    args = get_args()
    a = pd.read_csv(args.preds_a)
    b = pd.read_csv(args.preds_b)
    assert len(a) == len(b), "Predictions must have same rows/order."
    y_true = a["y_true"].values
    acc_a = accuracy(y_true, a["y_pred"].values)
    acc_b = accuracy(y_true, b["y_pred"].values)

    ci_a = bootstrap_ci(y_true, a["y_pred"].values, iters=args.boot_iters)
    ci_b = bootstrap_ci(y_true, b["y_pred"].values, iters=args.boot_iters)
    diff, p = permutation_test(y_true, a["y_pred"].values, b["y_pred"].values, iters=args.boot_iters)

    print(f"Model A accuracy: {acc_a:.4f} (95% CI {ci_a[0]:.4f}–{ci_a[1]:.4f})")
    print(f"Model B accuracy: {acc_b:.4f} (95% CI {ci_b[0]:.4f}–{ci_b[1]:.4f})")
    print(f"Accuracy difference (A−B): {diff:.4f}, permutation p-value={p:.4f}")

if __name__ == "__main__":
    main()
