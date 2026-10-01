import pandas as pd
from data_utils import CLASS_NAMES

preds = pd.read_csv("outputs/metrics/test_fusion_preds.csv")
test = pd.read_csv("data/splits/test.csv")

# Merge clip_dir so you can see which video failed
df = preds.copy()
df["clip_dir"] = test["clip_dir"].values

# Show mistakes
mistakes = df[df["y_true"] != df["y_pred"]].copy()
print("Mistakes:", len(mistakes))
if len(mistakes) > 0:
    for _, r in mistakes.iterrows():
        probs = {c: r[f"p_{c}"] for c in CLASS_NAMES}
        probs_sorted = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        print("\n---")
        print("clip:", r["clip_dir"])
        print("true:", CLASS_NAMES[int(r["y_true"])], "pred:", CLASS_NAMES[int(r["y_pred"])])
        print("probs:", probs_sorted)
