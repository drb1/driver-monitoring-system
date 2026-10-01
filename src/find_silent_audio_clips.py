# src/find_silent_audio_clips.py
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

MEL_ROOT = "data/processed_audio_mels"
SILENCE_STD_THRESH = 1e-4   # conservative threshold

rows = []

def is_silence(mel: np.ndarray, thresh=SILENCE_STD_THRESH):
    # mel shape: (T, M, 1) or (T, M)
    if mel.ndim == 3:
        mel = mel[..., 0]
    return np.std(mel) < thresh

def main():
    total = 0
    silent = 0

    per_class = {}

    for cls in sorted(os.listdir(MEL_ROOT)):
        cls_dir = os.path.join(MEL_ROOT, cls)
        if not os.path.isdir(cls_dir):
            continue

        per_class.setdefault(cls, {"total": 0, "silent": 0})

        for fn in tqdm(os.listdir(cls_dir), desc=f"Scanning {cls}", leave=False):
            if not fn.endswith(".npy"):
                continue

            path = os.path.join(cls_dir, fn)
            mel = np.load(path)

            total += 1
            per_class[cls]["total"] += 1

            silent_flag = is_silence(mel)
            if silent_flag:
                silent += 1
                per_class[cls]["silent"] += 1

            rows.append({
                "class": cls,
                "clip_id": fn.replace(".npy", ""),
                "path": path,
                "std": float(np.std(mel)),
                "is_silent": silent_flag
            })

    df = pd.DataFrame(rows)
    df.to_csv("outputs/metrics/audio_silence_report.csv", index=False)

    print("\n==== AUDIO SILENCE SUMMARY ====")
    print(f"Total clips: {total}")
    print(f"Silent clips: {silent} ({100*silent/total:.2f}%)\n")

    for cls, d in per_class.items():
        pct = 100 * d["silent"] / max(1, d["total"])
        print(f"{cls:12s}: {d['silent']:3d}/{d['total']:3d} silent ({pct:5.1f}%)")

    print("\nSaved detailed report to:")
    print("  outputs/metrics/audio_silence_report.csv")

if __name__ == "__main__":
    main()
