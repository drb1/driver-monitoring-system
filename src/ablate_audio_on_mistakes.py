import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import tensorflow as tf

from multimodal_data_utils import build_multimodal_dataset
from multimodal_models import build_fusion_model  # only if needed for custom objects
from data_utils import CLASS_NAMES
from models import TransformerEncoder, TemporalBlock

MODEL_PATH = "outputs/models/best_fusion_tcn.keras"
TEST_CSV   = "data/splits/test.csv"
BATCH      = 8
SEQ_LEN    = 16
IMG_SIZE   = 224

MISTAKES = [
    "data/processed/Normal/Normal77",
    "data/processed/Normal/Normal25",
    "data/processed/Distracted/Distraction24",
    "data/processed/Distracted/Distraction46",
]

def main():
    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    except Exception:
        model = tf.keras.models.load_model(
            MODEL_PATH, compile=False,
            custom_objects={"TransformerEncoder": TransformerEncoder, "TemporalBlock": TemporalBlock}
        )

    ds = build_multimodal_dataset(TEST_CSV, SEQ_LEN, IMG_SIZE, BATCH, shuffle=False, augment=False)

    # collect only the 4 clips
    test_df = pd.read_csv(TEST_CSV).reset_index(drop=True)
    wanted_idx = [i for i, p in enumerate(test_df["clip_dir"].tolist()) if p in set(MISTAKES)]
    wanted_idx = set(wanted_idx)

    all_preds = []
    cursor = 0
    for (v, a), y in ds:
        bs = v.shape[0]
        idxs = list(range(cursor, cursor + bs))
        cursor += bs

        keep = [k for k, gi in enumerate(idxs) if gi in wanted_idx]
        if not keep:
            continue

        v_sel = tf.gather(v, keep, axis=0)
        a_sel = tf.gather(a, keep, axis=0)
        y_sel = tf.gather(y, keep, axis=0)

        # real audio
        p_real = model.predict([v_sel, a_sel], verbose=0)

        # zero audio ablation
        a_zero = tf.zeros_like(a_sel)
        p_zero = model.predict([v_sel, a_zero], verbose=0)

        all_preds.append((v_sel, a_sel, y_sel, p_real, p_zero, keep, idxs))

    # print comparison
    for (_, _, y_sel, p_real, p_zero, keep, idxs) in all_preds:
        for j, k in enumerate(keep):
            global_i = idxs[k]
            clip = test_df.loc[global_i, "clip_dir"]
            yt = int(np.argmax(y_sel[j].numpy()))
            pr = int(np.argmax(p_real[j]))
            pz = int(np.argmax(p_zero[j]))
            print("\n---")
            print("clip:", clip)
            print("true:", CLASS_NAMES[yt])
            print("pred real:", CLASS_NAMES[pr], " top:", float(np.max(p_real[j])))
            print("pred zero:", CLASS_NAMES[pz], " top:", float(np.max(p_zero[j])))

if __name__ == "__main__":
    main()
