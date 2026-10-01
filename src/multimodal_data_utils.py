# src/multimodal_data_utils.py
import os, glob, random
from typing import List, Tuple

import numpy as np
import tensorflow as tf
import pandas as pd

from data_utils import (
    CLASS_NAMES,
    _load_sequence,
    _augment,
    _apply_mobilenet_preproc,
    _mixup_batch,
)

RAW_MEL_ROOT = "data/processed_audio_mels"


def _load_audio_npy(clip_dir: str, target_frames: int = 800, mel_bins: int = 128):
    """
    Loads cached log-mel npy created by build_audio_cache.py.
    Expected shape: (target_frames, mel_bins) e.g. (800, 128)

    Returns: (T, M, 1) float32
    """
    parts = clip_dir.replace("\\", "/").split("/")
    cls = parts[-2]      # e.g. Normal
    clip_id = parts[-1]  # e.g. Normal12

    npy_path = os.path.join(RAW_MEL_ROOT, cls, f"{clip_id}.npy")
    if not os.path.exists(npy_path):
        # fallback to silence if missing
        mel = np.zeros((target_frames, mel_bins), dtype=np.float32)
    else:
        mel = np.load(npy_path).astype(np.float32)
        # safety shape handling
        if mel.ndim != 2:
            mel = mel.reshape(target_frames, mel_bins).astype(np.float32)
        if mel.shape != (target_frames, mel_bins):
            # pad/crop if needed
            T, M = mel.shape
            if M != mel_bins:
                # fix mel bins mismatch (rare)
                mel2 = np.zeros((T, mel_bins), dtype=np.float32)
                mmin = min(M, mel_bins)
                mel2[:, :mmin] = mel[:, :mmin]
                mel = mel2
            if T < target_frames:
                pad = np.zeros((target_frames - T, mel_bins), dtype=np.float32)
                mel = np.concatenate([mel, pad], axis=0)
            elif T > target_frames:
                mel = mel[:target_frames]

    mel = np.expand_dims(mel, axis=-1)  # (800,128,1)
    return tf.convert_to_tensor(mel, dtype=tf.float32)


def build_multimodal_dataset(
    csv_path: str,
    seq_len: int,
    img_size: int,
    batch_size: int,
    shuffle: bool,
    augment: bool,
    audio_mixup: bool = False,  # kept for future use
):
    df = pd.read_csv(csv_path)
    paths = df["clip_dir"].tolist()
    labels = df["label"].tolist()

    jitter = 2 if augment else 0

    def gen():
        for p, y in zip(paths, labels):
            v = _load_sequence(p, seq_len, img_size, jitter=jitter)
            if augment:
                v = _augment(v)
            v = _apply_mobilenet_preproc(v)

            a = _load_audio_npy(p)  # (800,128,1)

            yield (v, a), tf.one_hot(y, depth=len(CLASS_NAMES))

    out_sig = (
        (
            tf.TensorSpec(shape=(seq_len, img_size, img_size, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(800, 128, 1), dtype=tf.float32),
        ),
        tf.TensorSpec(shape=(len(CLASS_NAMES),), dtype=tf.float32),
    )

    ds = tf.data.Dataset.from_generator(gen, output_signature=out_sig)

    if shuffle:
        ds = ds.shuffle(buffer_size=256, reshuffle_each_iteration=True)

    ds = ds.batch(batch_size, drop_remainder=False)

    # MixUp ONLY on video branch (audio kept aligned)
    if augment:
        def _mix(inputs, y):
            v, a = inputs
            v2, y2 = _mixup_batch(v, y, alpha=0.2, p=0.6)
            return (v2, a), y2

        ds = ds.map(_mix, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds
