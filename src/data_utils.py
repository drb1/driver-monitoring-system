# --- data_utils.py (drop-in replacement for the shown functions) ---
import os, glob, random
from typing import List, Tuple
import numpy as np
import tensorflow as tf
import pandas as pd

CLASS_NAMES = ["Normal", "Aggressive", "Distracted", "Drowsy"]
CLASS2IDX = {c:i for i,c in enumerate(CLASS_NAMES)}

def set_seed(seed: int = 42):
    random.seed(seed)
    tf.random.set_seed(seed)

def list_clip_dirs(processed_root: str) -> List[Tuple[str, int]]:
    items = []
    for cls in CLASS_NAMES:
        pattern = os.path.join(processed_root, cls, "*")
        for clip_dir in sorted(glob.glob(pattern)):
            if os.path.isdir(clip_dir):
                items.append((clip_dir, CLASS2IDX[cls]))
    return items

def _sample_indices(n: int, seq_len: int, jitter: int = 0):
    # Start with evenly spaced indices, then add small integer noise (temporal jitter).
    idxs = np.linspace(0, n-1, seq_len).astype(int)
    if jitter > 0 and n > 1:
        noise = np.random.randint(-jitter, jitter+1, size=seq_len)
        idxs = np.clip(idxs + noise, 0, n-1)
    return idxs

def _load_sequence(clip_dir: str, seq_len: int, img_size: int, jitter: int = 0):
    frames = sorted(glob.glob(os.path.join(clip_dir, "frame_*.jpg")))
    if len(frames) == 0:
        raise ValueError(f"No frames found in {clip_dir}")
    idxs = _sample_indices(len(frames), seq_len, jitter=jitter)
    imgs = []
    for fp in [frames[i] for i in idxs]:
        img = tf.io.read_file(fp)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, (img_size, img_size))
        imgs.append(img)
    video = tf.stack(imgs, axis=0)  # (T,H,W,3)
    video = tf.cast(video, tf.float32) / 255.0
    return video

def _gaussian_blur(frames, ksize=3):
    # Simple, cheap blur via depthwise conv; kernel size must be odd.
    if ksize <= 1: 
        return frames
    kernel_1d = tf.constant([[1., 2., 1.]], dtype=tf.float32)
    kernel_2d = tf.matmul(tf.transpose(kernel_1d), kernel_1d)
    kernel_2d = kernel_2d / tf.reduce_sum(kernel_2d)
    kernel_2d = tf.expand_dims(tf.expand_dims(kernel_2d, -1), -1)  # (k,k,1,1)
    # Apply per-channel
    frames = tf.nn.depthwise_conv2d(frames, tf.repeat(kernel_2d, 3, axis=2), strides=[1,1,1,1], padding="SAME")
    return frames

def _augment(video):
    # video: (T,H,W,3), values in [0,1]
    T = tf.shape(video)[0]
    H = tf.shape(video)[1]
    W = tf.shape(video)[2]

    # Horizontal flip
    if tf.random.uniform(()) > 0.5:
        video = tf.image.flip_left_right(video)

    # ----- Consistent color jitter across the whole clip -----
    # Brightness shift
    b_delta = tf.random.uniform([], -0.10, 0.10)
    video = tf.clip_by_value(video + b_delta, 0.0, 1.0)
    # Contrast
    c_scale = 1.0 + tf.random.uniform([], -0.20, 0.20)
    video = tf.image.adjust_contrast(video, c_scale)
    # Saturation & Hue (apply per-frame using map_fn but with same factors)
    s_scale = 1.0 + tf.random.uniform([], -0.30, 0.30)
    h_delta = tf.random.uniform([], -0.08, 0.08)
    def _color(f):
        f = tf.image.adjust_saturation(f, s_scale)
        f = tf.image.adjust_hue(f, h_delta)
        return f
    video = tf.map_fn(_color, video)

    # Random grayscale
    if tf.random.uniform(()) > 0.8:
        gray = tf.image.rgb_to_grayscale(video)
        video = tf.image.grayscale_to_rgb(gray)

    # Random spatial crop (mild) then resize back (keeps faces roughly in frame)
    if tf.random.uniform(()) > 0.5:
        crop_h = tf.cast(tf.cast(H, tf.float32) * tf.random.uniform([], 0.85, 1.0), tf.int32)
        crop_w = tf.cast(tf.cast(W, tf.float32) * tf.random.uniform([], 0.85, 1.0), tf.int32)
        video = tf.image.random_crop(video, size=(T, crop_h, crop_w, 3))
        video = tf.image.resize(video, (H, W))

    # Mild Gaussian blur (randomly)
    if tf.random.uniform(()) > 0.6:
        video = _gaussian_blur(video, ksize=3)

    # Small Gaussian noise
    if tf.random.uniform(()) > 0.5:
        noise = tf.random.normal(tf.shape(video), 0.0, 0.02)
        video = tf.clip_by_value(video + noise, 0.0, 1.0)

    # ----- Tiny time-crop to simulate frame drops -----
    if tf.random.uniform(()) > 0.6:
        keep = tf.random.uniform([], 0.85, 1.0)  # keep % of frames
        new_T = tf.maximum(2, tf.cast(keep * tf.cast(T, tf.float32), tf.int32))
        # Take a random contiguous segment and then resize back to T with linear resampling
        start = tf.random.uniform([], 0, T - new_T + 1, dtype=tf.int32)
        segment = video[start:start+new_T]  # (new_T, H, W, 3)
        # Resize temporally back to T by linear interpolation
        segment = tf.image.resize(segment, (H, W))  # spatial noop to keep shapes consistent
        # Temporal resize: use tf.image.resize on a fake spatial axis by stacking frames along width
        # Simpler: just gather indices to upsample back
        idxs = tf.cast(tf.linspace(0.0, tf.cast(new_T-1, tf.float32), T), tf.int32)
        video = tf.gather(segment, idxs, axis=0)

    return video

def _apply_mobilenet_preproc(video):
    # MobileNetV2 expects 0..255 then preprocess_input
    return tf.keras.applications.mobilenet_v2.preprocess_input(video * 255.0)

def _mixup_batch(x, y, alpha=0.2, p=0.6):
    # x: (B,T,H,W,3), y: (B,C)
    def _sample_beta(a, b):
        g1 = tf.random.gamma(shape=[], alpha=a, beta=1.0)
        g2 = tf.random.gamma(shape=[], alpha=b, beta=1.0)
        return g1 / (g1 + g2)
    def _do_mix():
        lam = tf.cast(_sample_beta(alpha, alpha), tf.float32)
        idx = tf.random.shuffle(tf.range(tf.shape(x)[0]))
        x2 = tf.gather(x, idx, axis=0)
        y2 = tf.gather(y, idx, axis=0)
        xm = lam * x + (1.0 - lam) * x2
        ym = lam * y + (1.0 - lam) * y2
        return xm, ym
    return tf.cond(tf.random.uniform([]) < p, _do_mix, lambda: (x, y))

def build_dataset(csv_path: str, seq_len: int, img_size: int, batch_size: int, shuffle: bool, augment: bool):
    df = pd.read_csv(csv_path)
    paths = df["clip_dir"].tolist()
    labels = df["label"].tolist()

    # More temporal randomness during training via jitter
    jitter = 2 if augment else 0

    def gen():
        for p, y in zip(paths, labels):
            v = _load_sequence(p, seq_len, img_size, jitter=jitter)
            if augment:
                v = _augment(v)
            v = _apply_mobilenet_preproc(v)
            yield v, tf.one_hot(y, depth=len(CLASS_NAMES))

    output_sig = (
        tf.TensorSpec(shape=(seq_len, img_size, img_size, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(len(CLASS_NAMES),), dtype=tf.float32),
    )
    ds = tf.data.Dataset.from_generator(gen, output_signature=output_sig)

    if shuffle:
        ds = ds.shuffle(buffer_size=256, reshuffle_each_iteration=True)

    ds = ds.batch(batch_size, drop_remainder=False)

    # Sequence MixUp at the batch level (training only)
    if augment:
        ds = ds.map(lambda x,y: _mixup_batch(x,y, alpha=0.2, p=0.6),
                    num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds
# --- end data_utils patch ---
