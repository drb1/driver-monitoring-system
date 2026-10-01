# src/build_audio_cache.py
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from audio_utils import extract_wav_ffmpeg, save_logmel_npy

RAW_ROOT = "data/raw"  # your mp4s live here
MEL_ROOT = "data/processed_audio_mels"
TMP_WAV  = "data/tmp_audio_wav"


def mp4_from_clipdir(clip_dir: str):
    """
    CSV clip_dir points to processed frame folders like:
      data/processed/Normal/Normal12
    Infer class + clip_id from that, then map to:
      data/raw/normal/Normal12.mp4
    NOTE: your raw class folders are lowercase and one is 'drowsiness'
    """
    parts = clip_dir.replace("\\", "/").split("/")
    cls = parts[-2]     # e.g., Normal
    clip_id = parts[-1] # e.g., Normal12

    raw_folder = {
        "Normal": "normal",
        "Aggressive": "aggressive",
        "Distracted": "distraction",
        "Drowsy": "drowsiness",
    }.get(cls)

    if raw_folder is None:
        raise ValueError(f"Unknown class folder in clip_dir: {clip_dir} (cls={cls})")

    mp4_path = os.path.join(RAW_ROOT, raw_folder, f"{clip_id}.mp4")
    return mp4_path, cls, clip_id


def ensure_parent_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def save_silence_mel(out_npy: str, target_frames: int = 800, mel_bins: int = 128):
    """
    Fallback: save a silence mel so multimodal pipeline does not break
    when video has no usable audio.
    """
    ensure_parent_dir(out_npy)
    mel = np.zeros((target_frames, mel_bins), dtype=np.float32)
    np.save(out_npy, mel)


def process_csv(csv_path: str, target_frames: int = 800):
    df = pd.read_csv(csv_path)

    clip_dirs = df["clip_dir"].tolist()
    for clip_dir in tqdm(clip_dirs, desc=f"Audio cache: {csv_path}"):
        mp4, cls, clip_id = mp4_from_clipdir(clip_dir)

        wav_path = os.path.join(TMP_WAV, cls, f"{clip_id}.wav")
        out_npy  = os.path.join(MEL_ROOT, cls, f"{clip_id}.npy")

        if os.path.exists(out_npy):
            continue

        # 1) If mp4 missing, write silence and continue
        if not os.path.exists(mp4):
            print(f"[warn] Missing MP4: {mp4} -> saving silence mel for {cls}/{clip_id}")
            save_silence_mel(out_npy, target_frames=target_frames, mel_bins=128)
            continue

        # 2) Try extracting audio (safe: returns True/False)
        ensure_parent_dir(wav_path)
        ok = extract_wav_ffmpeg(mp4, wav_path, sr=16000)

        # 3) If extraction failed (no audio / decode error), write silence mel
        if not ok:
            print(f"[warn] No usable audio in: {mp4} -> saving silence mel for {cls}/{clip_id}")
            save_silence_mel(out_npy, target_frames=target_frames, mel_bins=128)
            continue

        # 4) Convert WAV -> log-mel npy
        try:
            ensure_parent_dir(out_npy)
            save_logmel_npy(wav_path, out_npy, target_frames=target_frames)
        except Exception as e:
            # Any mel extraction error -> fallback to silence
            print(f"[warn] Mel extraction failed for: {wav_path} ({e}) -> saving silence mel")
            save_silence_mel(out_npy, target_frames=target_frames, mel_bins=128)
            continue


def main():
    os.makedirs(MEL_ROOT, exist_ok=True)
    os.makedirs(TMP_WAV, exist_ok=True)

    for p in ["data/splits/train.csv", "data/splits/val.csv", "data/splits/test.csv"]:
        if os.path.exists(p):
            process_csv(p, target_frames=800)
        else:
            print(f"[warn] CSV not found, skipping: {p}")

    print("Done. Audio mels saved to:", MEL_ROOT)


if __name__ == "__main__":
    main()
