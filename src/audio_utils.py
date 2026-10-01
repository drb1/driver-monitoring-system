# src/audio_utils.py
import os, subprocess
import numpy as np
import tensorflow as tf
import subprocess

def extract_wav_ffmpeg(mp4_path: str, wav_out: str, sr: int = 16000) -> bool:
    """
    Extract mono WAV @ sr from mp4.
    Returns True if audio extracted, False if mp4 has no audio (or extraction failed).
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", mp4_path,
        "-map", "0:a:0?",          # <- IMPORTANT: audio stream is optional
        "-ac", "1",
        "-ar", str(sr),
        "-vn",
        wav_out
    ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0:
        # Most likely no audio stream or decode error
        # Print a short message so you can see which file is failing
        err = p.stderr.decode("utf-8", errors="ignore")
        print(f"[ffmpeg] audio extract failed for: {mp4_path}\n{err.splitlines()[-8:]}")
        return False
    return True

def load_wav_tf(wav_path: str, sr: int = 16000):
    audio = tf.io.read_file(wav_path)
    wav, sample_rate = tf.audio.decode_wav(audio, desired_channels=1)
    wav = tf.squeeze(wav, axis=-1)  # (N,)
    # decode_wav returns float32 in [-1,1]
    return wav, sample_rate

def log_mel_spectrogram(
    wav: tf.Tensor,
    sr: int = 16000,
    n_fft: int = 1024,
    hop: int = 160,          # 10ms @16k
    win: int = 400,          # 25ms @16k
    n_mels: int = 128,
    fmin: float = 80.0,
    fmax: float = 7600.0,
):
    stft = tf.signal.stft(wav, frame_length=win, frame_step=hop, fft_length=n_fft,
                          window_fn=tf.signal.hann_window, pad_end=True)
    spec = tf.abs(stft) ** 2  # power
    mel_w = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=n_mels,
        num_spectrogram_bins=spec.shape[-1],
        sample_rate=sr,
        lower_edge_hertz=fmin,
        upper_edge_hertz=fmax
    )
    mel = tf.matmul(spec, mel_w)
    logmel = tf.math.log(mel + 1e-6)
    return logmel  # (time, n_mels)

def pad_or_trim_time(x: tf.Tensor, target_frames: int):
    """x: (T, n_mels) -> (target_frames, n_mels)"""
    t = tf.shape(x)[0]
    x = tf.cond(
        t >= target_frames,
        lambda: x[:target_frames],
        lambda: tf.pad(x, [[0, target_frames - t], [0, 0]])
    )
    return x

def save_logmel_npy(wav_path: str, out_npy: str, target_frames: int = 800):
    wav, sr = load_wav_tf(wav_path)
    # ensure sr matches expected; if not, still works but fmax may shift slightly
    logmel = log_mel_spectrogram(wav, sr=int(sr.numpy()))
    logmel = pad_or_trim_time(logmel, target_frames)
    arr = logmel.numpy().astype(np.float32)
    os.makedirs(os.path.dirname(out_npy), exist_ok=True)
    np.save(out_npy, arr)
