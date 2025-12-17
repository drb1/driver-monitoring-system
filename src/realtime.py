# src/realtime.py
import argparse, time, collections, sys, threading
import numpy as np
import cv2
import tensorflow as tf
from data_utils import CLASS_NAMES

# --- Custom TCN block registration (for loading best_tcn.keras) ---
CUSTOM_OBJECTS = {}
try:
    from models import TemporalBlock
    try:
        from keras.saving import register_keras_serializable
        register_keras_serializable(package="custom", name="TemporalBlock")(TemporalBlock)
    except Exception:
        pass
    CUSTOM_OBJECTS["TemporalBlock"] = TemporalBlock
except Exception:
    pass

def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--seq_len", type=int, default=16)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--source", type=str, default="0", help="0 for webcam, else path to video file")
    ap.add_argument("--smoothing", type=float, default=0.8, help="EMA [0-1), higher = smoother but slower")
    # Optional, backward-compatible extras:
    ap.add_argument("--min_seq", type=int, default=None, help="Predict once we have this many frames (pad the rest). Default = seq_len.")
    ap.add_argument("--infer_every", type=int, default=1, help="Run model every N frames to lift FPS.")
    ap.add_argument("--change_boost", type=float, default=0.4, help="When top class flips, trust new probs more (0..1).")
    ap.add_argument("--face_crop", action="store_true", help="Enable face cropping with margin.")
    ap.add_argument("--face_margin", type=float, default=0.25, help="Extra box margin around detected face (fraction of box).")
    ap.add_argument("--min_face", type=int, default=80, help="Min face size in pixels for detection.")
    return ap.parse_args()

def preprocess_frame(frame_bgr, img_size):
    # RGB + resize + MobileNetV2 preprocessing (matches training)
    frame = cv2.resize(frame_bgr, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32)
    frame = tf.keras.applications.mobilenet_v2.preprocess_input(frame)
    return frame

def pad_to_length(frames, target_len):
    if len(frames) == 0:
        raise ValueError("No frames to pad.")
    if len(frames) >= target_len:
        return frames[-target_len:]
    last = frames[-1]
    padded = list(frames) + [last] * (target_len - len(frames))
    return padded

def infer_input_specs(model):
    try:
        shp = model.input_shape  # (None, T, H, W, 3)
        if isinstance(shp, (list, tuple)):
            shp = shp[0]
        _, T, H, W, _ = shp
        return (None if T is None else int(T),
                None if H is None else int(H),
                None if W is None else int(W))
    except Exception:
        return None, None, None

class FrameGrabber:
    """Separate thread to keep reading the latest frame (reduces capture lag)."""
    def __init__(self, src):
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video source: {src}")
        # Try to keep buffer small so we get fresh frames
        try: self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception: pass
        self.lock = threading.Lock()
        self.frame = None
        self.stopped = False
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while not self.stopped:
            ok, f = self.cap.read()
            if not ok:
                self.stop()
                break
            with self.lock:
                self.frame = f

    def read(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def stop(self):
        self.stopped = True
        try: self.cap.release()
        except Exception: pass

def detect_face(frame_bgr, min_size, margin_frac):
    """Return a cropped BGR face ROI with margin; fallback to original frame if none."""
    try:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        face_cascade = cv2.CascadeClassifier(cascade_path)
        if face_cascade.empty():
            return frame_bgr
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                              minSize=(min_size, min_size))
        if len(faces) == 0:
            return frame_bgr
        # Pick largest face
        x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
        # Add margin
        mx = int(w * margin_frac)
        my = int(h * margin_frac)
        x0 = max(0, x - mx)
        y0 = max(0, y - my)
        x1 = min(frame_bgr.shape[1], x + w + mx)
        y1 = min(frame_bgr.shape[0], y + h + my)
        roi = frame_bgr[y0:y1, x0:x1]
        return roi if roi.size else frame_bgr
    except Exception:
        return frame_bgr

def main():
    args = get_args()
    min_seq = args.seq_len if args.min_seq is None else max(1, min(args.min_seq, args.seq_len))

    # Load model (register custom layer for TCN)
    with tf.keras.utils.custom_object_scope(CUSTOM_OBJECTS):
        model = tf.keras.models.load_model(args.model_path, compile=False)

    # Shape sanity
    exp_T, exp_H, exp_W = infer_input_specs(model)
    if exp_T and exp_T != args.seq_len:
        print(f"[warn] model expects seq_len={exp_T}, but --seq_len={args.seq_len}.", file=sys.stderr)
    if (exp_H and exp_H != args.img_size) or (exp_W and exp_W != args.img_size):
        print(f"[warn] model expects img_size≈({exp_H}x{exp_W}); --img_size={args.img_size}.", file=sys.stderr)

    # Video source & threaded grabber
    src = 0 if args.source == "0" else args.source
    grabber = FrameGrabber(src)

    # Rolling buffer of preprocessed frames
    seq = collections.deque(maxlen=args.seq_len)
    ema = None
    last_pred_idx = None
    last_infer = -1
    t_last = time.time()
    vis_label = "Warming up…"
    fps = 0.0

    # Warm-up one predict (CPU) to stabilize first call
    try:
        dummy = np.zeros((1, args.seq_len, args.img_size, args.img_size, 3), dtype=np.float32)
        _ = model.predict(dummy, verbose=0)
    except Exception:
        pass

    i = 0
    while True:
        frame = grabber.read()
        if frame is None:
            time.sleep(0.002)
            continue

        # Optional face crop
        if args.face_crop:
            roi = detect_face(frame, args.min_face, args.face_margin)
        else:
            roi = frame

        # Preprocess & push into sequence
        seq.append(preprocess_frame(roi, args.img_size))

        # Inference every N frames, pad if we have at least min_seq
        if (i - last_infer) >= max(1, args.infer_every) and len(seq) >= min_seq:
            last_infer = i
            frames_win = pad_to_length(list(seq), args.seq_len)
            batch = np.stack(frames_win, axis=0)[None, ...]  # (1, T, H, W, 3)

            probs = model.predict(batch, verbose=0)[0]  # (C,)
            top = int(np.argmax(probs))

            # Adaptive EMA: normal smoothing most of the time, but if class flips,
            # blend toward new probs more aggressively to reduce “stuck” feeling.
            if ema is None:
                ema = probs
            else:
                if last_pred_idx is not None and top != last_pred_idx and 0.0 <= args.change_boost <= 1.0:
                    alpha = max(0.0, min(args.smoothing, 0.99))
                    boost = max(0.0, min(args.change_boost, 1.0))
                    ema = (alpha * ema) + ((1 - alpha) * ((1 - boost) * ema + boost * probs))
                else:
                    alpha = max(0.0, min(args.smoothing, 0.99))
                    ema = alpha * ema + (1 - alpha) * probs

            pred_idx = int(np.argmax(ema))
            conf = float(ema[pred_idx])
            last_pred_idx = pred_idx
            vis_label = f"{CLASS_NAMES[pred_idx]} ({conf*100:.1f}%)"

        # FPS display from wall clock
        t_now = time.time()
        dt = t_now - t_last
        if dt > 0:
            fps = 1.0 / dt
        t_last = t_now

        # Draw UI
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 44), (0, 0, 0), -1)
        cv2.putText(frame, vis_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"FPS: {fps:.1f}", (w - 160, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("Driver Monitoring – Real-time", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break
        elif key == ord('r'):  # reset EMA/label if it feels stuck
            ema, last_pred_idx, vis_label = None, None, "Reset…"
        i += 1

    grabber.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
