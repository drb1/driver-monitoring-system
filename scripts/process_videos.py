import os, argparse, glob, cv2
from tqdm import tqdm

CANONICAL = {
    "Normal":     ["normal", "Normal"],
    "Aggressive": ["aggressive", "Aggressive", "aggression", "Aggression"],
    "Distracted": ["distracted", "Distracted", "distraction", "Distraction"],
    "Drowsy":     ["drowsy", "Drowsy", "drowsiness", "Drowsiness"]
}

VIDEO_EXTS = ("*.mp4","*.mov","*.avi","*.mkv","*.MP4","*.MOV","*.AVI","*.MKV")

def sample_indexes(n_frames, seq_len):
    import numpy as np
    return np.linspace(0, max(0, n_frames-1), seq_len).astype(int).tolist()

def find_class_videos(input_root, aliases):
    found = {canon: [] for canon in aliases.keys()}
    for canon, names in aliases.items():
        for name in names:
            folder = os.path.join(input_root, name)
            if not os.path.isdir(folder):
                continue
            vids = []
            for ext in VIDEO_EXTS:
                vids.extend(glob.glob(os.path.join(folder, ext)))
            found[canon].extend(sorted(vids))
    return found

def process_one(video_path, out_dir, seq_len=16, img_size=224):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap or not cap.isOpened():
        print(f"[WARN] Failed to open: {video_path}")
        return False
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (img_size, img_size))
        frames.append(frame)
    cap.release()
    if len(frames) == 0:
        print(f"[WARN] No frames read: {video_path}")
        return False

    idxs = sample_indexes(len(frames), seq_len)
    for i, idx in enumerate(idxs):
        fp = os.path.join(out_dir, f"frame_{i:03d}.jpg")
        cv2.imwrite(fp, cv2.cvtColor(frames[idx], cv2.COLOR_RGB2BGR))
    return True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_root", type=str, required=True)
    ap.add_argument("--output_root", type=str, required=True)
    ap.add_argument("--seq_len", type=int, default=16)
    ap.add_argument("--img_size", type=int, default=224)
    args = ap.parse_args()

    found = find_class_videos(args.input_root, CANONICAL)
    print("Discovered videos per class:")
    total = 0
    for k, v in found.items():
        print(f"  {k}: {len(v)}"); total += len(v)
    if total == 0:
        print("No videos found. Check your folders under data/raw.")
        return

    for canon, vids in found.items():
        for v in tqdm(vids, desc=f"Processing {canon}"):
            stem = os.path.splitext(os.path.basename(v))[0]
            out_dir = os.path.join(args.output_root, canon, stem)
            ok = process_one(v, out_dir, seq_len=args.seq_len, img_size=args.img_size)
            if not ok:
                print(f"[WARN] Skipped: {v}")

if __name__ == "__main__":
    main()
