# src/check_leakage.py
import os, glob, argparse, collections, hashlib, random
from typing import List, Dict, Tuple, Set
import numpy as np
import pandas as pd
import cv2

print("=== check_leakage.py: starting ===", flush=True)

if __name__ == "__main__":
    try:
        # import the real main from your file if needed
        # 
        from check_leakage import main  # <- only if you split files
        main()
    except Exception as e:
        import traceback, sys
        print("ERROR:", e, flush=True)
        traceback.print_exc()
        sys.exit(1)

def read_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "clip_dir" not in df.columns or "label" not in df.columns:
        raise ValueError(f"{csv_path} must have columns: clip_dir,label")
    # normalize clip paths
    df["clip_dir"] = df["clip_dir"].apply(lambda p: os.path.realpath(os.path.expanduser(str(p))))
    return df

def pick_group_id(clip_dir: str, mode: str, subject_depth: int, regex: str):
    """
    mode:
      - 'parent'   -> parent folder name of the clip_dir
      - 'basename' -> the clip folder name itself
      - 'regex'    -> first capture group from --regex
    subject_depth:
      - negative index from end of path parts (e.g., -2 = parent, -1 = basename)
        used only if mode='depth'
    """
    if mode == "parent":
        return os.path.basename(os.path.dirname(clip_dir))
    elif mode == "basename":
        return os.path.basename(clip_dir)
    elif mode == "depth":
        parts = [p for p in clip_dir.split(os.sep) if p]
        try:
            return parts[subject_depth]
        except Exception:
            return os.path.basename(os.path.dirname(clip_dir))
    elif mode == "regex":
        import re
        m = re.search(regex, clip_dir)
        return m.group(1) if m else os.path.basename(os.path.dirname(clip_dir))
    else:
        # fallback: parent
        return os.path.basename(os.path.dirname(clip_dir))

def load_frame_paths(clip_dir: str) -> List[str]:
    return sorted(glob.glob(os.path.join(clip_dir, "frame_*.jpg")))

def ahash_image(fp: str, size: int = 8) -> str:
    """Simple average hash (aHash) — compact & fast; good enough to catch duplicates."""
    img = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return ""
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    m = img.mean()
    bits = (img > m).astype(np.uint8).flatten()
    # pack 64 bits into hex string
    bitstring = ''.join('1' if b else '0' for b in bits.tolist())
    return f"{int(bitstring, 2):016x}"

def sample_frame_hashes(clip_dir: str, num_frames: int = 3) -> List[str]:
    fps = load_frame_paths(clip_dir)
    if len(fps) == 0:
        return []
    idxs = np.linspace(0, len(fps)-1, num_frames).astype(int)
    hashes = []
    for i in idxs:
        h = ahash_image(fps[i])
        if h:
            hashes.append(h)
    return hashes

def describe_overlap(name_a: str, set_a: Set[str], name_b: str, set_b: Set[str], max_show: int = 10):
    inter = sorted(set_a & set_b)
    print(f"\n[Overlap] {name_a} ∩ {name_b}: {len(inter)}")
    if inter:
        for s in inter[:max_show]:
            print("  -", s)
        if len(inter) > max_show:
            print(f"  ... and {len(inter) - max_show} more")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", type=str, default="data/splits/train.csv")
    ap.add_argument("--val_csv",   type=str, default="data/splits/val.csv")
    ap.add_argument("--test_csv",  type=str, default="data/splits/test.csv")
    # Group/subject extraction options
    ap.add_argument("--group_mode", type=str, choices=["parent","basename","depth","regex"],
                    default="parent",
                    help="How to derive a group/subject id from clip_dir for leakage checks.")
    ap.add_argument("--subject_depth", type=int, default=-2,
                    help="When --group_mode=depth, which path index to use (e.g., -2 = parent).")
    ap.add_argument("--regex", type=str, default=r".*/([^/]+)/[^/]+$",
                    help="When --group_mode=regex, 1st capture group is the subject id.")
    # Content duplicate scan
    ap.add_argument("--max_clips_per_split_for_hash", type=int, default=200,
                    help="Limit clips per split when hashing frames (performance).")
    ap.add_argument("--frames_per_clip", type=int, default=3,
                    help="How many frames to hash per clip for duplicate detection.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    pd.set_option("display.max_rows", 200)

    # ---------- Load splits ----------
    tr = read_csv(args.train_csv)
    va = read_csv(args.val_csv)
    te = read_csv(args.test_csv)

    print("=== Split sizes ===")
    print("Train:", len(tr), "clips")
    print("Val:", len(va), "clips")
    print("Test:", len(te), "clips")

    # ---------- 1) Exact clip_dir overlap ----------
    tr_set = set(tr.clip_dir.tolist())
    va_set = set(va.clip_dir.tolist())
    te_set = set(te.clip_dir.tolist())

    print("\n=== Exact clip_dir overlaps ===")
    describe_overlap("train", tr_set, "val", va_set)
    describe_overlap("train", tr_set, "test", te_set)
    describe_overlap("val", va_set, "test", te_set)

    # ---------- 2) Group/subject leakage ----------
    for df, name in [(tr,"train"), (va,"val"), (te,"test")]:
        df["group_id"] = df["clip_dir"].apply(lambda p: pick_group_id(
            p, args.group_mode, args.subject_depth, args.regex
        ))

    print("\n=== Group/Subject counts per split (top 20) ===")
    for df, name in [(tr,"train"), (va,"val"), (te,"test")]:
        cnt = df["group_id"].value_counts()
        print(f"\n[{name}] unique groups: {cnt.shape[0]}")
        print(cnt.head(20))

    tr_groups = set(tr.group_id.tolist())
    va_groups = set(va.group_id.tolist())
    te_groups = set(te.group_id.tolist())

    print("\n=== Group/Subject overlaps ===")
    describe_overlap("train(groups)", tr_groups, "val(groups)", va_groups)
    describe_overlap("train(groups)", tr_groups, "test(groups)", te_groups)
    describe_overlap("val(groups)", va_groups, "test(groups)", te_groups)

    # ---------- 3) Content-level duplicates via perceptual hash ----------
    # (hash a few frames from up to N clips per split)
    def pick_subset(df: pd.DataFrame, n: int) -> List[str]:
        clips = df.clip_dir.tolist()
        if len(clips) <= n:
            return clips
        return random.sample(clips, n)

    dup_map = collections.defaultdict(list)  # hash -> [(split, clip_dir)]
    print("\n=== Perceptual-hash scan (aHash) ===")
    for name, df in [("train", tr), ("val", va), ("test", te)]:
        subset = pick_subset(df, args.max_clips_per_split_for_hash)
        print(f"Hashing {len(subset)} clips from {name}...")
        for cd in subset:
            try:
                hs = sample_frame_hashes(cd, num_frames=args.frames_per_clip)
                for h in set(hs):  # unique per clip
                    if h:
                        dup_map[h].append((name, cd))
            except Exception as e:
                print(f"[WARN] hashing failed for {cd}: {e}")

    # Report cross-split collisions (hash identical across different splits)
    collisions = []
    for h, items in dup_map.items():
        splits = {s for s,_ in items}
        if len(splits) >= 2:
            collisions.append((h, items))
    print(f"\n[Duplicates across splits by aHash] Found {len(collisions)} hash keys colliding:")
    for h, items in collisions[:20]:
        print(f"- hash={h} ->")
        for s, cd in items:
            print(f"    [{s}] {cd}")
    if len(collisions) > 20:
        print(f"... and {len(collisions)-20} more")

    print("\n=== Summary ===")
    probs = []
    if (tr_set & va_set) or (tr_set & te_set) or (va_set & te_set):
        probs.append("Exact clip_dir overlap exists.")
    if (tr_groups & va_groups) or (tr_groups & te_groups) or (va_groups & te_groups):
        probs.append("Group/subject overlap exists (depends on group extraction).")
    if len(collisions) > 0:
        probs.append("Frame-level duplicate content found across splits (aHash collisions).")

    if probs:
        print("Potential leakage signals:\n - " + "\n - ".join(probs))
        print("\nIf group detection seems wrong, re-run with a different --group_mode (parent/basename/depth/regex)")
        print("Example: --group_mode depth --subject_depth -2   or   --group_mode regex --regex '.*?/(subject\\d+)/'")
    else:
        print("No leakage detected by these checks.")
