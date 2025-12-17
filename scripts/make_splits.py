import os, argparse, glob, random, csv

CLASS_NAMES = ["Normal", "Aggressive", "Distracted", "Drowsy"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_root", type=str, required=True)
    ap.add_argument("--splits_dir", type=str, default="data/splits")
    ap.add_argument("--train", type=float, default=0.7)
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    os.makedirs(args.splits_dir, exist_ok=True)

    subsets = {"train": [], "val": [], "test": []}
    for cls_idx, cls in enumerate(CLASS_NAMES):
        clip_dirs = [d for d in glob.glob(os.path.join(args.processed_root, cls, "*")) if os.path.isdir(d)]
        random.shuffle(clip_dirs)
        n = len(clip_dirs)
        n_tr = int(n * args.train)
        n_v  = int(n * args.val)
        tr = clip_dirs[:n_tr]
        va = clip_dirs[n_tr:n_tr+n_v]
        te = clip_dirs[n_tr+n_v:]

        subsets["train"] += [(p, cls_idx) for p in tr]
        subsets["val"]   += [(p, cls_idx) for p in va]
        subsets["test"]  += [(p, cls_idx) for p in te]

    for name in ["train","val","test"]:
        with open(os.path.join(args.splits_dir, f"{name}.csv"), "w", newline="") as f:
            w = csv.writer(f); w.writerow(["clip_dir","label"])
            for p, y in subsets[name]:
                w.writerow([p, y])

    print("Stratified counts per subset:")
    for name in ["train","val","test"]:
        counts = {i:0 for i in range(len(CLASS_NAMES))}
        for _, y in subsets[name]: counts[y]+=1
        print(f"  {name} ->", " ".join([f"{CLASS_NAMES[i]}:{counts[i]}" for i in range(4)]))

if __name__ == "__main__":
    main()
