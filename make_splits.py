import argparse
import os
import random


def collect_ids(ids_dir):
    ids = []
    for fn in sorted(os.listdir(ids_dir)):
        stem, ext = os.path.splitext(fn)
        if ext.lower() in (".png", ".bmp", ".tif", ".jpg"):
            ids.append(stem)
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids-dir", default=None)
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ratios", default="0.70,0.15,0.15")
    args = ap.parse_args()

    if args.ids_dir and os.path.isdir(args.ids_dir):
        ids = collect_ids(args.ids_dir)
    else:
        ids = [str(i) for i in range(10001, 10501)]        # canonical OCTA-500 subject ids

    r_tr, r_va, r_te = [float(x) for x in args.ratios.split(",")]
    rng = random.Random(args.seed)
    rng.shuffle(ids)
    n = len(ids)
    n_tr = int(round(n * r_tr))
    n_va = int(round(n * r_va))
    splits = {"train": ids[:n_tr], "val": ids[n_tr:n_tr + n_va], "test": ids[n_tr + n_va:]}

    os.makedirs(os.path.dirname(args.out_prefix) or ".", exist_ok=True)
    for name, lst in splits.items():
        path = f"{args.out_prefix}_{name}.txt"
        with open(path, "w") as f:
            f.write("\n".join(lst) + "\n")
        print(f"{path}: {len(lst)} ids")


if __name__ == "__main__":
    main()
