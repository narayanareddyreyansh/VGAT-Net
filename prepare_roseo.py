"""Convert the ROSE-O release into the images/vessel/faz layout and write the split files.

ROSE-O (Hao et al., IEEE TMI 2022) ships 39 participants with SVC, DVC and IVC angiograms and,
per participant, one vessel mask and one FAZ mask that are aligned with the SVC angiogram.
The released partition (30 training / 9 test participants) is kept; 5 training participants
are held out for validation with a fixed seed.

Usage:
    python scripts/prepare_roseo.py --src /path/to/ROSE-O --dst datasets/ROSE-O --seed 0
"""
import argparse
import os
import random

import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="folder containing train/ and test/ of the ROSE-O release")
    ap.add_argument("--dst", default="datasets/ROSE-O")
    ap.add_argument("--splits", default="data/splits")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-val", type=int, default=5)
    a = ap.parse_args()

    for sub in ("images", "vessel", "faz"):
        os.makedirs(os.path.join(a.dst, sub), exist_ok=True)
    ids = {"train": [], "test": []}
    for split, n, pref in (("train", 30, "tr"), ("test", 9, "te")):
        for i in range(1, n + 1):
            src = f"{i:02d}"
            _id = f"{pref}{i:02d}"
            img = cv2.imread(os.path.join(a.src, split, "img", "SVC", src + ".tif"), cv2.IMREAD_GRAYSCALE)
            ves = cv2.imread(os.path.join(a.src, split, "gt", "vessel", src + ".png"), cv2.IMREAD_GRAYSCALE)
            faz = cv2.imread(os.path.join(a.src, split, "gt", "FAZ", src + ".png"), cv2.IMREAD_GRAYSCALE)
            if img is None or ves is None or faz is None:
                raise FileNotFoundError(f"{split}/{src}")
            cv2.imwrite(os.path.join(a.dst, "images", _id + ".png"), img)
            cv2.imwrite(os.path.join(a.dst, "vessel", _id + ".png"), ((ves > 127) * 255).astype("uint8"))
            cv2.imwrite(os.path.join(a.dst, "faz", _id + ".png"), ((faz > 127) * 255).astype("uint8"))
            ids[split].append(_id)

    rng = random.Random(a.seed)
    tr = ids["train"][:]
    rng.shuffle(tr)
    val, train = sorted(tr[: a.n_val]), sorted(tr[a.n_val:])
    os.makedirs(a.splits, exist_ok=True)
    for name, lst in (("train", train), ("val", val), ("test", ids["test"])):
        with open(os.path.join(a.splits, f"roseo_{name}.txt"), "w") as f:
            f.write("\n".join(lst) + "\n")
    print(f"train {len(train)}  val {len(val)} {val}  test {len(ids['test'])}")


if __name__ == "__main__":
    main()
