import argparse
import os
import sys
import csv
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vgatnet.biomarkers import extract_all

try:
    import cv2
    def imread(p): return cv2.imread(p, cv2.IMREAD_GRAYSCALE)
except Exception:
    from PIL import Image
    def imread(p): return np.array(Image.open(p).convert("L"))

FIELDS = ["VD_percent", "PD_percent", "FAZ_area_mm2", "FAZ_perimeter_mm", "FAZ_circularity"]


def bland_altman(pred, ref):
    pred, ref = np.asarray(pred, float), np.asarray(ref, float)
    diff = pred - ref
    bias = float(np.mean(diff))
    sd = float(np.std(diff, ddof=1)) if len(diff) > 1 else 0.0
    return bias, (bias - 1.96 * sd, bias + 1.96 * sd)


def pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vessel-dir", required=True)
    ap.add_argument("--faz-dir", required=True)
    ap.add_argument("--fov-mm", type=float, default=3.0)
    ap.add_argument("--size-px", type=int, default=512)
    ap.add_argument("--out", default="biomarkers.csv")
    ap.add_argument("--reference", default=None, help="CSV with columns: id," + ",".join(FIELDS))
    args = ap.parse_args()

    rows = {}
    for fn in sorted(os.listdir(args.vessel_dir)):
        stem = os.path.splitext(fn)[0]
        vp = imread(os.path.join(args.vessel_dir, fn)) / 255.0
        fp_path = os.path.join(args.faz_dir, fn)
        if not os.path.exists(fp_path):
            continue
        fp = imread(fp_path) / 255.0
        rows[stem] = extract_all(vp, fp, args.fov_mm, args.size_px)

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id"] + FIELDS)
        for _id, r in rows.items():
            w.writerow([_id] + [f"{r[k]:.6f}" for k in FIELDS])
    print(f"Wrote {len(rows)} rows -> {args.out}")

    if args.reference and os.path.exists(args.reference):
        ref = {}
        with open(args.reference) as f:
            for r in csv.DictReader(f):
                ref[r["id"]] = {k: float(r[k]) for k in FIELDS}
        common = [i for i in rows if i in ref]
        print(f"\nAgreement on {len(common)} paired samples (Table 6 format):")
        print(f"{'Biomarker':22s} {'Pearson r':>10s} {'MAE':>10s} {'Bias (95% LoA)':>24s}")
        for k in FIELDS:
            p = [rows[i][k] for i in common]
            g = [ref[i][k] for i in common]
            r = pearson(p, g)
            mae = float(np.mean(np.abs(np.array(p) - np.array(g)))) if common else float("nan")
            bias, (lo, hi) = bland_altman(p, g) if common else (float("nan"), (float("nan"),) * 2)
            print(f"{k:22s} {r:10.3f} {mae:10.4f}   {bias:+.4f} ({lo:+.3f},{hi:+.3f})")


if __name__ == "__main__":
    main()
