import argparse
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vgatnet.data.preprocess import preprocess
from vgatnet.models import build_model
from vgatnet.models.baselines import build_baseline
from vgatnet.utils import load_config, load_checkpoint

try:
    import cv2
    def imread(p): return cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    def imwrite(p, a): cv2.imwrite(p, a)
    def resize(a, s, nn=False): return cv2.resize(a, s, interpolation=cv2.INTER_NEAREST if nn else cv2.INTER_LINEAR)
except Exception:
    from PIL import Image
    def imread(p): return np.array(Image.open(p).convert("L"))
    def imwrite(p, a): Image.fromarray(a).save(p)
    def resize(a, s, nn=False): return np.array(Image.fromarray(a).resize(s, Image.NEAREST if nn else Image.BILINEAR))


def make_model(cfg):
    return build_model(cfg) if cfg["model"]["name"] == "vgatnet" else build_baseline(cfg)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    size = cfg["data"]["image_size"]
    thr = cfg["eval"].get("threshold", 0.5)

    model = make_model(cfg).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    model.eval()

    os.makedirs(os.path.join(args.output, "vessel"), exist_ok=True)
    os.makedirs(os.path.join(args.output, "faz"), exist_ok=True)

    files = [f for f in sorted(os.listdir(args.input))
             if os.path.splitext(f)[1].lower() in (".png", ".bmp", ".tif", ".jpg")]
    for fn in files:
        raw = imread(os.path.join(args.input, fn))
        h0, w0 = raw.shape[:2]
        x = preprocess(resize(raw, (size, size)), clip_limit=cfg["data"].get("clip_limit", 2.0))
        x = torch.from_numpy(x).float().unsqueeze(0).unsqueeze(0).to(device)
        out = model(x)
        vp = (torch.sigmoid(out["vessel"]).cpu().numpy()[0, 0] >= thr).astype(np.uint8) * 255
        fp = (torch.sigmoid(out["faz"]).cpu().numpy()[0, 0] >= thr).astype(np.uint8) * 255
        stem = os.path.splitext(fn)[0]
        imwrite(os.path.join(args.output, "vessel", stem + ".png"), resize(vp, (w0, h0), nn=True))
        imwrite(os.path.join(args.output, "faz", stem + ".png"), resize(fp, (w0, h0), nn=True))
        print(f"predicted {fn}")
    print(f"Masks saved under {args.output}/")


if __name__ == "__main__":
    main()
