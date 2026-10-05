"""Per-image test metrics for one checkpoint -> CSV (needed for the paired significance tests)."""
import argparse, os, sys, csv
import torch
from torch.utils.data import DataLoader
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vgatnet.data import OCTASegDataset
from vgatnet.metrics import compute_all, ALL_METRICS
from vgatnet.models import build_model
from vgatnet.models.baselines import build_baseline
from vgatnet.utils import load_config, load_checkpoint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = load_config(a.config)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    d = cfg["data"]
    ds = OCTASegDataset(d["root"], d["test_split"], d["image_size"], train=False, clip_limit=d.get("clip_limit", 2.0))
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    model = build_model(cfg) if cfg["model"]["name"] == "vgatnet" else build_baseline(cfg)
    model = model.to(dev); load_checkpoint(a.checkpoint, model, map_location=dev); model.eval()
    thr = cfg["eval"].get("threshold", 0.5)
    rows = []
    with torch.no_grad():
        for b in dl:
            out = model(b["image"].to(dev))
            vp = torch.sigmoid(out["vessel"]).cpu().numpy()[0, 0] >= thr
            fp = torch.sigmoid(out["faz"]).cpu().numpy()[0, 0] >= thr
            rv = compute_all(vp, b["vessel"].numpy()[0, 0]); rf = compute_all(fp, b["faz"].numpy()[0, 0])
            rows.append({"id": b["id"][0], **{"vessel_" + k: rv[k] for k in ALL_METRICS}, **{"faz_" + k: rf[k] for k in ALL_METRICS}})
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("wrote", a.out, len(rows), "images")


if __name__ == "__main__":
    main()
