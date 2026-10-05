import argparse
import os
import sys
import json
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vgatnet.data import OCTASegDataset
from vgatnet.metrics import MetricAccumulator, ALL_METRICS
from vgatnet.models import build_model
from vgatnet.models.baselines import build_baseline
from vgatnet.utils import load_config, load_checkpoint


def make_model(cfg):
    return build_model(cfg) if cfg["model"]["name"] == "vgatnet" else build_baseline(cfg)


@torch.no_grad()
def evaluate_one(cfg, ckpt_path, device):
    d = cfg["data"]
    ds = OCTASegDataset(d["root"], d["test_split"], d["image_size"], train=False,
                        clip_limit=d.get("clip_limit", 2.0))
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=d.get("num_workers", 4))
    model = make_model(cfg).to(device)
    load_checkpoint(ckpt_path, model, map_location=device)
    model.eval()
    thr = cfg["eval"].get("threshold", 0.5)

    ves_acc, faz_acc = MetricAccumulator(), MetricAccumulator()
    for batch in loader:
        out = model(batch["image"].to(device))
        vp = (torch.sigmoid(out["vessel"]).cpu().numpy()[0, 0] >= thr)
        fp = (torch.sigmoid(out["faz"]).cpu().numpy()[0, 0] >= thr)
        ves_acc.update(vp, batch["vessel"].numpy()[0, 0])
        faz_acc.update(fp, batch["faz"].numpy()[0, 0])
    return ves_acc.summary(), faz_acc.summary()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ves_runs, faz_runs = [], []
    for ck in args.checkpoints:
        v, f = evaluate_one(cfg, ck, device)
        ves_runs.append({m: v[m][0] for m in ALL_METRICS})
        faz_runs.append({m: f[m][0] for m in ALL_METRICS})
        print(f"{ck}: vessel DSC={v['DSC'][0]:.2f}  FAZ DSC={f['DSC'][0]:.2f}")

    def agg(runs):
        return {m: (float(np.nanmean([r[m] for r in runs])),
                    float(np.nanstd([r[m] for r in runs]))) for m in ALL_METRICS}

    result = {"vessel_mean_std": agg(ves_runs), "faz_mean_std": agg(faz_runs),
              "n_seeds": len(args.checkpoints)}
    print("\n=== Vessel (mean +/- std over seeds) ===")
    for m in ALL_METRICS:
        mu, sd = result["vessel_mean_std"][m]
        print(f"  {m:5s}: {mu:6.2f} +/- {sd:.2f}")
    print("=== FAZ (mean +/- std over seeds) ===")
    for m in ALL_METRICS:
        mu, sd = result["faz_mean_std"][m]
        print(f"  {m:5s}: {mu:6.2f} +/- {sd:.2f}")

    out = args.out or os.path.join(cfg["output_dir"], "test_results.json")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
