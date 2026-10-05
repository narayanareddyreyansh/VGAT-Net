import argparse
import os
import sys
import json

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vgatnet.data import OCTASegDataset
from vgatnet.losses import TopologyPreservingHybridLoss
from vgatnet.metrics import MetricAccumulator
from vgatnet.models import build_model
from vgatnet.models.baselines import build_baseline
from vgatnet.utils import (set_seed, load_config, save_checkpoint,
                           AverageMeter, EarlyStopping)


def make_model(cfg):
    return build_model(cfg) if cfg["model"]["name"] == "vgatnet" else build_baseline(cfg)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    meter = AverageMeter()
    acc = MetricAccumulator()
    for batch in loader:
        img = batch["image"].to(device)
        tgt = {"vessel": batch["vessel"].to(device), "faz": batch["faz"].to(device)}
        out = model(img)
        loss, _ = criterion(out, tgt)
        meter.update(loss.item(), img.size(0))
        vp = torch.sigmoid(out["vessel"]).cpu().numpy()
        for i in range(vp.shape[0]):
            acc.update(vp[i, 0], batch["vessel"][i, 0].numpy())
    return meter.avg, acc.summary()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg.get("seed", 0)
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    d = cfg["data"]
    train_ds = OCTASegDataset(d["root"], d["train_split"], d["image_size"], train=True,
                              clip_limit=d.get("clip_limit", 2.0), seed=seed)
    val_ds = OCTASegDataset(d["root"], d["val_split"], d["image_size"], train=False,
                            clip_limit=d.get("clip_limit", 2.0))
    o = cfg["optim"]
    train_loader = DataLoader(train_ds, batch_size=o["batch_size"], shuffle=True,
                              num_workers=d.get("num_workers", 4), drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            num_workers=d.get("num_workers", 4))

    model = make_model(cfg).to(device)
    criterion = TopologyPreservingHybridLoss(cfg["loss"]["lambda1"], cfg["loss"]["lambda2"],
                                             cfg["loss"].get("topo_iters", 10))
    optimizer = torch.optim.Adam(model.parameters(), lr=o["lr"], weight_decay=o["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=o["epochs"])
    scaler = torch.cuda.amp.GradScaler(enabled=o.get("amp", False) and device == "cuda")
    stopper = EarlyStopping(o.get("early_stopping_patience", 15))

    out_dir = f"{cfg['output_dir']}/seed{seed}"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config_used.json"), "w") as f:
        json.dump({**cfg, "seed": seed}, f, indent=2, default=str)

    start_epoch, best_val = 0, float("inf")
    log = []
    for epoch in range(start_epoch, o["epochs"]):
        model.train()
        meter = AverageMeter()
        for batch in train_loader:
            img = batch["image"].to(device)
            tgt = {"vessel": batch["vessel"].to(device), "faz": batch["faz"].to(device)}
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                out = model(img)
                loss, _ = criterion(out, tgt)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            meter.update(loss.item(), img.size(0))
        scheduler.step()

        val_loss, val_metrics = validate(model, val_loader, criterion, device)
        improved = stopper.step(val_loss)
        log.append({"epoch": epoch, "train_loss": meter.avg, "val_loss": val_loss,
                    "val_DSC": val_metrics["DSC"][0]})
        print(f"[{epoch+1}/{o['epochs']}] train={meter.avg:.4f} val={val_loss:.4f} "
              f"val_DSC={val_metrics['DSC'][0]:.2f}", flush=True)

        if improved:
            best_val = val_loss
            save_checkpoint({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                             "epoch": epoch, "val_loss": val_loss, "config": cfg, "seed": seed},
                            os.path.join(out_dir, "best.pth"))
        save_checkpoint({"model": model.state_dict(), "epoch": epoch, "seed": seed},
                        os.path.join(out_dir, "last.pth"))
        with open(os.path.join(out_dir, "train_log.json"), "w") as f:
            json.dump(log, f, indent=2)

        if stopper.should_stop:
            print(f"Early stopping at epoch {epoch+1} (best val {best_val:.4f})")
            break

    print(f"Done. Best checkpoint: {os.path.join(out_dir, 'best.pth')}")


if __name__ == "__main__":
    main()
