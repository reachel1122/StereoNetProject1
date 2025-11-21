#!/usr/bin/env python
import argparse, yaml, torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import optim
from pathlib import Path
from src.models.stereonet_base import StereoNetBase
from src.datasets.kitti2015 import Kitti2015

def get_loader(cfg):
    name = cfg["dataset"]["name"].lower()
    root = cfg["dataset"]["root"]
    h, w = cfg["dataset"]["crop_hw"]
    bs   = cfg["train"]["batch_size"]
    nw   = cfg["train"]["num_workers"]
    if name in ("kitti","kitti2015","kitti_2015"):
        ds = Kitti2015(root=root, split="train", crop_hw=(h,w))
    else:
        raise ValueError(f"unknown dataset: {name}")
    return DataLoader(ds, batch_size=bs, shuffle=True, num_workers=nw, drop_last=True)

def main(cfg):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    net = StereoNetBase(**cfg["model"]).to(device).train()
    opt = optim.Adam(net.parameters(), lr=cfg["train"]["lr"])
    loader = get_loader(cfg)
    outdir = Path(cfg.get("out_dir","runs")); outdir.mkdir(parents=True, exist_ok=True)

    step = 0
    for epoch in range(cfg["train"]["epochs"]):
        for L,R,gt in loader:
            step += 1
            L,R,gt = L.to(device), R.to(device), gt.to(device)
            pred = net(L,R)
            loss = F.smooth_l1_loss(pred, gt)
            opt.zero_grad(); loss.backward(); opt.step()
            if step % 10 == 0:
                print(f"epoch {epoch+1} step {step} loss={loss.item():.4f}")

        # 每个 epoch 存一次
        ckpt = {"epoch": epoch+1, "state_dict": net.state_dict(), "opt": opt.state_dict()}
        torch.save(ckpt, outdir/f"teacher_e{epoch+1}.pt")
    print("✅ train done, saved to", outdir)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    with open(args.config, "r") as f: cfg = yaml.safe_load(f)
    main(cfg)
