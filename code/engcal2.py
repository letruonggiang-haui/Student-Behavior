#!/usr/bin/env python3
"""Hiệu chỉnh EngagementHead của checkpoint MTL mới (mtl.MTLDetectionModel).

Backbone đóng băng ở eval (BN running stats), cache đặc trưng, chỉ train MLP.

python engcal2.py --weights runs/mtl2_finetune_vn/weights/best.pt \
    --eng-data work/engagement2 --epochs 30 --out runs/mtl2_finetune_vn
"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn

from ultralytics import YOLO

import mtl  # noqa: F401 — cần cho unpickle MTLDetectionModel
from mtl import eng_dataloader, EngagementHead, as_mtl

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def extract_feats(model, loader):
    xs, ys = [], []
    for imgs, y in loader:
        model._predict_once(imgs.to(DEVICE))
        xs.append(model.eng_head.pool(model._feat).flatten(1).cpu())
        ys.append(y)
    return torch.cat(xs), torch.cat(ys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--eng-data", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--out", required=True)
    ap.add_argument("--eng-classes", type=int, default=None, help="mặc định: đếm thư mục lớp trong train")
    a = ap.parse_args()

    model = as_mtl(YOLO(a.weights).model).float().to(DEVICE).eval()
    n_cls = a.eng_classes or len([d for d in (Path(a.eng_data).expanduser() / "train").iterdir() if d.is_dir()])
    cur = model.eng_head.fc[-1].out_features if hasattr(model, "eng_head") else None
    if cur != n_cls:  # ckpt detection-only hoặc khác số lớp: gắn head mới
        model.attach_eng(n_cls)
        model.to(DEVICE).eval()
    print(f"engagement classes: {n_cls}")
    # checkpoint ultralytics lưu tham số với requires_grad=False — bật lại cho head
    for p in model.eng_head.parameters():
        p.requires_grad_(True)

    tr = eng_dataloader(a.eng_data, 32, 640, "train")
    va = eng_dataloader(a.eng_data, 32, 640, "val")
    Xtr, Ytr = extract_feats(model, tr)
    Xva, Yva = extract_feats(model, va)
    print(f"features: train {tuple(Xtr.shape)} val {tuple(Xva.shape)}")

    head = model.eng_head.fc.to(DEVICE)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    counts = torch.bincount(Ytr, minlength=n_cls).float()
    cw = (len(Ytr) / (n_cls * counts.clamp(min=1))).to(DEVICE)
    ce = nn.CrossEntropyLoss(weight=cw)
    best = 0.0
    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    for ep in range(a.epochs):
        head.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), 64):
            idx = perm[i:i + 64]
            opt.zero_grad()
            loss = ce(head(Xtr[idx].to(DEVICE)), Ytr[idx].to(DEVICE))
            loss.backward()
            opt.step()
        head.eval()
        with torch.no_grad():
            pv = head(Xva.to(DEVICE)).argmax(1).cpu()
        acc = (pv == Yva).float().mean().item()
        recalls = [((pv == c) & (Yva == c)).sum().item() / max((Yva == c).sum().item(), 1) for c in range(n_cls)]
        bal = sum(recalls) / n_cls
        print(json.dumps({"epoch": ep + 1, "val_acc": round(acc, 4), "val_bal": round(bal, 4)}), flush=True)
        acc = bal  # chọn theo balanced accuracy
        if acc >= best:
            best = acc
            torch.save({"eng_head": model.eng_head.state_dict(), "feat_idx": model.feat_idx},
                       out / "eng_calibrated.pt")
    print(f"ENGCAL_DONE best_val_bal={best:.4f}")


if __name__ == "__main__":
    main()
