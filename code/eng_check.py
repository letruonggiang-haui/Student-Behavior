"""Is the engagement head predicting anything, or just the majority class?"""
import os, json
from collections import Counter
import numpy as np, torch
from ultralytics import YOLO
import mtl
from mtl import eng_dataloader

R = os.path.expanduser("~/student_behavior/runs")
W = os.path.expanduser("~/student_behavior/work")
root = os.path.join(W, "engagement2")
dev = "cuda" if torch.cuda.is_available() else "cpu"

# class balance of the validation split
for split in ("train", "val"):
    d = os.path.join(root, split)
    if os.path.isdir(d):
        c = {k: len(os.listdir(os.path.join(d, k))) for k in sorted(os.listdir(d))
             if os.path.isdir(os.path.join(d, k))}
        n = sum(c.values())
        print(split, c, "majority = %.2f%%" % (100 * max(c.values()) / n) if n else "")

for name in ("mtl_scb5_s0", "mtl_l_s0"):
    w = os.path.join(R, name, "weights", "best.pt")
    e = os.path.join(R, name, "eng_calibrated.pt")
    if not os.path.exists(w):
        continue
    model = YOLO(w).model.float().to(dev).eval()
    loaded = False
    if os.path.exists(e):
        stt = torch.load(e, map_location="cpu", weights_only=False)
        model.eng_head.load_state_dict(stt["eng_head"]); model.to(dev); loaded = True
    preds, ys = [], []
    with torch.no_grad():
        for imgs, y in eng_dataloader(root, 32, 640, "val"):
            model._predict_once(imgs.to(dev))
            logits = model.eng_head(model._feat)
            preds += logits.argmax(1).cpu().tolist(); ys += y.tolist()
    preds, ys = np.array(preds), np.array(ys)
    acc = (preds == ys).mean() * 100
    print(f"\n{name}: calibrated head loaded={loaded}  acc={acc:.2f}")
    print("   prediction distribution:", dict(Counter(preds.tolist())))
    print("   ground-truth distribution:", dict(Counter(ys.tolist())))
    for c in sorted(set(ys.tolist())):
        m = ys == c
        print(f"   class {c}: recall {100*(preds[m]==c).mean():.1f}%  (n={m.sum()})")
