"""Evaluate the engagement head AS TRAINED JOINTLY in best.pt (no engcal2 refit)."""
import os, sys, json
from collections import Counter
import numpy as np, torch
from sklearn.metrics import balanced_accuracy_score
from ultralytics import YOLO
import mtl
from mtl import eng_dataloader
R = os.path.expanduser("~/student_behavior/runs"); W = os.path.expanduser("~/student_behavior/work")
root = os.path.join(W, "engagement2"); dev = "cuda"
for n in sys.argv[1:]:
    w = os.path.join(R, n, "weights", "best.pt")
    if not os.path.exists(w): print(n, "missing"); continue
    model = YOLO(w).model.float().to(dev).eval()
    res = {}
    for split in ("train", "val"):
        ps, ys, logit_std = [], [], []
        with torch.no_grad():
            for imgs, y in eng_dataloader(root, 32, 640, split):
                model._predict_once(imgs.to(dev)); lg = model.eng_head(model._feat)
                ps += lg.argmax(1).cpu().tolist(); ys += y.tolist(); logit_std.append(lg.std(0).mean().item())
        ps, ys = np.array(ps), np.array(ys)
        res[split] = {"acc": round(100*(ps==ys).mean(),2), "bal": round(100*balanced_accuracy_score(ys,ps),2),
                      "pred": dict(Counter(ps.tolist())), "logit_std": round(float(np.mean(logit_std)),4)}
    print(n, "RAW-HEAD", json.dumps(res), flush=True)
