"""Does the separate classifier's 82% survive on the val images that have NO near-duplicate in train?"""
import os, json
import numpy as np
from sklearn.metrics import balanced_accuracy_score
from ultralytics import YOLO
R = os.path.expanduser("~/student_behavior/runs"); W = os.path.expanduser("~/student_behavior/work")
root = os.path.join(W, "engagement2"); rows = json.load(open(os.path.join(W, "engagement2_dupcheck.json")))
dist = {r[0]: r[2] for r in rows}; names = ["engaged", "not_engaged"]
m = YOLO(os.path.join(R, "b2_engagement2_cls", "weights", "best.pt"))
recs = []
for ci, c in enumerate(names):
    files = sorted(os.listdir(os.path.join(root, "val", c)))
    paths = [os.path.join(root, "val", c, f) for f in files]
    for i in range(0, len(paths), 64):
        for f, r in zip(files[i:i+64], m.predict(paths[i:i+64], imgsz=224, verbose=False)):
            recs.append((dist.get(f, 99), ci, int(r.probs.top1)))
recs = np.array(recs)
for lo, hi, tag in ((0, 2, "near-dup d<=2"), (3, 4, "d 3-4"), (5, 6, "d 5-6"), (7, 99, "clean d>=7")):
    m_ = (recs[:, 0] >= lo) & (recs[:, 0] <= hi)
    if m_.sum() == 0: continue
    y, p = recs[m_, 1], recs[m_, 2]
    print(f"{tag:16s} n={m_.sum():4d}  acc {100*(y==p).mean():5.1f}  bal {100*balanced_accuracy_score(y,p) if len(set(y))>1 else float('nan'):5.1f}  gt1={int((y==1).sum())} pred1={int((p==1).sum())}")
