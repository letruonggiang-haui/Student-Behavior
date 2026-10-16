"""Is the separate engagement classifier's 83% real, or another artefact?"""
import os
from collections import Counter
import numpy as np
from ultralytics import YOLO
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix

R = os.path.expanduser("~/student_behavior/runs")
W = os.path.expanduser("~/student_behavior/work")
root = os.path.join(W, "engagement2")
w = os.path.join(R, "b2_engagement2_cls", "weights", "best.pt")
m = YOLO(w)

names = sorted(os.listdir(os.path.join(root, "val")))
print("class order:", names)
ys, ps = [], []
for ci, c in enumerate(names):
    d = os.path.join(root, "val", c)
    files = sorted(os.path.join(d, f) for f in os.listdir(d))
    for i in range(0, len(files), 64):
        for r in m.predict(files[i:i + 64], imgsz=224, verbose=False):
            ps.append(int(r.probs.top1)); ys.append(ci)
ys, ps = np.array(ys), np.array(ps)
print("n =", len(ys))
print("accuracy        %.2f" % (100 * (ys == ps).mean()))
print("balanced acc    %.2f" % (100 * balanced_accuracy_score(ys, ps)))
print("macro F1        %.2f" % (100 * f1_score(ys, ps, average="macro")))
print("majority floor  %.2f" % (100 * max(np.bincount(ys)) / len(ys)))
print("pred dist", dict(Counter(ps.tolist())), " gt dist", dict(Counter(ys.tolist())))
print("confusion (rows=gt):\n", confusion_matrix(ys, ps))
for ci, c in enumerate(names):
    msk = ys == ci
    print("  %-14s recall %.1f%%  (n=%d)" % (c, 100 * (ps[msk] == ci).mean(), msk.sum()))
