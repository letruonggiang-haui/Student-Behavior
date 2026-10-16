"""Are the detector's false positives correct detections of students hrw never labelled?

Matches images across packs by file name (the mosaiced and original releases share
names and labels; md5 differs because the faces are obscured).
"""
import os, glob, json
import numpy as np
from ultralytics import YOLO

W = os.path.expanduser("~/student_behavior/work")
R = os.path.expanduser("~/student_behavior/runs")
S5 = os.path.expanduser("~/student_behavior/scb5")
OTHER = ["SCB_BowTurnHead_20250509", "SCB5-Discuss-2024-9-17"]

other = {}
for d in OTHER:
    for lp in glob.glob(os.path.join(S5, d) + "/**/labels/**/*.txt", recursive=True):
        name = os.path.splitext(os.path.basename(lp))[0]
        bb = []
        for line in open(lp):
            t = line.split()
            if len(t) >= 5:
                bb.append([float(x) for x in t[1:5]])
        if bb:
            other.setdefault(name, []).extend(bb)
print("other-pack label files indexed:", len(other), flush=True)


def to_xyxy(b, w, h):
    b = np.asarray(b, float)
    x, y, bw, bh = b[:, 0] * w, b[:, 1] * h, b[:, 2] * w, b[:, 3] * h
    return np.stack([x - bw / 2, y - bh / 2, x + bw / 2, y + bh / 2], 1)


def iou_mat(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + bb[None, :] - inter + 1e-9)


m = YOLO(os.path.join(R, "det_scb5_s0", "weights", "best.pt"))
val_imgs = sorted(glob.glob(os.path.join(W, "scb5_hrw", "images", "val", "*")))
tot_fp = matched = 0
n_img = 0
cm, cu = [], []
for p in val_imgs:
    name = os.path.splitext(os.path.basename(p))[0]
    if name not in other:
        continue
    n_img += 1
    lp = os.path.join(W, "scb5_hrw", "labels", "val", name + ".txt")
    gt = []
    for line in open(lp):
        t = line.split()
        if len(t) >= 5: gt.append([float(x) for x in t[1:5]])
    r = m.predict(os.path.realpath(p), imgsz=640, conf=0.25, verbose=False)[0]
    if r.boxes is None or len(r.boxes) == 0: continue
    h, w = r.orig_shape
    pred = r.boxes.xyxy.cpu().numpy(); conf = r.boxes.conf.cpu().numpy()
    G = to_xyxy(gt, w, h) if gt else np.zeros((0, 4))
    O = to_xyxy(other[name], w, h)
    ig = iou_mat(pred, G).max(1) if len(G) else np.zeros(len(pred))
    fp = ig < 0.5
    tot_fp += int(fp.sum())
    if fp.sum():
        io = iou_mat(pred[fp], O).max(1) if len(O) else np.zeros(int(fp.sum()))
        matched += int((io >= 0.5).sum())
        cm += list(conf[fp][io >= 0.5]); cu += list(conf[fp][io < 0.5])

print("val images also annotated by another pack:", n_img)
print("false positives against hrw ground truth :", tot_fp)
print("...that hit a box another pack DID label :", matched,
      "= %.1f%%" % (100 * matched / max(tot_fp, 1)))
if cm: print("mean conf, FP explained by another pack  : %.3f" % np.mean(cm))
if cu: print("mean conf, FP unexplained                : %.3f" % np.mean(cu))
json.dump({"images": n_img, "fp": tot_fp, "fp_on_other_pack_boxes": matched,
           "pct": round(100 * matched / max(tot_fp, 1), 2)},
          open(os.path.join(R, "fp_missing_label_analysis.json"), "w"), indent=1)
