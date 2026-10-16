"""Are the detector's false positives actually correct detections of students the
hrw pack never labelled?

For every hrw val image that ALSO appears in the bth or dis packs, take the model's
false positives against the hrw ground truth and ask how many of them overlap a box
that those other packs did annotate. A high rate means the benchmark is punishing
correct detections, i.e. SCB5 is partially annotated rather than the model wrong.
"""
import os, glob, hashlib, json
import numpy as np
from ultralytics import YOLO

W = os.path.expanduser("~/student_behavior/work")
R = os.path.expanduser("~/student_behavior/runs")
S5 = os.path.expanduser("~/student_behavior/scb5")
OTHER = {"bth": "SCB_BowTurnHead_20250509", "dis": "SCB5-Discuss-2024-9-17"}


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        while True:
            b = f.read(1 << 20)
            if not b: break
            h.update(b)
    return h.hexdigest()


# index the other packs by image md5 -> their boxes (xywhn)
other_boxes = {}
for tag, d in OTHER.items():
    imgs = glob.glob(os.path.join(S5, d) + "/**/images/**/*.jpg", recursive=True)
    imgs += glob.glob(os.path.join(S5, d) + "/**/images/**/*.png", recursive=True)
    for p in imgs:
        lp = os.path.splitext(p.replace("/images/", "/labels/"))[0] + ".txt"
        if not os.path.exists(lp): continue
        bb = []
        for line in open(lp):
            t = line.split()
            if len(t) >= 5:
                bb.append([float(x) for x in t[1:5]])
        if bb:
            other_boxes.setdefault(md5(p), []).extend(bb)
print("other-pack images indexed:", len(other_boxes), flush=True)


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
tot_fp = matched_other = 0
n_overlap_imgs = 0
conf_matched, conf_unmatched = [], []
for p in val_imgs:
    real = os.path.realpath(p)
    h5 = md5(real)
    if h5 not in other_boxes:
        continue
    n_overlap_imgs += 1
    lp = os.path.join(W, "scb5_hrw", "labels", "val",
                      os.path.splitext(os.path.basename(p))[0] + ".txt")
    gt = []
    for line in open(lp):
        t = line.split()
        if len(t) >= 5: gt.append([float(x) for x in t[1:5]])
    r = m.predict(real, imgsz=640, conf=0.25, verbose=False)[0]
    if r.boxes is None or len(r.boxes) == 0: continue
    hgt, wgt = r.orig_shape
    pred = r.boxes.xyxy.cpu().numpy(); confs = r.boxes.conf.cpu().numpy()
    G = to_xyxy(gt, wgt, hgt) if gt else np.zeros((0, 4))
    O = to_xyxy(other_boxes[h5], wgt, hgt)
    ig = iou_mat(pred, G).max(1) if len(G) else np.zeros(len(pred))
    fp = ig < 0.5
    tot_fp += int(fp.sum())
    if fp.sum():
        io = iou_mat(pred[fp], O).max(1) if len(O) else np.zeros(int(fp.sum()))
        matched_other += int((io >= 0.5).sum())
        conf_matched += list(confs[fp][io >= 0.5]); conf_unmatched += list(confs[fp][io < 0.5])

print("val images also covered by bth/dis:", n_overlap_imgs)
print("false positives (IoU<0.5 vs hrw GT):", tot_fp)
print("...of which land on a box another pack DID annotate:", matched_other,
      "= %.1f%%" % (100 * matched_other / max(tot_fp, 1)))
if conf_matched:
    print("mean confidence, FP explained by other pack : %.3f" % np.mean(conf_matched))
if conf_unmatched:
    print("mean confidence, FP not explained          : %.3f" % np.mean(conf_unmatched))
json.dump({"images": n_overlap_imgs, "fp": tot_fp, "fp_on_other_pack_boxes": matched_other,
           "pct": round(100 * matched_other / max(tot_fp, 1), 2)},
          open(os.path.join(R, "fp_missing_label_analysis.json"), "w"), indent=1)
