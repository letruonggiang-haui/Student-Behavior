"""Evaluate our models on exactly the val subset YOLOv7 keeps.

YOLOv7 discards any image whose label file contains duplicate rows; ultralytics
de-duplicates the rows and keeps the image. The two frameworks therefore report
mAP on different val sets, which is why the authors' 74.65 is not comparable with
our 69.20 out of the box.
"""
import os, glob, json, shutil
from ultralytics import YOLO

W = os.path.expanduser("~/student_behavior/work")
R = os.path.expanduser("~/student_behavior/runs")
SRC = os.path.join(W, "scb5_hrw_orig")
OUT = os.path.join(W, "scb5_hrw_orig_v7subset")

kept = dropped = 0
for sp in ("train", "val"):
    for k in ("images", "labels"):
        os.makedirs(os.path.join(OUT, k, sp), exist_ok=True)
# train side is irrelevant for val-only comparison; link one image so yaml is valid
for sp in ("val",):
    for lp in sorted(glob.glob(os.path.join(SRC, "labels", sp, "*.txt"))):
        rows = []
        for line in open(lp):
            t = line.split()
            if len(t) >= 5:
                rows.append(tuple(t[:5]))
        name = os.path.splitext(os.path.basename(lp))[0]
        g = glob.glob(os.path.join(SRC, "images", sp, name + ".*"))
        if not g:
            continue
        if len(rows) != len(set(rows)):
            dropped += 1
            continue
        kept += 1
        di = os.path.join(OUT, "images", sp, os.path.basename(g[0]))
        if not os.path.islink(di):
            os.symlink(os.path.realpath(g[0]), di)
        shutil.copyfile(lp, os.path.join(OUT, "labels", sp, name + ".txt"))
print("val kept:", kept, " dropped (had duplicate rows):", dropped)

# train dir must be non-empty for the yaml loader
tr_src = sorted(glob.glob(os.path.join(SRC, "images", "train", "*")))[:8]
for p in tr_src:
    d = os.path.join(OUT, "images", "train", os.path.basename(p))
    if not os.path.islink(d):
        os.symlink(os.path.realpath(p), d)
    lp = os.path.join(SRC, "labels", "train", os.path.splitext(os.path.basename(p))[0] + ".txt")
    if os.path.exists(lp):
        shutil.copyfile(lp, os.path.join(OUT, "labels", "train", os.path.basename(lp)))

yml = os.path.join(W, "scb5_hrw_orig_v7subset.yaml")
with open(yml, "w") as f:
    f.write("path: %s\ntrain: images/train\nval: images/val\n\nnc: 3\n\n" % OUT)
    f.write("names: ['hand-raising', 'read', 'write']\n")

res = {}
for name in ("det_scb5orig_s0", "det_scb5orig_s1", "det_scb5orig_s2"):
    w = os.path.join(R, name, "weights", "best.pt")
    if not os.path.exists(w):
        continue
    for tag, data in (("full", os.path.join(W, "scb5_hrw_orig.yaml")), ("v7subset", yml)):
        r = YOLO(w).val(data=data, imgsz=640, batch=8, verbose=False)
        res.setdefault(name, {})[tag] = {"mAP50": round(float(r.box.map50) * 100, 2),
                                        "mAP50_95": round(float(r.box.map) * 100, 2)}
    print(name, res[name], flush=True)

import statistics as st
for tag in ("full", "v7subset"):
    v = [res[n][tag]["mAP50"] for n in res]
    print("%-9s mAP50 %.2f +/- %.2f  %s" % (tag, st.mean(v), st.stdev(v) if len(v) > 1 else 0, v))
d = [res[n]["v7subset"]["mAP50"] - res[n]["full"]["mAP50"] for n in res]
print("v7subset - full: %+.2f +/- %.2f  per seed %s" % (st.mean(d), st.stdev(d) if len(d) > 1 else 0, [round(x, 2) for x in d]))
json.dump(res, open(os.path.join(R, "v7subset_comparison.json"), "w"), indent=1)
