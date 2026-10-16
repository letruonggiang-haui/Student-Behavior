"""Build an ImageFolder engagement set from EngageNet frames, subject-disjoint by the
official split. One frame per clip by default (the middle frame); --train-frames all
uses start/middle/end for TRAIN only. SNP clips are dropped. Data stay local (EULA)."""
import os, csv, argparse

ap = argparse.ArgumentParser()
ap.add_argument("--frames", default=os.path.expanduser("~/student_behavior/engagenet_frames"))
ap.add_argument("--labels", default=os.path.expanduser("~/student_behavior/engagenet_frames/labels_all.csv"))
ap.add_argument("--out", default=os.path.expanduser("~/student_behavior/work/engagenet_frames"))
ap.add_argument("--train-frames", choices=["m", "all"], default="m")
a = ap.parse_args()

CLS = {"Not-Engaged": "0_not", "Barely-engaged": "1_barely", "Engaged": "2_engaged", "Highly-Engaged": "3_highly"}
SPLIT = {"Train": "train", "Validation": "val", "Test": "test"}
n = {}
with open(a.labels, newline="") as f:
    for r in csv.DictReader(f):
        if r["label"] not in CLS:
            continue
        sp = SPLIT[r["split"]]
        c = CLS[r["label"]]
        base = r["chunk"][:-4]
        d = os.path.join(a.out, sp, c)
        os.makedirs(d, exist_ok=True)
        tags = ("s", "m", "e") if (sp == "train" and a.train_frames == "all") else ("m",)
        for t in tags:
            src = os.path.join(a.frames, r["split"], base + "_" + t + ".jpg")
            if not os.path.exists(src):
                n["missing_" + sp] = n.get("missing_" + sp, 0) + 1
                continue
            dst = os.path.join(d, base + "_" + t + ".jpg")
            if not os.path.islink(dst):
                os.symlink(src, dst)
            n[sp + "/" + c] = n.get(sp + "/" + c, 0) + 1
for k in sorted(n):
    print(k, n[k])
