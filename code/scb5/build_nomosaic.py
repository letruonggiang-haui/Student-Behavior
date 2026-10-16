"""Build the SAME split from the non-mosaiced pack, to isolate the cost of face mosaicing.

Labels and splits are byte-identical between the two releases, so the only variable
is whether faces are obscured.
"""
import os, glob, shutil, json
from collections import Counter
NM = os.path.expanduser("~/student_behavior/scb5/SCB5-Handrise-Read-write-2024-9-17")
W = os.path.expanduser("~/student_behavior/work")
NAMES = ["hand-raising", "read", "write"]
out = os.path.join(W, "scb5_hrw_orig")
for sp in ("train", "val"):
    for k in ("images", "labels"):
        os.makedirs(os.path.join(out, k, sp), exist_ok=True)
imgs = sorted(glob.glob(NM + "/**/images/**/*.jpg", recursive=True) +
              glob.glob(NM + "/**/images/**/*.png", recursive=True))
stats, cls = Counter(), Counter()
for p in imgs:
    sp = "val" if "/val/" in p else "train"
    lp = os.path.splitext(p.replace("/images/", "/labels/"))[0] + ".txt"
    if not os.path.exists(lp):
        stats["no_label"] += 1
        continue
    base = os.path.basename(p)
    dl = os.path.join(out, "labels", sp, os.path.splitext(base)[0] + ".txt")
    if os.path.exists(dl):
        stats["dup"] += 1
        continue
    di = os.path.join(out, "images", sp, base)
    if not os.path.islink(di):
        os.symlink(os.path.abspath(p), di)
    shutil.copyfile(lp, dl)
    for line in open(lp):
        t = line.split()
        if len(t) >= 5:
            cls[NAMES[int(t[0])]] += 1
    stats[sp] += 1
with open(os.path.join(W, "scb5_hrw_orig.yaml"), "w") as f:
    f.write("path: %s\ntrain: images/train\nval: images/val\n\nnc: 3\n\n" % out)
    f.write("# Non-mosaiced SCB5 pack, used ONLY for the privacy-utility ablation.\n")
    f.write("names: %s\n" % NAMES)
print(dict(stats), dict(cls))
