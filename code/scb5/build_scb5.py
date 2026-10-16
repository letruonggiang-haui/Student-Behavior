"""Build a clean YOLO detection set from the SCB5 Handrise-Read-write pack.

No merging across packs: every pack annotates only its own behaviours, so a merged
set would carry missing-label noise (verified: only 130 of 13,328 images are covered
by all three student packs, and 2,977 images disagree on train/val across packs).

Output: ~/student_behavior/work/scb5_hrw/{images,labels}/{train,val} + scb5_hrw.yaml
Symlinks the images so no disk is duplicated.
"""
import os, glob, json, shutil
from collections import Counter

R = os.path.expanduser("~/student_behavior/scb5")
W = os.path.expanduser("~/student_behavior/work")
PACK = "SCB5-Handrise-Read-write-2024-9-17"
NAMES = ["hand-raising", "read", "write"]
OUT = os.path.join(W, "scb5_hrw")

for sp in ("train", "val"):
    for kind in ("images", "labels"):
        os.makedirs(os.path.join(OUT, kind, sp), exist_ok=True)

imgs = glob.glob(os.path.join(R, PACK) + "/**/images/**/*.jpg", recursive=True)
imgs += glob.glob(os.path.join(R, PACK) + "/**/images/**/*.png", recursive=True)
print("source images:", len(imgs))

stats = Counter()
cls = Counter()
for p in sorted(imgs):
    sp = "val" if "/val/" in p else "train"
    lp = os.path.splitext(p.replace("/images/", "/labels/"))[0] + ".txt"
    if not os.path.exists(lp):
        stats["no_label"] += 1
        continue
    base = os.path.basename(p)
    dst_i = os.path.join(OUT, "images", sp, base)
    dst_l = os.path.join(OUT, "labels", sp, os.path.splitext(base)[0] + ".txt")
    if os.path.exists(dst_i) or os.path.exists(dst_l):
        stats["dup_name"] += 1
        continue
    if not os.path.islink(dst_i):
        os.symlink(os.path.abspath(p), dst_i)
    shutil.copyfile(lp, dst_l)
    n = 0
    for line in open(lp):
        t = line.split()
        if len(t) >= 5:
            cls[NAMES[int(t[0])]] += 1
            n += 1
    stats[sp] += 1
    stats["boxes"] += n

yaml = os.path.join(W, "scb5_hrw.yaml")
with open(yaml, "w") as f:
    f.write("path: %s\n" % OUT)
    f.write("train: images/train\nval: images/val\n\n")
    f.write("nc: %d\n\n" % len(NAMES))
    f.write("# SCB-Dataset5, Handrise-Read-write pack, released by the SCB authors on\n")
    f.write("# huggingface.co/datasets/wintonYF/SCB-Dataset (academic, non-commercial use).\n")
    f.write("names: %s\n" % NAMES)

print("stats:", dict(stats))
print("class counts:", dict(cls))
print("yaml:", yaml)
json.dump({"stats": dict(stats), "classes": dict(cls)},
          open(os.path.join(W, "scb5_hrw_build.json"), "w"), indent=1)
