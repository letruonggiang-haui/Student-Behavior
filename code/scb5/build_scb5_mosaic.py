"""Build YOLO detection sets from the SCB authors' PUBLIC face-mosaiced release.

Source: huggingface.co/datasets/wintonYF/SCB-public-mosaic (SCB-2025-7-public-mosaic).
Labels and train/val splits are byte-identical to the non-mosaiced packs; only the
faces are obscured, so this is the artefact to publish against.

Packs are kept separate on purpose. They annotate only their own behaviours, and
only 130 of 13,328 images are covered by all three student packs, so a merged set
would carry missing-label noise. 2,977 images also disagree on train/val across
packs, which would leak.
"""
import os, glob, json, shutil, argparse
from collections import Counter

M = os.path.expanduser("~/student_behavior/scb_mosaic/extracted/SCB-2025-7-public-mosaic")
NM = os.path.expanduser("~/student_behavior/scb5")
W = os.path.expanduser("~/student_behavior/work")

PACKS = {
    "scb5_hrw": dict(mosaic="SCB5-Handrise-Read-write-2024-9-17",
                     names=["hand-raising", "read", "write"]),
    "scb5_bth": dict(mosaic="SCB5-Turn-Bow-Head-2024-9-17",
                     names=["BowHead", "TurnHead"]),
    # Discuss ships without labels in the mosaic release; take images from mosaic
    # and labels from the authors' non-mosaiced pack of the same name.
    "scb5_dis": dict(mosaic="SCB5-Discuss-2024-9-17",
                     labels_from=os.path.join(NM, "SCB5-Discuss-2024-9-17"),
                     names=["discuss"]),
}

ap = argparse.ArgumentParser()
ap.add_argument("--packs", nargs="*", default=list(PACKS))
a = ap.parse_args()

summary = {}
for key in a.packs:
    cfg = PACKS[key]
    src = os.path.join(M, cfg["mosaic"])
    out = os.path.join(W, key)
    for sp in ("train", "val"):
        for kind in ("images", "labels"):
            os.makedirs(os.path.join(out, kind, sp), exist_ok=True)

    imgs = sorted(glob.glob(src + "/**/images/**/*.jpg", recursive=True) +
                  glob.glob(src + "/**/images/**/*.png", recursive=True))
    stats, cls = Counter(), Counter()
    for p in imgs:
        sp = "val" if "/val/" in p else "train"
        lp = os.path.splitext(p.replace("/images/", "/labels/"))[0] + ".txt"
        if not os.path.exists(lp) and "labels_from" in cfg:
            alt = glob.glob(cfg["labels_from"] + "/**/labels/**/" +
                            os.path.splitext(os.path.basename(p))[0] + ".txt", recursive=True)
            lp = alt[0] if alt else lp
        if not os.path.exists(lp):
            stats["no_label"] += 1
            continue
        base = os.path.basename(p)
        di = os.path.join(out, "images", sp, base)
        dl = os.path.join(out, "labels", sp, os.path.splitext(base)[0] + ".txt")
        if os.path.exists(dl):
            stats["dup_name"] += 1
            continue
        if not os.path.islink(di):
            os.symlink(os.path.abspath(p), di)
        shutil.copyfile(lp, dl)
        n = 0
        for line in open(lp):
            t = line.split()
            if len(t) >= 5:
                cls[cfg["names"][int(t[0])]] += 1
                n += 1
        stats[sp] += 1
        stats["boxes"] += n

    yml = os.path.join(W, key + ".yaml")
    with open(yml, "w") as f:
        f.write("path: %s\ntrain: images/train\nval: images/val\n\n" % out)
        f.write("nc: %d\n\n" % len(cfg["names"]))
        f.write("# SCB-Dataset5, public face-mosaiced release by the SCB authors\n")
        f.write("# huggingface.co/datasets/wintonYF/SCB-public-mosaic\n")
        f.write("# Academic, non-commercial use. Not redistributed by us.\n")
        f.write("names: %s\n" % cfg["names"])
    summary[key] = {"stats": dict(stats), "classes": dict(cls), "yaml": yml}
    print(key, dict(stats), dict(cls), flush=True)

json.dump(summary, open(os.path.join(W, "scb5_mosaic_build.json"), "w"), indent=1)
print("written", os.path.join(W, "scb5_mosaic_build.json"))
