"""Measure object scale on SCB5-hrw: how small are the targets at 640 px?"""
import os, glob, json
import numpy as np
from PIL import Image
from collections import Counter

W = os.path.expanduser("~/student_behavior/work/scb5_hrw")
NAMES = ["hand-raising", "read", "write"]
rows = []
res = Counter()
for sp in ("train", "val"):
    for lp in sorted(glob.glob(os.path.join(W, "labels", sp, "*.txt"))):
        base = os.path.splitext(os.path.basename(lp))[0]
        ip = os.path.join(W, "images", sp, base + ".jpg")
        if not os.path.exists(ip):
            g = glob.glob(os.path.join(W, "images", sp, base + ".*"))
            if not g: continue
            ip = g[0]
        try:
            iw, ih = Image.open(ip).size
        except Exception:
            continue
        res[(iw, ih)] += 1
        seen = set()
        for line in open(lp):
            t = line.split()
            if len(t) < 5: continue
            k = tuple(t[:5])
            if k in seen: continue
            seen.add(k)
            c, _, _, w, h = int(t[0]), float(t[1]), float(t[2]), float(t[3]), float(t[4])
            rows.append((sp, c, w * iw, h * ih, w, h, iw, ih))

a = np.array([[r[2], r[3], r[4], r[5], r[6], r[7]] for r in rows])
cls = np.array([r[1] for r in rows])
print("boxes:", len(a))
print("\nnative image sizes (top):", res.most_common(6))

# object area as fraction of image; and side length after letterbox to S px
for S in (640, 960, 1280):
    scale = S / np.maximum(a[:, 4], a[:, 5])          # letterbox scale factor
    bw, bh = a[:, 0] * scale, a[:, 1] * scale          # box size in model pixels
    side = np.sqrt(bw * bh)
    print(f"\n--- at imgsz {S} ---")
    print("   box short side px: p10 %.1f  p25 %.1f  median %.1f  p75 %.1f  p90 %.1f" %
          tuple(np.percentile(np.minimum(bw, bh), [10, 25, 50, 75, 90])))
    print("   sqrt(area)  px   : p10 %.1f  p25 %.1f  median %.1f  p75 %.1f  p90 %.1f" %
          tuple(np.percentile(side, [10, 25, 50, 75, 90])))
    print("   COCO-small (<32px sqrt-area): %.1f%%   medium(32-96): %.1f%%   large(>96): %.1f%%" %
          (100 * (side < 32).mean(), 100 * ((side >= 32) & (side < 96)).mean(), 100 * (side >= 96).mean()))
    # how many boxes are smaller than the stride-8 receptive cell
    print("   below stride-8 cell (8px): %.1f%%   below stride-16: %.1f%%" %
          (100 * (side < 8).mean(), 100 * (side < 16).mean()))

print("\nper-class sqrt(area) fraction of image diagonal:")
for c, n in enumerate(NAMES):
    m = cls == c
    fr = np.sqrt(a[m, 2] * a[m, 3])
    print("   %-13s n=%6d  median %.4f  p25 %.4f  p75 %.4f" %
          (n, m.sum(), np.median(fr), np.percentile(fr, 25), np.percentile(fr, 75)))
