"""Near-duplicate audit of the SCB5 split, the same difference-hash audit already run on the
Kaggle engagement corpus and on the independent transfer set.

Two levels:
  1. crops: the single-student crops the within-domain probe control uses, built by importing
     probe_v2.scb_crops so the sample is identical (600 per class from the training split for
     fitting, 300 per class from the validation split for testing). Per test crop, the smallest
     64-bit difference-hash distance to any fitting crop.
  2. frames: every image of the training and validation splits. Per validation frame, the
     smallest distance to any training frame. This is the level at which the pack's own split
     is defined, so it is the level at which a leak would matter.
"""
import os, json, glob
import numpy as np
from PIL import Image

W = os.path.expanduser("~/student_behavior/work")
SCB = os.path.join(W, "scb5_hrw")
OUT = os.path.expanduser("~/student_behavior/runs/revision/scb_dupcheck.json")


def dhash_im(im, s=8):
    a = np.asarray(im.convert("L").resize((s + 1, s), Image.LANCZOS), dtype=np.int16)
    return np.packbits((a[:, 1:] > a[:, :-1]).flatten())


def ham(a, B):
    return np.unpackbits(np.bitwise_xor(a[None, :], B), axis=1).sum(1)


def summarise(dists, n_ref):
    d = np.array(dists)
    return {"n": int(d.size), "n_reference": int(n_ref),
            "median_min_distance": int(np.median(d)),
            "within": {str(t): {"count": int((d <= t).sum()), "pct": round(100.0 * (d <= t).mean(), 1)}
                       for t in (0, 2, 4, 6, 8, 10)}}


res = {"note": __doc__.strip()}

# 1. crops used by the within-domain probe control
import probe_v2  # noqa: E402  (imports torch; only scb_crops is used)

tr, ntr = probe_v2.scb_crops("train", 600)
te, nte = probe_v2.scb_crops("val", 300)
TH = np.stack([dhash_im(im) for im, _ in tr])
tr_lab = [c for _, c in tr]
d_crop, same = [], 0
for im, c in te:
    d = ham(dhash_im(im), TH)
    j = int(d.argmin())
    d_crop.append(int(d.min()))
    same += int(d.min() <= 4 and tr_lab[j] == c)
res["crops"] = summarise(d_crop, len(tr))
res["crops"]["counts"] = {"fit": ntr, "test": nte}
res["crops"]["near_duplicates_sharing_the_label"] = same
print("CROPS", json.dumps(res["crops"]), flush=True)

# 2. frames of the split itself
frames = {s: sorted(glob.glob(os.path.join(SCB, "images", s, "*.jpg"))) for s in ("train", "val")}
FH = np.stack([dhash_im(Image.open(p)) for p in frames["train"]])
d_frame = []
nearest = []
for p in frames["val"]:
    d = ham(dhash_im(Image.open(p)), FH)
    j = int(d.argmin())
    d_frame.append(int(d.min()))
    nearest.append((os.path.basename(p), int(d.min()), os.path.basename(frames["train"][j])))
res["frames"] = summarise(d_frame, len(frames["train"]))
res["frames"]["examples"] = sorted(nearest, key=lambda r: r[1])[:10]
print("FRAMES", json.dumps({k: v for k, v in res["frames"].items() if k != "examples"}), flush=True)

json.dump(res, open(OUT, "w"), indent=1)
print("SCB_DUPCHECK_DONE")
