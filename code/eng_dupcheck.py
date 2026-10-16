"""Near-duplicate audit of engagement2: per val image, min dHash Hamming distance to any train image."""
import os, glob, json
import numpy as np
from PIL import Image
W = os.path.expanduser("~/student_behavior/work"); root = os.path.join(W, "engagement2")
def dhash(p, s=8):
    im = Image.open(p).convert("L").resize((s+1, s), Image.LANCZOS); a = np.asarray(im, dtype=np.int16)
    return np.packbits((a[:, 1:] > a[:, :-1]).flatten())
def load(split):
    out = []
    for c in sorted(os.listdir(os.path.join(root, split))):
        for p in sorted(glob.glob(os.path.join(root, split, c, "*"))):
            try: out.append((p, c, dhash(p), Image.open(p).size))
            except Exception: pass
    return out
tr, va = load("train"), load("val")
TH = np.stack([t[2] for t in tr]); 
def ham(a, B): return np.unpackbits(np.bitwise_xor(a[None, :], B), axis=1).sum(1)
sizes = {}
for p, c, h, sz in tr + va: sizes[sz] = sizes.get(sz, 0) + 1
print("image sizes (top):", sorted(sizes.items(), key=lambda x: -x[1])[:5])
d_all, rows = [], []
for p, c, h, sz in va:
    d = ham(h, TH); j = int(d.argmin()); d_all.append(int(d.min()))
    rows.append((os.path.basename(p), c, int(d.min()), tr[j][1], os.path.basename(tr[j][0])))
d_all = np.array(d_all)
print("val images:", len(va), " train:", len(tr))
for t in (0, 2, 4, 6, 8, 10):
    print(f"  val images with a train image within dHash distance <= {t}: {(d_all<=t).sum()} ({100*(d_all<=t).mean():.1f}%)")
print("median min-distance:", int(np.median(d_all)))
# per class
for c in ("engaged", "not_engaged"):
    dd = np.array([r[2] for r in rows if r[1] == c]); print(f"  {c}: n={len(dd)} <=4: {(dd<=4).sum()}  median {int(np.median(dd))}")
# same-class near-dup vs cross-class near-dup
same = sum(1 for r in rows if r[2] <= 4 and r[1] == r[3]); cross = sum(1 for r in rows if r[2] <= 4 and r[1] != r[3])
print("near-dups (<=4) with SAME label as nearest train:", same, " with DIFFERENT label:", cross)
json.dump(rows, open(os.path.join(W, "engagement2_dupcheck.json"), "w"))
