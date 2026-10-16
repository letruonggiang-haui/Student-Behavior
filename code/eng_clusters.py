"""How many independent scene/subject clusters does the Student Engagement dataset really have,
when near-duplicates are merged ACROSS subfolders and across the two labels?"""
import os, glob, json
import numpy as np
from PIL import Image
from collections import Counter, defaultdict
W = os.path.expanduser("~/student_behavior/work"); root = os.path.join(W, "engagement2")
def dhash(p, s=8):
    a = np.asarray(Image.open(p).convert("L").resize((s+1, s), Image.LANCZOS), dtype=np.int16)
    return np.packbits((a[:, 1:] > a[:, :-1]).flatten())
items = []
for split in ("train", "val"):
    for c in ("engaged", "not_engaged"):
        for p in sorted(glob.glob(os.path.join(root, split, c, "*"))):
            items.append((p, split, c, dhash(p)))
H = np.stack([it[3] for it in items]); n = len(items)
bits = np.unpackbits(H, axis=1).astype(np.uint8)
par = list(range(n))
def find(x):
    while par[x] != x: par[x] = par[par[x]]; x = par[x]
    return x
for th in (4, 6, 10):
    par = list(range(n))
    for i in range(n):
        d = (bits[i] != bits[i+1:]).sum(1); js = np.where(d <= th)[0] + i + 1
        for j in js:
            a, b = find(i), find(int(j))
            if a != b: par[a] = b
    roots = [find(i) for i in range(n)]
    cl = defaultdict(list)
    for i, r in enumerate(roots): cl[r].append(i)
    sizes = sorted((len(v) for v in cl.values()), reverse=True)
    mixed_split = sum(1 for v in cl.values() if len({items[i][1] for i in v}) > 1)
    mixed_label = sum(1 for v in cl.values() if len({items[i][2] for i in v}) > 1)
    big = [v for v in cl.values() if len(v) >= 20]
    print(f"th<={th}: clusters={len(cl)}  clusters>=20 imgs={len(big)} (covering {sum(len(v) for v in big)}/{n})  "
          f"largest={sizes[:5]}  clusters spanning train+val={mixed_split}  clusters with both labels={mixed_label}")
