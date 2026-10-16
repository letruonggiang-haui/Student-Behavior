import os, glob, hashlib, json
from collections import Counter, defaultdict

R = os.path.expanduser("~/student_behavior/scb5")
PACKS = {
    "hrw": ("SCB5-Handrise-Read-write-2024-9-17", ["hand-raising", "read", "write"]),
    "dis": ("SCB5-Discuss-2024-9-17", ["discuss"]),
    "bth": ("SCB_BowTurnHead_20250509", ["BowHead", "TurnHead"]),
    "tea": ("SCB5_Teacher_Behavior_Stand_BlackBoard_Sreen_20250406-2",
            ["guide", "answer", "On-stage interaction", "blackboard-writing",
             "teacher", "stand", "screen", "blackBoard"]),
}


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


img_of = defaultdict(dict)
split_of = defaultdict(dict)
boxes = defaultdict(lambda: defaultdict(list))

for k, (d, names) in PACKS.items():
    imgs = glob.glob(os.path.join(R, d) + "/**/images/**/*.jpg", recursive=True)
    imgs += glob.glob(os.path.join(R, d) + "/**/images/**/*.png", recursive=True)
    for p in imgs:
        h = md5(p)
        img_of[h][k] = p
        split_of[h][k] = "val" if "/val/" in p else "train"
        lp = os.path.splitext(p.replace("/images/", "/labels/"))[0] + ".txt"
        if os.path.exists(lp):
            for line in open(lp):
                t = line.split()
                if len(t) >= 5:
                    boxes[h][k].append((names[int(t[0])], t[1:5]))
    print(k, "images", len(imgs), flush=True)

cov = Counter()
per = Counter()
for h, d in img_of.items():
    cov[tuple(sorted(d))] += 1
    per[len(d)] += 1

print("total unique images:", len(img_of))
print("images by number of packs covering them:", dict(sorted(per.items())))
print("--- coverage combinations ---")
for c, n in cov.most_common():
    print("  %-24s %d" % ("+".join(c), n))

inc = sum(1 for h, d in split_of.items() if len(set(d.values())) > 1)
print("images with inconsistent train/val across packs:", inc)

STU = ("bth", "dis", "hrw")
full = [h for h, d in img_of.items() if set(STU) <= set(d)]
print("images covered by ALL THREE student packs:", len(full))
cc = Counter()
tot = 0
for h in full:
    for k in STU:
        for nm, _ in boxes[h][k]:
            cc[nm] += 1
            tot += 1
print("boxes on that core:", tot, "per image:", round(tot / max(1, len(full)), 2))
print("class counts:", dict(cc))

full4 = [h for h in full if "tea" in img_of[h]]
print("of those, also in teacher pack:", len(full4))
cc2 = Counter()
tot2 = 0
for h in full4:
    for k in STU:
        for nm, _ in boxes[h][k]:
            cc2[nm] += 1
            tot2 += 1
    for nm, _ in boxes[h]["tea"]:
        if nm in ("stand", "guide"):
            cc2[nm] += 1
            tot2 += 1
print("8-class core boxes:", tot2, "per image:", round(tot2 / max(1, len(full4)), 2))
print("8-class counts:", dict(cc2))

# split composition of the cores
print("core3 split:", Counter(sorted(split_of[h].values())[0] for h in full))
print("core8 split:", Counter(sorted(split_of[h].values())[0] for h in full4))

json.dump({"n_unique": len(img_of), "core3": len(full), "core8": len(full4),
           "coverage": {"+".join(c): n for c, n in cov.items()}},
          open(R + "/merge_plan.json", "w"), indent=1)
