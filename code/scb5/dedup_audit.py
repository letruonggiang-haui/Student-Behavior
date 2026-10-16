"""Count duplicate boxes in the SCB5 packs.

Ultralytics silently de-duplicates identical label rows when it builds its cache,
so the raw box count and the instance count it reports during validation differ.
The paper's dataset table must quote the de-duplicated figure.
"""
import os, glob, json
from collections import Counter

W = os.path.expanduser("~/student_behavior/work")
PACKS = {"scb5_hrw": ["hand-raising", "read", "write"],
         "scb5_bth": ["BowHead", "TurnHead"],
         "scb5_dis": ["discuss"]}
out = {}
for pack, names in PACKS.items():
    rec = {}
    for sp in ("train", "val"):
        raw = dedup = 0
        imgs_with_dup = 0
        max_dup = 0
        cls_raw, cls_dedup = Counter(), Counter()
        for lp in sorted(glob.glob(os.path.join(W, pack, "labels", sp, "*.txt"))):
            rows = []
            for line in open(lp):
                t = line.split()
                if len(t) >= 5:
                    rows.append(tuple(t[:5]))
            uniq = set(rows)
            raw += len(rows)
            dedup += len(uniq)
            d = len(rows) - len(uniq)
            if d:
                imgs_with_dup += 1
                max_dup = max(max_dup, d)
            for r in rows:
                cls_raw[names[int(r[0])]] += 1
            for r in uniq:
                cls_dedup[names[int(r[0])]] += 1
        rec[sp] = dict(images=len(glob.glob(os.path.join(W, pack, "labels", sp, "*.txt"))),
                       boxes_raw=raw, boxes_dedup=dedup, duplicates=raw - dedup,
                       pct_dup=round(100 * (raw - dedup) / max(raw, 1), 2),
                       images_with_dup=imgs_with_dup, max_dup_in_one_image=max_dup,
                       class_raw=dict(cls_raw), class_dedup=dict(cls_dedup))
    tot_raw = sum(rec[s]["boxes_raw"] for s in rec)
    tot_ded = sum(rec[s]["boxes_dedup"] for s in rec)
    rec["total"] = dict(images=sum(rec[s]["images"] for s in ("train", "val")),
                        boxes_raw=tot_raw, boxes_dedup=tot_ded,
                        duplicates=tot_raw - tot_ded,
                        pct_dup=round(100 * (tot_raw - tot_ded) / max(tot_raw, 1), 2))
    out[pack] = rec
    print(pack, json.dumps(rec["total"]), flush=True)
    for sp in ("train", "val"):
        print("   ", sp, "images", rec[sp]["images"], "raw", rec[sp]["boxes_raw"],
              "dedup", rec[sp]["boxes_dedup"], "dup", rec[sp]["duplicates"],
              f"({rec[sp]['pct_dup']}%)", "imgs_with_dup", rec[sp]["images_with_dup"],
              "max_in_one", rec[sp]["max_dup_in_one_image"])
        print("     class dedup:", rec[sp]["class_dedup"])
json.dump(out, open(os.path.join(W, "scb5_dedup_audit.json"), "w"), indent=1)
print("written", os.path.join(W, "scb5_dedup_audit.json"))
