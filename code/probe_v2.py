"""Revision of the transfer probe (reviewers R1-M4, R2-M6, R3-M7).

1. Vietnamese probe (leave-one-subject-out, same protocol as cross_country_scb5.py) repeated
   for every available seed of each backbone, plus the multi-task model whose head was
   trained on EngageNet, so backbone variance and the engagement source are represented.
2. Within-domain control: the same frozen-feature linear probe on single-student crops cut
   from SCB5 ground-truth boxes (train-split crops for fitting, validation-split crops for
   testing), resized to 640 px like the Vietnamese images. If the COCO backbone also wins
   here, the gap is about task geometry (whole-image classification of one student from
   detection features), not about the country of the training data.
The 'generic' backbone is the COCO-pretrained YOLO11s checkpoint (yolo11s.pt), the same
initialisation every trained model starts from.
"""
import os, json, random
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

import mtl  # noqa: F401
import cross_country_scb5 as cc

R = Path(os.path.expanduser("~/student_behavior/runs"))
VN = Path(os.path.expanduser("~/student_behavior/vn_sample"))
SCB = Path(os.path.expanduser("~/student_behavior/work/scb5_hrw"))
OUT = R / "revision" / "probe_v2.json"
SCB_CLASSES = ["hand-raising", "read", "write"]

BACKBONES = {"COCO": ["yolo11s.pt"],
             "EngageNet frames, no gradient": [R / "mtl_en_s_lam0_s0" / "weights" / "best.pt"],
             "EngageNet frames, shuffled labels": [R / "mtl_en_s_shuf_s0" / "weights" / "best.pt"],
             "Detection-only": [R / f"det_scb5_s{i}" / "weights" / "best.pt" for i in range(3)],
             "Multi-task, Kaggle head": [R / f"mtl_scb5_s{i}" / "weights" / "best.pt" for i in range(3)],
             "Multi-task, EngageNet head": [R / f"mtl_en_s_s{i}" / "weights" / "best.pt" for i in range(3)]}


def scb_crops(split, per_class, min_side=48, seed=0):
    rng = random.Random(seed)
    by_cls = defaultdict(list)
    for lf in sorted((SCB / "labels" / split).glob("*.txt")):
        rows = [r.split() for r in lf.read_text().splitlines() if r.strip()]
        for r in rows:
            by_cls[int(r[0])].append((lf.stem, [float(v) for v in r[1:5]]))
    items = []
    for c, lst in sorted(by_cls.items()):
        rng.shuffle(lst)
        items += [(stem, box, c) for stem, box in lst[:per_class * 3]]
    out, cnt = [], defaultdict(int)
    for stem, (xc, yc, w, h), c in items:
        if cnt[c] >= per_class:
            continue
        im = Image.open(SCB / "images" / split / f"{stem}.jpg").convert("RGB")
        W, H = im.size
        x0, y0, x1, y1 = (xc - w / 2) * W, (yc - h / 2) * H, (xc + w / 2) * W, (yc + h / 2) * H
        if min(x1 - x0, y1 - y0) < min_side:
            continue
        pad = 0.1
        box = (max(0, x0 - pad * (x1 - x0)), max(0, y0 - pad * (y1 - y0)),
               min(W, x1 + pad * (x1 - x0)), min(H, y1 + pad * (y1 - y0)))
        out.append((im.crop(box).copy(), c)); cnt[c] += 1
    return out, dict(cnt)


TF = transforms.Compose([transforms.Resize((640, 640)), transforms.ToTensor()])


def feats(ex, data, bs=32):
    xs = []
    for i in range(0, len(data), bs):
        xs.append(ex(torch.stack([TF(d[0]) for d in data[i:i + bs]])))
    return torch.cat(xs), torch.tensor([d[1] for d in data])


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    res = {"note": __doc__.strip(), "vn": {}, "scb_within": {}}
    items, tf, bs = cc.load_dataset(VN)
    subjects = sorted(set(s for _, _, s in items))
    tr, ntr = scb_crops("train", 600)
    te, nte = scb_crops("val", 300)
    res["scb_within_counts"] = {"train": ntr, "val": nte}
    cc_classes = cc.CLASSES
    for fam, ws in BACKBONES.items():
        for w in ws:
            if not str(w).endswith("yolo11s.pt") and not Path(w).exists():
                continue
            tag = f"{fam} | {Path(w).parent.parent.name if fam != 'COCO' else 'yolo11s.pt'}"
            ex = cc.FeatureExtractor(w)
            X, y, ss = cc.extract_all(ex, items, tf, bs)
            folds = []
            for held in subjects:
                m = torch.tensor([s == held for s in ss])
                acc, f1, _ = cc.linear_probe(X[~m], y[~m], X[m], y[m])
                folds.append(round(acc * 100, 2))
            res["vn"][tag] = {"folds": folds, "mean": round(sum(folds) / len(folds), 2)}
            Xtr, ytr = feats(ex, tr)
            Xte, yte = feats(ex, te)
            cc.CLASSES = SCB_CLASSES
            acc, f1, rec = cc.linear_probe(Xtr, ytr, Xte, yte)
            cc.CLASSES = cc_classes
            res["scb_within"][tag] = {"acc": round(acc * 100, 2), "macro_f1": round(f1 * 100, 2), "recall": rec}
            print(tag, res["vn"][tag], res["scb_within"][tag], flush=True)
            del ex; torch.cuda.empty_cache()
            json.dump(res, open(OUT, "w"), indent=1)
    res["subjects"] = subjects
    json.dump(res, open(OUT, "w"), indent=1)
    print("PROBE_V2_DONE")


if __name__ == "__main__":
    main()
