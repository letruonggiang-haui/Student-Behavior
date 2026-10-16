"""Engagement on EngageNet frames: accuracy, balanced accuracy (UAR), macro-F1, with
95% subject-bootstrap intervals, on val (11 subjects) and test (26 subjects)."""
import os, csv, json, re, argparse
import numpy as np, torch
from sklearn.metrics import balanced_accuracy_score, f1_score
from ultralytics import YOLO
import mtl  # noqa: F401
from mtl import eng_dataloader, as_mtl

ap = argparse.ArgumentParser()
ap.add_argument("--weights", required=True)
ap.add_argument("--eng-weights", default=None)
ap.add_argument("--data", default=os.path.expanduser("~/student_behavior/work/engagenet_frames"))
ap.add_argument("--labels", default=os.path.expanduser("~/student_behavior/engagenet_frames/labels_all.csv"))
ap.add_argument("--imgsz", type=int, default=640)
ap.add_argument("--out", required=True)
ap.add_argument("--tag", default="")
ap.add_argument("--save-preds", default=None, help="npz with per-frame y, p, subject for paired bootstrap")
a = ap.parse_args()
dev = "cuda"

subj_of = {}
with open(a.labels, newline="") as f:
    for r in csv.DictReader(f):
        subj_of[r["chunk"][:-4]] = int(r["subj"])

model = as_mtl(YOLO(a.weights).model).float().to(dev).eval()
if a.eng_weights and os.path.exists(a.eng_weights):
    st = torch.load(a.eng_weights, map_location="cpu", weights_only=False)
    n_out = st["eng_head"]["fc.4.weight"].shape[0]
    if not hasattr(model, "eng_head") or model.eng_head.fc[-1].out_features != n_out:
        model.attach_eng(n_out)
        model.to(dev)
    model.eng_head.load_state_dict(st["eng_head"])
    model.to(dev).eval()


def boot(y, p, subj, fn, B=2000, seed=0):
    us = np.unique(subj)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(B):
        pick = rng.choice(us, len(us), replace=True)
        idx = np.concatenate([np.where(subj == s)[0] for s in pick])
        vals.append(fn(y[idx], p[idx]))
    return [round(float(np.percentile(vals, 2.5)) * 100, 1), round(float(np.percentile(vals, 97.5)) * 100, 1)]


acc = lambda y, p: float((y == p).mean())
bal = lambda y, p: float(balanced_accuracy_score(y, p))
mf1 = lambda y, p: float(f1_score(y, p, average="macro"))

out = {"weights": a.weights, "eng_weights": a.eng_weights, "tag": a.tag}
saved = {}
for split in ("val", "test"):
    dl = eng_dataloader(a.data, 32, a.imgsz, split)
    ps, ys = [], []
    with torch.no_grad():
        for imgs, y in dl:
            model._predict_once(imgs.to(dev))
            ps += model.eng_head(model._feat).argmax(1).cpu().tolist()
            ys += y.tolist()
    names = [os.path.basename(p) for p, _ in dl.dataset.samples]
    subj = np.array([subj_of[re.sub(r"_[sme]\.jpg$", "", nm)] for nm in names])
    y, p = np.array(ys), np.array(ps)
    floor = max(np.bincount(y)) / len(y)
    out[split] = {"n": int(len(y)), "subjects": int(len(np.unique(subj))), "floor": round(floor * 100, 2),
                  "acc": round(acc(y, p) * 100, 2), "acc_ci": boot(y, p, subj, acc),
                  "bal": round(bal(y, p) * 100, 2), "bal_ci": boot(y, p, subj, bal),
                  "macro_f1": round(mf1(y, p) * 100, 2),
                  "pred_dist": np.bincount(p, minlength=4).tolist(), "gt_dist": np.bincount(y, minlength=4).tolist()}
    print(split, json.dumps(out[split]), flush=True)
    saved[split + "_y"], saved[split + "_p"], saved[split + "_subj"] = y, p, subj
    saved[split + "_names"] = np.array(names)
json.dump(out, open(a.out, "w"), indent=1)
if a.save_preds:
    np.savez(a.save_preds, **saved)
