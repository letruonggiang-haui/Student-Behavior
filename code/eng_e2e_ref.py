"""Upper reference: an end-to-end fine-tuned image classifier on the same EngageNet frames."""
import os, json, csv, re
import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score
from ultralytics import YOLO

R = os.path.expanduser("~/student_behavior/runs")
E = os.path.expanduser("~/student_behavior/work/engagenet_frames")
L = os.path.expanduser("~/student_behavior/engagenet_frames/labels_all.csv")
subj_of = {r["chunk"][:-4]: int(r["subj"]) for r in csv.DictReader(open(L, newline=""))}

m = YOLO("yolo11s-cls.pt")
m.train(data=E, epochs=30, imgsz=224, batch=64, project=R, name="eng_e2e_cls_s", exist_ok=True, seed=0)
m = YOLO(os.path.join(R, "eng_e2e_cls_s", "weights", "best.pt"))
out = {}
for sp in ("val", "test"):
    classes = sorted(os.listdir(os.path.join(E, sp)))
    ys, ps, subj = [], [], []
    for ci, c in enumerate(classes):
        files = sorted(os.listdir(os.path.join(E, sp, c)))
        paths = [os.path.join(E, sp, c, f) for f in files]
        for i in range(0, len(paths), 128):
            for f, r in zip(files[i:i + 128], m.predict(paths[i:i + 128], imgsz=224, verbose=False)):
                ps.append(int(r.probs.top1)); ys.append(ci); subj.append(subj_of[re.sub(r"_[sme]\.jpg$", "", f)])
    y, p, s = np.array(ys), np.array(ps), np.array(subj)
    us = np.unique(s); rng = np.random.default_rng(0); bb = []
    for _ in range(2000):
        pick = rng.choice(us, len(us), replace=True); idx = np.concatenate([np.where(s == k)[0] for k in pick])
        bb.append(balanced_accuracy_score(y[idx], p[idx]))
    out[sp] = {"n": int(len(y)), "acc": round(float((y == p).mean()) * 100, 2),
               "bal": round(float(balanced_accuracy_score(y, p)) * 100, 2),
               "bal_ci": [round(float(np.percentile(bb, 2.5)) * 100, 1), round(float(np.percentile(bb, 97.5)) * 100, 1)],
               "macro_f1": round(float(f1_score(y, p, average="macro")) * 100, 2),
               "pred_dist": np.bincount(p, minlength=4).tolist()}
    print("E2E", sp, json.dumps(out[sp]), flush=True)
json.dump(out, open(os.path.join(R, "eng_e2e_cls_s.json"), "w"), indent=1)
