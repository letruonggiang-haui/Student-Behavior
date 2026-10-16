"""Per-frame predictions of the end-to-end EngageNet classifier (eng_e2e_cls_s), saved in the
same order and format as eval_engagenet.py --save-preds, for paired bootstrap."""
import os, csv, re, sys
import numpy as np
from ultralytics import YOLO

R = os.path.expanduser("~/student_behavior/runs")
E = os.path.expanduser("~/student_behavior/work/engagenet_frames")
L = os.path.expanduser("~/student_behavior/engagenet_frames/labels_all.csv")
out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(R, "revision", "preds_e2e.npz")
subj_of = {r["chunk"][:-4]: int(r["subj"]) for r in csv.DictReader(open(L, newline=""))}
m = YOLO(os.path.join(R, "eng_e2e_cls_s", "weights", "best.pt"))
saved = {}
for sp in ("val", "test"):
    classes = sorted(os.listdir(os.path.join(E, sp)))
    ys, ps, ss, names = [], [], [], []
    for ci, c in enumerate(classes):
        files = sorted(os.listdir(os.path.join(E, sp, c)))
        paths = [os.path.join(E, sp, c, f) for f in files]
        for i in range(0, len(paths), 128):
            for f, r in zip(files[i:i + 128], m.predict(paths[i:i + 128], imgsz=224, verbose=False)):
                ps.append(int(r.probs.top1)); ys.append(ci); names.append(f)
                ss.append(subj_of[re.sub(r"_[sme]\.jpg$", "", f)])
    saved[sp + "_y"], saved[sp + "_p"] = np.array(ys), np.array(ps)
    saved[sp + "_subj"], saved[sp + "_names"] = np.array(ss), np.array(names)
np.savez(out, **saved)
print("E2E_PREDS_DONE", out)
