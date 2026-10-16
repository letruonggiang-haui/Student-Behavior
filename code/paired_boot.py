"""Paired participant-bootstrap intervals for differences in balanced accuracy between
engagement configurations scored on the same EngageNet test participants (reviewer R1-M6).
Each resample draws participants with replacement and recomputes both configurations on
the identical frame set, so the interval is for the difference, not for each marginal."""
import os, json, glob, itertools
import numpy as np
from sklearn.metrics import balanced_accuracy_score

D = os.path.expanduser("~/student_behavior/runs/revision")
B = 5000
files = {os.path.basename(f)[6:-4]: f for f in sorted(glob.glob(os.path.join(D, "preds_*.npz")))}
Z = {k: np.load(f) for k, f in files.items()}
ref = next(iter(Z.values()))
names, subj, y = ref["test_names"], ref["test_subj"], ref["test_y"]
P = {}
for k, z in Z.items():
    pos = {n: i for i, n in enumerate(z["test_names"])}
    idx = np.array([pos[n] for n in names])
    assert (z["test_y"][idx] == y).all(), k
    P[k] = z["test_p"][idx]
us = np.unique(subj)
rng = np.random.default_rng(0)
draws = [np.concatenate([np.where(subj == s)[0] for s in rng.choice(us, len(us), replace=True)]) for _ in range(B)]
bal = lambda p, i: balanced_accuracy_score(y[i], p[i])
out = {"n_frames": int(len(y)), "n_participants": int(len(us)), "B": B,
       "point": {k: round(balanced_accuracy_score(y, p) * 100, 2) for k, p in P.items()}, "diff": {}}
for a, b in itertools.combinations(sorted(P), 2):
    d = np.array([bal(P[a], i) - bal(P[b], i) for i in draws]) * 100
    out["diff"][f"{a} - {b}"] = {"point": round((balanced_accuracy_score(y, P[a]) - balanced_accuracy_score(y, P[b])) * 100, 2),
                                 "ci95": [round(float(np.percentile(d, 2.5)), 2), round(float(np.percentile(d, 97.5)), 2)],
                                 "p_le_0": round(float((d <= 0).mean()), 4)}
json.dump(out, open(os.path.join(D, "paired_boot.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
print("PAIRED_BOOT_DONE")
