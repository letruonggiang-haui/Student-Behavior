#!/bin/bash
# Revision 2026-09, phase D: items raised by the second review round.
#  1. seeds 1 and 2 for the lambda = 0.2 configuration, which carries the paper's practical recommendation
#  2. batch-norm recalibration for the remaining large-scale runs, so that block has the same columns as the small one
#  3. a near-duplicate audit of the independent transfer set, the same audit already run on the Kaggle engagement set
set -x
PY=$HOME/student_behavior/venv/bin/python
export PYTHONUNBUFFERED=1
C=$HOME/student_behavior/code
W=$HOME/student_behavior/work
R=$HOME/student_behavior/runs
E=$W/engagenet_frames
V=$R/revision
cd $C

$PY - <<'PY'
import os, json, itertools
from pathlib import Path
from PIL import Image
import imagehash

root = Path(os.path.expanduser("~/student_behavior/vn_sample"))
items = [(f, f.parent.name, f.name.split("_")[0]) for f in sorted(root.glob("*/*.jpg"))]
h = {f: imagehash.dhash(Image.open(f), hash_size=8) for f, _, _ in items}
pairs = 0
cross_folder = 0
same_class_pairs = 0
n = len(items)
for (f1, c1, s1), (f2, c2, s2) in itertools.combinations(items, 2):
    d = h[f1] - h[f2]
    if d <= 4:
        pairs += 1
        if s1 != s2:
            cross_folder += 1
        if c1 == c2:
            same_class_pairs += 1
out = {"images": n, "near_duplicate_pairs_dhash_le_4": pairs,
       "pairs_across_identity_folders": cross_folder, "pairs_same_class": same_class_pairs}
json.dump(out, open(os.path.expanduser("~/student_behavior/runs/revision/vn_dupcheck.json"), "w"), indent=1)
print("VN_DUPCHECK", json.dumps(out))
PY

$PY bn_recal_eval.py --runs det_l_s1 det_l_s2 mtl_l_s0
$PY probe_v2.py

for S in 1 2; do
  N=mtl_en_s_lam02_s$S
  if [ ! -f $R/$N/weights/best.pt ]; then
    $PY train_mtl.py --phase finetune --det-data $W/scb5_hrw.yaml --eng-data $E --eng-classes 4 \
        --weights yolo11s.pt --epochs 60 --batch 16 --project $R --name $N --seed $S --eng-lambda 0.2 || exit 1
  fi
  $PY engcal2.py --weights $R/$N/weights/best.pt --eng-data $E --epochs 40 --out $R/$N --eng-classes 4 || true
  $PY eval_engagenet.py --weights $R/$N/weights/best.pt --eng-weights $R/$N/eng_calibrated.pt \
      --out $V/eng_${N}_refit.json --tag ${N}_refit --save-preds $V/preds_${N}_refit.npz
  $PY bn_recal_eval.py --runs $N
  echo "TRAINED $N $(date)"
done
echo "PHASE_D_DONE $(date)"
