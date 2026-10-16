#!/bin/bash
# Revision 2026-09, phase A: inference-only analyses on existing checkpoints (no training, no timing).
set -x
PY=$HOME/student_behavior/venv/bin/python
export PYTHONUNBUFFERED=1
C=$HOME/student_behavior/code
R=$HOME/student_behavior/runs
V=$R/revision
mkdir -p $V
cd $C

# per-frame engagement predictions for paired bootstrap (same numbers as Table 4, now saved)
$PY eval_engagenet.py --weights yolo11s.pt --eng-weights $R/eng_ref_coco_s/eng_calibrated.pt \
    --out $V/eng_coco_frozen.json --tag coco_frozen --save-preds $V/preds_coco_frozen.npz
$PY eval_engagenet.py --weights $R/det_scb5_s0/weights/best.pt --eng-weights $R/eng_ref_det_s/eng_calibrated.pt \
    --out $V/eng_det_frozen.json --tag det_frozen --save-preds $V/preds_det_frozen.npz
for M in s l; do
  $PY eval_engagenet.py --weights $R/mtl_en_${M}_s0/weights/best.pt \
      --out $V/eng_mtl_${M}_s0_joint.json --tag mtl_${M}_s0_joint --save-preds $V/preds_mtl_${M}_s0_joint.npz
  $PY eval_engagenet.py --weights $R/mtl_en_${M}_s0/weights/best.pt --eng-weights $R/mtl_en_${M}_s0/eng_calibrated.pt \
      --out $V/eng_mtl_${M}_s0_refit.json --tag mtl_${M}_s0_refit --save-preds $V/preds_mtl_${M}_s0_refit.npz
done
$PY eng_e2e_preds.py $V/preds_e2e.npz

# detection-side batch-norm recalibration on classroom images
$PY bn_recal_eval.py --runs det_scb5_s0 det_scb5_s1 det_scb5_s2 det_l_s0 mtl_scb5_s0 mtl_en_s_s0 mtl_en_l_s0

# transfer probe with all seeds + within-domain SCB5 crop control
$PY probe_v2.py

echo PHASE_A_DONE
