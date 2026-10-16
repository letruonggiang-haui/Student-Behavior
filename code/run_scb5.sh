#!/bin/bash
# ESWA rebuild on SCB-Dataset5 (public face-mosaiced release) after STBD-08 was
# withdrawn as an unlicensed redistribution.
#   tmux new -s scb5 -d 'bash ~/student_behavior/code/run_scb5.sh 2>&1 | tee ~/student_behavior/run_scb5.log'
set -x
cd ~/student_behavior/code
PY=~/student_behavior/venv/bin/python
export PYTHONUNBUFFERED=1
W=~/student_behavior/work
R=~/student_behavior/runs
DET=$W/scb5_hrw.yaml

echo "=== STEP 1: MTL one-phase, 3 seeds (COCO -> SCB5 det + engagement) ==="
for S in 0 1 2; do
  if [ ! -f $R/mtl_scb5_s$S/weights/best.pt ]; then
    $PY train_mtl.py --phase finetune --det-data $DET --eng-data $W/engagement2 \
        --weights yolo11s.pt --epochs 60 --batch 16 \
        --project $R --name mtl_scb5_s$S --seed $S || exit 1
  fi
done

echo "=== STEP 2: engagement head calibration (seed 0) ==="
[ -f $R/mtl_scb5_s0/eng_calibrated.pt ] || \
  $PY engcal2.py --weights $R/mtl_scb5_s0/weights/best.pt \
      --eng-data $W/engagement2 --epochs 30 --out $R/mtl_scb5_s0 || exit 1

echo "=== STEP 3: detection-only YOLO11s, 3 seeds ==="
for S in 0 1 2; do
  if [ ! -f $R/det_scb5_s$S/weights/best.pt ]; then
    $PY train_scb5.py --data $DET --weights yolo11s.pt \
        --name det_scb5_s$S --seed $S --epochs 60 --batch 16 || exit 1
  fi
done

echo "=== STEP 4: RT-DETR external SOTA control ==="
[ -f $R/rtdetr_scb5/weights/best.pt ] || \
  $PY train_scb5.py --data $DET --weights rtdetr-l.pt \
      --name rtdetr_scb5 --seed 0 --epochs 60 --batch 8 || exit 1

echo "=== STEP 5: Pareto points YOLO11n / YOLO11m ==="
[ -f $R/yolo11n_scb5/weights/best.pt ] || \
  $PY train_scb5.py --data $DET --weights yolo11n.pt --name yolo11n_scb5 --seed 0 --epochs 60 --batch 16 || exit 1
[ -f $R/yolo11m_scb5/weights/best.pt ] || \
  $PY train_scb5.py --data $DET --weights yolo11m.pt --name yolo11m_scb5 --seed 0 --epochs 60 --batch 8 || exit 1

echo "=== STEP 6: second benchmark, BowTurnHead pack ==="
[ -f $R/det_scb5bth_s0/weights/best.pt ] || \
  $PY train_scb5.py --data $W/scb5_bth.yaml --weights yolo11s.pt \
      --name det_scb5bth_s0 --seed 0 --epochs 60 --batch 16 || exit 1

echo "=== STEP 7: aggregate evaluation ==="
$PY evaluate_scb5.py --work $W --runs $R --vn-data $DET \
    --out $R/final_eval_scb5.json || exit 1

echo "=== STEP 8: cross-country LOSO probe (backbones now trained on SCB5) ==="
$PY cross_country_scb5.py --vn-dir ~/student_behavior/vn_sample --runs $R \
    --out $R/cross_country_scb5.json || echo "WARN cross_country failed"

echo "ALL_SCB5_DONE"
