#!/bin/bash
# Engagement head on EngageNet frames (subject-disjoint official split), 4 classes.
set -x
PY=$HOME/student_behavior/venv/bin/python
export PYTHONUNBUFFERED=1
C=$HOME/student_behavior/code
W=$HOME/student_behavior/work
R=$HOME/student_behavior/runs
E=$W/engagenet_frames
cd $C

# 1) reference: frozen COCO backbone + head (does detection pretraining even matter?)
$PY engcal2.py --weights yolo11s.pt --eng-data $E --epochs 40 --out $R/eng_ref_coco_s --eng-classes 4 || exit 1
$PY eval_engagenet.py --weights yolo11s.pt --eng-weights $R/eng_ref_coco_s/eng_calibrated.pt --out $R/eng_ref_coco_s.json --tag coco_s_frozen

# 2) reference: frozen SCB5 detection backbone + head
$PY engcal2.py --weights $R/det_scb5_s0/weights/best.pt --eng-data $E --epochs 40 --out $R/eng_ref_det_s --eng-classes 4 || exit 1
$PY eval_engagenet.py --weights $R/det_scb5_s0/weights/best.pt --eng-weights $R/eng_ref_det_s/eng_calibrated.pt --out $R/eng_ref_det_s.json --tag det_s_frozen

# 3) multi-task, YOLO11s and YOLO11l, seed 0, engagement source = EngageNet frames
for M in s l; do
  B=16
  [ "$M" = "l" ] && B=8
  if [ ! -f $R/mtl_en_${M}_s0/weights/best.pt ]; then
    $PY train_mtl.py --phase finetune --det-data $W/scb5_hrw.yaml --eng-data $E --eng-classes 4 \
      --weights yolo11$M.pt --epochs 60 --batch $B --project $R --name mtl_en_${M}_s0 --seed 0 || exit 1
  fi
  $PY engcal2.py --weights $R/mtl_en_${M}_s0/weights/best.pt --eng-data $E --epochs 40 --out $R/mtl_en_${M}_s0 --eng-classes 4 || true
  $PY eval_engagenet.py --weights $R/mtl_en_${M}_s0/weights/best.pt --out $R/mtl_en_${M}_s0_rawhead.json --tag mtl_${M}_raw
  $PY eval_engagenet.py --weights $R/mtl_en_${M}_s0/weights/best.pt --eng-weights $R/mtl_en_${M}_s0/eng_calibrated.pt --out $R/mtl_en_${M}_s0_cal.json --tag mtl_${M}_cal
done

# 4) upper reference: end-to-end fine-tuned classifier at 224 on the same frames
$PY $C/eng_e2e_ref.py

echo ENGAGENET_DONE
