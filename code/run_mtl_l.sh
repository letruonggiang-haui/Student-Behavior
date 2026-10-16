#!/bin/bash
set -x
PY=$HOME/student_behavior/venv/bin/python
export PYTHONUNBUFFERED=1
W=$HOME/student_behavior/work
R=$HOME/student_behavior/runs
# wait for the size-match chain to finish so the GPU is free
while tmux has-session -t =sizematch 2>/dev/null; do sleep 120; done
# multi-task on the YOLO11l backbone, mosaiced release (the paper benchmark), 3 seeds
for S in 0 1 2; do
  [ -f $R/mtl_l_s$S/weights/best.pt ] || $PY $HOME/student_behavior/code/train_mtl.py \
    --phase finetune --det-data $W/scb5_hrw.yaml --eng-data $W/engagement2 \
    --weights yolo11l.pt --epochs 60 --batch 8 --project $R --name mtl_l_s$S --seed $S || exit 1
done
$PY $HOME/student_behavior/code/engcal2.py --weights $R/mtl_l_s0/weights/best.pt \
  --eng-data $W/engagement2 --epochs 30 --out $R/mtl_l_s0 || true
# detection-only YOLO11l on the same mosaiced benchmark, for the paired gap
for S in 0 1 2; do
  [ -f $R/det_l_s$S/weights/best.pt ] || $PY $HOME/student_behavior/code/train_scb5.py \
    --data $W/scb5_hrw.yaml --weights yolo11l.pt --name det_l_s$S --seed $S --epochs 60 --batch 8 || exit 1
done
echo MTL_L_DONE
