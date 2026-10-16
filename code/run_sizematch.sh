#!/bin/bash
set -x
PY=$HOME/student_behavior/venv/bin/python
export PYTHONUNBUFFERED=1
W=$HOME/student_behavior/work
R=$HOME/student_behavior/runs
# size-matched comparison against YOLOv7 (37.2M): original images, same split
[ -f $R/y11l_orig/weights/best.pt ] || $PY $HOME/student_behavior/code/train_scb5.py \
  --data $W/scb5_hrw_orig.yaml --weights yolo11l.pt --name y11l_orig --seed 0 --epochs 60 --batch 8
[ -f $R/y11x_orig/weights/best.pt ] || $PY $HOME/student_behavior/code/train_scb5.py \
  --data $W/scb5_hrw_orig.yaml --weights yolo11x.pt --name y11x_orig --seed 0 --epochs 60 --batch 4
[ -f $R/y11m_orig/weights/best.pt ] || $PY $HOME/student_behavior/code/train_scb5.py \
  --data $W/scb5_hrw_orig.yaml --weights yolo11m.pt --name y11m_orig --seed 0 --epochs 60 --batch 8
echo SIZEMATCH_DONE
