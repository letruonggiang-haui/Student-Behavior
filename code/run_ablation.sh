#!/bin/bash
# Privacy-utility ablation: identical split, identical labels, faces mosaiced or not.
set -x
cd ~/student_behavior/code
PY=~/student_behavior/venv/bin/python
export PYTHONUNBUFFERED=1
W=~/student_behavior/work
R=~/student_behavior/runs
for S in 0 1 2; do
  if [ ! -f $R/det_scb5orig_s$S/weights/best.pt ]; then
    $PY train_scb5.py --data $W/scb5_hrw_orig.yaml --weights yolo11s.pt \
        --name det_scb5orig_s$S --seed $S --epochs 60 --batch 16 || exit 1
  fi
done
echo ABLATION_DONE
