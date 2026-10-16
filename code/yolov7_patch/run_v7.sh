#!/bin/bash
set -x
cd $HOME/yolov7
PY=$HOME/student_behavior/venv/bin/python
export PYTHONUNBUFFERED=1
# authors config: batch 4, 40 epochs, hyp.scratch.p5, non-mosaiced images
$PY train.py --workers 4 --device 0 --batch-size 4 --data data/scb5_hrw_orig.yaml \
  --img 640 640 --cfg cfg/training/yolov7.yaml --weights yolov7.pt \
  --name scb5_v7_b4 --hyp data/hyp.scratch.p5.yaml --epochs 40 --exist-ok
echo V7_B4_DONE
# our config: batch 16, 60 epochs, same repo
$PY train.py --workers 8 --device 0 --batch-size 16 --data data/scb5_hrw_orig.yaml \
  --img 640 640 --cfg cfg/training/yolov7.yaml --weights yolov7.pt \
  --name scb5_v7_b16 --hyp data/hyp.scratch.p5.yaml --epochs 60 --exist-ok
echo V7_ALL_DONE
