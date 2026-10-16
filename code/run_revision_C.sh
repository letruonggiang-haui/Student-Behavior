#!/bin/bash
# Revision 2026-09, phase C: the prospective batch-normalisation control.
# Engagement frames pass through the backbone with normalisation layers in inference mode,
# so they never contribute to the running statistics, while everything else is unchanged.
# Waits for enough free GPU memory so that it never competes for memory with other projects.
set -x
PY=$HOME/student_behavior/venv/bin/python
export PYTHONUNBUFFERED=1
C=$HOME/student_behavior/code
W=$HOME/student_behavior/work
R=$HOME/student_behavior/runs
E=$W/engagenet_frames
V=$R/revision
cd $C

while true; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)
  free=$((total - used))
  [ "$free" -ge 11000 ] && break
  sleep 300
done
echo "GPU has $free MiB free $(date)"

N=mtl_en_s_bnfrozen_s0
if [ ! -f $R/$N/weights/best.pt ]; then
  $PY train_mtl.py --phase finetune --det-data $W/scb5_hrw.yaml --eng-data $E --eng-classes 4 \
      --weights yolo11s.pt --epochs 60 --batch 16 --project $R --name $N --seed 0 --eng-bn-eval --no-amp || exit 1
fi
$PY engcal2.py --weights $R/$N/weights/best.pt --eng-data $E --epochs 40 --out $R/$N --eng-classes 4 || true
$PY eval_engagenet.py --weights $R/$N/weights/best.pt --eng-weights $R/$N/eng_calibrated.pt \
    --out $V/eng_${N}_refit.json --tag ${N}_refit --save-preds $V/preds_${N}_refit.npz
$PY bn_recal_eval.py --runs $N
echo "PHASE_C_DONE $(date)"
