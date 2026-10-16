#!/bin/bash
# Revision 2026-09, phase B: throughput re-measurement and new training runs.
# Waits for the other project's G3 queue on this shared L4 and for an idle GPU, so that
# timing is taken on an otherwise idle device and training does not compete with it.
set -x
PY=$HOME/student_behavior/venv/bin/python
export PYTHONUNBUFFERED=1
C=$HOME/student_behavior/code
W=$HOME/student_behavior/work
R=$HOME/student_behavior/runs
E=$W/engagenet_frames
V=$R/revision
mkdir -p $V
cd $C

wait_idle() {
  until grep -q "G3_DONE" $HOME/engagenet_p2_crossdomain/logs/g3.log 2>/dev/null; do sleep 300; done
  local ok=0
  while [ $ok -lt 3 ]; do
    n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)
    if [ "$n" = "0" ]; then ok=$((ok+1)); else ok=0; fi
    sleep 60
  done
}

wait_idle
echo "GPU idle $(date)"
for t in 1 2 3; do
  $PY fps_v2.py && break
  sleep 600; wait_idle
done

train_eng() {  # name, then extra train_mtl args
  N=$1; shift
  if [ ! -f $R/$N/weights/best.pt ]; then
    $PY train_mtl.py --phase finetune --det-data $W/scb5_hrw.yaml --eng-data $E --eng-classes 4 \
        --weights yolo11s.pt --epochs 60 --batch 16 --project $R --name $N "$@" || return 1
  fi
  $PY engcal2.py --weights $R/$N/weights/best.pt --eng-data $E --epochs 40 --out $R/$N --eng-classes 4 || true
  $PY eval_engagenet.py --weights $R/$N/weights/best.pt --out $V/eng_${N}_joint.json --tag ${N}_joint \
      --save-preds $V/preds_${N}_joint.npz
  $PY eval_engagenet.py --weights $R/$N/weights/best.pt --eng-weights $R/$N/eng_calibrated.pt \
      --out $V/eng_${N}_refit.json --tag ${N}_refit --save-preds $V/preds_${N}_refit.npz
  $PY bn_recal_eval.py --runs $N
  echo "TRAINED $N $(date)"
}

# seeds for the EngageNet-head model at the small scale
train_eng mtl_en_s_s1 --seed 1
train_eng mtl_en_s_s2 --seed 2
# controls, seed 0: engagement batches still interleaved but no engagement gradient
train_eng mtl_en_s_lam0_s0 --seed 0 --eng-lambda 0
# controls, seed 0: same frames and head width, labels permuted (no signal)
train_eng mtl_en_s_shuf_s0 --seed 0 --eng-shuffle
# ratio control: 16 engagement images per 16 detection images (the 1:1 ratio of the large-scale run)
train_eng mtl_en_s_eb16_s0 --seed 0 --eng-batch 16
# loss-weight sweep on the EngageNet head
train_eng mtl_en_s_lam02_s0 --seed 0 --eng-lambda 0.2
train_eng mtl_en_s_lam5_s0 --seed 0 --eng-lambda 5

# probe again with all seeds present, then paired bootstrap over every saved prediction set
$PY probe_v2.py
$PY paired_boot.py
echo PHASE_B_DONE
