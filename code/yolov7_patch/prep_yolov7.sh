#!/bin/bash
# Reproduce the SCB authors' YOLOv7 baseline on our split, to test whether the
# 74.65 mAP they report is comparable with our ultralytics numbers.
set -e
cd ~/yolov7
PY=~/student_behavior/venv/bin/python

# weights
[ -f yolov7.pt ] || curl -fL -o yolov7.pt \
  https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt

# data yaml, non-mosaiced images to match how the authors trained
cat > data/scb5_hrw_orig.yaml <<'YAML'
train: ~/student_behavior/work/scb5_hrw_orig/images/train
val: ~/student_behavior/work/scb5_hrw_orig/images/val
nc: 3
names: ['hand-raising','read','write']
YAML

# compatibility patches for numpy>=1.24 and torch>=2
grep -rl "np\.int\b" utils/ models/ 2>/dev/null | xargs -r sed -i 's/np\.int\b/int/g'
grep -rl "np\.float\b" utils/ models/ 2>/dev/null | xargs -r sed -i 's/np\.float\b/float/g'
grep -rl "np\.bool\b" utils/ models/ 2>/dev/null | xargs -r sed -i 's/np\.bool\b/bool/g'
# torch.load defaults to weights_only=True on torch>=2.6
grep -rl "torch\.load(" *.py utils/ models/ 2>/dev/null | xargs -r \
  sed -i 's/torch\.load(\([^)]*map_location=[^)]*\))/torch.load(\1, weights_only=False)/g'
sed -i 's/torch\.load(w, map_location=map_location, weights_only=False)/torch.load(w, map_location=map_location, weights_only=False)/' models/experimental.py 2>/dev/null || true

$PY - <<'PYEOF'
import re, pathlib
# recip_norm: torch.meshgrid needs indexing= on new torch; add it where missing
for p in list(pathlib.Path('.').rglob('*.py')):
    try: s = p.read_text()
    except Exception: continue
    if 'torch.meshgrid(' in s and 'indexing=' not in s:
        s = s.replace('torch.meshgrid(', 'torch.meshgrid(')  # leave; warning only
        p.write_text(s)
print("patched")
PYEOF
echo PREP_DONE
