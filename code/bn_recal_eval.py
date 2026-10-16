"""Detection-side batch-norm recalibration (revision control for reviewer R2-M1).

Joint training passes webcam engagement frames through the shared backbone in training
mode, so the running BN statistics used at detection inference mix two domains. This
script re-estimates every BN running mean/variance on classroom training images only
(letterboxed 640 px, no augmentation, cumulative average), leaves all weights unchanged,
and re-scores SCB5 validation mAP. The same procedure is applied to the detection-only
models so that the effect of recalibration itself is measured, not assumed.
"""
import os, sys, json, argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox

import mtl  # noqa: F401

R = Path(os.path.expanduser("~/student_behavior/runs"))
Y = os.path.expanduser("~/student_behavior/work/scb5_hrw.yaml")
TRAIN = Path(os.path.expanduser("~/student_behavior/work/scb5_hrw/images/train"))


def val_map(m):
    r = m.val(data=Y, imgsz=640, batch=8, half=False, verbose=False, plots=False)
    return round(float(r.box.map50) * 100, 2), round(float(r.box.map) * 100, 2)


@torch.no_grad()
def recalibrate(model, n_img=None, bs=16):
    bns = [mm for mm in model.modules() if isinstance(mm, nn.modules.batchnorm._BatchNorm)]
    for b in bns:
        b.reset_running_stats(); b.momentum = None
    model.eval()
    for b in bns:
        b.train()
    lb = LetterBox(new_shape=(640, 640), auto=False)
    files = sorted(TRAIN.glob("*.jpg"))[:n_img]
    dev = next(model.parameters()).device
    for i in range(0, len(files), bs):
        ims = [lb(image=cv2.imread(str(f)))[:, :, ::-1] for f in files[i:i + bs]]
        x = torch.from_numpy(np.ascontiguousarray(np.stack(ims).transpose(0, 3, 1, 2))).float().div(255).to(dev)
        model._predict_once(x) if hasattr(model, "_predict_once") else model(x)
    model.eval()
    return len(bns), len(files)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", default=str(R / "revision" / "bn_recal.json"))
    a = ap.parse_args()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    for run in a.runs:
        w = R / run / "weights" / "best.pt"
        if not w.exists():
            print("missing", w); continue
        m = YOLO(str(w))
        before = val_map(m)
        m = YOLO(str(w))                      # fresh copy (val may fuse/half the first one)
        core = m.model.float().cuda()
        nb, ni = recalibrate(core)
        after = val_map(m)
        res[run] = {"map50_before": before[0], "map5095_before": before[1],
                    "map50_after": after[0], "map5095_after": after[1],
                    "bn_layers": nb, "recal_images": ni}
        print(run, json.dumps(res[run]), flush=True)
        json.dump(res, open(a.out, "w"), indent=1)
    print("BN_RECAL_DONE")


if __name__ == "__main__":
    main()
