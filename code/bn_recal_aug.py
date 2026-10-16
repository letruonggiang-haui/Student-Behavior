"""Control for the reviewer question: is the recalibration pass itself neutral?

bn_recal_eval.py re-estimates the batch-norm statistics on letterboxed training images with
no augmentation. That pass lowers the detection-only models, which never saw out-of-domain
frames, so part of the like-for-like cost could come from degrading the control arm rather
than from correcting the treated one.

This script repeats the pass for the detection-only models using the framework's own
training dataloader with the augmentation the run ended with (mosaic closed for the last
ten epochs, every other augmentation active), so the statistics are re-estimated on the
stream the model actually trained on. If mAP then returns to the trained value, the drop
seen in the plain pass is an artefact of the pass, not of the model.
"""
import os, json, argparse
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from ultralytics import YOLO
from ultralytics.data import build_yolo_dataset, build_dataloader
from ultralytics.cfg import get_cfg

import mtl  # noqa: F401

R = Path(os.path.expanduser("~/student_behavior/runs"))
Y = os.path.expanduser("~/student_behavior/work/scb5_hrw.yaml")
TRAIN = os.path.expanduser("~/student_behavior/work/scb5_hrw/images/train")


def val_map(m):
    r = m.val(data=Y, imgsz=640, batch=8, half=False, verbose=False, plots=False)
    return round(float(r.box.map50) * 100, 2), round(float(r.box.map) * 100, 2)


@torch.no_grad()
def recal_with_training_stream(core, run, bs=16):
    args = yaml.safe_load(open(R / run / "args.yaml"))
    cfg = get_cfg(overrides={k: args[k] for k in args if k in get_cfg().__dict__})
    cfg.mosaic = 0.0          # the run closed mosaic for its last ten epochs
    cfg.mixup = 0.0
    cfg.copy_paste = 0.0
    data = yaml.safe_load(open(Y))
    data.setdefault("channels", 3)
    ds = build_yolo_dataset(cfg, TRAIN, bs, data, mode="train", rect=False, stride=32)
    dl = build_dataloader(ds, bs, workers=4, shuffle=False)

    bns = [m for m in core.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    for b in bns:
        b.reset_running_stats(); b.momentum = None
    core.eval()
    for b in bns:
        b.train()
    dev = next(core.parameters()).device
    n = 0
    for batch in dl:
        x = batch["img"].to(dev, non_blocking=True).float() / 255
        core._predict_once(x) if hasattr(core, "_predict_once") else core(x)
        n += x.shape[0]
    core.eval()
    return len(bns), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", default=str(R / "revision" / "bn_recal_aug.json"))
    a = ap.parse_args()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    for run in a.runs:
        w = R / run / "weights" / "best.pt"
        if not w.exists():
            print("missing", w, flush=True); continue
        m = YOLO(str(w))
        before = val_map(m)
        m = YOLO(str(w))
        core = m.model.float().cuda()
        nb, ni = recal_with_training_stream(core, run)
        after = val_map(m)
        res[run] = {"map50_before": before[0], "map5095_before": before[1],
                    "map50_after": after[0], "map5095_after": after[1],
                    "bn_layers": nb, "recal_images": ni,
                    "stream": "training dataloader, mosaic closed, other augmentation on"}
        print(run, json.dumps(res[run]), flush=True)
        json.dump(res, open(a.out, "w"), indent=1)
    print("BN_RECAL_AUG_DONE")


if __name__ == "__main__":
    main()
