"""Can the shared backbone carry engagement at all if its last stage is allowed to move
for the engagement task alone?  Start from an MTL checkpoint, unfreeze the last K
backbone blocks up to the hooked feature layer + head, train with class-weighted CE on
engagement2/train, report BALANCED accuracy on val per epoch. Then re-measure detection
mAP to see what the detector paid.  K=0 reproduces the frozen-feature result."""
import os, sys, json, copy
import numpy as np, torch, torch.nn as nn
from sklearn.metrics import balanced_accuracy_score
from ultralytics import YOLO
import mtl
from mtl import eng_dataloader
R = os.path.expanduser("~/student_behavior/runs"); W = os.path.expanduser("~/student_behavior/work")
root = os.path.join(W, "engagement2"); dev = "cuda"
src = sys.argv[1] if len(sys.argv) > 1 else "mtl_scb5_s0"
Ks = [int(k) for k in (sys.argv[2:] or ["2", "4"])]
tr = eng_dataloader(root, 16, 640, "train"); va = eng_dataloader(root, 32, 640, "val")
ytr = np.array([y for _, y in tr.dataset.samples]); cw = torch.tensor(len(ytr) / (2 * np.bincount(ytr)), dtype=torch.float32, device=dev)

def evaluate(model):
    ps, ys = [], []
    model.eval()
    with torch.no_grad():
        for imgs, y in va:
            model._predict_once(imgs.to(dev)); ps += model.eng_head(model._feat).argmax(1).cpu().tolist(); ys += y.tolist()
    ps, ys = np.array(ps), np.array(ys)
    return round(100 * balanced_accuracy_score(ys, ps), 2), round(100 * (ps == ys).mean(), 2), int((ps == 1).sum())

out = {}
for K in Ks:
    model = YOLO(os.path.join(R, src, "weights", "best.pt")).model.float().to(dev)
    fi = model.feat_idx
    for p in model.parameters(): p.requires_grad_(False)
    train_params = list(model.eng_head.parameters())
    for i in range(max(0, fi - K + 1), fi + 1):
        for p in model.model[i].parameters(): p.requires_grad_(True); train_params.append(p)
    opt = torch.optim.AdamW([{"params": model.eng_head.parameters(), "lr": 1e-3},
                             {"params": [p for i in range(max(0, fi - K + 1), fi + 1) for p in model.model[i].parameters()], "lr": 1e-4}],
                            weight_decay=1e-4)
    ce = nn.CrossEntropyLoss(weight=cw); hist = []; best = (0, None)
    for ep in range(12):
        model.train()
        for m in model.model[:max(0, fi - K + 1)]: m.eval()
        for imgs, y in tr:
            model._predict_once(imgs.to(dev)); loss = ce(model.eng_head(model._feat), y.to(dev))
            opt.zero_grad(); loss.backward(); opt.step()
        b, a, n1 = evaluate(model); hist.append((ep, b, a, n1))
        if b > best[0]: best = (b, copy.deepcopy(model.state_dict()))
        print(f"{src} K={K} ep{ep:02d} val bal {b:5.2f} acc {a:5.2f} pred1 {n1}", flush=True)
    model.load_state_dict(best[1]); model.eval()
    torch.save({"state_dict": model.state_dict()}, os.path.join(R, f"unfreeze_{src}_K{K}.pt"))
    # detection cost: run ultralytics val with these weights via a temp YOLO wrapper
    y = YOLO(os.path.join(R, src, "weights", "best.pt")); y.model.load_state_dict(best[1]); y.model.to(dev).float()
    r = y.val(data=os.path.join(W, "scb5_hrw.yaml"), imgsz=640, batch=8, verbose=False)
    out[f"K{K}"] = {"best_val_bal": best[0], "hist": hist, "det_mAP50_after": round(float(r.box.map50) * 100, 2)}
    print(f"RESULT {src} K={K} best_bal={best[0]} det_mAP50_after={out[f'K{K}']['det_mAP50_after']}", flush=True)
json.dump(out, open(os.path.join(R, f"unfreeze_{src}.json"), "w"), indent=1)
