#!/usr/bin/env python3
"""Đánh giá chuyển giao chéo quốc gia: đặc trưng học từ lớp học Trung Quốc
(SCB) có nhận diện được hành vi trong lớp học Việt Nam không?

Thiết kế: linear probe trên đặc trưng backbone đóng băng, đánh giá
leave-one-subject-out (4 chủ thể VN) -> so sánh các backbone:
  - coco      : YOLO11s trọng số COCO (không thấy lớp học nào)
  - scb_det   : detection-only huấn luyện trên SCB (lớp học TQ)
  - scb_mtl   : mô hình đa nhiệm đề xuất (SCB + engagement)

Giữ nguyên khung hình giữa các nhánh so sánh (đều là ảnh crop VN), nên
chênh lệch phản ánh chất lượng đặc trưng chứ không phải khác biệt bố cục.

python cross_country.py --vn-dir ~/student_behavior/vn_sample \
    --runs ~/student_behavior/runs --out ~/student_behavior/runs/cross_country.json
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from ultralytics import YOLO

import mtl  # noqa: F401 — cần cho unpickle MTLDetectionModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASSES = ["handrise", "look_forward", "read", "sleep", "stand", "turn_head", "write"]


class FeatureExtractor:
    """Lấy đặc trưng gộp toàn cục ở tầng stride-32 của bất kỳ DetectionModel nào."""

    def __init__(self, weights):
        self.model = YOLO(str(weights)).model.float().to(DEVICE).eval()
        idx = None
        for i, m in enumerate(self.model.model):
            if m.__class__.__name__ in {"C2PSA", "SPPF"}:
                idx = i
        assert idx is not None, f"không tìm thấy tầng C2PSA/SPPF trong {weights}"
        self._feat = None
        self.model.model[idx].register_forward_hook(self._hook)

    def _hook(self, module, inp, out):
        self._feat = out

    @torch.no_grad()
    def __call__(self, batch):
        self.model._predict_once(batch.to(DEVICE))
        return self._feat.mean(dim=(2, 3)).cpu()  # global average pool


def load_dataset(vn_dir, imgsz=640, batch=32):
    """Trả về (X, y, subject) — mỗi ảnh là 1 clip (frame giữa), tránh rò rỉ trong clip."""
    tf = transforms.Compose([transforms.Resize((imgsz, imgsz)), transforms.ToTensor()])
    items = []
    for ci, cls in enumerate(CLASSES):
        for f in sorted((Path(vn_dir) / cls).glob("*.jpg")):
            items.append((f, ci, f.name.split("_")[0]))  # ID nằm ở đầu tên file
    print(f"nạp {len(items)} ảnh, {len(CLASSES)} lớp, "
          f"{len(set(s for _, _, s in items))} chủ thể")
    return items, tf, batch


def extract_all(extractor, items, tf, batch):
    xs, ys, ss = [], [], []
    buf = []
    for f, ci, sid in items:
        buf.append((tf(Image.open(f).convert("RGB")), ci, sid))
        if len(buf) == batch:
            xs.append(extractor(torch.stack([b[0] for b in buf])))
            ys += [b[1] for b in buf]; ss += [b[2] for b in buf]; buf = []
    if buf:
        xs.append(extractor(torch.stack([b[0] for b in buf])))
        ys += [b[1] for b in buf]; ss += [b[2] for b in buf]
    return torch.cat(xs), torch.tensor(ys), ss


def linear_probe(Xtr, ytr, Xte, yte, epochs=300, lr=1e-2, wd=1e-4, seed=0):
    """Hồi quy logistic đa lớp trên đặc trưng đã chuẩn hoá."""
    torch.manual_seed(seed)
    mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    clf = nn.Linear(Xtr.shape[1], len(CLASSES)).to(DEVICE)
    opt = torch.optim.AdamW(clf.parameters(), lr=lr, weight_decay=wd)
    ce = nn.CrossEntropyLoss()
    Xtr, ytr = Xtr.to(DEVICE), ytr.to(DEVICE)
    for _ in range(epochs):
        opt.zero_grad()
        ce(clf(Xtr), ytr).backward()
        opt.step()
    with torch.no_grad():
        pred = clf(Xte.to(DEVICE)).argmax(1).cpu()
    acc = (pred == yte).float().mean().item()
    # macro-F1
    f1s = []
    for c in range(len(CLASSES)):
        tp = ((pred == c) & (yte == c)).sum().item()
        fp = ((pred == c) & (yte != c)).sum().item()
        fn = ((pred != c) & (yte == c)).sum().item()
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
    per_cls_recall = {}
    for c, name in enumerate(CLASSES):
        m = yte == c
        per_cls_recall[name] = round(((pred == c) & m).sum().item() / max(m.sum().item(), 1), 4)
    return acc, sum(f1s) / len(f1s), per_cls_recall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vn-dir", type=Path, required=True)
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    runs = a.runs.expanduser()

    backbones = {
        "COCO (no classroom data)": "yolo11s.pt",
        "Detection-only (SCB5, 3-class)": runs / "det_scb5_s0" / "weights" / "best.pt",
        "Proposed multi-task (SCB5, 3-class)": runs / "mtl_scb5_s0" / "weights" / "best.pt",
    }
    items, tf, batch = load_dataset(a.vn_dir.expanduser())
    subjects = sorted(set(s for _, _, s in items))
    R = {"subjects": subjects, "n_samples": len(items), "classes": CLASSES, "results": {}}

    for name, w in backbones.items():
        if not Path(w).exists() and not str(w).endswith(".pt"):
            print(f"bỏ qua {name}: không có {w}")
            continue
        print(f"\n=== backbone: {name} ===", flush=True)
        try:
            ex = FeatureExtractor(w)
            X, y, ss = extract_all(ex, items, tf, batch)
            del ex
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"LỖI {name}: {e}")
            continue
        folds, f1s, recalls = [], [], defaultdict(list)
        for held in subjects:                       # leave-one-subject-out
            te = torch.tensor([s == held for s in ss])
            acc, f1, rec = linear_probe(X[~te], y[~te], X[te], y[te])
            folds.append(acc); f1s.append(f1)
            for k, v in rec.items():
                recalls[k].append(v)
            print(f"  held-out {held}: acc={acc:.4f} macroF1={f1:.4f}", flush=True)
        mean = sum(folds) / len(folds)
        std = (sum((x - mean) ** 2 for x in folds) / (len(folds) - 1)) ** 0.5
        R["results"][name] = {
            "acc_mean": round(mean, 4), "acc_std": round(std, 4),
            "acc_folds": [round(x, 4) for x in folds],
            "macroF1_mean": round(sum(f1s) / len(f1s), 4),
            "per_class_recall_mean": {k: round(sum(v) / len(v), 4) for k, v in recalls.items()},
        }
        print(f"  -> {mean:.4f} ± {std:.4f}")

    a.out.expanduser().write_text(json.dumps(R, indent=2, ensure_ascii=False))
    print(json.dumps(R["results"], indent=2, ensure_ascii=False))
    print("CROSS_COUNTRY_DONE")


if __name__ == "__main__":
    main()
