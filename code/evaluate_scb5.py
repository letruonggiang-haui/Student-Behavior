#!/usr/bin/env python3
"""Đánh giá cho bản ESWA: gộp nhiều seed (mean±std), đối chứng SOTA ngoài
(RT-DETR), các điểm Pareto (YOLO11n/s/m), FPS + số tham số cho từng mô hình.

python evaluate_eswa.py --work ~/student_behavior/work --runs ~/student_behavior/runs \
    --vn-data ~/student_behavior/work/vn.yaml --out ~/student_behavior/runs/final_eval_eswa.json
"""
import argparse
import json
import statistics as st
import time
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO, RTDETR

import mtl  # noqa: F401 — cần cho unpickle MTLDetectionModel
from mtl import eng_dataloader

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load(weights):
    """RTDETR checkpoint cần lớp RTDETR; còn lại dùng YOLO."""
    return RTDETR(str(weights)) if "rtdetr" in str(weights) else YOLO(str(weights))


def val_map(weights, data_yaml, per_class=False):
    m = load(weights)
    r = m.val(data=str(data_yaml), imgsz=640, batch=8, verbose=False)
    d = {"mAP50": round(float(r.box.map50), 4), "mAP50_95": round(float(r.box.map), 4),
         "precision": round(float(r.box.mp), 4), "recall": round(float(r.box.mr), 4),
         "params_M": round(sum(p.numel() for p in m.model.parameters()) / 1e6, 2)}
    if per_class:
        names = m.model.names if hasattr(m.model, "names") else {}
        d["per_class_AP50"] = {names.get(int(c), str(int(c))): round(float(ap), 4)
                               for c, ap in zip(r.box.ap_class_index, r.box.ap50)}
    return d


@torch.no_grad()
def fps_single(weights, img_dir, n=200, warmup=20):
    """FPS batch-1 của một mô hình detection đơn (predict API)."""
    m = load(weights)
    imgs = sorted(Path(img_dir).glob("*"))[:n + warmup]
    for p in imgs[:warmup]:
        m.predict(str(p), imgsz=640, verbose=False)
    torch.cuda.synchronize()
    t0 = time.time()
    for p in imgs[warmup:]:
        m.predict(str(p), imgsz=640, verbose=False)
    torch.cuda.synchronize()
    return round(n / (time.time() - t0), 2)


@torch.no_grad()
def fps_multitask(weights, img_dir, n=200, warmup=20):
    """FPS của mô hình đa nhiệm: 1 forward phục vụ cả detection lẫn engagement."""
    from PIL import Image
    import torchvision.transforms.functional as TF
    model = YOLO(str(weights)).model.float().to(DEVICE).eval()
    imgs = sorted(Path(img_dir).glob("*"))[:n + warmup]
    ts = [TF.to_tensor(Image.open(p).convert("RGB").resize((640, 640))).unsqueeze(0)
          for p in imgs]
    for t in ts[:warmup]:
        model._predict_once(t.to(DEVICE)); model.eng_head(model._feat)
    torch.cuda.synchronize()
    t0 = time.time()
    for t in ts[warmup:]:
        model._predict_once(t.to(DEVICE)); model.eng_head(model._feat)
    torch.cuda.synchronize()
    return round(n / (time.time() - t0), 2)


@torch.no_grad()
def eng_acc_mtl(weights, eng_w, eng_root):
    model = YOLO(str(weights)).model.float().to(DEVICE).eval()
    if Path(eng_w).exists():
        stt = torch.load(eng_w, map_location="cpu", weights_only=False)
        model.eng_head.load_state_dict(stt["eng_head"])
        model.to(DEVICE)
    loader = eng_dataloader(eng_root, 32, 640, "val")
    correct = total = 0
    for imgs, ys in loader:
        model._predict_once(imgs.to(DEVICE))
        correct += (model.eng_head(model._feat).argmax(1).cpu() == ys).sum().item()
        total += len(ys)
    return round(correct / max(total, 1), 4)


def agg(vals):
    """mean±std cho danh sách giá trị (std mẫu; 0 nếu chỉ 1 seed)."""
    return {"mean": round(st.mean(vals), 4),
            "std": round(st.stdev(vals), 4) if len(vals) > 1 else 0.0,
            "n": len(vals), "runs": [round(v, 4) for v in vals]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--vn-data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    work, runs = a.work.expanduser(), a.runs.expanduser()
    y = yaml.safe_load(open(a.vn_data))
    vn_val_dir = Path(y["path"]) / y["val"]
    R = {"seeds": {}, "sota": {}, "pareto": [], "engagement": {}, "per_class": {}}

    # --- các họ mô hình chạy nhiều seed ---
    # Thiết kế sạch: MTL một pha (COCO co-train STBD-08+engagement, KHÔNG pretrain 481);
    # baseline detection-only = B3 (COCO->STBD-08); bỏ B1 (nguồn 481 đã loại).
    families = {
        "MTL (proposed)": ["mtl_scb5_s0", "mtl_scb5_s1", "mtl_scb5_s2"],
        "Detection-only": ["det_scb5_s0", "det_scb5_s1", "det_scb5_s2"],
    }
    for fam, names in families.items():
        m50, m95 = [], []
        for nm in names:
            w = runs / nm / "weights" / "best.pt"
            if w.exists():
                try:
                    d = val_map(w, a.vn_data)
                    m50.append(d["mAP50"]); m95.append(d["mAP50_95"])
                except Exception as e:
                    print(f"WARN {nm}: {e}")
        if m50:
            R["seeds"][fam] = {"mAP50": agg(m50), "mAP50_95": agg(m95)}

    # --- per-class của mô hình đề xuất (seed 0) ---
    w0 = runs / "mtl_scb5_s0" / "weights" / "best.pt"
    if w0.exists():
        R["per_class"] = val_map(w0, a.vn_data, per_class=True).get("per_class_AP50", {})

    # --- đối chứng SOTA ngoài + điểm Pareto ---
    cands = [
        ("RT-DETR-l", runs / "rtdetr_scb5" / "weights" / "best.pt", "single"),
        ("YOLO11n", runs / "yolo11n_scb5" / "weights" / "best.pt", "single"),
        ("YOLO11m", runs / "yolo11m_scb5" / "weights" / "best.pt", "single"),
        ("YOLO11s (B3)", runs / "det_scb5_s0" / "weights" / "best.pt", "single"),
        ("MTL (proposed)", w0, "multitask"),
    ]
    for name, w, kind in cands:
        if not w.exists():
            continue
        try:
            d = val_map(w, a.vn_data)
            d["fps"] = (fps_multitask(w, vn_val_dir) if kind == "multitask"
                        else fps_single(w, vn_val_dir))
            d["name"], d["kind"] = name, kind
            R["pareto"].append(d)
            if name.startswith("RT-DETR"):
                R["sota"][name] = d
        except Exception as e:
            print(f"WARN pareto {name}: {e}")

    # --- FPS pipeline rời rạc (detection B3 + B2 engagement classifier) ---
    b1w = runs / "det_scb5_s0" / "weights" / "best.pt"
    b2w = runs / "b2_engagement2_cls" / "weights" / "best.pt"
    if b1w.exists() and b2w.exists():
        det, cls = YOLO(str(b1w)), YOLO(str(b2w))
        imgs = sorted(vn_val_dir.glob("*"))[:220]
        for p in imgs[:20]:
            det.predict(str(p), imgsz=640, verbose=False)
            cls.predict(str(p), imgsz=224, verbose=False)
        torch.cuda.synchronize(); t0 = time.time()
        for p in imgs[20:]:
            det.predict(str(p), imgsz=640, verbose=False)
            cls.predict(str(p), imgsz=224, verbose=False)
        torch.cuda.synchronize()
        R["fps_separate_pipeline"] = round(200 / (time.time() - t0), 2)

    # --- engagement (split subject-disjoint) ---
    engw = runs / "mtl_scb5_s0" / "eng_calibrated.pt"
    if w0.exists():
        R["engagement"]["mtl_val_acc"] = eng_acc_mtl(w0, engw, work / "engagement2")
    if b2w.exists():
        r = YOLO(str(b2w)).val(data=str(work / "engagement2"), imgsz=224, verbose=False)
        R["engagement"]["b2_top1"] = round(float(r.top1), 4)

    a.out.expanduser().write_text(json.dumps(R, indent=2, ensure_ascii=False))
    print(json.dumps(R, indent=2, ensure_ascii=False))
    print("EVAL_ESWA_DONE")


if __name__ == "__main__":
    main()
