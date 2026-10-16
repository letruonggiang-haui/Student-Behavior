"""Throughput under ONE timed region for every configuration (revision, 2026-09).

Timed region per frame, identical for all rows: read the JPEG from disk (cv2.imread),
Ultralytics predict() on the array (letterbox, normalise, forward, NMS, result objects),
then, for multi-task models, the engagement head on the hooked stride-32 map, and for the
two-model pipeline, the engagement classifier on the same array at 224 px.
FP32 (Ultralytics default, no half precision), batch 1, conf 0.25, IoU 0.7, max_det 300 (Ultralytics defaults).
20 warm-up frames, then R repeats of 200 frames; repeats are interleaved across
configurations so thermal drift affects all rows alike. Reports median FPS over repeats,
min/max, and per-frame latency percentiles. Aborts if another process holds the GPU.
"""
import os, json, time, subprocess, statistics as st
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO, RTDETR

import mtl  # noqa: F401  (unpickles MTLDetectionModel)

R = Path(os.path.expanduser("~/student_behavior/runs"))
VAL = Path(os.path.expanduser("~/student_behavior/work/scb5_hrw/images/val"))
OUT = R / "revision" / "fps_v2.json"
N, WARM, REPEATS = 200, 20, 5
CLS_W = R / "eng_e2e_cls_s" / "weights" / "best.pt"  # YOLO11s-cls trained on EngageNet frames

CONFIGS = [
    ("YOLO11n", "det", R / "yolo11n_scb5" / "weights" / "best.pt"),
    ("Detection-only YOLO11s", "det", R / "det_scb5_s0" / "weights" / "best.pt"),
    ("YOLO11m", "det", R / "yolo11m_scb5" / "weights" / "best.pt"),
    ("Detection-only YOLO11l", "det", R / "det_l_s0" / "weights" / "best.pt"),
    ("RT-DETR-l", "rtdetr", R / "rtdetr_scb5" / "weights" / "best.pt"),
    ("Multi-task YOLO11s, Kaggle head", "mtl", R / "mtl_scb5_s0" / "weights" / "best.pt"),
    ("Multi-task YOLO11l, Kaggle head", "mtl", R / "mtl_l_s0" / "weights" / "best.pt"),
    ("Multi-task YOLO11s, EngageNet head", "mtl", R / "mtl_en_s_s0" / "weights" / "best.pt"),
    ("Multi-task YOLO11l, EngageNet head", "mtl", R / "mtl_en_l_s0" / "weights" / "best.pt"),
    ("Two-model pipeline, YOLO11s + classifier", "pipe", R / "det_scb5_s0" / "weights" / "best.pt"),
    ("Two-model pipeline, YOLO11l + classifier", "pipe", R / "det_l_s0" / "weights" / "best.pt"),
]


def gpu_others():
    q = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                       capture_output=True, text=True).stdout.split()
    return [p for p in q if p.strip() and int(p) != os.getpid()]


def ensure_hook(model):
    if getattr(model, "_feat", None) is not None:
        return
    idx = max(i for i, m in enumerate(model.model) if m.__class__.__name__ in {"C2PSA", "SPPF"})
    model.model[idx].register_forward_hook(lambda mod, i, o: setattr(model, "_feat", o))


def build(kind, w, cls):
    m = RTDETR(str(w)) if kind == "rtdetr" else YOLO(str(w))

    def det(img):
        return m.predict(img, imgsz=640, verbose=False)

    if kind in ("det", "rtdetr"):
        return lambda img: det(img)[0].boxes.data.shape
    if kind == "mtl":
        def f(img):
            r = det(img)
            core = m.predictor.model.model if hasattr(m.predictor.model, "model") else m.model
            feat = getattr(m.model, "_feat", None)
            if feat is None:
                feat = getattr(core, "_feat", None)
            head = m.model.eng_head
            return int(head(feat.float()).argmax(1).item()), r[0].boxes.data.shape
        return f
    if kind == "pipe":
        def f(img):
            r = det(img)
            c = cls.predict(img, imgsz=224, verbose=False)
            return int(c[0].probs.top1), r[0].boxes.data.shape
        return f
    raise ValueError(kind)


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    others = gpu_others()
    if others:
        raise SystemExit(f"GPU busy (pids {others}); throughput not measured")
    imgs = sorted(VAL.glob("*.jpg"))[:N + WARM]
    cls = YOLO(str(CLS_W))
    fns = {}
    for name, kind, w in CONFIGS:
        if not w.exists():
            print("missing", name, w); continue
        fns[name] = build(kind, w, cls)
        for p in imgs[:WARM]:           # warm-up, also creates the predictor
            fns[name](cv2.imread(str(p)))
        if kind == "mtl":
            m_ok = fns[name](cv2.imread(str(imgs[0])))
            print(name, "engagement output ok:", m_ok[0], flush=True)
    lat = {k: [] for k in fns}
    fps = {k: [] for k in fns}
    for rep in range(REPEATS):
        for name, f in fns.items():
            if gpu_others():
                raise SystemExit("another process took the GPU during timing; aborting")
            torch.cuda.synchronize()
            t_all = time.perf_counter()
            for p in imgs[WARM:]:
                t0 = time.perf_counter()
                f(cv2.imread(str(p)))
                torch.cuda.synchronize()
                lat[name].append((time.perf_counter() - t0) * 1000)
            fps[name].append(N / (time.perf_counter() - t_all))
            print(f"rep {rep} {name:45s} {fps[name][-1]:7.2f} FPS", flush=True)
    out = {"protocol": __doc__.strip(), "gpu": torch.cuda.get_device_name(0),
           "torch": torch.__version__, "n_frames": N, "warmup": WARM, "repeats": REPEATS, "rows": {}}
    for name, _, w in CONFIGS:
        if name not in fps:
            continue
        L = np.array(lat[name])
        out["rows"][name] = {"fps_median": round(st.median(fps[name]), 2),
                             "fps_min": round(min(fps[name]), 2), "fps_max": round(max(fps[name]), 2),
                             "fps_repeats": [round(x, 2) for x in fps[name]],
                             "latency_ms_p50": round(float(np.percentile(L, 50)), 2),
                             "latency_ms_p90": round(float(np.percentile(L, 90)), 2),
                             "weights": str(w)}
    json.dump(out, open(OUT, "w"), indent=1)
    print(json.dumps(out["rows"], indent=1))
    print("FPS_V2_DONE")


if __name__ == "__main__":
    main()
