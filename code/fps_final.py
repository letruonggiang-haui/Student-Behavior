"""Throughput on an idle GPU for every candidate configuration."""
import os, json, time, yaml, statistics as st
import torch
from pathlib import Path
from ultralytics import YOLO
import mtl  # noqa: F401 — unpickles MTLDetectionModel
from evaluate_scb5 import fps_single, fps_multitask  # reuse the exact timing code

W = Path(os.path.expanduser("~/student_behavior/work"))
R = Path(os.path.expanduser("~/student_behavior/runs"))
y = yaml.safe_load(open(W / "scb5_hrw.yaml"))
val_dir = Path(y["path"]) / y["val"]

CAND = [
    ("MTL YOLO11s (proposed, fast)", R / "mtl_scb5_s0" / "weights" / "best.pt", "multitask"),
    ("MTL YOLO11l (proposed, accurate)", R / "mtl_l_s0" / "weights" / "best.pt", "multitask"),
    ("Detection-only YOLO11s", R / "det_scb5_s0" / "weights" / "best.pt", "single"),
    ("Detection-only YOLO11l", R / "det_l_s0" / "weights" / "best.pt", "single"),
    ("YOLO11m", R / "yolo11m_scb5" / "weights" / "best.pt", "single"),
    ("RT-DETR-l", R / "rtdetr_scb5" / "weights" / "best.pt", "single"),
]
out = {}
for name, w, kind in CAND:
    if not w.exists():
        print("missing", name); continue
    f = fps_multitask(w, val_dir) if kind == "multitask" else fps_single(w, val_dir)
    m = YOLO(str(w))
    p = sum(x.numel() for x in m.model.parameters()) / 1e6
    out[name] = {"fps": f, "params_M": round(p, 2), "kind": kind}
    print(f"{name:34s} {f:8.2f} FPS   {p:6.2f} M", flush=True)
    del m; torch.cuda.empty_cache()

# two-model pipeline: detector + separate engagement classifier
b2 = R / "b2_engagement2_cls" / "weights" / "best.pt"
for det_name, det_w in (("YOLO11s", R / "det_scb5_s0" / "weights" / "best.pt"),
                        ("YOLO11l", R / "det_l_s0" / "weights" / "best.pt")):
    if not (det_w.exists() and b2.exists()):
        continue
    det, cls = YOLO(str(det_w)), YOLO(str(b2))
    imgs = sorted(val_dir.glob("*"))[:220]
    for p in imgs[:20]:
        det.predict(str(p), imgsz=640, verbose=False); cls.predict(str(p), imgsz=224, verbose=False)
    torch.cuda.synchronize(); t0 = time.time()
    for p in imgs[20:]:
        det.predict(str(p), imgsz=640, verbose=False); cls.predict(str(p), imgsz=224, verbose=False)
    torch.cuda.synchronize()
    f = round(200 / (time.time() - t0), 2)
    out[f"Two-model pipeline ({det_name} + classifier)"] = {"fps": f, "kind": "pipeline"}
    print(f"{'Two-model pipeline (' + det_name + ')':34s} {f:8.2f} FPS", flush=True)
    del det, cls; torch.cuda.empty_cache()

json.dump(out, open(R / "fps_final.json", "w"), indent=1)
print("written", R / "fps_final.json")
