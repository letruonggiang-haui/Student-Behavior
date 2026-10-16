"""Detection baselines on the SCB5 Handrise-Read-write pack."""
import argparse, os, json
from ultralytics import YOLO

W = os.path.expanduser("~/student_behavior/work")
R = os.path.expanduser("~/student_behavior/runs")

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=os.path.join(W, "scb5_hrw.yaml"))
ap.add_argument("--weights", default="yolo11s.pt")
ap.add_argument("--name", default="scb5_det_s0")
ap.add_argument("--epochs", type=int, default=60)
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

m = YOLO(a.weights)
m.train(data=a.data, epochs=a.epochs, batch=a.batch, seed=a.seed,
        project=R, name=a.name, exist_ok=True, verbose=True)
res = m.val(data=a.data, project=R, name=a.name + "_val", exist_ok=True)
out = {"name": a.name, "weights": a.weights, "seed": a.seed,
       "mAP50": float(res.box.map50), "mAP50_95": float(res.box.map),
       "per_class_map50": {res.names[i]: float(v) for i, v in zip(res.box.ap_class_index, res.box.ap50)}}
print("RESULT " + json.dumps(out))
json.dump(out, open(os.path.join(R, a.name + ".json"), "w"), indent=1)
