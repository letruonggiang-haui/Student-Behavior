#!/usr/bin/env python3
"""Huấn luyện MTL bằng MTLTrainer (trainer chuẩn ultralytics + nhánh engagement).

python train_mtl.py --phase pretrain --det-data work/scb.yaml --eng-data work/engagement2 \
    --epochs 60 --batch 16 --project runs --name mtl2_pretrain
python train_mtl.py --phase finetune --det-data work/vn.yaml --eng-data work/engagement2 \
    --weights runs/mtl2_pretrain/weights/best.pt --epochs 40 --project runs --name mtl2_finetune_vn
"""
import argparse

from mtl import MTLTrainer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["pretrain", "finetune"], required=True)
    ap.add_argument("--det-data", required=True)
    ap.add_argument("--eng-data", required=True)
    ap.add_argument("--weights", default="yolo11s.pt")
    ap.add_argument("--eng-lambda", type=float, default=1.0)
    ap.add_argument("--eng-classes", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--project", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eng-batch", type=int, default=8)
    ap.add_argument("--eng-shuffle", action="store_true")
    ap.add_argument("--eng-bn-eval", action="store_true")
    ap.add_argument("--no-amp", action="store_true")
    args = ap.parse_args()

    overrides = dict(model=args.weights, data=args.det_data, epochs=args.epochs,
                     batch=args.batch, imgsz=args.imgsz, project=args.project,
                     name=args.name, exist_ok=True, patience=20, seed=args.seed,
                     amp=not args.no_amp)
    trainer = MTLTrainer(overrides=overrides)
    trainer.eng_root = args.eng_data
    trainer.eng_lambda = args.eng_lambda
    trainer.eng_classes = args.eng_classes
    trainer.eng_batch_size = args.eng_batch
    trainer.eng_shuffle = args.eng_shuffle
    trainer.eng_bn_eval = args.eng_bn_eval
    trainer.train()
    print(f"MTL_TRAIN_DONE phase={args.phase} best={trainer.best}")


if __name__ == "__main__":
    main()
