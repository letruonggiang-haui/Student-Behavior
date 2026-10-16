#!/usr/bin/env python3
"""MTL trên trainer chuẩn ultralytics (mosaic/EMA/warmup/fitness-mAP có sẵn).

- MTLDetectionModel: DetectionModel + EngagementHead trên đặc trưng stride-32
  (hook C2PSA/SPPF cuối backbone). loss() = det_loss + lambda * CE(engagement).
- MTLTrainer: DetectionTrainer nạp xen 1 batch engagement mỗi iteration qua
  preprocess_batch; best.pt vẫn chọn theo fitness mAP của validator chuẩn.

Module này phải import được (không định nghĩa trong __main__) để checkpoint
unpickle được ở mọi script: torch pickle tham chiếu mtl.MTLDetectionModel.
"""
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel


class EngagementHead(nn.Module):
    """Global pooled MLP head on stride-32 backbone features."""

    def __init__(self, c_in: int, n_cls: int = 2, hidden: int = 256):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c_in, hidden),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, n_cls),
        )

    def forward(self, feat):
        return self.fc(self.pool(feat))


class MTLDetectionModel(DetectionModel):
    """DetectionModel + engagement head. Batch engagement được trainer gắn vào
    self._eng_batch trước mỗi lần gọi loss(); tiêu thụ xong thì xoá."""

    def attach_eng(self, n_eng: int = 2):
        idx = None
        for i, m in enumerate(self.model):
            if m.__class__.__name__ in {"C2PSA", "SPPF"}:
                idx = i
        assert idx is not None, "no C2PSA/SPPF layer found"
        self.feat_idx = idx
        self._feat = None
        self._eng_batch = None
        self._eng_loss_val = 0.0
        self.eng_lambda = 1.0
        self.model[idx].register_forward_hook(self._hook)
        was_training = self.training
        self.eval()
        with torch.no_grad():
            dev = next(self.parameters()).device
            self._predict_once(torch.zeros(1, 3, 640, 640, device=dev))
        if was_training:
            self.train()
        self.eng_head = EngagementHead(self._feat.shape[1], n_eng).to(next(self.parameters()).device)

    def _hook(self, module, inp, out):
        self._feat = out

    def eng_logits(self, imgs):
        """imgs float 0-1 (B,3,H,W): một forward detection, hook nuôi eng head."""
        self._predict_once(imgs)
        return self.eng_head(self._feat)

    def loss(self, batch, preds=None):
        dloss, items = super().loss(batch, preds)
        eng = getattr(self, "_eng_batch", None)
        if eng is not None and self.training:
            self._eng_batch = None
            imgs, ys = eng
            bn_eval = getattr(self, "eng_bn_eval", False)
            if bn_eval:
                import torch.nn as _nn
                bns = [mm for mm in self.modules() if isinstance(mm, _nn.modules.batchnorm._BatchNorm)]
                for b in bns:
                    b.eval()
            logits = self.eng_logits(imgs)
            if bn_eval:
                for b in bns:
                    b.train()
            closs = F.cross_entropy(logits, ys)
            self._eng_loss_val = float(closs.detach())
            dloss = dloss + self.eng_lambda * closs * imgs.shape[0]
        self._feat = None  # keep no graph-attached tensor on the module (EMA deepcopy)
        return dloss, items


def eng_dataloader(root, batch, imgsz=640, split="train", shuffle_labels=False, seed=0):
    tf = transforms.Compose([
        transforms.Resize((imgsz, imgsz)),
        *([transforms.RandomHorizontalFlip()] if split == "train" else []),
        transforms.ToTensor(),
    ])
    ds = datasets.ImageFolder(str(Path(root) / split), transform=tf)
    if shuffle_labels:
        # control: same images and class frequencies, labels permuted once (no signal)
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(len(ds.targets), generator=g).tolist()
        tg = [ds.targets[i] for i in perm]
        ds.samples = [(pth, t) for (pth, _), t in zip(ds.samples, tg)]
        ds.imgs = ds.samples
        ds.targets = tg
    return DataLoader(ds, batch_size=batch, shuffle=(split == "train"), num_workers=4,
                      pin_memory=True, drop_last=(split == "train"))


def as_mtl(model):
    """Ép một DetectionModel thuần (checkpoint YOLO/ultralytics) sang MTLDetectionModel
    để gắn được eng_head; không đổi tham số, chỉ đổi lớp."""
    if not isinstance(model, MTLDetectionModel):
        model.__class__ = MTLDetectionModel
    return model


class MTLTrainer(DetectionTrainer):
    """DetectionTrainer + nạp xen batch engagement. Tham số riêng truyền qua
    class attributes (đặt trước khi train) vì get_cfg không nhận key lạ."""

    eng_root: str = None
    eng_lambda: float = 1.0
    eng_batch_size: int = 8
    eng_weights: str = None  # eng_head state_dict (.pt with key "eng_head")
    eng_classes: int = 2      # số lớp engagement (2 cho binary, 4 cho EngageNet)
    eng_shuffle: bool = False  # control: permuted engagement labels
    eng_bn_eval: bool = False  # control: engagement frames do not update batch-norm statistics

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = MTLDetectionModel(cfg, nc=self.data["nc"], verbose=verbose)
        model.attach_eng(self.eng_classes)
        if weights:
            model.load(weights)  # intersect: nạp cả eng_head nếu ckpt nguồn có
        if self.eng_weights:
            st = torch.load(self.eng_weights, map_location="cpu", weights_only=False)
            model.eng_head.load_state_dict(st["eng_head"])
        model.eng_lambda = self.eng_lambda
        model.eng_bn_eval = self.eng_bn_eval
        self._eng_iter = None
        return model

    def _next_eng(self):
        if self._eng_iter is None:
            self._eng_loader = eng_dataloader(self.eng_root, self.eng_batch_size,
                                              self.args.imgsz, "train",
                                              shuffle_labels=self.eng_shuffle)
            self._eng_iter = iter(self._eng_loader)
        try:
            imgs, ys = next(self._eng_iter)
        except StopIteration:
            self._eng_iter = iter(self._eng_loader)
            imgs, ys = next(self._eng_iter)
        return imgs.to(self.device, non_blocking=True), ys.to(self.device)

    def preprocess_batch(self, batch):
        batch = super().preprocess_batch(batch)
        if self.eng_root:
            m = self.model.module if hasattr(self.model, "module") else self.model
            m._eng_batch = self._next_eng()
        return batch
