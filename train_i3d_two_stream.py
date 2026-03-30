"""
Two-stream I3D training script.

Trains RGB and optical flow I3D streams jointly.  Each stream produces its own
logits; the training loss is the average of the per-stream cross-entropy losses.
At evaluation, stream logits are averaged before computing accuracy (late fusion).

Usage
-----
CUDA_VISIBLE_DEVICES=0 python train_i3d_two_stream.py \
    --dataset ucf101 \
    --run_name i3d_twostream_ucf101

All other arguments have sensible defaults (see parse_args below).

Data pipeline
-------------
Uses the existing FlowDatasetWrapper which requires pre-cached optical flow
(flow.npy files next to the frame directories).  If flow is not cached it will
be computed automatically on the first run via ensure_flows().
"""

import argparse
import atexit
import contextlib
import os
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloaders.dataset import VideoDataset
from network.video.i3d import I3DTwoStream
from utils.data_factory import FlowDatasetWrapper


def parse_args():
    parser = argparse.ArgumentParser(
        description="Two-stream I3D training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["ucf10", "ucf11", "ucf101", "hmdb51"])
    parser.add_argument("--run_name", type=str, required=True)

    parser.add_argument("--epochs",       type=int,   default=100)
    parser.add_argument("--batch_size",   type=int,   default=16)
    parser.add_argument("--lr",           type=float, default=1e-4)
    parser.add_argument("--fc_lr_mult",   type=float, default=10.0,
                        help="LR multiplier for the FC classifiers (both streams)")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--dropout",      type=float, default=0.5)
    parser.add_argument("--clip_len",     type=int,   default=16,
                        help="Frames per clip.  Paper uses 64; 16 works for short clips.")

    parser.add_argument("--num_workers",  type=int,   default=8)
    parser.add_argument("--device",       type=str,   default="auto",
                        choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--no_amp",       action="store_true",
                        help="Disable AMP (float16).  Not usually needed for I3D.")
    parser.add_argument("--resume",       type=str,   default=None,
                        help="Path to checkpoint to resume from.")
    parser.add_argument("--test_only",    action="store_true",
                        help="Skip training; evaluate the loaded checkpoint on the test set.")

    args = parser.parse_args()

    ds_classes = {"ucf10": 10, "ucf11": 11, "ucf101": 101, "hmdb51": 51}
    args.num_classes = ds_classes[args.dataset]

    return args


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_loaders(args):
    """Build train / val / test DataLoaders with the FlowDatasetWrapper."""
    pin_memory = torch.cuda.is_available()
    use_workers = args.num_workers > 0
    worker_kwargs = (dict(persistent_workers=True, prefetch_factor=2)
                     if use_workers else {})

    def _loader(split, shuffle):
        ds = FlowDatasetWrapper(
            VideoDataset(dataset=args.dataset, split=split,
                         clip_len=args.clip_len, preprocess=False,
                         augment=(split == "train"))
        )
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                          num_workers=args.num_workers, pin_memory=pin_memory,
                          **worker_kwargs)

    return {
        "train": _loader("train", shuffle=True),
        "val":   _loader("val",   shuffle=False),
        "test":  _loader("test",  shuffle=False),
    }


def _build_optimizer(model: I3DTwoStream, args):
    """AdamW with BN/bias excluded from weight decay; 10× LR on FC layers."""

    def split_wd(params):
        decay, no_decay = [], []
        for p in params:
            (no_decay if p.ndim <= 1 else decay).append(p)
        return decay, no_decay

    decay_1x, no_decay_1x = split_wd(model.get_1x_lr_params())
    decay_fc, no_decay_fc = split_wd(model.get_10x_lr_params())
    fc_lr = args.lr * args.fc_lr_mult

    param_groups = [
        {"params": decay_1x,    "lr": args.lr,  "weight_decay": args.weight_decay},
        {"params": no_decay_1x, "lr": args.lr,  "weight_decay": 0.0},
        {"params": decay_fc,    "lr": fc_lr,    "weight_decay": args.weight_decay},
        {"params": no_decay_fc, "lr": fc_lr,    "weight_decay": 0.0},
    ]
    optimizer = optim.AdamW(param_groups, lr=args.lr,
                            weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)
    return optimizer, scheduler


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class I3DTrainer:
    def __init__(self, args):
        self.args = args
        self.start_epoch = 0
        self.best_acc = 0.0

        # Device
        if args.device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(args.device)
        print(f"==> Device: {self.device}")

        # W&B
        import wandb
        self.wandb = wandb
        timestamp = datetime.now().strftime("%m%d-%H%M")
        self.out_dir = os.path.join("runs", f"{args.run_name}_{timestamp}")
        os.makedirs(os.path.join(self.out_dir, "checkpoints"), exist_ok=True)
        wandb.init(name=args.run_name, dir=self.out_dir, config=vars(args))
        atexit.register(lambda: wandb.finish() if wandb else None)

        # Model
        print(f"==> Building I3D two-stream (clip_len={args.clip_len})")
        self.model = I3DTwoStream(
            num_classes=args.num_classes,
            dropout_prob=args.dropout,
            clip_len=args.clip_len,
        ).to(self.device)

        total = sum(p.numel() for p in self.model.parameters())
        print(f"==> Total parameters: {total:,}")
        wandb.config.update({"total_params": total})

        # Data
        print("==> Loading data (flow will be cached on first run)...")
        self.loaders = _make_loaders(args)

        # Loss / optimiser / schedule
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=args.label_smoothing).to(self.device)
        self.optimizer, self.scheduler = _build_optimizer(self.model, args)

        # AMP
        self.amp_enabled = (self.device.type == "cuda") and not args.no_amp
        self.scaler = GradScaler(device="cuda") if self.amp_enabled else None

        # Resume
        if args.resume and os.path.isfile(args.resume):
            print(f"==> Resuming from {args.resume}")
            ckpt = torch.load(args.resume, map_location=self.device)
            self.start_epoch = ckpt["epoch"]
            self.best_acc = ckpt.get("best_acc", 0.0)
            self.model.load_state_dict(ckpt["state_dict"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            if "scheduler" in ckpt:
                self.scheduler.load_state_dict(ckpt["scheduler"])

    # -----------------------------------------------------------------------

    def _run_epoch(self, epoch: int, split: str):
        is_train = split == "train"
        self.model.train() if is_train else self.model.eval()
        loader = self.loaders[split]

        stats = {"loss": 0.0, "correct": 0, "total": 0, "batches": 0,
                 "grad_norm": 0.0, "grad_batches": 0, "nan_grad_batches": 0,
                 "skipped": 0}

        pbar = tqdm(loader, desc=f"Ep {epoch+1} [{split.upper()}]")

        for inputs, targets in pbar:
            # inputs is [rgb, flow] from FlowDatasetWrapper
            rgb  = inputs[0].to(self.device)
            flow = inputs[1].to(self.device)
            targets = targets.to(self.device, dtype=torch.long).view(-1)

            # Skip non-finite inputs
            if not (torch.isfinite(rgb).all() and torch.isfinite(flow).all()):
                stats["skipped"] += 1
                continue

            if is_train:
                self.optimizer.zero_grad()

            amp_ctx = (autocast(device_type=self.device.type)
                       if self.amp_enabled else contextlib.nullcontext())
            grad_ctx = (contextlib.nullcontext() if is_train
                        else torch.no_grad())

            with amp_ctx, grad_ctx:
                rgb_logits, flow_logits = self.model([rgb, flow])
                loss_rgb  = self.criterion(rgb_logits,  targets)
                loss_flow = self.criterion(flow_logits, targets)
                loss = (loss_rgb + loss_flow) * 0.5

            if not torch.isfinite(loss):
                stats["skipped"] += 1
                if is_train:
                    self.optimizer.zero_grad(set_to_none=True)
                continue

            if is_train:
                if self.scaler:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                else:
                    loss.backward()

                bad_grads = any(
                    p.grad is not None and not torch.isfinite(p.grad).all()
                    for p in self.model.parameters()
                )
                if bad_grads:
                    stats["nan_grad_batches"] += 1
                    self.optimizer.zero_grad(set_to_none=True)
                    if self.scaler:
                        self.scaler.update()
                else:
                    gn = nn.utils.clip_grad_norm_(self.model.parameters(), 1.0).item()
                    stats["grad_norm"]    += gn
                    stats["grad_batches"] += 1
                    if self.scaler:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()

            # Accuracy: average the two stream logits (late fusion)
            fused_logits = (rgb_logits + flow_logits) * 0.5
            preds = fused_logits.argmax(dim=1)

            stats["loss"]    += loss.item()
            stats["batches"] += 1
            stats["total"]   += targets.size(0)
            stats["correct"] += preds.eq(targets).sum().item()

            postfix = {
                "L":  f"{stats['loss']/stats['batches']:.3f}",
                "A":  f"{100.*stats['correct']/stats['total']:.1f}%",
            }
            if is_train and stats["grad_batches"]:
                postfix["gn"] = f"{stats['grad_norm']/stats['grad_batches']:.2f}"
            pbar.set_postfix(postfix)

        n = stats["batches"]
        result = {
            "loss": stats["loss"] / n if n else 0.0,
            "acc":  100. * stats["correct"] / stats["total"] if stats["total"] else 0.0,
        }
        if is_train:
            result["grad_norm"]        = stats["grad_norm"] / max(1, stats["grad_batches"])
            result["nan_grad_batches"] = stats["nan_grad_batches"]
            result["skipped"]          = stats["skipped"]
        return result

    def _save(self, epoch: int, tag: str):
        torch.save(
            {"epoch": epoch, "state_dict": self.model.state_dict(),
             "optimizer": self.optimizer.state_dict(),
             "scheduler": self.scheduler.state_dict(),
             "best_acc": self.best_acc},
            os.path.join(self.out_dir, "checkpoints", f"{tag}.pth"),
        )

    # -----------------------------------------------------------------------

    def run(self):
        if self.args.test_only:
            test = self._run_epoch(0, "test")
            print(f"Test | Loss: {test['loss']:.3f} | Acc: {test['acc']:.2f}%")
            self.wandb.finish()
            return

        start = time.time()
        for epoch in range(self.start_epoch, self.args.epochs):
            t = self._run_epoch(epoch, "train")
            v = self._run_epoch(epoch, "val")
            self.scheduler.step()

            print(
                f"Ep {epoch+1} | "
                f"T_Loss: {t['loss']:.3f} T_Acc: {t['acc']:.1f}% | "
                f"V_Loss: {v['loss']:.3f} V_Acc: {v['acc']:.1f}% | "
                f"GN: {t['grad_norm']:.3f} | "
                f"NaN_Grad: {t['nan_grad_batches']} Skip: {t['skipped']}"
            )

            log = {f"train/{k}": val for k, val in t.items()}
            log.update({f"val/{k}": val for k, val in v.items()})
            log["epoch"] = epoch + 1
            log["lr"]    = self.optimizer.param_groups[0]["lr"]
            if self.scaler:
                log["amp/scale"] = self.scaler.get_scale()
            self.wandb.log(log)

            if v["acc"] > self.best_acc:
                self.best_acc = v["acc"]
                self._save(epoch + 1, "best_model")

        self._save(self.args.epochs, "last_model")

        print("==> Running final test evaluation...")
        test = self._run_epoch(self.args.epochs - 1, "test")
        print(f"Test | Loss: {test['loss']:.3f} | Acc: {test['acc']:.2f}%")
        self.wandb.summary.update({f"test/{k}": v for k, v in test.items()})
        self.wandb.summary["total_runtime_sec"] = time.time() - start
        self.wandb.finish()


if __name__ == "__main__":
    I3DTrainer(parse_args()).run()
