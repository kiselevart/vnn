"""
Two-stream I3D training script — DDP-capable.

Single GPU:
    CUDA_VISIBLE_DEVICES=0 python tools/train_i3d_two_stream.py \
        --dataset ucf101 --run_name i3d_twostream_ucf101

4-GPU DDP (scale --lr linearly: 1 GPU = 1e-4, 4 GPUs = 4e-4):
    NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=4,5,6,7 \
    torchrun --nproc_per_node=4 --master_port=29502 tools/train_i3d_two_stream.py \
        --dataset ucf101 --run_name i3d_twostream_ucf101 --lr 4e-4

Data pipeline
-------------
Uses FlowDatasetWrapper which requires pre-cached optical flow (flow.npy files next
to the frame directories).  Flow is computed automatically on the first run.
"""

import argparse
import atexit
import contextlib
import os
import sys
import time
from datetime import datetime

# Allow running as `torchrun tools/train_i3d_two_stream.py` from the vnn/ root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from dataloaders.dataset import VideoDataset
from network.video.i3d import I3DTwoStream
from utils.data_factory import FlowDatasetWrapper


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Two-stream I3D training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["ucf10", "ucf11", "ucf101", "hmdb51"])
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--split",       type=int,   default=1, choices=[1, 2, 3])

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
    parser.add_argument("--no_wandb",     action="store_true")
    parser.add_argument("--wandb_group",  type=str,   default=None,
                        help="W&B run group for grouping multi-split runs.")
    parser.add_argument("--resume",       type=str,   default=None,
                        help="Path to checkpoint to resume from.")
    parser.add_argument("--test_only",    action="store_true",
                        help="Skip training; evaluate the loaded checkpoint on the test set.")

    args = parser.parse_args()

    ds_classes = {"ucf10": 10, "ucf11": 11, "ucf101": 101, "hmdb51": 51}
    args.num_classes = ds_classes[args.dataset]

    return args


# ---------------------------------------------------------------------------
# W&B no-op stub for non-main ranks
# ---------------------------------------------------------------------------

class _SummaryStub:
    def __setitem__(self, k, v): pass
    def update(self, d=None, **kw): pass


class _WandbStub:
    class config:
        @staticmethod
        def update(*a, **kw): pass

    summary: _SummaryStub

    def __init__(self):
        self.summary = _SummaryStub()

    def log(self, *a, **kw): pass
    def finish(self, **kw): pass
    def init(self, **kw): pass


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class I3DTrainer:

    @property
    def raw_model(self):
        return self.model.module if isinstance(self.model, DDP) else self.model

    # ------------------------------------------------------------------
    # DDP / device / logging setup
    # ------------------------------------------------------------------

    def _setup_ddp(self):
        local_rank = int(os.environ.get("LOCAL_RANK", -1))
        self.ddp = local_rank >= 0
        self.local_rank = max(local_rank, 0)
        if self.ddp:
            dist.init_process_group(backend="nccl")
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            atexit.register(dist.destroy_process_group)
        else:
            self.rank = 0
            self.world_size = 1
        self.is_main = (self.rank == 0)

    def _setup_device(self, device_pref):
        if self.ddp:
            self.device = torch.device(f"cuda:{self.local_rank}")
            torch.cuda.set_device(self.local_rank)
        elif device_pref == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device_pref)
        if self.is_main:
            print(f"==> Device: {self.device}  World size: {self.world_size}")

    def _setup_logging(self, args):
        timestamp = datetime.now().strftime("%m%d-%H%M")
        self.out_dir = os.path.join("runs", f"{args.run_name}_{timestamp}")

        if self.is_main and not args.no_wandb:
            os.makedirs(os.path.join(self.out_dir, "checkpoints"), exist_ok=True)
            import wandb
            self.wandb = wandb
            self.wandb.init(name=args.run_name, group=args.wandb_group,
                            dir=self.out_dir, config=vars(args))
            atexit.register(lambda: self.wandb.finish() if self.wandb else None)
        else:
            if self.is_main:
                os.makedirs(os.path.join(self.out_dir, "checkpoints"), exist_ok=True)
            self.wandb = _WandbStub()

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def _make_loaders(self, args):
        use_workers = args.num_workers > 0
        worker_kwargs = (dict(persistent_workers=True, prefetch_factor=2)
                         if use_workers else {})

        loaders = {}
        self.train_sampler = None

        for split, is_train in [("train", True), ("val", False), ("test", False)]:
            ds = FlowDatasetWrapper(
                VideoDataset(dataset=args.dataset, split=split,
                             clip_len=args.clip_len, preprocess=False,
                             augment=is_train, ucf_split=args.split)
            )
            if self.ddp:
                sampler = DistributedSampler(
                    ds,
                    num_replicas=self.world_size,
                    rank=self.rank,
                    shuffle=is_train,
                    drop_last=is_train,
                )
                if is_train:
                    self.train_sampler = sampler
                loaders[split] = DataLoader(
                    ds, batch_size=args.batch_size, sampler=sampler,
                    num_workers=args.num_workers, pin_memory=True,
                    **worker_kwargs,
                )
            else:
                loaders[split] = DataLoader(
                    ds, batch_size=args.batch_size, shuffle=is_train,
                    num_workers=args.num_workers,
                    pin_memory=torch.cuda.is_available(),
                    **worker_kwargs,
                )

        return loaders

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------

    def _build_optimizer(self, args):
        def split_wd(params):
            decay, no_decay = [], []
            for p in params:
                (no_decay if p.ndim <= 1 else decay).append(p)
            return decay, no_decay

        decay_1x, no_decay_1x = split_wd(self.raw_model.get_1x_lr_params())
        decay_fc, no_decay_fc = split_wd(self.raw_model.get_10x_lr_params())
        fc_lr = args.lr * args.fc_lr_mult

        param_groups = [
            {"params": decay_1x,    "lr": args.lr,  "weight_decay": args.weight_decay},
            {"params": no_decay_1x, "lr": args.lr,  "weight_decay": 0.0},
            {"params": decay_fc,    "lr": fc_lr,    "weight_decay": args.weight_decay},
            {"params": no_decay_fc, "lr": fc_lr,    "weight_decay": 0.0},
        ]
        self.optimizer = optim.AdamW(param_groups, lr=args.lr,
                                     weight_decay=args.weight_decay)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=args.epochs, eta_min=1e-6)

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------

    def __init__(self, args):
        self.args = args
        self.start_epoch = 0
        self.best_acc = 0.0

        self._setup_ddp()
        self._setup_device(args.device)
        self._setup_logging(args)

        if self.is_main:
            print("==> Loading data (flow will be cached on first run)...")
        self.loaders = self._make_loaders(args)

        if self.is_main:
            print(f"==> Building I3D two-stream (clip_len={args.clip_len})")
        self.model = I3DTwoStream(
            num_classes=args.num_classes,
            dropout_prob=args.dropout,
            clip_len=args.clip_len,
        ).to(self.device)

        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=args.label_smoothing).to(self.device)

        # Build optimizer before DDP wrapping so param groups reference raw tensors
        self._build_optimizer(args)

        if self.ddp:
            self.model = DDP(self.model, device_ids=[self.local_rank])

        self.amp_enabled = (self.device.type == "cuda") and not args.no_amp
        self.scaler = GradScaler("cuda") if self.amp_enabled else None

        if self.is_main:
            total = sum(p.numel() for p in self.raw_model.parameters())
            print(f"==> Total parameters: {total:,}")
            self.wandb.config.update({"total_params": total, "world_size": self.world_size})

        if args.resume and os.path.isfile(args.resume):
            if self.is_main:
                print(f"==> Resuming from {args.resume}")
            ckpt = torch.load(args.resume, map_location=self.device)
            self.start_epoch = ckpt["epoch"]
            self.best_acc = ckpt.get("best_acc", 0.0)
            self.raw_model.load_state_dict(ckpt["state_dict"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            if "scheduler" in ckpt:
                self.scheduler.load_state_dict(ckpt["scheduler"])
            if self.ddp:
                dist.barrier()

    # ------------------------------------------------------------------
    # Epoch loop
    # ------------------------------------------------------------------

    def _run_epoch(self, epoch: int, split: str):
        is_train = split == "train"
        self.model.train() if is_train else self.model.eval()
        loader = self.loaders[split]

        if is_train and self.train_sampler is not None:
            self.train_sampler.set_epoch(epoch)

        stats = {"loss": 0.0, "correct": 0, "total": 0, "batches": 0,
                 "grad_norm": 0.0, "grad_batches": 0, "nan_grad_batches": 0,
                 "skipped": 0}

        pbar = tqdm(loader, desc=f"Ep {epoch+1} [{split.upper()}]",
                    disable=not self.is_main)

        for inputs, targets in pbar:
            rgb  = inputs[0].to(self.device)
            flow = inputs[1].to(self.device)
            targets = targets.to(self.device, dtype=torch.long).view(-1)

            if not (torch.isfinite(rgb).all() and torch.isfinite(flow).all()):
                stats["skipped"] += 1
                continue

            if is_train:
                self.optimizer.zero_grad()

            amp_ctx  = (autocast(device_type=self.device.type)
                        if self.amp_enabled else contextlib.nullcontext())
            grad_ctx = (contextlib.nullcontext() if is_train else torch.no_grad())

            with amp_ctx, grad_ctx:
                out = self.model([rgb, flow])
                if is_train:
                    rgb_logits, flow_logits, aux_logits = out
                else:
                    rgb_logits, flow_logits = out
                    aux_logits = []
                loss = (self.criterion(rgb_logits, targets) +
                        self.criterion(flow_logits, targets)) * 0.5
                for aux in aux_logits:
                    loss = loss + 0.3 * self.criterion(aux, targets)

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
                    for p in self.raw_model.parameters()
                )
                if bad_grads:
                    stats["nan_grad_batches"] += 1
                    self.optimizer.zero_grad(set_to_none=True)
                    if self.scaler:
                        self.scaler.update()
                else:
                    gn = nn.utils.clip_grad_norm_(
                        self.raw_model.parameters(), 1.0).item()
                    stats["grad_norm"]    += gn
                    stats["grad_batches"] += 1
                    if self.scaler:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()

            fused_logits = (rgb_logits + flow_logits) * 0.5
            stats["loss"]    += loss.item()
            stats["batches"] += 1
            stats["total"]   += targets.size(0)
            stats["correct"] += fused_logits.argmax(dim=1).eq(targets).sum().item()

            if self.is_main:
                postfix = {
                    "L": f"{stats['loss']/stats['batches']:.3f}",
                    "A": f"{100.*stats['correct']/stats['total']:.1f}%",
                }
                if is_train and stats["grad_batches"]:
                    postfix["gn"] = f"{stats['grad_norm']/stats['grad_batches']:.2f}"
                pbar.set_postfix(postfix)

        # Aggregate across all ranks
        if self.ddp:
            agg = torch.tensor(
                [stats["loss"], float(stats["correct"]),
                 float(stats["total"]), float(stats["batches"])],
                device=self.device, dtype=torch.float64,
            )
            dist.all_reduce(agg, op=dist.ReduceOp.SUM)
            stats["loss"]    = agg[0].item()
            stats["correct"] = int(agg[1].item())
            stats["total"]   = int(agg[2].item())
            stats["batches"] = int(agg[3].item())

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

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def _save(self, epoch: int, tag: str):
        if not self.is_main:
            return
        torch.save(
            {"epoch": epoch, "state_dict": self.raw_model.state_dict(),
             "optimizer": self.optimizer.state_dict(),
             "scheduler": self.scheduler.state_dict(),
             "best_acc": self.best_acc},
            os.path.join(self.out_dir, "checkpoints", f"{tag}.pth"),
        )

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self):
        if self.args.test_only:
            test = self._run_epoch(0, "test")
            if self.is_main:
                print(f"Test Result | Loss: {test['loss']:.3f} | Top-1: {test['acc']:.2f}%")
            self.wandb.finish()
            return

        start = time.time()
        for epoch in range(self.start_epoch, self.args.epochs):
            t = self._run_epoch(epoch, "train")
            v = self._run_epoch(epoch, "val")
            self.scheduler.step()

            if self.is_main:
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

        if self.is_main:
            print("==> Running final test evaluation...")
        test = self._run_epoch(self.args.epochs - 1, "test")
        if self.is_main:
            print(f"Test Result | Loss: {test['loss']:.3f} | Top-1: {test['acc']:.2f}%")
            self.wandb.summary.update({f"test/{k}": v for k, v in test.items()})
            self.wandb.summary["total_runtime_sec"] = time.time() - start
        self.wandb.finish()


if __name__ == "__main__":
    I3DTrainer(parse_args()).run()
