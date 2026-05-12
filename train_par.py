"""
train_par.py — Multi-GPU training via DistributedDataParallel.

Launch with torchrun:
    torchrun --nproc_per_node=4 train_par.py --dataset ssv2 --model vnn_fusion_ho --run_name my_run --batch_size 8

Effective batch size = batch_size × nproc_per_node.
Scale --lr linearly with world size (e.g. --lr 4e-4 for 4 GPUs with default 1e-4).
"""

import argparse
import atexit
import contextlib
import os
import time
from datetime import datetime

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler  # type: ignore[attr-defined]
from torch.amp import autocast  # type: ignore[attr-defined]
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from utils.data_factory import get_dataloaders
from utils.model_factory import get_model


# ---------------------------------------------------------------------------
# Args — identical to train.py
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--task", type=str, default=None, choices=["cifar", "video", "timeseries", "mnist"])
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--fc_lr_mult", type=float, default=10.0)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--Q", type=int, default=2)
    parser.add_argument("--clip_len", type=int, default=16)
    parser.add_argument("--n_lag", type=int, default=None)
    parser.add_argument("--n_lag_t", type=int, default=None)
    parser.add_argument("--n_lag_s", type=int, default=None)
    parser.add_argument("--disable_cubic", action="store_true")
    parser.add_argument("--cubic_mode", type=str, default="symmetric", choices=["symmetric", "general"])
    parser.add_argument("--in_ch", type=int, default=None)
    parser.add_argument("--base_ch", type=int, default=8)
    parser.add_argument("--Qc", type=int, default=1)
    parser.add_argument("--poly_degrees", type=int, nargs="+", default=None)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--jitter_sigma", type=float, default=0.03)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--split", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--test_only", action="store_true")

    args = parser.parse_args()

    _video_ds   = {"ucf10", "ucf11", "ucf101", "hmdb51", "ssv2"}
    _ds_classes = {"cifar10": 10, "ucf11": 11, "ucf101": 101, "hmdb51": 51, "ucf10": 10, "ssv2": 174, "mnist": 10}
    if args.task is None:
        if args.dataset == "cifar10":
            args.task = "cifar"
        elif args.dataset == "mnist":
            args.task = "mnist"
        elif args.dataset in _video_ds:
            args.task = "video"
        else:
            args.task = "timeseries"
    args.num_classes = _ds_classes.get(args.dataset, None)
    args.ucf_split = args.split
    return args


# ---------------------------------------------------------------------------
# W&B no-op stub for non-main ranks
# ---------------------------------------------------------------------------

class _SummaryStub:
    def __setitem__(self, k: object, v: object) -> None: pass
    def update(self, d: object = None, **kw: object) -> None: pass  # type: ignore[override]


class _WandbStub:
    class config:
        @staticmethod
        def update(*a: object, **kw: object) -> None: pass

    summary: _SummaryStub

    def __init__(self) -> None:
        self.summary = _SummaryStub()

    def __setitem__(self, k: object, v: object) -> None: pass
    def log(self, *a: object, **kw: object) -> None: pass
    def finish(self, **kw: object) -> None: pass
    def init(self, **kw: object) -> None: pass


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:

    @property
    def raw_model(self):
        return self.model.module if isinstance(self.model, DDP) else self.model

    # ------------------------------------------------------------------
    # Setup
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
        timestamp = datetime.now().strftime('%m%d-%H%M')
        self.run_name = args.run_name
        self.out_dir = os.path.join("runs", f"{self.run_name}_{timestamp}")

        if self.is_main:
            os.makedirs(os.path.join(self.out_dir, "checkpoints"), exist_ok=True)
            import wandb
            self.wandb = wandb
            self.wandb.init(
                name=self.run_name,
                dir=self.out_dir,
                config=vars(args),
                group=getattr(args, "wandb_group", None),
            )
            atexit.register(lambda: self.wandb.finish() if self.wandb else None)
        else:
            self.wandb = _WandbStub()

    def _setup_data(self, args):
        loaders = get_dataloaders(args)
        if not self.ddp:
            return loaders

        use_workers = args.num_workers > 0

        new_loaders = {}
        for split, loader in loaders.items():
            sampler = DistributedSampler(
                loader.dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=(split == "train"),
                drop_last=(split == "train"),
            )
            new_loaders[split] = DataLoader(
                loader.dataset,
                batch_size=args.batch_size,
                sampler=sampler,
                num_workers=args.num_workers,
                pin_memory=True,
                persistent_workers=use_workers,
                prefetch_factor=2 if use_workers else None,
            )
        return new_loaders

    def _setup_optimizer(self, args):
        if args.task in ("video", "timeseries", "mnist"):
            get_1x  = getattr(self.raw_model, "get_1x_lr_params", None)
            get_10x = getattr(self.raw_model, "get_10x_lr_params", None)

            def split_wd(param_iter):
                decay, no_decay = [], []
                for p in param_iter:
                    (no_decay if p.ndim <= 1 else decay).append(p)
                return decay, no_decay

            if callable(get_1x):
                d1, nd1 = split_wd(get_1x())
                params = [
                    {"params": d1,  "lr": args.lr, "weight_decay": args.weight_decay},
                    {"params": nd1, "lr": args.lr, "weight_decay": 0.0},
                ]
            else:
                d, nd = split_wd(self.raw_model.parameters())
                params = [
                    {"params": d,  "lr": args.lr, "weight_decay": args.weight_decay},
                    {"params": nd, "lr": args.lr, "weight_decay": 0.0},
                ]

            if callable(get_10x):
                dfc, ndfc = split_wd(get_10x())
                fc_lr = args.lr * args.fc_lr_mult
                params += [
                    {"params": dfc,  "lr": fc_lr, "weight_decay": args.weight_decay},
                    {"params": ndfc, "lr": fc_lr, "weight_decay": 0.0},
                ]

            self.optimizer = optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.epochs, eta_min=1e-6)
        else:
            self.optimizer = optim.SGD(self.raw_model.parameters(), lr=args.lr,
                                       momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.epochs)

    def __init__(self, args):
        self.args = args
        self.start_epoch = 0
        self.best_acc = 0.0

        self._setup_ddp()
        self._setup_device(args.device)
        self._setup_logging(args)

        self.loaders = self._setup_data(args)
        self.model = get_model(args, self.device)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing).to(self.device)
        self._setup_optimizer(args)  # before DDP wrap — operates on raw model

        if self.ddp:
            self.model = DDP(self.model, device_ids=[self.local_rank])

        self.amp_enabled = (self.device.type == "cuda") and not args.no_amp
        self.scaler = GradScaler("cuda") if self.amp_enabled else None

        self.skipped_stats = {"total": 0, "input": 0, "output": 0, "loss": 0}
        self.finite_debug_prints = 0
        self.total_params = sum(p.numel() for p in self.raw_model.parameters())
        self.wandb.config.update({"total_params": self.total_params, "world_size": self.world_size})
        if self.is_main:
            print(f"==> Total Parameters: {self.total_params:,}")

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

    # ------------------------------------------------------------------
    # Stats helpers
    # ------------------------------------------------------------------

    def _get_gate_stats(self):
        stats = {}
        idx = 0
        for module in self.raw_model.modules():
            has_quad = hasattr(module, 'quad_gate')
            has_cubic = hasattr(module, 'cubic_gate')
            has_poly = hasattr(module, 'poly_gates') and hasattr(module, 'poly_degrees')
            if has_quad:
                stats[f"gates/b{idx}/quad"] = getattr(module, 'quad_gate').abs().mean().item()
            if has_cubic:
                stats[f"gates/b{idx}/cubic"] = getattr(module, 'cubic_gate').abs().mean().item()
            if has_poly:
                for gate, deg in zip(  # type: ignore[arg-type]
                    getattr(module, 'poly_gates'), getattr(module, 'poly_degrees')
                ):
                    stats[f"gates/b{idx}/lag{deg}"] = gate.abs().mean().item()
            if has_quad or has_cubic or has_poly:
                idx += 1
        return stats

    def _get_weight_stats(self):
        w_sum, w_count, w_max = 0.0, 0, 0.0
        with torch.no_grad():
            for p in self.raw_model.parameters():
                w_sum += p.data.sum().item()
                w_count += p.data.numel()
                w_max = max(w_max, p.data.abs().max().item())
        return w_sum / w_count if w_count > 0 else 0.0, w_max

    def _check_finite(self, tensor, name, epoch, batch_idx, mode):
        tensors = tensor if isinstance(tensor, (list, tuple)) else [tensor]
        if all(torch.isfinite(t).all() for t in tensors):
            return True
        self.skipped_stats["total"] += 1
        self.skipped_stats[name] += 1
        if self.finite_debug_prints < 3 and self.is_main:
            print(f"[{mode.upper()}][Ep {epoch+1}][Batch {batch_idx}] Skipping: non-finite {name}")
            self.finite_debug_prints += 1
        return False

    # ------------------------------------------------------------------
    # Epoch loop
    # ------------------------------------------------------------------

    def _run_epoch(self, epoch, mode="train"):
        is_train = (mode == "train")
        self.model.train() if is_train else self.model.eval()
        loader = self.loaders[mode]

        if is_train and hasattr(loader.sampler, "set_epoch"):
            loader.sampler.set_epoch(epoch)

        stats = {"loss": 0.0, "correct": 0, "total": 0, "batches": 0,
                 "grad_norm": 0.0, "grad_batches": 0, "nan_grad_batches": 0}
        self.finite_debug_prints = 0
        if is_train:
            self.skipped_stats = {"total": 0, "input": 0, "output": 0, "loss": 0}

        pbar = tqdm(enumerate(loader), total=len(loader),
                    desc=f"Ep {epoch+1} [{mode.upper()}]", disable=not self.is_main)

        for batch_idx, (inputs, targets) in pbar:
            inputs = [x.to(self.device) for x in inputs] if isinstance(inputs, (list, tuple)) else inputs.to(self.device)
            targets = targets.to(self.device, dtype=torch.long).view(-1)

            if not self._check_finite(inputs, "input", epoch, batch_idx, mode):
                continue

            if is_train:
                self.optimizer.zero_grad()

            with (autocast(device_type=self.device.type) if self.amp_enabled else contextlib.nullcontext()):
                with contextlib.nullcontext() if is_train else torch.no_grad():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)

            if not self._check_finite(outputs, "output", epoch, batch_idx, mode) or \
               not self._check_finite(loss,    "loss",   epoch, batch_idx, mode):
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
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.raw_model.parameters(), 1.0).item()
                    stats["grad_norm"]   += grad_norm
                    stats["grad_batches"] += 1
                    if self.scaler:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()

            stats["loss"]    += loss.item()
            stats["batches"] += 1
            stats["total"]   += targets.size(0)
            stats["correct"] += outputs.max(1)[1].eq(targets).sum().item()

            if self.is_main:
                postfix = {
                    "L": f"{stats['loss']/stats['batches']:.3f}",
                    "A": f"{100.*stats['correct']/stats['total']:.1f}%",
                }
                if is_train:
                    postfix["gn"] = f"{stats['grad_norm']/max(1, stats['grad_batches']):.2f}"
                    if stats["nan_grad_batches"]:
                        postfix["nan_g"] = stats["nan_grad_batches"]
                pbar.set_postfix(postfix)

        # Aggregate across all ranks
        if self.ddp:
            agg = torch.tensor(
                [stats["loss"], float(stats["correct"]), float(stats["total"]), float(stats["batches"])],
                device=self.device, dtype=torch.float64,
            )
            dist.all_reduce(agg, op=dist.ReduceOp.SUM)
            stats["loss"]    = agg[0].item()
            stats["correct"] = int(agg[1].item())
            stats["total"]   = int(agg[2].item())
            stats["batches"] = int(agg[3].item())

        n = stats["batches"]
        res = {
            "loss": stats["loss"] / n if n else 0.0,
            "acc":  100. * stats["correct"] / stats["total"] if stats["total"] else 0.0,
        }
        if is_train:
            res["grad_norm"]        = stats["grad_norm"] / max(1, stats["grad_batches"])
            res["nan_grad_batches"] = stats["nan_grad_batches"]
        return res

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def _load_best_model(self):
        best_ckpt = os.path.join(self.out_dir, "checkpoints", "best_model.pth")
        if os.path.exists(best_ckpt):
            ckpt = torch.load(best_ckpt, map_location=self.device)
            self.raw_model.load_state_dict(ckpt["state_dict"])
            if self.is_main:
                print(f"==> Loaded best checkpoint (val acc {self.best_acc:.2f}%) for test evaluation.")

    def run(self):
        if self.args.test_only:
            if self.is_main:
                print("==> Running Test Evaluation...")
            test_stats = self._run_epoch(0, "test")
            if self.is_main:
                print(f"Test Result | Loss: {test_stats['loss']:.3f} | Acc: {test_stats['acc']:.2f}%")
                self.wandb.finish()
            return test_stats["acc"]

        start_time_total = time.time()
        for epoch in range(self.start_epoch, self.args.epochs):
            t_stats = self._run_epoch(epoch, "train")
            v_stats = self._run_epoch(epoch, "val")
            self.scheduler.step()

            if self.is_main:
                w_mean, w_max = self._get_weight_stats()
                print(
                    f"Ep {epoch+1} | "
                    f"T_Loss: {t_stats['loss']:.3f} T_Acc: {t_stats['acc']:.1f}% | "
                    f"V_Loss: {v_stats['loss']:.3f} V_Acc: {v_stats['acc']:.1f}% | "
                    f"GN: {t_stats['grad_norm']:.3f} | "
                    f"NaN_Grad: {t_stats['nan_grad_batches']} Skipped: {self.skipped_stats['total']} | "
                    f"W_Max: {w_max:.2e}"
                )

                log_data = {f"train/{k}": v for k, v in t_stats.items()}
                log_data.update({f"val/{k}": v for k, v in v_stats.items()})
                log_data.update({
                    "epoch": epoch + 1,
                    "lr": self.optimizer.param_groups[0]["lr"],
                    "weights/mean": w_mean,
                    "weights/max": w_max,
                    "skip/total":  self.skipped_stats["total"],
                    "skip/input":  self.skipped_stats["input"],
                    "skip/output": self.skipped_stats["output"],
                    "skip/loss":   self.skipped_stats["loss"],
                })
                if self.scaler:
                    log_data["amp/scale"] = self.scaler.get_scale()
                if hasattr(self.raw_model, "cross_abs_max"):
                    log_data["train/cross_abs_max"] = self.raw_model.cross_abs_max
                log_data.update(self._get_gate_stats())
                self.wandb.log(log_data)

                if v_stats["acc"] > self.best_acc:
                    self.best_acc = v_stats["acc"]
                    torch.save(
                        {"epoch": epoch + 1, "state_dict": self.raw_model.state_dict(),
                         "optimizer": self.optimizer.state_dict(),
                         "scheduler": self.scheduler.state_dict(),
                         "best_acc": self.best_acc},
                        os.path.join(self.out_dir, "checkpoints", "best_model.pth"),
                    )

                if epoch + 1 == self.args.epochs:
                    torch.save(
                        {"epoch": epoch + 1, "state_dict": self.raw_model.state_dict(),
                         "optimizer": self.optimizer.state_dict(),
                         "scheduler": self.scheduler.state_dict(),
                         "best_acc": self.best_acc},
                        os.path.join(self.out_dir, "checkpoints", "last_model.pth"),
                    )

        if self.ddp:
            dist.barrier()

        if self.is_main:
            total_runtime = time.time() - start_time_total
            self.wandb.summary["total_runtime_sec"] = total_runtime
            print(f"==> Training Complete. Total Runtime: {total_runtime/60:.2f} mins")

        self._load_best_model()
        if self.is_main:
            print("==> Running Final Test Evaluation...")
        test_stats = self._run_epoch(self.args.epochs - 1, "test")
        if self.is_main:
            print(f"Test Result | Loss: {test_stats['loss']:.3f} | Acc: {test_stats['acc']:.2f}%")
            self.wandb.summary.update({f"test/{k}": v for k, v in test_stats.items()})
            self.wandb.finish()
        return test_stats["acc"]


if __name__ == "__main__":
    args = parse_args()
    Trainer(args).run()
