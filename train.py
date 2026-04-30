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
from tqdm import tqdm

from utils.data_factory import get_dataloaders
from utils.model_factory import get_model

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    
    # Core Training Args
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name: cifar10 | ucf101 | hmdb51 | ucf10 | ucf11 | <UCR/UEA dataset name>")
    parser.add_argument("--task", type=str, default=None, choices=["cifar", "video", "timeseries", "mnist"],
                        help="Task type. Auto-detected from --dataset if not set.")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--fc_lr_mult", type=float, default=10.0, help="LR multiplier for the final FC classifier layer")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--Q", type=int, default=2)
    parser.add_argument("--clip_len", type=int, default=16,
                        help="Number of frames per clip. Must be a multiple of 16.")
    parser.add_argument("--n_lag", type=int, default=None,
                        help="Laguerre basis size for LaguerreConv3d. None = full (= kernel T dim).")
    parser.add_argument("--disable_cubic", action="store_true")
    parser.add_argument("--cubic_mode", type=str, default="symmetric",
                        choices=["symmetric", "general"])
    parser.add_argument("--in_ch", type=int, default=None,
                        help="Input channels for timeseries models (auto-detected from dataset if not set).")
    parser.add_argument("--base_ch", type=int, default=8,
                        help="Base channel width for vnn_1d. Channels per block = 3/4/8/12 × base_ch.")
    parser.add_argument("--Qc", type=int, default=1,
                        help="Cubic rank for vnn_1d (default 1).")
    parser.add_argument("--poly_degrees", type=int, nargs="+", default=None,
                        help="Laguerre polynomial degrees for laguerre_vnn_1d, e.g. --poly_degrees 2 3. "
                             "Defaults to [2, 3] (quad+cubic equivalent).")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Softplus scale for laguerre_vnn_1d. Lower values (e.g. 0.5) help "
                             "stability when using degrees >= 4.")
    parser.add_argument("--jitter_sigma", type=float, default=0.03,
                        help="Gaussian noise std for timeseries jitter augmentation.")
    parser.add_argument("--wandb_group", type=str, default=None,
                        help="W&B group name to cluster related runs (e.g. a benchmark sweep).")
    parser.add_argument("--split", type=int, default=1, choices=[1, 2, 3],
                        help="UCF101/HMDB51 official dataset split number (1–3). Default: 1.")
    parser.add_argument("--avg_splits", action="store_true",
                        help="Train on all 3 UCF101/HMDB51 splits sequentially and report the mean test accuracy.")

    # Logging & System
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--no_amp", action="store_true", help="Disable AMP (float16). Use for models without output clamping.")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--test_only", action="store_true", help="Only run evaluation on the test set")
    
    args = parser.parse_args()
    
    # Auto-determine Task and Classes
    _video_ds   = {"ucf10", "ucf11", "ucf101", "hmdb51"}
    _ds_classes = {"cifar10": 10, "ucf11": 11, "ucf101": 101, "hmdb51": 51, "ucf10": 10, "mnist": 10}
    if args.task is None:
        if args.dataset == "cifar10":
            args.task = "cifar"
        elif args.dataset == "mnist":
            args.task = "mnist"
        elif args.dataset in _video_ds:
            args.task = "video"
        else:
            args.task = "timeseries"
    args.num_classes = _ds_classes.get(args.dataset, None)  # None for UCR (set later from data)
    args.ucf_split = args.split  # propagated to VideoDataset via data_factory

    return args

class Trainer:
    def __init__(self, args):
        self.args = args
        self.start_epoch = 0
        self.best_acc = 0.0
        
        self._setup_device(args.device)
        self._setup_logging(args)
        
        # Load data first — for timeseries, this sets args.num_classes and args.in_ch.
        self.loaders = get_dataloaders(args)
        self.model = get_model(args, self.device)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing).to(self.device)
        self._setup_optimizer(args)
        
        # Mixed Precision Setup
        self.amp_enabled = (self.device.type == "cuda") and not args.no_amp
        self.scaler = GradScaler(device="cuda") if self.amp_enabled else None
        
        # Numerical Stability & Parameter Tracking
        self.skipped_stats = {"total": 0, "input": 0, "output": 0, "loss": 0}
        self.finite_debug_prints = 0
        self.total_params = sum(p.numel() for p in self.model.parameters())
        self.wandb.config.update({"total_params": self.total_params})
        print(f"==> Total Parameters: {self.total_params:,}")

        # Optional Resume
        if args.resume and os.path.isfile(args.resume):
            print(f"==> Resuming from {args.resume}")
            ckpt = torch.load(args.resume, map_location=self.device)
            self.start_epoch = ckpt["epoch"]
            self.best_acc = ckpt.get("best_acc", 0.0)
            self.model.load_state_dict(ckpt["state_dict"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            if "scheduler" in ckpt:
                self.scheduler.load_state_dict(ckpt["scheduler"])

    def _setup_device(self, device_pref):
        if device_pref == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device_pref)
        print(f"==> Device: {self.device}")

    def _setup_logging(self, args):
        import wandb
        self.wandb = wandb
        
        timestamp = datetime.now().strftime('%m%d-%H%M')
        self.run_name = args.run_name
        self.out_dir = os.path.join("runs", f"{self.run_name}_{timestamp}")
        os.makedirs(os.path.join(self.out_dir, "checkpoints"), exist_ok=True)

        init_kwargs = {
            "name": self.run_name,
            "dir": self.out_dir,
            "config": vars(args),
            "group": getattr(args, "wandb_group", None),
        }

        self.wandb.init(**init_kwargs)
        atexit.register(lambda: self.wandb.finish() if self.wandb else None)

    def _setup_optimizer(self, args):
        if args.task in ("video", "timeseries", "mnist"):
            get_1x = getattr(self.model, "get_1x_lr_params", None)
            get_10x = getattr(self.model, "get_10x_lr_params", None)

            # Separate BN/bias params from weight params — BN γ/β should not be
            # decayed, as penalising γ toward zero fights against feature scaling.
            def split_wd(param_iter):
                decay, no_decay = [], []
                for p in param_iter:
                    if p.ndim <= 1:  # BN γ/β and biases are 1-D
                        no_decay.append(p)
                    else:
                        decay.append(p)
                return decay, no_decay

            if callable(get_1x):
                decay_1x, no_decay_1x = split_wd(get_1x())
                params = [
                    {"params": decay_1x,    "lr": args.lr, "weight_decay": args.weight_decay},
                    {"params": no_decay_1x, "lr": args.lr, "weight_decay": 0.0},
                ]
            else:
                decay_all, no_decay_all = split_wd(self.model.parameters())
                params = [
                    {"params": decay_all,    "lr": args.lr, "weight_decay": args.weight_decay},
                    {"params": no_decay_all, "lr": args.lr, "weight_decay": 0.0},
                ]

            if callable(get_10x):
                decay_fc, no_decay_fc = split_wd(get_10x())
                fc_lr = args.lr * args.fc_lr_mult
                params.append({"params": decay_fc,    "lr": fc_lr, "weight_decay": args.weight_decay})
                params.append({"params": no_decay_fc, "lr": fc_lr, "weight_decay": 0.0})

            self.optimizer = optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.epochs, eta_min=1e-6)
        else:
            self.optimizer = optim.SGD(self.model.parameters(), lr=args.lr, momentum=args.momentum, 
                                     weight_decay=args.weight_decay, nesterov=True)
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.epochs)

    def _get_gate_stats(self):
        stats = {}
        idx = 0
        for module in self.model.modules():
            has_quad  = hasattr(module, 'quad_gate')
            has_cubic = hasattr(module, 'cubic_gate')
            has_poly  = hasattr(module, 'poly_gates') and hasattr(module, 'poly_degrees')
            if has_quad:
                stats[f"gates/b{idx}/quad"] = module.quad_gate.abs().mean().item()  # type: ignore[union-attr]
            if has_cubic:
                stats[f"gates/b{idx}/cubic"] = module.cubic_gate.abs().mean().item()  # type: ignore[union-attr]
            if has_poly:
                for gate, deg in zip(module.poly_gates, module.poly_degrees):  # type: ignore[union-attr]
                    stats[f"gates/b{idx}/lag{deg}"] = gate.abs().mean().item()
            if has_quad or has_cubic or has_poly:
                idx += 1
        return stats

    def _get_weight_stats(self):
        """Calculates global mean and max absolute weight across the model."""
        w_sum, w_count, w_max = 0.0, 0, 0.0
        with torch.no_grad():
            for p in self.model.parameters():
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
        
        if self.finite_debug_prints < 3:
            print(f"[{mode.upper()}][Ep {epoch+1}][Batch {batch_idx}] Skipping: non-finite {name}")
            self.finite_debug_prints += 1
        return False

    def _run_epoch(self, epoch, mode="train"):
        is_train = (mode == "train")
        self.model.train() if is_train else self.model.eval()
        loader = self.loaders[mode if is_train else "val" if mode == "val" else "test"]

        stats = {"loss": 0.0, "correct": 0, "total": 0, "batches": 0,
                 "grad_norm": 0.0, "grad_batches": 0, "nan_grad_batches": 0}
        self.finite_debug_prints = 0
        if is_train:
            self.skipped_stats = {"total": 0, "input": 0, "output": 0, "loss": 0}

        pbar = tqdm(enumerate(loader), total=len(loader), desc=f"Ep {epoch+1} [{mode.upper()}]")

        for batch_idx, (inputs, targets) in pbar:
            inputs = [x.to(self.device) for x in inputs] if isinstance(inputs, (list, tuple)) else inputs.to(self.device)
            targets = targets.to(self.device, dtype=torch.long).view(-1)

            if not self._check_finite(inputs, "input", epoch, batch_idx, mode): continue

            if is_train: self.optimizer.zero_grad()

            with (autocast(device_type=self.device.type) if self.amp_enabled else contextlib.nullcontext()):
                with contextlib.nullcontext() if is_train else torch.no_grad():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)

            if not self._check_finite(outputs, "output", epoch, batch_idx, mode) or \
               not self._check_finite(loss, "loss", epoch, batch_idx, mode):
                if is_train: self.optimizer.zero_grad(set_to_none=True)
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
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0).item()
                    stats["grad_norm"] += grad_norm
                    stats["grad_batches"] += 1
                    if self.scaler:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()

            stats["loss"] += loss.item()
            stats["batches"] += 1
            stats["total"] += targets.size(0)
            stats["correct"] += outputs.max(1)[1].eq(targets).sum().item()

            postfix = {
                "L": f"{stats['loss']/stats['batches']:.3f}",
                "A": f"{100.*stats['correct']/stats['total']:.1f}%",
            }
            if is_train:
                postfix["gn"] = f"{stats['grad_norm']/max(1, stats['grad_batches']):.2f}"
                if stats["nan_grad_batches"]:
                    postfix["nan_g"] = stats["nan_grad_batches"]
            pbar.set_postfix(postfix)

        n = stats["batches"]
        res = {
            "loss": stats["loss"] / n if n else 0,
            "acc":  100. * stats["correct"] / stats["total"] if stats["total"] else 0,
        }
        if is_train:
            res["grad_norm"] = stats["grad_norm"] / max(1, stats["grad_batches"])
            res["nan_grad_batches"] = stats["nan_grad_batches"]
        return res

    def _load_best_model(self):
        best_ckpt = os.path.join(self.out_dir, "checkpoints", "best_model.pth")
        if os.path.exists(best_ckpt):
            ckpt = torch.load(best_ckpt, map_location=self.device)
            self.model.load_state_dict(ckpt["state_dict"])
            print(f"==> Loaded best checkpoint (val acc {self.best_acc:.2f}%) for test evaluation.")

    def run(self):
        if self.args.test_only:
            print(f"==> Running Test Evaluation...")
            test_stats = self._run_epoch(0, "test")
            print(f"Test Result | Loss: {test_stats['loss']:.3f} | Acc: {test_stats['acc']:.2f}%")
            self.wandb.finish()
            return test_stats["acc"]

        start_time_total = time.time()
        for epoch in range(self.start_epoch, self.args.epochs):
            t_stats, v_stats = self._run_epoch(epoch, "train"), self._run_epoch(epoch, "val")
            self.scheduler.step()
            
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
                "epoch": epoch+1,
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
            if hasattr(self.model, "cross_abs_max"):
                log_data["train/cross_abs_max"] = self.model.cross_abs_max
            log_data.update(self._get_gate_stats())
            self.wandb.log(log_data)

            if v_stats["acc"] > self.best_acc:
                self.best_acc = v_stats["acc"]
                torch.save({"epoch": epoch+1, "state_dict": self.model.state_dict(), "optimizer": self.optimizer.state_dict(), "scheduler": self.scheduler.state_dict(), "best_acc": self.best_acc},
                           os.path.join(self.out_dir, "checkpoints", "best_model.pth"))

            if epoch + 1 == self.args.epochs:
                torch.save(
                    {"epoch": epoch+1, "state_dict": self.model.state_dict(),
                     "optimizer": self.optimizer.state_dict(), "scheduler": self.scheduler.state_dict(),
                     "best_acc": self.best_acc},
                    os.path.join(self.out_dir, "checkpoints", "last_model.pth"),
                )

        total_runtime = time.time() - start_time_total
        self.wandb.summary["total_runtime_sec"] = total_runtime
        print(f"==> Training Complete. Total Runtime: {total_runtime/60:.2f} mins")

        self._load_best_model()
        print(f"==> Running Final Test Evaluation...")
        test_stats = self._run_epoch(self.args.epochs - 1, "test")
        print(f"Test Result | Loss: {test_stats['loss']:.3f} | Acc: {test_stats['acc']:.2f}%")
        self.wandb.summary.update({f"test/{k}": v for k, v in test_stats.items()})
        self.wandb.finish()
        return test_stats["acc"]

if __name__ == "__main__":
    args = parse_args()

    if args.avg_splits and args.task == "video":
        if args.resume:
            raise ValueError("--resume cannot be used together with --avg_splits.")
        base_run_name = args.run_name
        # Group all 3 split runs together in W&B automatically.
        if args.wandb_group is None:
            args.wandb_group = base_run_name
        test_accs = []
        for s in [1, 2, 3]:
            args.ucf_split = s
            args.run_name = f"{base_run_name}_s{s}"
            acc = Trainer(args).run()
            test_accs.append(acc)
        mean_acc = sum(test_accs) / 3
        split_str = " | ".join(f"S{i+1}: {a:.2f}%" for i, a in enumerate(test_accs))
        print(f"\n{'='*60}")
        print(f"3-Split Results: {split_str}")
        print(f"3-Split Mean Accuracy: {mean_acc:.2f}%")
        print(f"{'='*60}")
    else:
        Trainer(args).run()
