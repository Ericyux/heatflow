"""Unified training driver.

Example:
    python -m src.train --data data/heat.npz --model fno --epochs 30
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import PDEData, PairDataset, add_coords
from .metrics import StepPredictor, one_step_errors, relative_l2
from .models import build_model, count_params
from .throttle import pace, set_duty

RESULTS_DIR = Path("results")


def seed_everything(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_split(model, frames, mean, std, periodic, device):
    """Physical-space one-step relative L2 over all pairs in `frames`."""
    predictor = StepPredictor(model, mean, std, periodic, device)
    errs = one_step_errors(predictor, frames)
    model.train()
    return errs.mean().item()


def train(args):
    seed_everything(args.seed)
    set_duty(args.gpu_duty)
    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = True

    data = PDEData(args.data)
    mean, std = data.norm_stats()
    train_frames = data.frames("train")
    val_frames = data.frames("val")
    pde = data.meta["pde"]

    loader = DataLoader(
        PairDataset(train_frames), batch_size=args.batch_size,
        shuffle=True, num_workers=0, pin_memory=(device.type == "cuda"),
    )

    model = build_model(args.model, periodic=data.periodic).to(device)
    n_params = count_params(model)
    print(f"{args.model} on {pde}: {n_params:,} params, "
          f"{len(loader.dataset)} training pairs, device {device}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(loader)
    )

    out_dir = RESULTS_DIR / f"{pde}_{args.model}"
    out_dir.mkdir(parents=True, exist_ok=True)

    history = {"train_loss": [], "val_rel_l2": [], "epoch_seconds": []}
    best_val = float("inf")
    for epoch in range(args.epochs):
        model.train()
        t0 = time.perf_counter()
        running, seen = 0.0, 0
        for inputs, targets in loader:
            t_batch = time.perf_counter()
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            x = add_coords((inputs - mean) / std, data.periodic)
            y = (targets - mean) / std
            pred = model(x)
            loss = relative_l2(pred, y).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            running += loss.item() * inputs.shape[0]
            seen += inputs.shape[0]
            pace(t_batch)

        val_rel = evaluate_split(model, val_frames, mean, std, data.periodic, device)
        secs = time.perf_counter() - t0
        history["train_loss"].append(running / seen)
        history["val_rel_l2"].append(val_rel)
        history["epoch_seconds"].append(secs)
        print(f"epoch {epoch + 1:3d}/{args.epochs}  "
              f"train {running / seen:.4f}  val rel-L2 {val_rel:.4f}  {secs:.1f}s")

        if val_rel < best_val:
            best_val = val_rel
            torch.save(model.state_dict(), out_dir / "best.pt")

    config = {
        "pde": pde,
        "model": args.model,
        "data": str(args.data),
        "n_params": n_params,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "norm_mean": mean,
        "norm_std": std,
        "best_val_rel_l2": best_val,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(f"best val rel-L2 {best_val:.4f} -> {out_dir / 'best.pt'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", required=True, choices=["fno", "unet", "cnn"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--gpu-duty", type=float, default=0.6,
                        help="fraction of wall time the GPU is kept busy "
                             "(thermal headroom for laptops; 1.0 = flat out)")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
