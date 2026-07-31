"""Dataset generation and loading.

Datasets are stored as .npz files holding trajectory tensors of shape
(n_traj, n_frames + 1, N, N) plus metadata. Models are trained on one-step
pairs u(t) -> u(t + dt) built from the training trajectories; splits are by
trajectory, never by pair, to avoid leakage.
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from . import solvers

DATA_DIR = Path("data")


def add_coords(u, periodic):
    """Append normalised coordinate channels to a (B, 1, N, N) batch.

    Coordinates are generated from the resolution at call time, so the same
    model input convention works at any grid size (required for zero-shot
    super-resolution).
    """
    b, _, n, _ = u.shape
    if periodic:
        xs = torch.arange(n, device=u.device, dtype=u.dtype) / n
    else:
        xs = torch.linspace(0.0, 1.0, n, device=u.device, dtype=u.dtype)
    x_grid, y_grid = torch.meshgrid(xs, xs, indexing="ij")
    coords = torch.stack([x_grid, y_grid]).unsqueeze(0).expand(b, -1, -1, -1)
    return torch.cat([u, coords], dim=1)


class PairDataset(Dataset):
    """All consecutive one-step pairs (u_t, u_{t+1}) from a trajectory array."""

    def __init__(self, frames):
        # frames: (n_traj, T + 1, N, N) float32 tensor (kept on CPU)
        self.frames = frames
        self.horizon = frames.shape[1] - 1

    def __len__(self):
        return self.frames.shape[0] * self.horizon

    def __getitem__(self, idx):
        traj, t = divmod(idx, self.horizon)
        return self.frames[traj, t].unsqueeze(0), self.frames[traj, t + 1].unsqueeze(0)


class TripletDataset(Dataset):
    """Consecutive (u_t, u_{t+1}, u_{t+2}) triplets, for pushforward training."""

    def __init__(self, frames):
        self.frames = frames
        self.horizon = frames.shape[1] - 2

    def __len__(self):
        return self.frames.shape[0] * self.horizon

    def __getitem__(self, idx):
        traj, t = divmod(idx, self.horizon)
        return (self.frames[traj, t].unsqueeze(0),
                self.frames[traj, t + 1].unsqueeze(0),
                self.frames[traj, t + 2].unsqueeze(0))


class PDEData:
    """Loads a generated .npz dataset and exposes splits and metadata."""

    def __init__(self, path):
        self.path = Path(path)
        raw = np.load(self.path, allow_pickle=False)
        self.meta = json.loads(str(raw["meta"]))
        self.arrays = {k: raw[k] for k in raw.files if k != "meta"}

    @property
    def periodic(self):
        return bool(self.meta["periodic"])

    def frames(self, split, res=None):
        """(n_traj, T + 1, N, N) float32 tensor for a split at a resolution."""
        res = res if res is not None else self.meta["res_train"]
        key = f"{split}_{res}"
        if key not in self.arrays:
            raise KeyError(f"no array '{key}' in {self.path.name}; have {sorted(self.arrays)}")
        return torch.from_numpy(self.arrays[key])

    def norm_stats(self):
        """Scalar mean/std over the training frames (resolution-independent)."""
        train = self.arrays[f"train_{self.meta['res_train']}"]
        return float(train.mean()), float(train.std())

    def super_resolutions(self):
        return [int(r) for r in self.meta["res_super"]]


def _save(path, meta, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, meta=json.dumps(meta), **{
        k: v.astype(np.float32) for k, v in arrays.items()
    })
    sizes = {k: tuple(v.shape) for k, v in arrays.items()}
    print(f"saved {path} {sizes}")


def generate_heat(
    out,
    n_train=800,
    n_val=100,
    n_test=100,
    res_train=64,
    res_super=(128, 256),
    alpha=0.1,
    t_final=1.0,
    n_frames=20,
    seed=0,
    device="cpu",
    chunk=64,
):
    """Heat-equation dataset. Test trajectories are additionally solved at the
    super-resolution grids from the *same* analytic initial conditions, so the
    higher-resolution test sets discretise identical continuum trajectories.
    """
    gen = torch.Generator().manual_seed(seed)
    n_total = n_train + n_val + n_test
    params = solvers.sample_peak_params(n_total, generator=gen)

    def solve_at(par, res):
        outs = []
        for i in range(0, par.shape[0], chunk):
            ic = solvers.gaussian_peaks_ic(par[i : i + chunk], res, device=device)
            frames = solvers.solve_heat_2d(ic, alpha=alpha, t_final=t_final, n_frames=n_frames)
            outs.append(frames.cpu().numpy())
        return np.concatenate(outs)

    splits = {
        "train": params[:n_train],
        "val": params[n_train : n_train + n_val],
        "test": params[n_train + n_val :],
    }
    arrays = {}
    for name, par in splits.items():
        arrays[f"{name}_{res_train}"] = solve_at(par, res_train)
        print(f"heat: solved {name} at {res_train}^2 ({par.shape[0]} traj)")
    for res in res_super:
        arrays[f"test_{res}"] = solve_at(splits["test"], res)
        print(f"heat: solved test at {res}^2")

    meta = {
        "pde": "heat",
        "periodic": False,
        "res_train": res_train,
        "res_super": list(res_super),
        "alpha": alpha,
        "t_final": t_final,
        "n_frames": n_frames,
        "dt_record": t_final / n_frames,
        "seed": seed,
    }
    _save(out, meta, **arrays)


def generate_ns(
    out,
    n_train=200,
    n_val=25,
    n_test=25,
    res_gen=256,
    res_train=64,
    visc=1e-3,
    t_final=30.0,
    n_frames=30,
    forcing_amp=0.1,
    dt_max=1e-3,
    seed=0,
    device="cpu",
    chunk=16,
):
    """Navier-Stokes dataset, generated at res_gen and subsampled.

    Training/val/test frames are kept at res_train; test frames are also kept
    at every intermediate power-of-two resolution up to res_gen for zero-shot
    super-resolution evaluation.
    """
    gen = torch.Generator().manual_seed(seed)
    n_total = n_train + n_val + n_test
    w0 = solvers.gaussian_random_field_2d(n_total, res_gen, generator=gen)

    frames = []
    for i in range(0, n_total, chunk):
        batch = w0[i : i + chunk].to(device)
        sol = solvers.solve_navier_stokes_2d(
            batch, visc=visc, t_final=t_final, n_frames=n_frames,
            forcing_amp=forcing_amp, dt_max=dt_max,
        )
        frames.append(sol.cpu().numpy())
        print(f"ns: solved trajectories {i + 1}-{min(i + chunk, n_total)} / {n_total}")
    frames = np.concatenate(frames)

    def sub(a, res):
        stride = res_gen // res
        return a[:, :, ::stride, ::stride]

    res_super = []
    r = res_train * 2
    while r <= res_gen:
        res_super.append(r)
        r *= 2

    arrays = {
        f"train_{res_train}": sub(frames[:n_train], res_train),
        f"val_{res_train}": sub(frames[n_train : n_train + n_val], res_train),
        f"test_{res_train}": sub(frames[n_train + n_val :], res_train),
    }
    for res in res_super:
        arrays[f"test_{res}"] = sub(frames[n_train + n_val :], res)

    meta = {
        "pde": "ns",
        "periodic": True,
        "res_train": res_train,
        "res_gen": res_gen,
        "res_super": res_super,
        "visc": visc,
        "t_final": t_final,
        "n_frames": n_frames,
        "dt_record": t_final / n_frames,
        "forcing_amp": forcing_amp,
        "dt_max": dt_max,
        "seed": seed,
    }
    _save(out, meta, **arrays)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate PDE datasets")
    sub = parser.add_subparsers(dest="pde", required=True)

    p_heat = sub.add_parser("heat")
    p_heat.add_argument("--out", default=str(DATA_DIR / "heat.npz"))
    p_heat.add_argument("--n-train", type=int, default=800)
    p_heat.add_argument("--n-val", type=int, default=100)
    p_heat.add_argument("--n-test", type=int, default=100)
    p_heat.add_argument("--res-train", type=int, default=64)
    p_heat.add_argument("--res-super", type=int, nargs="*", default=[128, 256])
    p_heat.add_argument("--n-frames", type=int, default=20)
    p_heat.add_argument("--seed", type=int, default=0)
    p_heat.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    p_ns = sub.add_parser("ns")
    p_ns.add_argument("--out", default=str(DATA_DIR / "ns.npz"))
    p_ns.add_argument("--n-train", type=int, default=200)
    p_ns.add_argument("--n-val", type=int, default=25)
    p_ns.add_argument("--n-test", type=int, default=25)
    p_ns.add_argument("--res-gen", type=int, default=256)
    p_ns.add_argument("--res-train", type=int, default=64)
    p_ns.add_argument("--visc", type=float, default=1e-3)
    p_ns.add_argument("--t-final", type=float, default=30.0)
    p_ns.add_argument("--n-frames", type=int, default=30)
    p_ns.add_argument("--chunk", type=int, default=16)
    p_ns.add_argument("--seed", type=int, default=0)
    p_ns.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()
    kwargs = {k: v for k, v in vars(args).items() if k != "pde"}
    if args.pde == "heat":
        kwargs["res_super"] = tuple(kwargs["res_super"])
        generate_heat(**kwargs)
    else:
        generate_ns(**kwargs)


if __name__ == "__main__":
    main()
