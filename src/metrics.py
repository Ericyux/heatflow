"""Metrics and the autoregressive rollout wrapper."""

import time

import torch

from .data import add_coords
from .throttle import pace


def relative_l2(pred, target, eps=1e-12):
    """Per-sample relative L2 error ||pred - target|| / ||target||.

    The standard metric of the neural-operator literature (Li et al., 2021),
    which makes numbers comparable across papers. pred/target: (B, ...);
    returns (B,).
    """
    diff = (pred - target).flatten(1).norm(dim=1)
    denom = target.flatten(1).norm(dim=1).clamp_min(eps)
    return diff / denom


class StepPredictor:
    """Wraps a trained model into a physical-space one-step map.

    Handles normalisation and coordinate channels so callers can work purely
    with physical fields of any resolution.
    """

    def __init__(self, model, mean, std, periodic, device):
        self.model = model.to(device).eval()
        self.mean = mean
        self.std = std
        self.periodic = periodic
        self.device = device

    @torch.no_grad()
    def step(self, u):
        """u: (B, N, N) physical -> (B, N, N) physical at t + dt."""
        x = (u.unsqueeze(1) - self.mean) / self.std
        x = add_coords(x, self.periodic)
        y = self.model(x)
        return (y * self.std + self.mean).squeeze(1)

    @torch.no_grad()
    def rollout(self, u0, n_steps):
        """Autoregressive rollout. u0: (B, N, N) -> (B, n_steps + 1, N, N)."""
        u = u0
        frames = [u]
        for _ in range(n_steps):
            u = self.step(u)
            frames.append(u)
        return torch.stack(frames, dim=1)


@torch.no_grad()
def rollout_errors(predictor, true_frames, batch=8):
    """Mean relative L2 at every rollout step, starting from frame 0.

    true_frames: (n_traj, T + 1, N, N). Returns a (T,) tensor where entry t
    is the mean relative L2 of the model state after t + 1 autoregressive
    steps against the reference trajectory.
    """
    horizon = true_frames.shape[1] - 1
    totals = torch.zeros(horizon)
    count = 0
    for i in range(0, true_frames.shape[0], batch):
        t_chunk = time.perf_counter()
        chunk = true_frames[i : i + batch].to(predictor.device)
        pred = predictor.rollout(chunk[:, 0], horizon)
        for t in range(horizon):
            err = relative_l2(pred[:, t + 1], chunk[:, t + 1])
            totals[t] += err.sum().cpu()
        count += chunk.shape[0]
        pace(t_chunk)
    return totals / count


@torch.no_grad()
def one_step_errors(predictor, frames, pair_batch=None):
    """Relative L2 over all consecutive one-step pairs in `frames`.

    Pairs are processed in chunks sized inversely to the grid area so peak
    memory stays roughly constant across resolutions. Returns a flat tensor
    of per-pair errors.
    """
    n = frames.shape[-1]
    if pair_batch is None:
        pair_batch = max(4, (64 * 64 * 64) // (n * n))
    inputs = frames[:, :-1].reshape(-1, n, n)
    targets = frames[:, 1:].reshape(-1, n, n)
    errs = []
    for i in range(0, inputs.shape[0], pair_batch):
        t_chunk = time.perf_counter()
        pred = predictor.step(inputs[i : i + pair_batch].to(predictor.device))
        err = relative_l2(pred, targets[i : i + pair_batch].to(predictor.device))
        errs.append(err.cpu())
        pace(t_chunk)
    return torch.cat(errs)


def enstrophy_spectrum(w):
    """Radially binned enstrophy spectrum of a vorticity field.

    w: (B, N, N). Returns (k_bins, spectrum) with spectrum averaged over the
    batch — used to check whether models preserve small-scale structure.
    """
    n = w.shape[-1]
    w_hat = torch.fft.fftshift(torch.fft.fft2(w), dim=(-2, -1)) / (n * n)
    power = w_hat.abs() ** 2
    k = torch.arange(n) - n // 2
    kx, ky = torch.meshgrid(k, k, indexing="ij")
    k_mag = torch.sqrt(kx.float() ** 2 + ky.float() ** 2).round().long()
    k_max = n // 2
    spectrum = torch.zeros(w.shape[0], k_max)
    for i in range(1, k_max + 1):
        mask = k_mag == i
        spectrum[:, i - 1] = power[:, mask].sum(dim=1)
    return torch.arange(1, k_max + 1), spectrum.mean(dim=0)
