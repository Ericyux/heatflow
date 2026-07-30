"""Evaluation suite: one-step error, autoregressive rollout stability,
zero-shot super-resolution, and wall-clock timing against the numerical
solver that generated the data.

Example:
    python -m src.evaluate --data data/heat.npz
Writes results/<pde>_eval.json plus results/<pde>_fields.npz (sample fields
and curves consumed by src.figures).
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from . import solvers, throttle
from .data import PDEData
from .metrics import (StepPredictor, enstrophy_spectrum, one_step_errors,
                      relative_l2, rollout_errors)
from .models import build_model, count_params

RESULTS_DIR = Path("results")
MODELS = ["fno", "unet", "cnn"]


def load_predictor(pde, name, periodic, device):
    out_dir = RESULTS_DIR / f"{pde}_{name}"
    config = json.loads((out_dir / "config.json").read_text())
    model = build_model(name, periodic=periodic)
    model.load_state_dict(torch.load(out_dir / "best.pt", map_location=device))
    predictor = StepPredictor(model, config["norm_mean"], config["norm_std"],
                              periodic, device)
    return predictor, config


def _timeit(fn, reps, warmup, sync):
    if not sync:
        warmup = 0  # no CUDA context / kernel-launch cost to amortise on CPU
    for _ in range(warmup):
        fn()
    if sync:
        torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        if sync:
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        # thermal cooldown between reps; outside the timed window, so it
        # does not affect the measurements themselves
        throttle.pace(t0)
    return statistics.median(times)


def time_solver(data, res, device, reps, dt_max=None):
    """Median wall-clock seconds to produce one full trajectory."""
    meta = data.meta
    sync = device.type == "cuda"
    if meta["pde"] == "heat":
        gen = torch.Generator().manual_seed(1234)
        ic = solvers.gaussian_peaks_ic(solvers.sample_peak_params(1, generator=gen),
                                       res, device=device)
        fn = lambda: solvers.solve_heat_2d(
            ic, alpha=meta["alpha"], t_final=meta["t_final"],
            n_frames=meta["n_frames"],
        )
    else:
        w0 = solvers.gaussian_random_field_2d(
            1, res, generator=torch.Generator().manual_seed(1234)
        ).to(device)
        fn = lambda: solvers.solve_navier_stokes_2d(
            w0, visc=meta["visc"], t_final=meta["t_final"],
            n_frames=meta["n_frames"], forcing_amp=meta["forcing_amp"],
            dt_max=dt_max if dt_max is not None else meta["dt_max"],
        )
    return _timeit(fn, reps=reps, warmup=1, sync=sync)


def time_model(predictor, res, n_steps, device, reps=50):
    """(one-step seconds, full-rollout seconds) at batch size 1."""
    sync = device.type == "cuda"
    u = torch.randn(1, res, res, device=device)
    one = _timeit(lambda: predictor.step(u), reps=reps, warmup=5, sync=sync)
    roll = _timeit(lambda: predictor.rollout(u, n_steps), reps=max(3, reps // 10),
                   warmup=1, sync=sync)
    return one, roll


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-cpu-timing", action="store_true")
    parser.add_argument("--gpu-duty", type=float, default=0.6,
                        help="fraction of wall time the GPU is kept busy "
                             "(thermal headroom for laptops; 1.0 = flat out)")
    args = parser.parse_args()

    throttle.set_duty(args.gpu_duty)
    device = torch.device(args.device)
    data = PDEData(args.data)
    meta = data.meta
    pde = meta["pde"]
    res_train = meta["res_train"]
    horizon = meta["n_frames"]
    test_frames = data.frames("test")

    results = {"pde": pde, "meta": meta, "models": {}}
    fields = {"truth": test_frames[0].numpy()}  # sample trajectory for figures

    predictors = {}
    for name in MODELS:
        predictor, config = load_predictor(pde, name, data.periodic, device)
        predictors[name] = predictor
        entry = {"n_params": config["n_params"]}

        # --- one-step accuracy at training resolution
        errs = one_step_errors(predictor, test_frames)
        entry["one_step_rel_l2"] = errs.mean().item()
        entry["one_step_rel_l2_std"] = errs.std().item()

        # --- autoregressive rollout stability
        curve = rollout_errors(predictor, test_frames)
        entry["rollout_rel_l2"] = curve.tolist()
        entry["rollout_final_rel_l2"] = curve[-1].item()

        # --- zero-shot super-resolution (trained at res_train only)
        entry["super_resolution"] = {str(res_train): entry["one_step_rel_l2"]}
        for res in data.super_resolutions():
            errs_r = one_step_errors(predictor, data.frames("test", res))
            entry["super_resolution"][str(res)] = errs_r.mean().item()

        # --- sample rollout fields for qualitative figures
        sample = predictor.rollout(test_frames[:1, 0].to(device), horizon)
        fields[name] = sample[0].cpu().numpy()

        results["models"][name] = entry
        print(f"{pde}/{name}: one-step {entry['one_step_rel_l2']:.4f}, "
              f"rollout final {entry['rollout_final_rel_l2']:.4f}, "
              f"super-res {entry['super_resolution']}")

    # --- enstrophy spectra at the final rollout step (NS only)
    if pde == "ns":
        spectra = {}
        k, true_spec = enstrophy_spectrum(test_frames[:, -1])
        spectra["k"] = k.tolist()
        spectra["truth"] = true_spec.tolist()
        for name, predictor in predictors.items():
            rolled = []
            for i in range(0, test_frames.shape[0], 8):
                t_chunk = time.perf_counter()
                chunk = test_frames[i : i + 8, 0].to(device)
                rolled.append(predictor.rollout(chunk, horizon)[:, -1].cpu())
                throttle.pace(t_chunk)
            _, spec = enstrophy_spectrum(torch.cat(rolled))
            spectra[name] = spec.tolist()
        results["spectra"] = spectra

    # --- wall-clock timing per trajectory
    timing = {"n_frames": horizon}
    res_gen = meta.get("res_gen", res_train)
    reps_gpu = 3 if pde == "ns" else 20
    timing[f"solver_{res_gen}_{device.type}"] = time_solver(data, res_gen, device, reps=reps_gpu)
    if res_gen != res_train:
        # solver run directly at the model's resolution (CFL-limited dt)
        timing[f"solver_{res_train}_{device.type}"] = time_solver(
            data, res_train, device, reps=reps_gpu, dt_max=float("inf"),
        )
    if not args.skip_cpu_timing:
        cpu = torch.device("cpu")
        timing[f"solver_{res_gen}_cpu"] = time_solver(data, res_gen, cpu, reps=1 if pde == "ns" else 3)
    for name, predictor in predictors.items():
        one, roll = time_model(predictor, res_train, horizon, device)
        timing[f"{name}_step_{res_train}_{device.type}"] = one
        timing[f"{name}_rollout_{res_train}_{device.type}"] = roll
        if not args.skip_cpu_timing:
            cpu_pred = StepPredictor(predictor.model.to("cpu"), predictor.mean,
                                     predictor.std, predictor.periodic, torch.device("cpu"))
            one_c, roll_c = time_model(cpu_pred, res_train, horizon,
                                       torch.device("cpu"), reps=10)
            timing[f"{name}_step_{res_train}_cpu"] = one_c
            timing[f"{name}_rollout_{res_train}_cpu"] = roll_c
            predictor.model.to(device)
    results["timing_seconds"] = timing

    solver_ref = timing[f"solver_{res_gen}_{device.type}"]
    for name in MODELS:
        roll = timing[f"{name}_rollout_{res_train}_{device.type}"]
        results["models"][name]["speedup_vs_solver"] = solver_ref / roll

    out_json = RESULTS_DIR / f"{pde}_eval.json"
    out_json.write_text(json.dumps(results, indent=2))
    np.savez_compressed(RESULTS_DIR / f"{pde}_fields.npz", **fields)
    print(f"wrote {out_json}")
    for name in MODELS:
        print(f"  {name}: {results['models'][name]['speedup_vs_solver']:.1f}x faster "
              f"than solver at {res_gen}^2 (per trajectory, {device.type})")


if __name__ == "__main__":
    main()
