# Neural operators vs. convolutional baselines for PDE surrogate modeling

Can a learned operator replace a numerical PDE solver? This repo benchmarks a
**Fourier Neural Operator** (implemented from scratch) against two
**parameter-matched** convolutional baselines (U-Net, plain CNN) on two 2-D
problems: the heat equation and incompressible **Navier-Stokes** in vorticity
form. All models learn the one-step solution operator `u(t) → u(t + Δt)` and
are evaluated on one-step accuracy, autoregressive rollout stability,
**zero-shot super-resolution**, and wall-clock cost against the solver that
generated the data.

**Headline results (Navier-Stokes, ν = 10⁻³):**

- FNO reaches **0.19%** one-step relative L2 error — **3.7× more accurate than
  a parameter-matched U-Net** (0.70%) and 12× more accurate than a plain CNN
  (2.3%).
- A full 30-frame FNO rollout takes **226 ms** on an RTX 4050 Laptop GPU vs.
  **54.5 s** for the pseudo-spectral solver at data-generation fidelity
  (256², CFL-limited steps) on the same GPU — a **241× speedup** (5.1× vs.
  the same solver run coarsely at 64²).
- Evaluated **zero-shot at 4× the training resolution** (trained at 64²,
  tested at 256²), FNO degrades only to 1.0% error while the U-Net and CNN
  collapse to ~13%: convolution kernels are fixed in *pixels*, Fourier modes
  are fixed in *physical wavenumbers*.
- After 30 autoregressive steps, FNO error is 3.7%; the CNN diverges (84%),
  visibly shredding the flow into receptive-field-sized artifacts.

| | one-step rel. L2 ↓ | 30-step rollout ↓ | zero-shot 256² ↓ | speedup vs. solver ↑ | params |
|---|---|---|---|---|---|
| **FNO** | **0.0019** | **0.037** | **0.010** | 241× | 2.37M |
| U-Net | 0.0070 | 0.087 | 0.136 | 207× | 2.44M |
| CNN | 0.0227 | 0.835 | 0.130 | 454×* | 2.35M |

\*fast but wrong — it has diverged by the end of the rollout.

<p align="center">
<img src="results/figures/ns_fields.png" width="70%">
</p>

<p align="center">
<img src="results/figures/ns_rollout.png" width="49%"><img src="results/figures/ns_superres.png" width="49%">
</p>

<p align="center">
<img src="results/figures/ns_spectrum.png" width="49%"><img src="results/figures/ns_timing.png" width="49%">
</p>

The enstrophy spectrum after a full rollout shows *why* the baselines fail:
all three models track the true spectrum through the energy-containing scales,
but the CNN pumps spurious enstrophy into high wavenumbers (its sawtooth
artifacts), while FNO stays closest to the truth until the dissipation range
that lies below its retained modes.

## Heat equation results

The original version of this project benchmarked only the 2-D heat equation.
Heat is the friendliest possible PDE for a learned surrogate — linear,
dissipative, and error-forgiving (diffusion damps a model's own mistakes,
visible in the CNN row of the rollout figure below). It is kept as a sanity
benchmark, and the same ranking holds:

| | one-step rel. L2 ↓ | 20-step rollout ↓ | zero-shot 256² ↓ | params |
|---|---|---|---|---|
| **FNO** | **0.0006** | **0.0028** | **0.010** | 2.37M |
| U-Net | 0.0015 | 0.0075 | 0.077 | 2.44M |
| CNN | 0.0038 | 0.098 | 0.063 | 2.35M |

Honest caveat: at 64² the explicit finite-difference heat solver costs only
0.71 s/trajectory on GPU (0.31 s on CPU), so learned surrogates win just ~4×
on GPU and actually *lose* to the CPU solver when run on CPU. Speed claims
for operator learning are only meaningful when the reference solver is
genuinely expensive — which is exactly why the Navier-Stokes benchmark
exists: there the solver needs ~1.8 s of CFL-limited spectral substeps per
recorded frame, while the FNO jumps Δt = 1.0 in one 8 ms forward pass.

<p align="center">
<img src="results/figures/heat_fields.png" width="70%">
</p>

## What is being tested

All models learn the **one-step solution operator**: given the field at time
t (plus two coordinate channels), predict the field at t + Δt. Long horizons
are produced **autoregressively** — the model eats its own predictions — which
is the honest test of a surrogate: one-step error only has to be slightly
biased for a rollout to drift or explode. (The first version of this repo
instead mapped a whole space-time volume to a shifted copy of itself with 3-D
convolutions, which is non-causal — the input contains the future — and
structurally cannot be rolled out or evaluated across resolutions.)

**Models** (all ~2.4M parameters, same training protocol — AdamW, cosine
schedule, relative-L2 loss, batch 32, no per-model tuning):

- `FNO2d` — 4 spectral layers, 12 Fourier modes, width 32, implemented from
  scratch in [src/models.py](src/models.py) (~60 lines for the spectral
  convolution; no external operator-learning library). For the non-periodic
  heat problem the lifted representation is zero-padded by a fixed *fraction
  of the domain*, keeping the architecture resolution-agnostic.
- `UNet2d` — 3-level U-Net, the serious conventional baseline: downsampling
  gives it a global receptive field.
- `CNN2d` — 8 stacked 3×3 conv layers (the 2-D analogue of the original
  project's baseline). Its 17-pixel receptive field cannot propagate
  information across the domain in one step, which is fatal for
  advection-dominated dynamics.

**Benchmarks:**

- **Heat**: ∂u/∂t = 0.1 ∇²u on [0,1]², homogeneous Dirichlet boundaries,
  random sum-of-Gaussian initial conditions. 800/100/100
  train/val/test trajectories, 21 frames over T = 1. Ground truth from an
  explicit CFL-limited finite-difference solver. Because the initial
  conditions are analytic, the *same* continuum trajectories are re-solved at
  128² and 256² for super-resolution evaluation.
- **Navier-Stokes** (vorticity form): ∂ω/∂t + u·∇ω = ν∇²ω + f on the periodic
  unit torus, ν = 10⁻³, fixed sinusoidal forcing, Gaussian-random-field
  initial vorticity — the setup of Li et al. (2021). 200/25/25 trajectories,
  31 frames over T = 30, generated at 256² by a pseudo-spectral solver
  (Crank-Nicolson viscosity, dealiased advection, adaptive CFL time step,
  GPU-batched) and subsampled to 64² for training.

**Metrics**: relative L2 error ‖pred − true‖₂/‖true‖₂ (the standard metric of
the neural-operator literature, so numbers are directly comparable to Li et
al.), reported one-step over all test pairs and per-step along rollouts.

**Timing protocol**: median wall-clock per full trajectory at batch size 1,
same machine (RTX 4050 Laptop 6 GB / torch 2.11 cu128), `torch.cuda.synchronize`
around every timed region, cooldown sleeps *between* timed reps. Solver and
models are both implemented in PyTorch, so the comparison stays within one
framework. The 241× headline compares the FNO rollout at its 64² operating
resolution against the solver at the 256² fidelity used to generate ground
truth; against the solver run at the model's own 64² resolution the speedup
is 5.1×. Both numbers are in [results/ns_eval.json](results/ns_eval.json).

## Reproduce

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128   # pick your CUDA
pip install -r requirements.txt

# data (heat ~3 min, NS ~40 min on a laptop GPU)
python -m src.data heat --out data/heat.npz
python -m src.data ns   --out data/ns.npz

# train (3 models x 2 PDEs)
python -m src.train --data data/heat.npz --model fno  --epochs 30
python -m src.train --data data/heat.npz --model unet --epochs 30
python -m src.train --data data/heat.npz --model cnn  --epochs 30
python -m src.train --data data/ns.npz   --model fno  --epochs 40
python -m src.train --data data/ns.npz   --model unet --epochs 40
python -m src.train --data data/ns.npz   --model cnn  --epochs 40

# evaluate + figures
python -m src.evaluate --data data/heat.npz
python -m src.evaluate --data data/ns.npz
python -m src.figures --pde heat
python -m src.figures --pde ns
```

Training and evaluation default to `--gpu-duty 0.6`, which duty-cycles GPU
work to keep thin laptops from thermal-throttling (or shutting down). On a
desktop GPU pass `--gpu-duty 1.0` for full speed; results are identical
either way.

## Repository layout

```
src/
  solvers.py    finite-difference heat + pseudo-spectral Navier-Stokes (torch, batched)
  data.py       dataset generation CLI, one-step pair datasets, normalisation
  models.py     FNO2d (from scratch), UNet2d, CNN2d — parameter-matched
  metrics.py    relative L2, rollout wrapper, enstrophy spectra
  train.py      unified training driver
  evaluate.py   one-step / rollout / super-resolution / timing suite
  figures.py    renders results/figures/*.png from saved eval results
  throttle.py   GPU duty-cycle pacing for thermally limited machines
results/        eval JSONs, sample fields, figures (checkpoints gitignored)
legacy/         original v1 scripts (3-D volume-to-volume TFNO vs. CNN, MSE)
```

## Known limitations

- Single seed per configuration; differences between FNO and U-Net (3-4×)
  are far larger than epoch-to-epoch noise, but error bars would need ~5
  seeds.
- The Navier-Stokes regime (ν = 10⁻³, smooth forcing) is mildly turbulent,
  not a hard-turbulence benchmark; ν = 10⁻⁴ at longer horizons would need
  more data and training than a 6 GB laptop GPU comfortably provides.
- Super-resolution evaluation feeds models the *true* high-resolution state
  and measures one-step error; it does not test high-resolution rollouts.
