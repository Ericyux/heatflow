"""Render all figures from the JSON/npz files written by src.evaluate.

    python -m src.figures --pde heat
    python -m src.figures --pde ns
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path("results")
FIG_DIR = RESULTS_DIR / "figures"
STYLE = {
    "fno": dict(color="#d62728", label="FNO"),
    "fnopf": dict(color="#9467bd", label="FNO + pushforward"),
    "unet": dict(color="#1f77b4", label="U-Net"),
    "cnn": dict(color="#7f7f7f", label="CNN"),
}
PDE_TITLE = {"heat": "2-D heat equation", "ns": "2-D Navier-Stokes (vorticity)"}


def _save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def fig_rollout(results, pde):
    """Median rollout curve per model, with every individual seed drawn as a
    thin line. Seed curves beat a mean +/- std band here: rollout-final
    distributions are skewed (occasional diverging seeds), so a symmetric
    band on a log axis misleads.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, entry in results["models"].items():
        per_seed = np.array(entry.get("rollout_rel_l2_per_seed",
                                      [entry["rollout_rel_l2"]]))
        steps = np.arange(1, per_seed.shape[1] + 1)
        for seed_curve in per_seed:
            ax.semilogy(steps, seed_curve, color=STYLE[name]["color"],
                        alpha=0.35, linewidth=0.9)
        ax.semilogy(steps, np.median(per_seed, axis=0), marker="o",
                    markersize=3, **STYLE[name])
    ax.set_xlabel("autoregressive step")
    ax.set_ylabel("relative L2 error")
    n_seeds = max(e.get("n_seeds", 1) for e in results["models"].values())
    ax.set_title(f"Rollout stability — {PDE_TITLE[pde]} "
                 f"(median of {n_seeds} seeds; thin lines = seeds)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    _save(fig, f"{pde}_rollout.png")


def fig_superres(results, pde):
    fig, ax = plt.subplots(figsize=(6, 4))
    res_train = results["meta"]["res_train"]
    for name, entry in results["models"].items():
        table = entry["super_resolution"]
        stds = entry.get("super_resolution_std", {})
        res = sorted(int(r) for r in table)
        errs = [table[str(r)] for r in res]
        yerr = [stds.get(str(r), 0.0) for r in res]
        ax.errorbar(res, errs, yerr=yerr, marker="o", capsize=3, **STYLE[name])
        ax.set_yscale("log")
    ax.axvline(res_train, color="k", linestyle=":", alpha=0.6)
    ax.text(res_train * 1.02, ax.get_ylim()[0] * 1.3, "training\nresolution",
            fontsize=8, va="bottom")
    ax.set_xscale("log", base=2)
    ax.set_xticks([res_train] + [int(r) for r in results["meta"]["res_super"]])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("evaluation grid resolution")
    ax.set_ylabel("one-step relative L2 error")
    ax.set_title(f"Zero-shot super-resolution — {PDE_TITLE[pde]}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    _save(fig, f"{pde}_superres.png")


def fig_fields(results, pde):
    fields = np.load(RESULTS_DIR / f"{pde}_fields.npz")
    truth = fields["truth"]
    horizon = truth.shape[0] - 1
    steps = [horizon // 4, horizon // 2, horizon]
    rows = ["truth"] + [m for m in STYLE if m in fields]
    row_labels = ["Ground truth"] + [STYLE[m]["label"] for m in rows[1:]]

    cmap = "RdBu_r" if pde == "ns" else "inferno"

    fig, axes = plt.subplots(len(rows), len(steps), figsize=(3.2 * len(steps), 3.0 * len(rows)))
    for i, row in enumerate(rows):
        arr = fields[row]
        for j, t in enumerate(steps):
            ax = axes[i, j]
            if pde == "ns":
                vmax = np.abs(truth).max()
                vmin = -vmax
            else:
                # heat decays ~10x over the rollout: scale each time column
                # by the ground truth at that step so late frames stay visible
                vmax = truth[t].max()
                vmin = 0.0
            im = ax.imshow(arr[t], cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(f"t = step {t}")
            if j == 0:
                ax.set_ylabel(row_labels[i], fontsize=11)
            if row != "truth":
                err = np.linalg.norm(arr[t] - truth[t]) / np.linalg.norm(truth[t])
                ax.set_xlabel(f"rel L2 {err:.3f}", fontsize=9)
    if pde == "ns":
        fig.colorbar(im, ax=axes, shrink=0.6, label="vorticity")
        fig.suptitle(f"Autoregressive rollout — {PDE_TITLE[pde]}", y=0.92)
    else:
        fig.suptitle(f"Autoregressive rollout — {PDE_TITLE[pde]} "
                     "(columns scaled to ground truth per step)", y=0.92)
    _save(fig, f"{pde}_fields.png")


def fig_spectrum(results):
    spectra = results["spectra"]
    k = np.array(spectra["k"])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(k, spectra["truth"], "k-", linewidth=2, label="Ground truth")
    for name in STYLE:
        if name in spectra:
            ax.loglog(k, spectra[name], **STYLE[name])
    ax.set_xlabel("wavenumber |k|")
    ax.set_ylabel("enstrophy")
    ax.set_title("Enstrophy spectrum after full rollout — Navier-Stokes")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    _save(fig, "ns_spectrum.png")


def fig_timing(results, pde):
    timing = results["timing_seconds"]
    meta = results["meta"]
    res_train = meta["res_train"]
    res_gen = meta.get("res_gen", res_train)

    bars = []
    if f"solver_{res_gen}_cpu" in timing:
        bars.append((f"solver {res_gen}² (CPU)", timing[f"solver_{res_gen}_cpu"], "#444444"))
    key_gpu = [k for k in timing if k.startswith(f"solver_{res_gen}_") and not k.endswith("cpu")]
    for k in key_gpu:
        bars.append((f"solver {res_gen}² (GPU)", timing[k], "#444444"))
    if res_gen != res_train:
        k = f"solver_{res_train}_cuda"
        if k in timing:
            bars.append((f"solver {res_train}² (GPU)", timing[k], "#999999"))
    for name in results["models"]:
        k = f"{name}_rollout_{res_train}_cuda"
        if k not in timing:
            k = f"{name}_rollout_{res_train}_cpu"
        bars.append((f"{STYLE[name]['label']} rollout (GPU)", timing[k], STYLE[name]["color"]))

    fig, ax = plt.subplots(figsize=(7, 3.5))
    labels = [b[0] for b in bars]
    vals = [b[1] * 1e3 for b in bars]
    colors = [b[2] for b in bars]
    y = np.arange(len(bars))
    ax.barh(y, vals, color=colors)
    ax.set_yticks(y, labels)
    ax.set_xscale("log")
    ax.set_xlabel("wall-clock per trajectory (ms, log scale)")
    ax.set_title(f"Inference cost — {PDE_TITLE[pde]}")
    for yi, v in zip(y, vals):
        ax.text(v * 1.15, yi, f"{v:,.1f} ms", va="center", fontsize=8)
    ax.set_xlim(right=max(vals) * 8)
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.3)
    _save(fig, f"{pde}_timing.png")


def fig_learning_curves(pde):
    fig, ax = plt.subplots(figsize=(6, 4))
    for name in STYLE:
        run_dirs = sorted(RESULTS_DIR.glob(f"{pde}_{name}_s*"))
        for i, run_dir in enumerate(run_dirs):
            hist_path = run_dir / "history.json"
            if not hist_path.exists():
                continue
            hist = json.loads(hist_path.read_text())
            style = dict(STYLE[name])
            if i > 0:
                style.pop("label")  # one legend entry per model
            ax.semilogy(np.arange(1, len(hist["val_rel_l2"]) + 1),
                        hist["val_rel_l2"], alpha=0.6, linewidth=1.2, **style)
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation one-step relative L2")
    ax.set_title(f"Learning curves — {PDE_TITLE[pde]}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    _save(fig, f"{pde}_learning.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pde", required=True, choices=["heat", "ns"])
    args = parser.parse_args()

    results = json.loads((RESULTS_DIR / f"{args.pde}_eval.json").read_text())
    fig_rollout(results, args.pde)
    fig_superres(results, args.pde)
    fig_fields(results, args.pde)
    fig_timing(results, args.pde)
    fig_learning_curves(args.pde)
    if args.pde == "ns":
        fig_spectrum(results)


if __name__ == "__main__":
    main()
