"""Ground-truth PDE solvers.

Both solvers are written in PyTorch, batched over trajectories, and run on CPU
or GPU. They serve two roles: generating training/evaluation data, and acting
as the wall-clock baselines that the learned operators are benchmarked
against.
"""

import math

import torch

# ---------------------------------------------------------------------------
# 2-D heat equation, homogeneous Dirichlet boundaries, explicit FD
# ---------------------------------------------------------------------------


def sample_peak_params(n_traj, n_peaks=5, generator=None):
    """Sample parameters (x0, y0, width, amplitude) for sum-of-Gaussian ICs.

    Returns a (n_traj, n_peaks, 4) tensor. The parameters — not the gridded
    fields — define a trajectory, so the same trajectory can be discretised
    at any resolution (needed for zero-shot super-resolution evaluation).
    """
    pos = torch.rand(n_traj, n_peaks, 2, generator=generator)
    width = 0.1 + 0.2 * torch.rand(n_traj, n_peaks, 1, generator=generator)
    amp = 0.5 + 0.5 * torch.rand(n_traj, n_peaks, 1, generator=generator)
    return torch.cat([pos, width, amp], dim=-1)


def gaussian_peaks_ic(params, n, device=None):
    """Evaluate sum-of-Gaussian initial conditions on an n x n grid.

    params: (B, n_peaks, 4) rows of (x0, y0, width, amplitude).
    Returns (B, n, n) fields on the inclusive grid linspace(0, 1, n).
    """
    device = device if device is not None else params.device
    params = params.to(device)
    xs = torch.linspace(0.0, 1.0, n, device=device)
    x_grid, y_grid = torch.meshgrid(xs, xs, indexing="ij")
    x0 = params[:, :, 0, None, None]
    y0 = params[:, :, 1, None, None]
    width = params[:, :, 2, None, None]
    amp = params[:, :, 3, None, None]
    field = amp * torch.exp(-((x_grid - x0) ** 2 + (y_grid - y0) ** 2) / width**2)
    return field.sum(dim=1)


@torch.no_grad()
def solve_heat_2d(u0, alpha=0.1, t_final=1.0, n_frames=20, cfl=0.2):
    """Solve u_t = alpha * (u_xx + u_yy) on [0,1]^2 with u = 0 on the boundary.

    Explicit 5-point finite differences; dt is set from the stability limit
    dt <= cfl * dx^2 / alpha (stable for cfl <= 0.25).

    u0: (B, N, N). Returns (B, n_frames + 1, N, N) recorded at uniform
    intervals t_final / n_frames, including the initial frame.
    """
    n = u0.shape[-1]
    dx = 1.0 / (n - 1)
    dt_rec = t_final / n_frames
    dt_stable = cfl * dx * dx / alpha
    n_sub = max(1, math.ceil(dt_rec / dt_stable))
    lam = alpha * (dt_rec / n_sub) / (dx * dx)

    u = u0.clone()
    u[:, 0, :] = u[:, -1, :] = 0
    u[:, :, 0] = u[:, :, -1] = 0
    frames = [u.clone()]
    for _ in range(n_frames):
        for _ in range(n_sub):
            u[:, 1:-1, 1:-1] += lam * (
                u[:, 2:, 1:-1]
                + u[:, :-2, 1:-1]
                + u[:, 1:-1, 2:]
                + u[:, 1:-1, :-2]
                - 4.0 * u[:, 1:-1, 1:-1]
            )
        frames.append(u.clone())
    return torch.stack(frames, dim=1)


# ---------------------------------------------------------------------------
# 2-D incompressible Navier-Stokes (vorticity form), periodic, pseudo-spectral
# ---------------------------------------------------------------------------


def gaussian_random_field_2d(n_traj, n, alpha=2.5, tau=7.0, generator=None):
    """Sample w0 ~ N(0, sigma^2 (-Lap + tau^2 I)^(-alpha)) on the unit torus.

    Same initial-condition distribution as Li et al. (2021). Returns
    (n_traj, n, n) real fields with zero mean.
    """
    sigma = tau ** (0.5 * (2.0 * alpha - 2.0))
    k1 = torch.fft.fftfreq(n, d=1.0 / n)[:, None]
    k2 = torch.fft.rfftfreq(n, d=1.0 / n)[None, :]
    sqrt_eig = (
        (n * n)
        * math.sqrt(2.0)
        * sigma
        * (4.0 * math.pi**2 * (k1**2 + k2**2) + tau**2) ** (-alpha / 2.0)
    )
    sqrt_eig[0, 0] = 0.0  # zero-mean field

    real = torch.randn(n_traj, n, n // 2 + 1, generator=generator)
    imag = torch.randn(n_traj, n, n // 2 + 1, generator=generator)
    coeff = sqrt_eig * torch.complex(real, imag) / math.sqrt(2.0)
    return torch.fft.irfft2(coeff, s=(n, n))


def ns_forcing(n, amplitude=0.1, device=None):
    """Fixed forcing f = amplitude * (sin(2 pi (x+y)) + cos(2 pi (x+y)))."""
    xs = torch.arange(n, device=device) / n
    x_grid, y_grid = torch.meshgrid(xs, xs, indexing="ij")
    phase = 2.0 * math.pi * (x_grid + y_grid)
    return amplitude * (torch.sin(phase) + torch.cos(phase))


@torch.no_grad()
def solve_navier_stokes_2d(
    w0,
    visc=1e-3,
    t_final=30.0,
    n_frames=30,
    forcing_amp=0.1,
    dt_max=1e-3,
    cfl=0.5,
):
    """Pseudo-spectral solver for 2-D incompressible Navier-Stokes on the
    periodic unit torus, in vorticity form:

        dw/dt + u . grad(w) = visc * Lap(w) + f,   div(u) = 0

    Crank-Nicolson for the viscous term, explicit advection evaluated in
    physical space with 2/3-rule dealiasing, adaptive CFL-limited time step.
    Setup follows Li et al. (2021), "Fourier Neural Operator for Parametric
    Partial Differential Equations".

    w0: (B, N, N) initial vorticity. Returns (B, n_frames + 1, N, N).
    """
    batch, n, _ = w0.shape
    device = w0.device

    k1 = torch.fft.fftfreq(n, d=1.0 / n, device=device)[:, None]
    k2 = torch.fft.rfftfreq(n, d=1.0 / n, device=device)[None, :]
    lap = 4.0 * math.pi**2 * (k1**2 + k2**2)  # eigenvalues of -Lap
    lap_pois = lap.clone()
    lap_pois[0, 0] = 1.0  # avoid div-by-zero; psi mean is a gauge choice
    # 2/3-rule dealiasing mask for the quadratic advection term.
    dealias = ((k1.abs() < n / 3.0) & (k2.abs() < n / 3.0)).to(w0.dtype)

    f_hat = torch.fft.rfft2(ns_forcing(n, forcing_amp, device=device))

    two_pi_i_k1 = 2.0j * math.pi * k1
    two_pi_i_k2 = 2.0j * math.pi * k2
    dx = 1.0 / n

    w_hat = torch.fft.rfft2(w0)
    frames = [w0.clone()]
    dt_rec = t_final / n_frames

    for _ in range(n_frames):
        t_left = dt_rec
        while t_left > 1e-12:
            psi_hat = w_hat / lap_pois
            u = torch.fft.irfft2(two_pi_i_k2 * psi_hat, s=(n, n))  # dpsi/dy
            v = torch.fft.irfft2(-two_pi_i_k1 * psi_hat, s=(n, n))  # -dpsi/dx
            w_x = torch.fft.irfft2(two_pi_i_k1 * w_hat, s=(n, n))
            w_y = torch.fft.irfft2(two_pi_i_k2 * w_hat, s=(n, n))

            u_max = torch.maximum(u.abs().max(), v.abs().max()).clamp_min(1e-8)
            dt = min(dt_max, cfl * dx / u_max.item(), t_left)

            adv_hat = torch.fft.rfft2(u * w_x + v * w_y) * dealias
            visc_half = 0.5 * dt * visc * lap
            w_hat = (
                w_hat * (1.0 - visc_half) + dt * (f_hat - adv_hat)
            ) / (1.0 + visc_half)
            t_left -= dt
        frames.append(torch.fft.irfft2(w_hat, s=(n, n)))
    return torch.stack(frames, dim=1)
