"""GPU duty-cycle throttling.

Sustained 100% GPU load can thermally trip thin laptops. Hot loops call
pace(t_start) after each unit of work; when a duty cycle < 1 is configured,
it sleeps long enough that work occupies roughly that fraction of wall time.
No-ops entirely on CPU-only machines or at duty >= 1.
"""

import time

import torch

_DUTY = 1.0


def set_duty(duty):
    global _DUTY
    _DUTY = max(0.05, min(1.0, float(duty)))


def pace(t_start):
    """Sleep so the work since t_start amounts to the configured duty cycle."""
    if _DUTY >= 1.0 or not torch.cuda.is_available():
        return
    torch.cuda.synchronize()
    work = time.perf_counter() - t_start
    if work > 0:
        time.sleep(work * (1.0 / _DUTY - 1.0))
