"""Diagnostics that distinguish a limit cycle from a fixed point or diffusion."""

import numpy as np


def bout_onsets(cell_series, resource_cell, refractory=8):
    """Onset steps of feeding bouts = returns to the resource cell, merging
    visits closer than `refractory` steps into one bout."""
    at_res = np.where(cell_series == resource_cell)[0]
    if len(at_res) == 0:
        return np.array([], dtype=int)
    onsets = [at_res[0]]
    for t in at_res[1:]:
        if t - onsets[-1] > refractory:
            onsets.append(t)
    return np.array(onsets, dtype=int)


def venture_onsets(phase, refractory=4, burn=200):
    """Steps where the agent flips into a venture phase (consolidate->venture),
    merging flips closer than `refractory` steps."""
    v = (np.asarray(phase[burn:]) == "venture").astype(int)
    flips = np.where(np.diff(v) == 1)[0] + 1 + burn
    if len(flips) == 0:
        return flips
    keep = [flips[0]]
    for f in flips[1:]:
        if f - keep[-1] > refractory:
            keep.append(f)
    return np.array(keep, dtype=int)


def cycle_period(out, resource_cell=None, refractory=4, burn=200):
    """Mean steps between consecutive venture onsets = the consolidate->venture
    rhythm period. Returns (period, n_cycles). Robust to per-step jitter."""
    onsets = venture_onsets(out["phase"], refractory=refractory, burn=burn)
    if len(onsets) < 3:
        return np.nan, 0
    gaps = np.diff(onsets)
    return float(np.median(gaps)), len(gaps)


def phase_autocorr(x, maxlag=300):
    x = np.asarray(x, float)
    x = x - x.mean()
    if x.std() < 1e-9:
        return np.zeros(maxlag)
    ac = np.correlate(x, x, "full")[len(x) - 1:]
    return ac[:maxlag] / ac[0]


def power_spectrum(x):
    x = np.asarray(x, float) - np.mean(x)
    w = np.hanning(len(x))
    f = np.fft.rfft(x * w)
    freqs = np.fft.rfftfreq(len(x))
    p = np.abs(f) ** 2
    return freqs, p


def phase_orbit_area(b, u):
    """Signed area enclosed by the (b,u) trajectory (shoelace on the mean-centered
    loop). ~0 for a point attractor or back-and-forth line; large for an orbit."""
    b = np.asarray(b, float); u = np.asarray(u, float)
    b = (b - b.min()) / (np.ptp(b) + 1e-12)
    u = (u - u.min()) / (np.ptp(u) + 1e-12)
    area = 0.5 * np.sum(b[:-1] * u[1:] - b[1:] * u[:-1])
    return abs(area) / len(b)                       # per-step enclosed area
