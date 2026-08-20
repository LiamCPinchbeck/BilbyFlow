"""
bilbyflow.data.noise — noise realisations and the injection x-combiner.

Three noise models matching training __getitem__ exactly, plus injection_to_x
which assembles a full training-matched x from a projected FD signal + one
noise realisation. RNG is threaded explicitly so the paths are reproducible
(pass a numpy Generator; the global np.random module also works).

PSD estimation routes through canonical.welch_psd — this module holds no
Welch conventions of its own. estimate_psd_from_strain is kept as an alias.
"""

import os
import glob
import numpy as np

from .canonical import (
    window_fd, canonical_bn, canonical_valid_mask, whiten_fd, var_q_of,
    signal_to_whitened, build_x_strain, welch_psd,
)

__all__ = [
    "noise_gaussian_physical",
    "noise_real_segment", "noise_gaussian_whitened",
    "injection_to_x",
    "build_noise_index", "load_noise_strain",
    "estimate_psd_from_strain",
    "draw_real_noise_pair",
]


# ── real-noise segment sourcing (injection runs) ─────────────────────────────

def build_noise_index(noise_data_dir, eras_filter=None):
    """{era: {det: [files]}} over *_H1_noise.npy / *_L1_noise.npy pairs."""
    index = {}
    for era in sorted(os.listdir(noise_data_dir)):
        ed = os.path.join(noise_data_dir, era)
        if not os.path.isdir(ed):
            continue
        if eras_filter is not None and era not in eras_filter:
            continue
        h1 = sorted(glob.glob(os.path.join(ed, "*_H1_noise.npy")))
        l1 = sorted(glob.glob(os.path.join(ed, "*_L1_noise.npy")))
        if h1 and l1:
            index[era] = {"H1": h1, "L1": l1}
    if not index:
        raise SystemExit(f"No complete (era, H1, L1) noise segments under "
                         f"{noise_data_dir} (filter: {eras_filter}).")
    return index


def load_noise_strain(fp):
    """(strain, sr) from a stored (t0, dt, strain) .npy noise file."""
    stored = np.load(fp, allow_pickle=True)
    dt_s = float(stored[1])
    return np.asarray(stored[2], dtype=np.float64), int(round(1.0 / dt_s))


# Canonical Welch estimation lives in canonical.welch_psd; alias kept so
# existing imports keep working.
estimate_psd_from_strain = welch_psd


def _randint(rng, n):
    """Uniform int in [0, n) for both Generator and the np.random module."""
    return int(rng.integers(n)) if hasattr(rng, "integers") \
        else int(rng.randint(n))


def draw_real_noise_pair(noise_index, cfg, g, psd_scope="full-file",
                         psd_cache=None, rng=None):
    """Random (era-consistent) H1/L1 noise segments + PSDs.

    psd_scope='full-file': PSD over the whole strain (training-matched,
    cacheable); 'off-source': excise the drawn segment (FIX-5, no cache).

    psd_cache: optional dict held by the caller; keyed by (file, duration,
    f_min) so one cache can serve multiple configs safely.
    """
    if psd_scope not in ("full-file", "off-source"):
        raise ValueError(f"psd_scope={psd_scope!r}")
    if rng is None:
        rng = np.random
    if psd_cache is None:
        psd_cache = {}

    n_td, sr = g["n_td"], g["sr"]

    era = str(rng.choice(list(noise_index.keys())))
    det_noise_td, det_psds = {}, {}

    for det in ["H1", "L1"]:
        files = noise_index[era][det]
        for _ in range(10):
            fp = files[_randint(rng, len(files))]
            strain, sr_actual = load_noise_strain(fp)
            if sr_actual != sr or len(strain) < n_td:
                continue
            start = _randint(rng, len(strain) - n_td + 1)
            det_noise_td[det] = strain[start:start + n_td]

            if psd_scope == "full-file":
                key = (fp, g["duration"], g["f_min"])
                if key not in psd_cache:   # segment-independent -> cacheable
                    psd_cache[key] = welch_psd(strain, cfg, g)
                det_psds[det] = psd_cache[key]
            else:                          # off-source: depends on start
                det_psds[det] = welch_psd(strain, cfg, g,
                                          exclude=(start, start + n_td))
            break
        else:
            raise RuntimeError(f"No usable {det} noise segment in era {era}")

    return det_noise_td, det_psds, era


# ── the three noise realisations (training-matched) ──────────────────────────

def noise_gaussian_physical(bn, valid, freq_mask, tukey_window, n_td,
                            n_fd_full, rng=None, cdtype=np.complex128):
    """Gaussian-physical noise matching training: white FD -> * bn ->
    window_fd -> / bn. Returns (whitened_noise_fd, physical_noise_fd)."""
    if rng is None:
        rng = np.random
    nv = int(valid.sum())
    white = np.zeros(n_fd_full, dtype=cdtype)
    white[valid] = rng.normal(size=nv) + 1j * rng.normal(size=nv)
    phys_windowed = window_fd(bn * white, freq_mask, tukey_window, n_td)
    noise_w = np.zeros(n_fd_full, dtype=phys_windowed.dtype)
    noise_w[valid] = phys_windowed[valid] / bn[valid]
    return noise_w, phys_windowed


def noise_real_segment(segment_td, bn, valid, freq_mask, tukey_window, n_td,
                       sr, td_dtype=np.float64):
    """Real noise segment -> whitened, matching training:
    rfft(segment * tukey) / sr, then whiten.
    Returns (whitened_noise_fd, physical_noise_fd)."""
    noise_fd = np.fft.rfft(np.asarray(segment_td, dtype=td_dtype)
                           * tukey_window) / sr
    noise_w = np.zeros_like(noise_fd)
    noise_w[valid] = noise_fd[valid] / bn[valid]
    return noise_w, noise_fd


def noise_gaussian_whitened(valid, n_fd_full, rng=None):
    """Unit-variance whitened noise (legacy, NOT the training/evaluation
    default anymore). Sometimes fun to see the sensitivity of the flow to
    the different Gaussian noise realisations though."""
    if rng is None:
        rng = np.random
    nv = int(valid.sum())
    w = np.zeros(n_fd_full, dtype=complex)
    w[valid] = rng.normal(size=nv) + 1j * rng.normal(size=nv)
    return w


# ── injection x-combiner ─────────────────────────────────────────────────────

def injection_to_x(signal_fd_by_det, det_psds, cfg, g, tukey_window, td_norm,
                   noise_kind="gaussian_physical", noise_segments=None,
                   rng=None):
    """Build x from an injection + noise, matching training conventions.

    Returns (x_strain, sig_w_by_det, total_fd_by_det, var_q)."""
    freq_mask = g["freq_mask"]
    n_fd_full, n_td, df, sr = g["n_fd_full"], g["n_td"], g["df"], g["sr"]
    sig_w_by_det, noise_w_by_det, total_fd_by_det, var_q = {}, {}, {}, {}
    for det in ["H1", "L1"]:
        bn = canonical_bn(det_psds[det], df)
        valid = canonical_valid_mask(bn)
        sig_w, sig_windowed = signal_to_whitened(
            signal_fd_by_det[det], bn, freq_mask, tukey_window, n_td)
        sig_w_by_det[det] = sig_w

        if noise_kind == "gaussian_physical":
            noise_w, noise_phys = noise_gaussian_physical(
                bn, valid, freq_mask, tukey_window, n_td, n_fd_full, rng=rng)
        elif noise_kind == "gaussian_whitened":
            noise_w = noise_gaussian_whitened(valid, n_fd_full, rng=rng)
            noise_phys = np.zeros(n_fd_full, dtype=complex)
            noise_phys[valid] = bn[valid] * noise_w[valid]
        elif noise_kind == "real":
            if noise_segments is None or det not in noise_segments:
                raise ValueError(f"noise_kind='real' requires "
                                 f"noise_segments['{det}']")
            noise_w, noise_phys = noise_real_segment(
                noise_segments[det], bn, valid, freq_mask, tukey_window,
                n_td, sr)
        else:
            raise ValueError(f"Unknown noise_kind: {noise_kind}")
        noise_w_by_det[det] = noise_w
        var_q[det] = var_q_of(noise_w, freq_mask)

        total = np.zeros(n_fd_full, dtype=complex)
        total[valid] = sig_windowed[valid] + noise_phys[valid]
        total_fd_by_det[det] = total

    x_strain = build_x_strain(sig_w_by_det, noise_w_by_det, freq_mask, n_td,
                              td_norm)
    return x_strain, sig_w_by_det, total_fd_by_det, var_q