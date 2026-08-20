"""
bilbyflow.data.psd — PSD bank construction for on-the-fly whitening.

One builder in active use:
  precompute_psd_bank_from_segments — Welch estimates from noise segments,
                                      era-balanced, with a per-frequency
                                      median-ASD floor. All Welch conventions
                                      route through canonical.welch_psd.

precompute_psd_bank (GP sampling) is retired: GP-sampled PSDs are smooth and
lineless and never captured the detail needed. The name is kept as a stub so
old configs fail loudly instead of importing-then-crashing.
"""

import os
import glob
import numpy as np

from .canonical import welch_psd

__all__ = ["precompute_psd_bank", "precompute_psd_bank_from_segments"]

# sub-segment lengths (s) cut from each noise file, and cuts per length
_WINDOW_LENGTHS = (32, 64, 128, 256)
_N_PER_LENGTH = 4
_BANK_SEED = 42


def _freq_grid(cfg):
    """Grid dict (welch_psd-compatible) from the config."""
    duration = float(cfg["duration"])
    sr = int(cfg["sampling_frequency"])
    n_fd = int(duration * sr / 2) + 1
    freq_array = np.arange(n_fd) / duration
    return dict(duration=duration, sr=sr, f_min=float(cfg["f_min"]),
                n_fd=n_fd, freq_array=freq_array, df=1.0 / duration)


def _segment_psds(fp, cfg, g):
    """All Welch PSD estimates from one noise file, on the analysis grid.
    Returns a list (possibly empty)."""
    sr = g["sr"]
    nperseg = int(g["duration"] * sr)

    stored = np.load(fp, allow_pickle=True)
    if int(round(1.0 / float(stored[1]))) != sr:
        return []
    strain = stored[2].astype(np.float64)
    seg_duration = len(strain) / sr

    out = []
    for win_len in _WINDOW_LENGTHS:
        if win_len > seg_duration:
            continue
        n_win = int(win_len * sr)
        max_start = len(strain) - n_win
        starts = ([max_start // 2] if _N_PER_LENGTH == 1
                  else np.linspace(0, max_start, _N_PER_LENGTH, dtype=int))
        for start in starts:
            sub = strain[start:start + n_win]
            if len(sub) < nperseg:
                continue
            psd = welch_psd(sub, cfg, g)
            if np.isfinite(psd[g["freq_array"] >= g["f_min"]]).any():
                out.append(psd)
    return out


def _collect_raw_psds(noise_data_dir, era_dirs, cfg, g):
    """{era: {det: [psd, ...]}} over every noise file of every era."""
    raw = {era: {"H1": [], "L1": []} for era in era_dirs}
    for era in era_dirs:
        for det in ("H1", "L1"):
            files = sorted(glob.glob(
                os.path.join(noise_data_dir, era, f"*_{det}_noise.npy")))
            if not files:
                print(f"  WARNING: no files for {era} {det}")
                continue
            for fp in files:
                try:
                    raw[era][det] += _segment_psds(fp, cfg, g)
                except Exception as e:
                    print(f"  WARNING: {fp}: {e}")
        print(f"  {era}: {len(raw[era]['H1'])} H1, "
              f"{len(raw[era]['L1'])} L1 PSDs")
    return raw


def _era_counts(n_psds, valid_eras, era_weights):
    """PSDs per era: weighted when era_weights is given, else balanced."""
    if era_weights:
        w = np.array([float(era_weights.get(e, 0.0)) for e in valid_eras])
        if w.sum() <= 0:
            raise ValueError(
                f"era_weights {era_weights} give zero total over {valid_eras}")
        counts = np.floor(n_psds * w / w.sum()).astype(int)
        counts[np.argmax(w)] += n_psds - counts.sum()
        print(f"  PSD bank era weights {dict(zip(valid_eras, counts))}")
    else:
        base, rem = divmod(n_psds, len(valid_eras))
        counts = np.array([base + (i < rem) for i in range(len(valid_eras))])
    return counts


def _apply_asd_floor(bank, in_band, factor, det_name):
    """Raise per-frequency ASDs below median/factor to the floor, in place."""
    asd = np.sqrt(bank)
    finite_in = np.isfinite(asd[:, in_band]) & (asd[:, in_band] > 0)
    asd_median = np.nanmedian(
        np.where(finite_in, asd[:, in_band], np.nan), axis=0)
    floor = np.zeros(bank.shape[1])
    floor[in_band] = asd_median / factor
    fb = np.broadcast_to(floor[None, :], asd.shape)
    finite = np.isfinite(asd) & (asd > 0)
    low = finite & (asd < fb)
    print(f"  {det_name} ASD floor: raised {int(low.sum())}/{int(finite.sum())} "
          f"({100.0 * low.sum() / max(finite.sum(), 1):.2f}%) "
          f"below median/{factor}")
    bank[:] = np.where(low, fb, asd) ** 2


def precompute_psd_bank_from_segments(cfg, noise_data_dir):
    """Estimate PSDs directly from all noise segments (canonical Welch
    conventions), era-balanced, with a per-frequency median-ASD floor to
    prevent whitening blow-up."""
    bank_cfg = cfg.get("psd_bank", {})
    n_psds = int(bank_cfg.get("n_psds", 5000))
    eras_filter = bank_cfg.get("eras", None)
    asd_floor_factor = float(bank_cfg.get("asd_floor_factor", 5.0))

    g = _freq_grid(cfg)
    freq_array, n_fd = g["freq_array"], g["n_fd"]

    era_dirs = sorted(d for d in os.listdir(noise_data_dir)
                      if os.path.isdir(os.path.join(noise_data_dir, d))
                      and (eras_filter is None or d in eras_filter))
    if not era_dirs:
        raise RuntimeError(
            f"No era directories in {noise_data_dir} (filter {eras_filter})")

    print(f"Building real-segment PSD bank from {noise_data_dir}")
    print(f"  Eras {era_dirs} | n_psds {n_psds} | "
          f"asd_floor_factor {asd_floor_factor}")

    raw = _collect_raw_psds(noise_data_dir, era_dirs, cfg, g)
    valid_eras = [e for e in era_dirs if raw[e]["H1"] and raw[e]["L1"]]
    if not valid_eras:
        raise RuntimeError("No eras have both H1 and L1 PSDs")
    print(f"  Valid eras: {valid_eras}")

    counts = _era_counts(n_psds, valid_eras, bank_cfg.get("era_weights"))

    rng = np.random.default_rng(_BANK_SEED)
    psd_H1 = np.empty((n_psds, n_fd))
    psd_L1 = np.empty((n_psds, n_fd))
    era_labels = np.empty(n_psds, dtype=object)
    idx = 0
    for era, n_this in zip(valid_eras, counts):
        n_this = int(n_this)
        for pool, bank in ((raw[era]["H1"], psd_H1), (raw[era]["L1"], psd_L1)):
            bank[idx:idx + n_this] = np.asarray(pool)[
                rng.integers(0, len(pool), size=n_this)]
        era_labels[idx:idx + n_this] = era
        idx += n_this
        print(f"  {era}: sampled {n_this} PSDs")
    assert idx == n_psds, f"Filled {idx} but expected {n_psds}"

    in_band = freq_array >= g["f_min"]
    _apply_asd_floor(psd_H1, in_band, asd_floor_factor, "H1")
    _apply_asd_floor(psd_L1, in_band, asd_floor_factor, "L1")

    perm = rng.permutation(n_psds)
    return dict(psd_H1=psd_H1[perm], psd_L1=psd_L1[perm],
                era=era_labels[perm], freq_array=freq_array, df=g["df"])


def precompute_psd_bank(cfg, gp_dir):
    """RETIRED (2026-08-20): GP-sampled PSDs are smooth and lineless and do
    not capture real-detector detail. Use precompute_psd_bank_from_segments.
    Kept so old configs fail with a clear message."""
    raise NotImplementedError(
        "precompute_psd_bank (GP sampling) is retired; use "
        "precompute_psd_bank_from_segments with a noise-segment directory.")