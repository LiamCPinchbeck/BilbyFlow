"""
bilbyflow.io.config — config loading and the derived analysis grid.

grid_quantities / window_quantities are the single source of truth for the
FD/TD grid; training, inference and diagnostics all import them here instead
of recomputing per-module (they previously drifted between scripts).
"""

import numpy as np
import yaml
import torch

__all__ = [
    "load_config", "get_prior_bounds",
    "grid_quantities", "window_quantities", "td_norm_from",
    "get_reference_detector_data",
]


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_prior_bounds(cfg):
    """(low, high) tensors over inferred_parameters, in the training
    coordinate (ln for dL / Mc when configured)."""
    dL_log = str(cfg.get("dL_param", "linear")).lower() == "log"
    Mc_log = str(cfg.get("Mc_param", "linear")).lower() == "log"
    lows, highs = [], []
    for p in cfg["inferred_parameters"]:
        pr = cfg["priors"][p]
        lo, hi = float(pr["min"]), float(pr["max"])
        if p == "luminosity_distance" and dL_log:
            lo, hi = float(np.log(lo)), float(np.log(hi))
        if p == "chirp_mass" and Mc_log:
            lo, hi = float(np.log(lo)), float(np.log(hi))
        lows.append(lo)
        highs.append(hi)
    return (torch.tensor(lows, dtype=torch.float32),
            torch.tensor(highs, dtype=torch.float32))


def grid_quantities(cfg):
    """The FD/TD analysis grid derived from duration / sampling_frequency /
    f_min. Returns a dict used everywhere x is packed or split."""
    duration = float(cfg["duration"])
    sr = int(cfg["sampling_frequency"])
    f_min = float(cfg["f_min"])
    df = 1.0 / duration
    n_fd_full = int(duration * sr / 2) + 1
    n_td = 2 * (n_fd_full - 1)
    freq_array = np.arange(n_fd_full) / duration
    freq_mask = (freq_array >= f_min) & (freq_array <= sr / 2.0)
    n_masked = int(np.sum(freq_mask))
    return dict(duration=duration, sr=sr, f_min=f_min, df=df,
                n_fd_full=n_fd_full, n_td=n_td, freq_array=freq_array,
                freq_mask=freq_mask, n_masked=n_masked)


def window_quantities(cfg, n_td):
    """Tukey window over the TD segment and its mean-square factor."""
    from scipy.signal.windows import tukey
    duration = float(cfg["duration"])
    roll_off = float(cfg.get("tukey_roll_off", 0.2))
    w = tukey(n_td, alpha=2.0 * roll_off / duration)
    return w, float(np.mean(w ** 2))


def td_norm_from(g, window_factor=None):
    """TD normalisation constant sqrt(n_masked) / freq_window_factor."""
    freq_window_factor = g["n_masked"] / len(g["freq_mask"])
    return np.sqrt(g["n_masked"]) / freq_window_factor


def get_reference_detector_data(cfg):
    """Reference (H1) frequency array / mask / PSD for building the on-the-fly
    dataset. Uses the fixed bilby design PSD."""
    import bilby
    ifo = bilby.gw.detector.InterferometerList(["H1"])[0]
    ifo.minimum_frequency = float(cfg["f_min"])
    ifo.set_strain_data_from_power_spectral_density(
        sampling_frequency=float(cfg["sampling_frequency"]),
        duration=float(cfg["duration"]),
        start_time=0,
    )
    psd = ifo.power_spectral_density_array.copy()
    asd = np.sqrt(psd)
    asd[asd == 0] = np.inf
    return dict(
        freq_array=ifo.strain_data.frequency_array.copy(),
        freq_mask=ifo.frequency_mask.copy(),
        psd=psd, asd=asd,
        df=ifo.strain_data.frequency_array[1] - ifo.strain_data.frequency_array[0],
    )