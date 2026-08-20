"""
bilbyflow.io.config — config loading and the derived analysis grid.

grid_quantities is the single source of truth for the FD/TD grid; training,
inference and diagnostics all import it here instead of recomputing
per-module (they previously drifted between scripts).

Window / td_norm constructions live in data.canonical (canonical_tukey,
canonical_td_norm); window_quantities / td_norm_from are kept here as thin
back-compat wrappers with lazy imports (io.config cannot import
data.canonical at module level — canonical imports this module).
"""

import numpy as np
import yaml

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
    import torch

    log_params = {
        "luminosity_distance": str(cfg.get("dL_param", "linear")).lower() == "log",
        "chirp_mass": str(cfg.get("Mc_param", "linear")).lower() == "log",
    }

    lows, highs = [], []
    for p in cfg["inferred_parameters"]:
        pr = cfg["priors"][p]
        lo, hi = float(pr["min"]), float(pr["max"])
        if log_params.get(p, False):
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
    
    n_fd_full = int(duration * sr / 2) + 1

    freq_array = np.arange(n_fd_full) / duration
    freq_mask = (freq_array >= f_min) & (freq_array <= sr / 2.0)

    return dict(duration=duration, sr=sr, f_min=f_min, df=1.0 / duration,
                n_fd_full=n_fd_full, n_td=2 * (n_fd_full - 1),
                freq_array=freq_array, freq_mask=freq_mask,
                n_masked=int(np.sum(freq_mask)))


def window_quantities(cfg, n_td):
    """Back-compat: (tukey window, mean-square window factor).
    Canonical definition: data.canonical.canonical_tukey."""
    from ..data.canonical import canonical_tukey

    w = canonical_tukey(cfg, n_td)
    return w, float(np.mean(w ** 2))


def td_norm_from(g, window_factor=None):
    """Back-compat: TD normalisation constant.
    Canonical definition: data.canonical.canonical_td_norm."""

    from ..data.canonical import canonical_td_norm

    return canonical_td_norm(g)


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
        df=ifo.strain_data.frequency_array[1]
           - ifo.strain_data.frequency_array[0],
    )