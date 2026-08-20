"""
bilbyflow.io.samples — published-PE sample loading and comparison helpers.

Reads bilby .json and pesummary .h5/.hdf5 posteriors, extracts a MAP point,
and packs published samples into the inferred-parameter array used for the
corner overlays.
"""

import os
import glob
import numpy as np
import bilby

try:
    from pesummary.io import read as pesummary_read
    HAS_PESUMMARY = True
except ImportError:
    HAS_PESUMMARY = False

__all__ = [
    "find_sample_files", "load_published_samples",
    "extract_map_params", "published_samples_to_array",
]

# pesummary analysis-label preference when no approximant is requested
_LABEL_PREFERENCE = ("C01:IMRPhenomXPHM", "C01:IMRPhenomPv2", "C01:Mixed")

# parameters extract_map_params reports (0.0 when absent)
_BBH_PARAMS = ("chirp_mass", "mass_ratio", "luminosity_distance", "theta_jn",
               "ra", "dec", "geocent_time", "a_1", "a_2", "tilt_1", "tilt_2",
               "phi_12", "phi_jl", "psi", "phase")


def find_sample_files(samples_dir, event_names=None):
    """{event: file} over .h5/.hdf5/.json posteriors under samples_dir.
    With event_names, matches each name against basenames then full paths;
    without, infers the event from the GW* token in the filename."""

    all_files = []
    for ext in ("**/*.h5", "**/*.hdf5", "**/*.json"):
        all_files.extend(glob.glob(os.path.join(samples_dir, ext),
                                   recursive=True))
    
    sample_files = {}
    if event_names:
        for event in event_names:
            matches = [f for f in all_files if event in os.path.basename(f)] \
                or [f for f in all_files if event in f]
            if matches:
                sample_files[event] = matches[0]
        return sample_files

    for f in all_files:
        basename = os.path.basename(f)
        name = next((part for part in basename.split("_")
                     if part.startswith("GW")),
                    os.path.splitext(basename)[0])
        sample_files[name] = f
    return sample_files


def load_published_samples(filepath, approximant=None):
    """{parameter: samples} from a bilby .json or pesummary .h5/.hdf5."""

    if filepath.endswith(".json"):
        result = bilby.core.result.read_in_result(filename=filepath)
        return {col: result.posterior[col].values
                for col in result.posterior.columns}

    if filepath.endswith((".h5", ".hdf5")):

        if not HAS_PESUMMARY:
            raise RuntimeError("pesummary required to read .h5 files")
    
        f = pesummary_read(filepath)
        labels = f.labels
        label = next((l for l in ((approximant,) + _LABEL_PREFERENCE
                                  if approximant else _LABEL_PREFERENCE)
                      if l in labels), labels[0])
        
        print(f"  Using label: {label} (available: {labels})")

        samples = f.samples_dict[label]

        return {p: np.array(samples[p]) for p in samples.parameters}

    raise ValueError(f"Unknown file format: {filepath}")


def extract_map_params(samples):
    """Per-parameter MAP point: 1-D KDE argmax when there are enough samples,
    median as fallback, 0.0 when the parameter is absent."""
    from scipy.stats import gaussian_kde

    params = {}
    for p in _BBH_PARAMS:
        if p not in samples or len(samples[p]) == 0:
            params[p] = 0.0
            continue
        vals = np.array(samples[p], dtype=float)
        if len(vals) > 10:
            try:
                kde = gaussian_kde(vals)
                grid = np.linspace(vals.min(), vals.max(), 1000)
                params[p] = float(grid[np.argmax(kde(grid))])
                continue
            except Exception:
                pass
        params[p] = float(np.median(vals))
    return params


def published_samples_to_array(samples, param_names, event_tc, cfg):
    """Pack published samples into an (n, n_params) array in inferred-parameter
    order. geocent_time becomes an offset from event_tc when it is inferred,
    else zeroed (it is marginalised)."""
    from ..inference.reweight import is_geocent_inferred

    n = min(len(v) for k, v in samples.items()
            if k in param_names and len(v) > 0)
    arr = np.zeros((n, len(param_names)))

    for j, p in enumerate(param_names):
        if p in samples and len(samples[p]) >= n:
            vals = np.array(samples[p][:n])
            if p == "geocent_time":
                vals = ((vals - event_tc) if is_geocent_inferred(cfg)
                        else np.zeros(n))
            arr[:, j] = vals
    return arr