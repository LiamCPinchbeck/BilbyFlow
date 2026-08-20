"""
bilbyflow.coordinates.params: parameter-vector <-> full-params dict, and the
log-coordinate (ln dL, ln Mc) exponentiation used at output.

Currently don't recommend ln Mc based on some A/B testing, but it should be implemented.

Sky handling (ra<->dt_HL, dec<->phi_det and the sidereal offset) lives in
coordinates.sky (from sky_coord_utils_b); 
this module covers the intrinsic plus geocent_time packing that both training and reweighting share.


"""

import numpy as np

__all__ = ["theta_to_full_params", "dL_to_physical"]

_ALL_BBH_PARAMS = [
    "chirp_mass", "mass_ratio", 
    "luminosity_distance", "theta_jn",
    "ra", "dec", 
    "geocent_time",  # recommend making a flow nuisance parameter
    "a_1", "a_2", 
    "tilt_1", "tilt_2",
    "phi_12", "phi_jl", 
    "psi", "phase" # recommend making a flow nuisance parameters
    ]


def theta_to_full_params(theta_vec, cfg, reference_params=None, tc_gps=None):
    """Map an inferred-parameter vector to a full 15-param dict for bilby /
    the synthetic likelihood. geocent_time is stored as an offset and shifted
    back onto the event GPS time; unset params default to 0 (fill values)."""
    inferred = cfg["inferred_parameters"]
    ref_tc = float(cfg["ref_geocent_time"])
    event_tc = tc_gps if tc_gps is not None else ref_tc
    params = {}
    for i, p in enumerate(inferred):
        params[p] = (float(theta_vec[i]) + event_tc) if p == "geocent_time" else float(theta_vec[i])
    if reference_params:
        for p in reference_params:
            params.setdefault(p, reference_params[p])
    for p in _ALL_BBH_PARAMS:
        if p not in params:
            params[p] = event_tc if p == "geocent_time" else 0.0
    return params


def dL_to_physical(samples, param_names, cfg):
    """Exponentiate log coordinates (optionally dL AND/OR Mc) in place-safe copy.
    Name kept from the training script so call sites don't need edits."""
    out = np.array(samples, copy=True)
    if str(cfg.get("dL_param", "linear")).lower() == "log" \
            and "luminosity_distance" in param_names:
        di = param_names.index("luminosity_distance")
        out[:, di] = np.exp(out[:, di])
    if str(cfg.get("Mc_param", "linear")).lower() == "log" \
            and "chirp_mass" in param_names:
        mi = param_names.index("chirp_mass")
        out[:, mi] = np.exp(out[:, mi])
    return out