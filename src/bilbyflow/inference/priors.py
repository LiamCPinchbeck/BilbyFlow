"""
bilbyflow.inference.priors — the physical (target) prior.

make_prior_dict builds the bilby PriorDict for the physical analysis prior.
Mostly to work as an inbetween for the way I did my config files originally (and
will later be changed) to talk nicely with the Bilby configs.

The priors are generally (may change later to account for selection effects):
- UniformInComponentsChirpMass / MassRatio, 
- PowerLaw(2) distance, 
- isotropic angles. 

This is the target pi in the importance weight w = L pi / q, and it is
deliberately independent of any training-side proposal reshaping (the dL/Mc
PowerLaw pads applied when building the banks) — the prior swap in
inference.sample corrects proposal-vs-target, so this stays the physical prior.
"""

import numpy as np
import bilby
from ..coordinates.sky import MAX_DT_HL

__all__ = ["make_prior_dict", "dL_bounds", "make_injection_priors",
           "sky_prior_log_terms"]


def dL_bounds(cfg):
    pr = cfg["priors"]["luminosity_distance"]
    return float(pr["min"]), float(pr["max"])


def make_injection_priors(cfg, tc_gps, mode="physical", max_inj_dL=5000.0):
    """Prior dict the INJECTIONS are drawn from. The IS target prior stays
    make_prior_dict (physical) regardless — only the injected population
    changes:
      physical  -> target prior itself (population-weighted efficiency)
      training  -> dL log-uniform, mirroring the training proposal (raw-flow
                   PP calibration only; mean efficiency NOT population-weighted)
      nearby    -> dL uniform on [dL_min, max_inj_dL] (test population)"""
    
    priors = make_prior_dict(cfg, tc_gps=tc_gps)

    lo, hi = dL_bounds(cfg)

    if mode == "physical":
        return priors
    priors = priors.copy()

    if mode == "training":
        priors["luminosity_distance"] = bilby.core.prior.LogUniform(
            minimum=lo, maximum=hi, name="luminosity_distance",
            latex_label="$d_L$", unit="Mpc")
        
    elif mode == "nearby":
        priors["luminosity_distance"] = bilby.core.prior.Uniform(
            minimum=lo, maximum=float(max_inj_dL), name="luminosity_distance",
            latex_label="$d_L$", unit="Mpc")
    else:
        raise ValueError(f"unknown injection-prior mode '{mode}'")

    return priors

def detector_to_radec_log_jacobian(samples, inferred_parameters, tc):
    """log|d(dt_HL, phi_det)/d(ra, dec)| = log(MAX_DT_HL * |cos(dec)|).
    The (ra,dec)->(theta,phi) rotation is area-preserving on the sphere and
    the sin(theta) factors cancel, leaving only the dt stretch and the
    dec-colatitude factor. `samples` holds physical (ra, dec) columns."""
    dec = np.asarray(samples[:, inferred_parameters.index("dec")], dtype=float)
    return np.log(MAX_DT_HL) + np.log(np.abs(np.cos(dec)) + 1e-300)


def sky_prior_log_terms(mode, npe_samples, sample_dicts, priors, cfg, tc_gps):
    """explicit sky-prior convention for the IS weights. 

    'detector-uniform': target prior uniform in (dt_HL, phi_det) — matches
        training; prior and proposal share coordinates, the constant cancels
        -> 0.
    'isotropic': target isotropic on the sphere; needs the detector->(ra,dec)
        log-Jacobian so the proposal density can be expressed in (ra, dec)."""

    n = len(sample_dicts)

    if mode == "detector-uniform":
        return np.zeros(n)

    if mode == "isotropic":    
        ra = np.array([s["ra"] for s in sample_dicts])
        dec = np.array([s["dec"] for s in sample_dicts])
    
        lp = np.zeros(n)
        for p, v in (("ra", ra), ("dec", dec)):
            if p in priors and not isinstance(priors[p], (float, int)):
                lp += priors[p].ln_prob(v)

        log_jac = detector_to_radec_log_jacobian(
            npe_samples, cfg["inferred_parameters"], tc=float(tc_gps))

        return lp - np.asarray(log_jac, dtype=float)
    
    raise ValueError(f"unknown sky-prior mode '{mode}'")


def make_prior_dict(cfg, tc_gps=None):
    """bilby PriorDict for the physical prior over inferred + nuisance
    parameters. geocent_time is centred on tc_gps if given, else on
    ref_geocent_time."""
    from bilby.core.prior import PriorDict, Uniform, Cosine, Sine, PowerLaw


    ref_tc = float(cfg["ref_geocent_time"])
    event_tc = tc_gps if tc_gps is not None else ref_tc
    pr = cfg["priors"]


    nuisance_params = cfg.get("nuisance_parameters", []) or []
    all_params = cfg["inferred_parameters"] + nuisance_params


    # Below just handles typical prior construction (from tutorials, please propose changes if
        # this is not standard/optimal) handling mostly min/max values.
    priors = PriorDict()
    for p in all_params:
        if p == "geocent_time":
            priors[p] = Uniform(minimum=event_tc + pr[p]["min"],
                                maximum=event_tc + pr[p]["max"],
                                name=p, latex_label="$t_c$")
        elif p == "chirp_mass":
            priors[p] = bilby.gw.prior.UniformInComponentsChirpMass(
                minimum=pr[p]["min"], maximum=pr[p]["max"],
                name=p, latex_label="$\\mathcal{M}$", unit="$M_{\\odot}$")
        elif p == "mass_ratio":
            priors[p] = bilby.gw.prior.UniformInComponentsMassRatio(
                minimum=pr[p]["min"], maximum=pr[p]["max"], name=p)
        elif p == "luminosity_distance":
            priors[p] = PowerLaw(alpha=2, minimum=pr[p]["min"], maximum=pr[p]["max"],
                                 name=p, latex_label="$d_L$", unit="Mpc")
        elif p == "dec":
            priors[p] = Cosine(name=p)
        elif p in ("tilt_1", "tilt_2", "theta_jn"):
            priors[p] = Sine(name=p)
        elif p in ("ra", "phi_12", "phi_jl", "phase", "psi"):
            priors[p] = Uniform(minimum=pr[p]["min"], maximum=pr[p]["max"],
                                name=p, boundary="periodic")
        elif p in pr:
            priors[p] = Uniform(minimum=pr[p]["min"], maximum=pr[p]["max"], name=p)
    return priors