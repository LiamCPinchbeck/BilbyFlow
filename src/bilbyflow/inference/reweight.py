"""
bilbyflow.inference.reweight — importance reweighting of NPE posteriors.

reweight_event evaluates the exact (marginalised) likelihood on flow draws and
forms the importance weights log_w = logL + log_prior - log_draw_prob, then
hands off to the two-stage (or single-stage) HM scheme in inference.two_stage. 
Headline n_eff / khat come from the single-stage weights(immediately used HM lnlike).
Possible speedups come from the two stage but injects extra variance in the weights
which decreases the effective sample size.


Marginalisation is delegated to inference.synthetic_phase
(SyntheticExtrinsicLikelihood): any of {phase, psi, geocent_time,
luminosity_distance} that is a nuisance (not in cfg["inferred_parameters"]) is
numerically marginalised in the IS target; 
the corresponding bilby prior factor is therefore skipped from log_prior. 

With synthetic marginalisation off, those parameters are fixed at fill values 
(the old, biased mode) and a warning is printed.

result["log_weight_final"] is the plotting/publishing weight: -inf everywhere
except the equal-weight two-stage survivors (weight 0.0 in log space), falling
back to the raw stage-1 log_weight when two-stage did not run.

The waveform generator, SNR utility, PSIS estimator, parameter fill, and the
marginal likelihood all come from sibling package modules; only the
reweighting orchestration lives here.
"""

import time
import multiprocessing as mp

import numpy as np
from scipy.special import logsumexp
import bilby

from ..likelihood.waveform import make_waveform_generator
from ..likelihood.snr import rho_opt_mf, optimal_snr_squared
from ..coordinates.params import theta_to_full_params
from ..diagnostics.psis import psis_khat
from ..io.config import grid_quantities, window_quantities
from .priors import sky_prior_log_terms
from .synthetic_phase import SyntheticExtrinsicLikelihood
from .two_stage import two_stage_reweight, single_stage_hm

__all__ = [
    "MARGABLE", "is_geocent_inferred", "marg_flags", "reweight_event",
]

# These are variables that are marginalizable either due to simplicity
    # or that in some regimes they have analytical relationships with
    # the likelihood and can be exactly marginalized out. This is only
    # the case with waveform models that don't include higher order mode
    # content e.g. you can't do analytical marginalization with XPHM waveforms
MARGABLE = ("phase", "psi", "geocent_time", "luminosity_distance")


# ── marginalisation bookkeeping ──────────────────────────────────────────────

def is_geocent_inferred(cfg):
    return "geocent_time" in cfg["inferred_parameters"]


def marg_flags(cfg):
    """{param: is_nuisance} over MARGABLE — nuisance == not inferred by the flow."""
    inf = cfg["inferred_parameters"]
    return {p: (p not in inf) for p in MARGABLE}


# ── multiprocessing workers (fork; module-level globals) ─────────────────────

_bilby_likelihood = None
_sample_dicts = None
_sp_like = None
_worker_seed_base = 0


def _get_log_likelihood_worker(i):
    try:
        _bilby_likelihood.parameters.update(_sample_dicts[i])
        return (_bilby_likelihood.log_likelihood_ratio(),
                np.nan, np.nan, np.nan, np.nan)
    except (ValueError, RuntimeError):
        return (-np.inf, np.nan, np.nan, np.nan, np.nan)


def _synth_worker(i):
    """Exact (phase, psi, tc, dL)-marginal logLR + joint recovery draw."""
    try:
        rng = np.random.default_rng(_worker_seed_base + 1_000_003 * (i + 1))
        return _sp_like.marginal_loglr(_sample_dicts[i],
                                       return_extrinsic=True, rng=rng)
    except (ValueError, RuntimeError):
        return (-np.inf, np.nan, np.nan, np.nan, np.nan)



###############################################################################################################
###############################################################################################################
# --------------------------------        MAIN ENTRY PONIT           ------------------------------------------
###############################################################################################################
###############################################################################################################

# this function is a prime example of me using to many print statements. Things went
    # wrong quite often with this function (I spent almost more time debugging this 
    # thing than the actuall cool ML architecture), hence the print statements. As 
    # with a lot of things in this package, hopefully soon more of this will be off-
    # loaded to a dedicated logger.
def reweight_event(
        npe_samples,  # in the paper we usually refer to these are proposal samples from the flow
        log_draw_prob, # low prob that the flow assigns the npe_samples
        ifos, cfg, priors,
        npool=1, tc_gps=None, map_params=None,
        approximant=None, synthetic_phase=True, sp_kwargs=None,
        do_two_stage=True,
        injection_params=None, 
        nuisance_reference="truth", # if they are not marginalized over what values to give them
        sky_prior="detector-uniform", n_selfcheck=0,
        event_seed=None, window_template=False):
    """Reweight one event's flow draws to the exact (marginalised) target.

    Real-data mode (defaults) and injection mode share this function; the
    injection extras are all optional:
      injection_params    known truths -> logL(truth)/deficit + batch
                          self-check + SNR gate diagnostics
      nuisance_reference  'truth' fills unfilled nuisance params from the
                          injection (FIX-7-adjacent), anything else -> zeros
      sky_prior           'detector-uniform' (training-matched, term = 0) or
                          'isotropic' (adds ln pi(ra,dec) + detector->radec
                          log-Jacobian) — FIX-2
      n_selfcheck         batch size for the marginalisation self-check
                          (FIX-3; needs a likelihood with .self_check, the
                          pure-v5 evaluator skips it and only tracks
                          refine_delta)
      event_seed          decorrelates worker rngs across injections (FIX-6)
      window_template     template windowed like the injected signal (FIX-1)

    Returns a result dict with the headline n_eff / efficiency / khat (plus
    n_eff_raw pre-smoothing), the per-sample log_weight / log_weight_final /
    log_weight_weighted, recovered extrinsics, two-stage bookkeeping, and SNR
    diagnostics.
    """

    global _bilby_likelihood, _sample_dicts, _sp_like, _worker_seed_base



    n_samples = len(npe_samples)
    start_time = ifos[0].strain_data.start_time
    flags = marg_flags(cfg)
    do_synth = synthetic_phase and any(flags.values())
    seed_base = int(event_seed) if event_seed is not None else int(tc_gps)

    if window_template:
        g = grid_quantities(cfg)
        tukey_window, _ = window_quantities(cfg, g["n_td"])
        wfg = make_waveform_generator(cfg, start_time=start_time,
                                      approximant=approximant, windowed=True,
                                      g=g, tukey_window=tukey_window)
    else:
        wfg = make_waveform_generator(cfg, start_time=start_time,
                                      approximant=approximant)

    bilby_likelihood = bilby.gw.GravitationalWaveTransient(
        interferometers=ifos, waveform_generator=wfg, priors=priors,
        phase_marginalization=False)


    # filling values with truths and calculating optimal lnlike and 
        # SNR values for diagnostics. Should not be part of the main
        # reweighting loop
    truth_extra = {}
    if injection_params is not None:
        bilby_likelihood.parameters.update(injection_params)
        log_l_truth = bilby_likelihood.log_likelihood_ratio()
        rho2 = optimal_snr_squared(ifos, wfg, injection_params)
        # FIX-1 diagnostic: E[logLR at truth] = rho^2/2; a persistent deficit
        # >> sqrt(rho^2/2) indicates data/template mismatch (e.g. windowing).
        print(f"  log_L_ratio at truth: {log_l_truth:.1f}  "
              f"(rho_opt^2/2 = {rho2 / 2:.1f}, deficit = {rho2 / 2 - log_l_truth:+.1f}, "
              f"expected fluctuation ~ {np.sqrt(rho2 / 2):.1f})")
        truth_extra = dict(log_l_truth=float(log_l_truth), rho2_opt=float(rho2),
                           window_template=bool(window_template),
                           sky_prior=sky_prior)

    # uses MAP injection params if truths are not known (e.g. real events)
    elif map_params is not None:
        try:
            bilby_likelihood.parameters.update(map_params)
            print(f"  log_L_ratio at published MAP: {bilby_likelihood.log_likelihood_ratio():.1f}")
        except Exception as e:
            print(f"  (could not evaluate log_L at MAP: {e})")


    ref_p = injection_params if (injection_params is not None
                                 and nuisance_reference == "truth") else None

    sample_dicts = [theta_to_full_params(npe_samples[i], 
                                         cfg,
                                         reference_params=ref_p, 
                                         tc_gps=tc_gps)
                                for i in range(n_samples)]


    # skip the bilby prior factor for anything marginalised in the target
    skip = ("ra", "dec") + tuple(p for p in MARGABLE if flags[p] and do_synth)
    log_prior = np.zeros(n_samples)
    for param in priors.keys():
        if param in skip:
            continue
        if not isinstance(priors[param], (float, int)):
            vals = np.array([s[param] for s in sample_dicts])
            log_prior += priors[param].ln_prob(vals)

    if sky_prior != "detector-uniform":
        print(f"  sky prior convention: {sky_prior}")


    log_prior += sky_prior_log_terms(sky_prior, npe_samples, sample_dicts,
                                     priors, cfg, tc_gps)
    log_prior = np.nan_to_num(log_prior, nan=-np.inf)


    _bilby_likelihood = bilby_likelihood
    _sample_dicts = sample_dicts
    _sp_like = None
    _worker_seed_base = seed_base * 2_000_000_011


    if do_synth and do_two_stage:
        marg_list = [p for p in MARGABLE if flags[p]]
        print(f"  synthetic extrinsics: exact marginal over {marg_list}")
        # bit of this reweighting that does the actual marginalization
        _sp_like = SyntheticExtrinsicLikelihood(
            ifos, wfg, priors,
            marg_phase=flags["phase"], marg_psi=flags["psi"],
            marg_time=flags["geocent_time"], marg_dist=flags["luminosity_distance"],
            tc0=float(tc_gps), **(sp_kwargs or {}))
        import time as _time
        _t0 = _time.time()

        # the marginalized log-like. Make sure the grids you use for doing the integration
        #   are fine enough for the result to be trustworthy. Otherwise you might get inflated
        #   values or general gobbledy-gook 
        m = _sp_like.marginal_loglr(sample_dicts[0]) 
        print(f"  first-sample marginal logLR = {m:.1f} "
              f"(refine_delta={_sp_like.last_refine_delta:+.3f}, "
              f"{_time.time() - _t0:.3f}s/sample)")

        # batch self-check over random posterior samples (injections)
            # an attempt to check that the marginal lnlike results are stable
        if injection_params is not None and n_selfcheck > 0:
            rng_sc = np.random.default_rng(seed_base)
            idx_sc = rng_sc.choice(n_samples, size=min(n_selfcheck, n_samples),
                                   replace=False)
            has_sc = hasattr(_sp_like, "self_check")
            e0s, eTs, deltas = [], [], []
            for j in idx_sc:
                if has_sc:
                    e0, eT = _sp_like.self_check(sample_dicts[j],
                                                 bilby_likelihood, n_test=4)
                    e0s.append(e0)
                    eTs.append(eT)
                _sp_like.marginal_loglr(sample_dicts[j])
                deltas.append(abs(_sp_like.last_refine_delta))
            dm = max(deltas)
            truth_extra.update(selfcheck_idx=idx_sc,
                               selfcheck_refine_delta=np.array(deltas))
            if has_sc:
                e0m, eTm = max(e0s), max(eTs)
                truth_extra.update(selfcheck_err0=np.array(e0s),
                                   selfcheck_errT=np.array(eTs))
                print(f"  self-check over {len(idx_sc)} samples: "
                      f"max err(dt=0)={e0m:.2e} "
                      + ("(OK)" if e0m < 1e-2 else "** RAISE --n-phase-basis **")
                      + f" | max err(dt=max)={eTm:.2e}", end="")
            else:
                print(f"  self-check ({len(idx_sc)} samples): evaluator has no "
                      f".self_check (pure-v5), refine_delta only", end="")
            print(f" | max |refine_delta|={dm:.3f}"
                  + ("" if dm < 0.5 else "  ** raise --n-phi/--n-psi/--refine-fac **"))
    
    elif any(flags.values()) and not do_synth:
        print("  ** WARNING: synthetic marginalization OFF but "
              f"{[p for p in MARGABLE if flags[p]]} are nuisance -> FIXED at "
              f"fill values (old, biased mode).")


    skip_main = do_synth and not do_two_stage    # single-stage: HM pass IS the loop
    if not skip_main:
        worker = _synth_worker if do_synth else _get_log_likelihood_worker
        print(f"  Likelihood for {n_samples} samples (npool={npool})"
              + (" [synthetic-extrinsic marginal]" if do_synth else "") + "...")
        if npool > 1:
            chunksize = max(1, n_samples // (npool * 4))
            with mp.Pool(processes=npool) as pool:
                out = list(pool.imap(worker, range(n_samples), chunksize=chunksize))
        else:
            out = [worker(i) for i in range(n_samples)]
        out = np.asarray(out, dtype=np.float64)
        log_likelihood = out[:, 0]
        rec = dict(recovered_phase=out[:, 1], recovered_psi=out[:, 2],
                   recovered_tc_offset=out[:, 3], recovered_dL=out[:, 4])
    else:
        print("  [1-stage] skipping v6 main loop (log_likelihood = HM marginal)")
        log_likelihood = None
        rec = dict(recovered_phase=np.full(n_samples, np.nan),
                   recovered_psi=np.full(n_samples, np.nan),
                   recovered_tc_offset=np.full(n_samples, np.nan),
                   recovered_dL=np.full(n_samples, np.nan))


    # ── two-stage (or single-stage) HM resample ──
    ts = None
    if do_synth:
        if do_two_stage:
            ts = two_stage_reweight(
                log_l22=log_likelihood, log_prior=log_prior,
                log_draw_prob=log_draw_prob, sample_dicts=sample_dicts,
                sp_like_v6=_sp_like, ifos=ifos, wfg=wfg, priors=priors,
                flags=flags, tc_gps=tc_gps, sp_kwargs=sp_kwargs,
                window_fn=None, npool=npool, seed=seed_base)
        else:
            ts = single_stage_hm(
                log_prior=log_prior, log_draw_prob=log_draw_prob,
                sample_dicts=sample_dicts, ifos=ifos, wfg=wfg, priors=priors,
                flags=flags, tc_gps=tc_gps, sp_kwargs=sp_kwargs,
                window_fn=None, npool=npool, seed=seed_base,
                log_l22_v6=None)
            log_likelihood = ts["log_l_hm"]      # HM marginal IS the likelihood
    else:
        print("  ** two-stage skipped: requires synthetic marginalisation")

    log_weight = log_likelihood + log_prior - log_draw_prob


    base_extra = {}
    if ts is not None:
        base_extra = dict(idx_final=ts["idx_final"], idx_stage1=ts["idx_stage1"],
                          eff_stage1=ts["eff1"], eff_stage2=ts["eff2"],
                          eff_total=ts["eff_total"], kish1=ts["kish1"],
                          kish2=ts["kish2"], t_stage2=ts["t_stage2"],
                          n_unique_final=ts.get("n_unique_final"),
                          log_w2=ts["log_w2"], t_window=ts["t_window"])
        # full-length WEIGHTED HM weights (idx_stage1-aligned; -inf elsewhere)
        if ts.get("log_w_final") is not None:
            lw_hm = np.full(n_samples, -np.inf)
            lw_hm[ts["idx_stage1"]] = np.asarray(ts["log_w_final"])
            base_extra["log_weight_weighted"] = lw_hm


    valid = np.isfinite(log_weight)
    n_valid = int(valid.sum())


    # publishing weight: equal-weight survivors (log 0.0), else raw stage-1
    log_weight_final = np.full(n_samples, -np.inf)
    if ts is not None and "idx_final" in base_extra and len(base_extra["idx_final"]) > 0:
        log_weight_final[base_extra["idx_final"]] = 0.0
    else:
        log_weight_final = log_weight


    base = dict(log_likelihood=log_likelihood, log_prior=log_prior,
                log_weight=log_weight, log_draw_prob=log_draw_prob,
                n_samples=n_samples, synthetic_phase=do_synth,
                log_weight_final=log_weight_final,
                marg_flags=flags, tc0=float(tc_gps),
                **truth_extra, **rec, **base_extra)
    if n_valid < 2:
        return dict(n_eff=0.0, efficiency=0.0, n_valid=n_valid, khat=np.inf, **base)


    # headline stats: idx_final (equal-weight, unique) when two-stage ran,
    # else PSIS on the stage-1 weights.
    n_eff_psis = None            # set only on the PSIS path with khat < 0.7
    if ts is not None and len(base_extra.get("idx_final", [])) > 0:        
        n_final = int(base_extra["n_unique_final"])
        n_eff = float(ts["kish2"] * len(base_extra["idx_stage1"]) / 100.0)
        khat = float(ts["khat2"] if ts.get("khat2") is not None
                     else psis_khat(log_weight[valid]))
        n_eff_raw = float(n_eff)      # kish2 is computed on the RAW combined weights
        smoothed = False
        log_w_use = None
    else:
        log_w_valid = log_weight[valid]
        khat = float(psis_khat(log_w_valid))
        n_eff_raw = float(np.exp(2 * logsumexp(log_w_valid)
                                 - logsumexp(2 * log_w_valid)))
        log_w_use, smoothed = log_w_valid, False
        if khat < 0.7:
            try:
                from arviz import psislw
                log_w_smooth, _ = psislw(log_w_valid)
                log_w_use, smoothed = np.asarray(log_w_smooth), True
            except ImportError:
                pass

        # Two ESS numbers answering two questions:
        #   n_eff_raw   raw weights — proposal quality (Dingo-comparable)
        #   n_eff_psis  smoothed weights (khat < 0.7 only) — the variance
        #               proxy for the PSIS-smoothed estimates
        # HEADLINE n_eff = psis when the smoothing is trustworthy, raw
        # otherwise. NOTE this makes the headline discontinuous at
        # khat = 0.7 and non-comparable to raw-reporting pipelines —
        # n_eff_raw is always in the result dict for the paper table.
        n_eff_psis = (float(np.exp(2 * logsumexp(log_w_use)
                                   - logsumexp(2 * log_w_use)))
                      if smoothed else None)
        n_eff = n_eff_psis if n_eff_psis is not None else n_eff_raw

    base["log_weight_smoothed"] = log_w_use if smoothed else None
    base["psis_smoothed"] = smoothed
    base["n_eff_raw"] = n_eff_raw
    base["n_eff_psis"] = n_eff_psis
        
    # SNR bookkeeping at the published MAP and the max-weight posterior sample
    snr = {}
    if map_params is not None:
        try:
            snr["rho_opt_map"], snr["rho_mf_map"] = rho_opt_mf(ifos, wfg, map_params)
        except Exception as e:
            print(f"  (rho at MAP failed: {e})")

    try:
        jmax = int(np.nanargmax(np.where(valid, log_weight, -np.inf)))
        snr["rho_opt_maxw"], snr["rho_mf_maxw"] = rho_opt_mf(ifos, wfg, sample_dicts[jmax])
    except Exception as e:
        print(f"  (rho at max-weight sample failed: {e})")

    if snr:
        print("  SNR  " + "  ".join(f"{k}={v:.1f}" for k, v in snr.items()))


    return dict(n_eff=float(n_eff), efficiency=float(n_eff / n_valid * 100),
                n_valid=n_valid, khat=khat, **snr, **base)