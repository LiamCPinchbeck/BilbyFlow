"""
two_stage.py — Two-phase importance sampling with higher-mode correction.

Pipeline:
  Stage 1: log w₁ = log L_{22} + log \pi - log q   (all N proposal draws)
           Systematic resample --> ~Kish(n_eff) unique (2,2)-posterior samples

  Stage 2: log w_2 = log L_HM - log L_{22}         (survivors only, v5 numerics)
           Rejection sample at w_2/max(w_2) --> equal-weight HM-corrected samples

Stage-2 denominator uses a mode_array=[[2,2],[2,-2]] deep-copied waveform
generator with identical numerics to the HM numerator — the ratio isolates
the pure higher-mode correction.
"""

import copy
import time
import multiprocessing as mp

import numpy as np
from scipy.special import logsumexp


# ── Pool worker state (fork semantics) ──────────────────────────────────────

_V5_HM = None
_V5_22 = None
_DICTS = None


def _worker_pair(i):
    """Evaluate (log L_HM, log L_22) for sample i on v5 numerics."""
    try:
        lhm = float(_V5_HM.marginal_loglr(_DICTS[i]))
    except (ValueError, RuntimeError):
        lhm = -np.inf
    try:
        l22 = float(_V5_22.marginal_loglr(_DICTS[i]))
    except (ValueError, RuntimeError):
        l22 = -np.inf
    return (lhm, l22)


def _pool_eval_pairs(v5_hm, v5_22, sample_dicts, idx, npool):
    """Single pool pass returning (log L_HM, log L_22) arrays for idx."""
    global _V5_HM, _V5_22, _DICTS
    _V5_HM, _V5_22, _DICTS = v5_hm, v5_22, sample_dicts
    try:
        idx_list = list(idx)
        if npool > 1 and len(idx_list) > npool:
            chunksize = max(1, len(idx_list) // (npool * 4))
            with mp.Pool(npool) as pool:
                out = pool.map(_worker_pair, idx_list, chunksize=chunksize)
        else:
            out = [_worker_pair(i) for i in idx_list]
    finally:
        _V5_HM = _V5_22 = _DICTS = None
    out = np.asarray(out, dtype=np.float64)
    return out[:, 0], out[:, 1]


# ── Resampling utilities ────────────────────────────────────────────────────

def systematic_resample(log_w, n_out, rng):
    """Systematic resampling: lower variance than multinomial. May contain
    duplicates."""
    lw = np.asarray(log_w, dtype=np.float64)
    ok = np.isfinite(lw)
    w = np.zeros(lw.shape)
    w[ok] = np.exp(lw[ok] - logsumexp(lw[ok]))
    cumw = np.cumsum(w)
    cumw[-1] = 1.0
    u = (rng.uniform() + np.arange(n_out)) / n_out
    return np.searchsorted(cumw, u)


def rejection_sample(log_w, rng):
    """Accept index i with probability w_i / max(w). Returns accepted indices."""
    lw = np.asarray(log_w, dtype=np.float64)
    ok = np.isfinite(lw)
    accept = np.zeros(lw.shape, dtype=bool)
    if ok.any():
        accept[ok] = np.log(rng.uniform(size=ok.sum())) < (lw[ok] - lw[ok].max())
    return np.flatnonzero(accept)


def kish_eff(log_w):
    """Kish effective sample size as a percentage of finite weights."""
    lw = np.asarray(log_w, dtype=np.float64)
    lw = lw[np.isfinite(lw)]
    if lw.size == 0:
        return 0.0
    w = np.exp(lw - lw.max())
    return 100.0 * w.sum() ** 2 / (np.sum(w ** 2) * len(w))


# ── Waveform generator helpers ──────────────────────────────────────────────

def make_22_generator(wfg):
    """Deep-copy a waveform generator restricted to (2,\pm 2) modes only."""
    wfg_22 = copy.deepcopy(wfg)
    wa = dict(getattr(wfg, "waveform_arguments", {}) or {})
    wa["mode_array"] = [[2, 2], [2, -2]]
    wfg_22.waveform_arguments = wa
    return wfg_22


# ── Main pipeline ───────────────────────────────────────────────────────────

def two_stage_reweight(log_l22, log_prior, log_draw_prob, sample_dicts,
                       sp_like_v6, ifos, wfg, priors, flags, tc_gps,
                       sp_kwargs=None, window_fn=None, npool=1,
                       t_delta=15.0, seed=0, verbose=True):
    """Two-phase IS: (2,2) reweight --> resample --> HM correction --> rejection.

    Parameters
    ----------
    log_l22 : array (N,)
        Stage-1 (2,2) marginal log-likelihood ratios (already evaluated).
    log_prior, log_draw_prob : arrays (N,)
        Target prior and proposal log-densities.
    sample_dicts : list[dict]
        Full parameter dictionaries for each of the N proposal draws.
    sp_like_v6 : SyntheticExtrinsicLikelihood
        The v6 likelihood used in stage 1 (needed for time_support).
    ifos, wfg, priors : bilby objects
        Interferometers, waveform generator, prior dict for v5 construction.
    flags : dict
        Marginalisation flags {param: bool} for extrinsic parameters.
    tc_gps : float
        Trigger time.

    Returns
    -------
    dict with keys:
        idx_stage1, idx_final, log_w1, log_w2, log_l_hm, log_l_22_v5,
        eff1, eff2, eff_total, kish1, kish2, t_stage2, t_window,
        n_unique_final, counts_stage1
    """
    from synthetic_phase_v5 import SyntheticExtrinsicLikelihood as V5

    rng = np.random.default_rng(seed)
    N = len(log_l22)

    # ── Stage 1: systematic resample from (2,2) weights ─────────────────
    log_w1 = np.asarray(log_l22) + np.asarray(log_prior) - np.asarray(log_draw_prob)
    kish1 = kish_eff(log_w1)
    n_target = max(int(kish1 * N / 100.0), 50)

    idx1_raw = systematic_resample(log_w1, n_target, rng)
    idx1, inv1, counts1 = np.unique(idx1_raw, return_inverse=True,
                                    return_counts=True)

    result = dict(log_w1=log_w1, idx_stage1=idx1, counts_stage1=counts1,
                  idx_stage1_raw=idx1_raw, kish1=kish1,
                  eff1=100.0 * idx1.size / N)

    if verbose:
        print(f"  [2-stage] stage 1: systematic resample {n_target} draws -> "
              f"{idx1.size} unique ({result['eff1']:.1f}% of N; Kish {kish1:.1f}%)")

    if idx1.size == 0:
        result.update(idx_final=idx1, log_w2=np.array([]),
                      log_l_hm=np.array([]), log_l_22_v5=np.array([]),
                      eff2=0.0, eff_total=0.0, kish2=0.0,
                      t_stage2=0.0, t_window=None, n_unique_final=0)
        return result

    # ── Narrow the t_c window from (2,2) support ───────────────────────
    t_window = None
    if flags.get("geocent_time", False):
        jmax = int(idx1[np.argmax(log_w1[idx1])])
        t_window = sp_like_v6.time_support(sample_dicts[jmax], delta=t_delta)
        if verbose:
            print(f"  [2-stage] t_c window: [{t_window[0]*1e3:+.1f}, "
                  f"{t_window[1]*1e3:+.1f}] ms "
                  f"(full prior [{sp_like_v6.t_lo*1e3:+.0f}, "
                  f"{sp_like_v6.t_hi*1e3:+.0f}] ms)")

    # ── Stage 2: v5 HM numerator / v5 (2,2) denominator ────────────────
    sk = dict(sp_kwargs or {})
    sk.update(n_basis=sk.get("n_basis", 5),
              positive_harmonics=sk.get("positive_harmonics", True),
              n_phi=sk.get("n_phi", 64), n_psi=sk.get("n_psi", 32),
              refine_fac=sk.get("refine_fac", 8),
              dt_res=sk.get("dt_res", 2.5e-5))
    sk.pop("phase_basis", None)

    common = dict(marg_phase=flags["phase"], marg_psi=flags["psi"],
                  marg_time=flags["geocent_time"],
                  marg_dist=flags["luminosity_distance"],
                  tc0=float(tc_gps), window_fn=window_fn, t_window=t_window)

    v5_hm = V5(ifos, wfg, priors, **common, **sk)
    v5_22 = V5(ifos, make_22_generator(wfg), priors, **common, **sk)

    if verbose:
        print(f"  [2-stage] stage 2: {idx1.size} unique survivors, "
              f"2x{sk['n_basis']}-call FFT basis (HM + 22, one pool)")

    t0 = time.perf_counter()
    log_l_hm, log_l_22_v5 = _pool_eval_pairs(v5_hm, v5_22, sample_dicts,
                                              idx1, npool)
    t_stage2 = time.perf_counter() - t0

    log_w2 = log_l_hm - log_l_22_v5

    # ── Outlier clipping: prevent single extreme value from dominating ──
    fin = np.isfinite(log_w2)
    if fin.sum() > 10:
        med = np.median(log_w2[fin])
        mad = np.median(np.abs(log_w2[fin] - med)) + 1e-12
        bad = fin & (np.abs(log_w2 - med) > 10.0 * mad)
        if bad.any() and verbose:
            print(f"  [2-stage] ** {bad.sum()} outlier log_w2 "
                  f"(|dlogL - med| > 10 MAD, max={np.abs(log_w2[bad]-med).max():.1f}); "
                  f"clipping for rejection max")
            log_w2 = np.where(bad, med + 10.0 * mad * np.sign(log_w2 - med),
                              log_w2)

    # ── Rejection sample ────────────────────────────────────────────────
    log_w2_raw = log_w2[inv1]
    idx2_raw = rejection_sample(log_w2_raw, rng)
    idx_final = idx1_raw[idx2_raw]
    n_uniq_final = int(np.unique(idx_final).size)

    result.update(
        idx_final=idx_final, log_w2=log_w2,
        log_l_hm=log_l_hm, log_l_22_v5=log_l_22_v5,
        kish2=kish_eff(log_w2_raw),
        n_unique_final=n_uniq_final,
        eff2=100.0 * idx2_raw.size / max(n_target, 1),
        eff_total=100.0 * idx_final.size / N,
        t_stage2=t_stage2, t_window=t_window)

    if verbose:
        d = log_w2[np.isfinite(log_w2)]
        msg = (f"  [2-stage] stage 2: {idx2_raw.size}/{n_target} accepted "
               f"({result['eff2']:.1f}%; Kish {result['kish2']:.1f}%) "
               f"in {t_stage2:.1f}s")
        if d.size:
            msg += (f" | dlogL mean={d.mean():+.3f} "
                    f"max|.|={np.abs(d - d.mean()).max():.3f}")
        print(msg)
        print(f"  [2-stage] final: {idx_final.size}/{N} equal-weight HM "
              f"samples ({result['eff_total']:.2f}%)")
        print(f"  [2-stage] mean multiplicity {counts1.mean():.1f} "
              f"({n_target} draws -> {idx1.size} unique); "
              f"final {idx_final.size} draws, {n_uniq_final} distinct theta")

    return result