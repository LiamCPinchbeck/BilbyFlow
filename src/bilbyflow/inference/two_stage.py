"""
bilbyflow.inference.two_stage — two-phase importance sampling.

  Stage 1  log_w1 = logL_22 + log_prior - log_q          (v6, all N draws)
           Bernoulli thinning to ~Kish n_eff unique survivors
  Stage 2  direct v5 HM weight on the survivors,
           HT-corrected, PSIS-smoothed if k-hat < 0.7,
           then rejection sample -> equal-weight unique HM samples.

The combined weight is the DIRECT higher-mode weight,
    log_w_comb = logL_HM + log_prior[idx] - log_q[idx] - log_p_thin,
so the v6 (2,2) screen enters only through the thinning probability p_i,
which cancels in the Horvitz-Thompson estimator. The v6-v5 (2,2) residual
therefore never rides on the weight (see the reweighting-math note, section 8).

Stage-2 denominator (logL_22_v5) is retained only for the dlogL diagnostic,
computed for free from the same v5 basis (marginal_loglr_pair).

single_stage_hm is the expensive reference: the same v5 HM evaluator on ALL
samples with direct weights; its Kish is what the two-stage efficiency should
reproduce.
"""

import time
import numpy as np
import multiprocessing as mp
from scipy.special import logsumexp

__all__ = ["two_stage_reweight", "single_stage_hm", "kish_eff",
           "rejection_sample", "thin_indices"]

# v5 evaluator defaults, shared by both pipelines
_V5_DEFAULTS = dict(n_basis=5, positive_harmonics=True, n_phi=64, n_psi=32,
                    refine_fac=8, dt_res=2.5e-5)

# module-level state for pool workers (fork start method)
_V5_HM = None
_DICTS = None
_PAIR = True          # compute the (2,2)-collapse denominator too?




# ── pool evaluation ──────────────────────────────────────────────────────────

def _worker_pair(i):
    try:
        if _PAIR:
            lhm, l22 = _V5_HM.marginal_loglr_pair(_DICTS[i])
        else:
            lhm, l22 = _V5_HM.marginal_loglr(_DICTS[i]), np.nan
        return (float(lhm), float(l22))
    except (ValueError, RuntimeError):
        return (-np.inf, -np.inf)




def _pool_eval_pairs(v5_hm, sample_dicts, idx, npool, pair=True):
    """One pool pass: (logL_HM, logL_22) arrays for sample_dicts[idx].
    pair=False skips the (2,2)-collapse grid pass (~2x); log_l_22 is NaN."""
    global _V5_HM, _DICTS, _PAIR
    _V5_HM, _DICTS, _PAIR = v5_hm, sample_dicts, pair
    try:
        idx_list = list(idx)
        if npool > 1 and len(idx_list) > npool:
            with mp.Pool(npool) as pool:
                out = pool.map(_worker_pair, idx_list,
                               chunksize=max(1, len(idx_list) // (npool * 4)))
        else:
            out = [_worker_pair(i) for i in idx_list]
    finally:
        _V5_HM = _DICTS = None
    out = np.asarray(out, dtype=np.float64)
    return out[:, 0], out[:, 1]



# mostly for multi-processing
def _build_v5(ifos, wfg, priors, flags, tc_gps, sp_kwargs, window_fn,
              t_window):
    """The v5 HM evaluator with the shared kwarg defaults applied."""
    from .synthetic_phase import SyntheticExtrinsicLikelihood as V5
    sk = {**_V5_DEFAULTS, **(sp_kwargs or {})}
    sk.pop("phase_basis", None)
    return V5(ifos, wfg, priors,
              marg_phase=flags["phase"], marg_psi=flags["psi"],
              marg_time=flags["geocent_time"],
              marg_dist=flags["luminosity_distance"],
              tc0=float(tc_gps), window_fn=window_fn, t_window=t_window,
              **sk)





# --------------------- weight utilities ---------------------
# just does the rejection sampling

def rejection_sample(log_w, rng):
    """Accept i with probability w_i / max(w). Returns positions of accepted
    entries. Uniqueness is a property of the input, not of this function."""
    lw = np.asarray(log_w, dtype=np.float64)
    ok = np.isfinite(lw)
    accept = np.zeros(lw.shape, dtype=bool)
    if ok.any():
        accept[ok] = np.log(rng.uniform(size=ok.sum())) < (lw[ok] - lw[ok].max())
    return np.flatnonzero(accept)





def thin_indices(log_w, n_target, rng):
    """Bernoulli thinning: keep i w.p. min(1, w_i/c), c set so E[#kept]=n_target.
    Returns (idx, log_w_kept, log_p) with log_w_kept = log w_i - log p_i
    (Horvitz-Thompson). Indices unique by construction; estimator unbiased."""
    lw = np.asarray(log_w, dtype=np.float64)
    ok = np.isfinite(lw)
    if not ok.any():
        return np.array([], dtype=int), np.array([]), np.array([])
    lwn = np.full(lw.shape, -np.inf)
    lwn[ok] = lw[ok] - logsumexp(lw[ok])
    lo, hi = lwn[ok].min() - 50.0, lwn[ok].max()
    for _ in range(80):                                   # bisect on log c
        mid = 0.5 * (lo + hi)
        n = np.exp(np.minimum(0.0, lwn[ok] - mid)).sum()
        lo, hi = (mid, hi) if n > n_target else (lo, mid)
    log_c = 0.5 * (lo + hi)
    log_p = np.minimum(0.0, lwn - log_c)
    keep = np.zeros(lw.shape, dtype=bool)
    keep[ok] = np.log(rng.uniform(size=ok.sum())) < log_p[ok]
    idx = np.flatnonzero(keep)
    return idx, lwn[idx] - log_p[idx], log_p[idx]




def kish_eff(log_w):
    """Kish n_eff / n of finite log-weights, in percent."""
    lw = np.asarray(log_w, dtype=np.float64)
    lw = lw[np.isfinite(lw)]
    if lw.size == 0:
        return 0.0
    w = np.exp(lw - lw.max())
    return 100.0 * w.sum() ** 2 / (np.sum(w ** 2) * len(w))





def _psis_smooth(log_w, tag, verbose):
    """PSIS-smooth log_w when k-hat < 0.7 (needs >25 finite weights and
    arviz). Returns (log_w_for_rejection, khat) — the input array unchanged
    when smoothing does not apply."""
    fin = np.isfinite(log_w)
    if fin.sum() <= 25:
        return log_w, None
    try:
        from arviz import psislw
    except ImportError:
        return log_w, None
    lw_s, k = psislw(log_w[fin])
    khat = float(k)
    if khat < 0.7:
        out = np.full_like(log_w, -np.inf)
        out[fin] = np.asarray(lw_s)
        if verbose:
            print(f"  [{tag}] PSIS-smoothed weights (k-hat={khat:.2f})")
        return out, khat
    if verbose:
        print(f"  [{tag}] k-hat={khat:.2f} >= 0.7, using raw weights")
    return log_w, khat


###########################################################################################
###########################################################################################
# ------------------------------- single-stage reference ----------------------------------
###########################################################################################
###########################################################################################


def single_stage_hm(log_prior, log_draw_prob, sample_dicts,
                    ifos, wfg, priors, flags, tc_gps,
                    sp_kwargs=None, window_fn=None, npool=1,
                    seed=0, verbose=True, log_l22_v6=None):
    """Diagnostic: HM v5 marginal on ALL samples, full t_c window, direct
    weights, single PSIS+rejection. No v6, no window narrowing. Its Kish is
    the reference the two-stage theoretical efficiency should reproduce."""
    rng = np.random.default_rng(seed)
    N = len(log_prior)

    v5_hm = _build_v5(ifos, wfg, priors, flags, tc_gps, sp_kwargs,
                      window_fn, t_window=None)

    if verbose:
        print(f"  [1-stage HM] evaluating v5 HM on all {N} samples (full t_c window)")
    t0 = time.perf_counter()
    log_l_hm, _ = _pool_eval_pairs(v5_hm, sample_dicts, np.arange(N), npool,
                                   pair=False)
    t_eval = time.perf_counter() - t0

    log_w = log_l_hm + np.asarray(log_prior) - np.asarray(log_draw_prob)
    kish = kish_eff(log_w)
    log_w_rej, khat = _psis_smooth(log_w, "1-stage HM", verbose)

    _print_tail_diagnostics(log_w, log_w_rej, khat, log_l_hm, log_l22_v6)

    idx_final = rejection_sample(log_w_rej, rng)
    eff_total = 100.0 * len(idx_final) / N
    if verbose:
        print(f"  [1-stage HM] Kish={kish:.2f}%  accepted {len(idx_final)}/{N} "
              f"({eff_total:.2f}%) in {t_eval:.0f}s")

    return dict(idx_final=idx_final,
                idx_stage1=np.arange(N),           # kish2*len/100 = Kish ESS
                kish2=kish,                        # real Kish%, not 100
                log_w_final=log_w_rej,             # WEIGHTED (full length N)
                log_w1=log_w, log_w2=np.zeros(len(idx_final)),
                log_l_hm=log_l_hm, log_l_22_v5=None,
                kish1=kish, khat2=khat,
                eff1=eff_total, eff2=100.0, eff_total=eff_total,
                n_unique_final=len(idx_final),
                t_stage2=t_eval, t_window=None)




def _print_tail_diagnostics(log_w, log_w_rej, khat, log_l_hm, log_l22_v6):
    """Single-stage tail printouts: top weights, the three n_eff notions, and
    the v5-vs-v6 dlogL residual when the caller supplies log_l22_v6.
    
    As in the name, figures out how the weights are distributed and is a quick
    diagnostic check. Preferably you would just look at the weight distribution
    if you wanted something more robust.
    
    """

    fin = np.isfinite(log_w)
    lw = log_w[fin]
    med = np.median(lw)
    lws = np.sort(lw)[::-1]
    print(f"  [1-stage HM] top-10 log_w - median: "
          + " ".join(f"{v - med:+.1f}" for v in lws[:10]))
    n_eff_kish = np.exp(2 * logsumexp(lw) - logsumexp(2 * lw))
    n_eff_rej = np.exp(logsumexp(lw) - lw.max())          # = expected #accepted
    print(f"  [1-stage HM] n_eff: Kish={n_eff_kish:.0f}  "
          f"rejection(raw)={n_eff_rej:.0f}", end="")
    if khat is not None and khat < 0.7:
        lw2 = log_w_rej[np.isfinite(log_w_rej)]
        print(f"  rejection(PSIS)={np.exp(logsumexp(lw2) - lw2.max()):.0f}")
    else:
        print()
    if log_l22_v6 is not None:
        d = log_l_hm - np.asarray(log_l22_v6)
        dfin = d[np.isfinite(d)]
        print(f"  [1-stage HM] dlogL(HM_v5 - 22_v6): med={np.median(dfin):+.2f} "
              f"std={dfin.std():.2f} max|.-med|={np.abs(dfin - np.median(dfin)).max():.1f}")
        top = np.argsort(log_w)[::-1][:10]
        print(f"  [1-stage HM] top-10 weights' dlogL: "
              + " ".join(f"{d[j] - np.median(dfin):+.1f}" for j in top))







###########################################################################
###########################################################################
# --------------           two-stage pipeline      ------------------------
###########################################################################
###########################################################################

# Currently not in use because of the injected noise that comes with
    # reweighting twice. Even with the fun Horvitz-Thompson

def two_stage_reweight(log_l22, log_prior, log_draw_prob, sample_dicts,
                       sp_like_v6, ifos, wfg, priors, flags, tc_gps,
                       sp_kwargs=None, window_fn=None, npool=1,
                       t_delta=15.0, seed=0, verbose=True):
    """Full two-phase pipeline. Stage-1 logL (v6) is supplied by the caller.

    Returns a dict with idx_stage1, idx_final, log_w1, log_w2, log_w_comb,
    log_w_final, log_l_hm, log_l_22_v5, eff1/eff2/eff_total, kish1/kish2,
    khat2, t_stage2, t_window, n_unique_final.
    """
    rng = np.random.default_rng(seed)
    N = len(log_l22)

    # ── Stage 1: weights + Bernoulli thinning (unique survivors) ──
    log_w1 = np.asarray(log_l22) + np.asarray(log_prior) - np.asarray(log_draw_prob)
    kish1 = kish_eff(log_w1)
    n_target = max(int(kish1 * N / 100.0), 50)
    idx1, lw1_kept, log_p1 = thin_indices(log_w1, n_target, rng)

    out = dict(log_w1=log_w1, idx_stage1=idx1,
               counts_stage1=np.ones(idx1.size, dtype=int),
               log_w1_kept=lw1_kept,
               kish1=kish1, eff1=100.0 * idx1.size / N)
    if verbose:
        print(f"  [2-stage] stage 1: thinned to {idx1.size} unique "
              f"(target {n_target}; {out['eff1']:.1f}% of N; Kish {kish1:.1f}%)")
    if idx1.size == 0:
        out.update(idx_final=idx1, log_w2=np.array([]),
                   log_l_hm=np.array([]), log_l_22_v5=np.array([]),
                   eff2=0.0, eff_total=0.0, kish2=0.0, t_stage2=0.0,
                   t_window=None, n_unique_final=0)
        return out

    # ── Stage 2: v5 HM numerator + v5 (2,2)-collapse denominator ──
    v5_hm = _build_v5(ifos, wfg, priors, flags, tc_gps, sp_kwargs,
                      window_fn, t_window=None)
    if verbose:
        print(f"  [2-stage] stage 2: {idx1.size} unique survivors, ...")

    t0 = time.perf_counter()
    log_l_hm, log_l_22_v5 = _pool_eval_pairs(v5_hm, sample_dicts, idx1, npool)
    t_stage2 = time.perf_counter() - t0

    log_w2 = log_l_hm - log_l_22_v5

    # ── direct v5 HM weight with Horvitz-Thompson thinning correction ──
    # log_w = logL_HM_v5 + log pi - log q - log p_thin.
    # The v5 (2,2) terms cancel identically; the v6 (2,2) enters only through
    # the thinning probabilities, so the v6-v5 residual never touches the
    # weight. For above-threshold samples this IS the single-stage weight.
    log_w_comb = (log_l_hm + np.asarray(log_prior)[idx1]
                  - np.asarray(log_draw_prob)[idx1] - log_p1)

    log_w_final, khat2 = _psis_smooth(log_w_comb, "2-stage", verbose)
    acc = rejection_sample(log_w_final, rng)
    idx_final = idx1[acc]

    kish2 = kish_eff(log_w_comb)
    out.update(idx_final=idx_final, log_w2=log_w2,
               log_w_comb=log_w_comb, log_w_final=log_w_final,
               log_l_hm=log_l_hm, log_l_22_v5=log_l_22_v5,
               khat2=khat2, kish2=kish2, n_unique_final=idx_final.size,
               eff2=100.0 * idx_final.size / max(idx1.size, 1),
               eff_total=100.0 * idx_final.size / N,
               t_stage2=t_stage2, t_window=None)

    if verbose:
        d = log_w2[np.isfinite(log_w2)]
        msg = (f"  [2-stage] stage 2: {idx_final.size}/{idx1.size} accepted "
               f"({out['eff2']:.1f}%; Kish {kish2:.1f}%) in {t_stage2:.1f}s")
        if d.size:
            msg += (f" | dlogL mean={d.mean():+.3f} "
                    f"max|.|={np.abs(d - d.mean()).max():.3f}")
        print(msg)
        print(f"  [2-stage] final: {idx_final.size}/{N} equal-weight HM samples "
              f"({out['eff_total']:.2f}%), all distinct")
    return out

