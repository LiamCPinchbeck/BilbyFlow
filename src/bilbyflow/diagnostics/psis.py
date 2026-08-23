"""
bilbyflow.diagnostics.psis — Pareto-smoothed IS tail diagnostic.

Essentially, if the efficiency is low it doesn't necessarily mean that the
reweighting is unreliable. If the khat is high it implies that the variance
in the weights is so large that you can't draw good samples. Generally only
been a thing in low samples when it's unreliable in the first place (^u^). 
(more on that below)

psis_khat is the single k-hat estimator imported by both the reweighting
headline stats and the summary plotting. Prefers arviz.psislw; falls back to
a direct Generalised-Pareto fit on the tail excesses (_tail_khat — also the
estimator the reliability report's jitter refits use, so the "stability"
panel measures the same fit it questions).

Note on the reliability metrics (from O3 real-/on-source-event runs): 
k-hat is estimated from the upper tail, 
so when only a few tens of samples carry weight the fit is noisy
and can even return k-hat < 0 with a heavy empirical tail. 
Gate any "reliable/unreliable" flag on n_eff, not on k-hat alone.
"""

import glob
import os
import pickle
import numpy as np
from scipy.stats import genpareto

__all__ = ["psis_khat", "psis_reliability_report"]


def _tail_khat(lw):
    """Direct GenPD shape fit on the upper-tail excesses of finite log-weights
    (M = min(0.2 S, 3 sqrt(S)) tail samples, Vehtari et al. arXiv:1507.02646 or arXiv:1507.04544 choice).
    Returns inf when the tail is too small or the fit fails."""
    S = lw.size
    M = int(min(0.2 * S, 3.0 * np.sqrt(S)))
    ws = np.sort(np.exp(lw - lw.max()))
    excess = ws[-M:] - ws[-M - 1]
    excess = excess[excess > 0]
    if len(excess) < 5:
        return np.inf
    try:
        return float(genpareto.fit(excess, floc=0)[0])
    except Exception:
        return np.inf


def psis_khat(log_w):
    """PSIS k-hat (Vehtari et al., arXiv:1507.02646)."""
    lw = np.asarray(log_w, dtype=np.float64)
    lw = lw[np.isfinite(lw)]
    if lw.size < 25:
        return np.inf
    try:
        from arviz import psislw
        _, k = psislw(lw)
        return float(k)
    except ImportError:
        return _tail_khat(lw)


# ── reliability report ───────────────────────────────────────────────────────

def _khat_jitter(lw, n_refit=20, frac=0.8, seed=0):
    """Std of _tail_khat over random subsets — the fit-stability proxy."""
    rng = np.random.default_rng(seed)
    ks = []
    for _ in range(n_refit):
        k = _tail_khat(rng.choice(lw, size=int(frac * lw.size), replace=False))
        if np.isfinite(k):
            ks.append(k)
    return float(np.std(ks)) if len(ks) >= 5 else np.nan


def _event_record(pkl):
    """Fills out a dict with the reliability metrics from a pickle result file."""

    with open(pkl, "rb") as f:
        c = pickle.load(f)
    
    r = c["result"]
    lw = np.asarray(r.get("log_weight_weighted", r["log_weight"]), float)
    lw = lw[np.isfinite(lw)]

    if lw.size < 25:
        return None


    w = np.exp(lw - lw.max())
    w /= w.sum()
    ws = np.sort(w)

    n_eff = float(r.get("n_eff", np.nan))
    n_raw = float(r.get("n_eff_raw", n_eff))

    return dict(
        event=str(r.get("event", os.path.basename(pkl)[:16])),
        khat=float(r.get("khat", np.nan)), khat_jitter=_khat_jitter(lw),
        n=lw.size, n_eff=n_eff, n_eff_raw=n_raw,
        inflation=(n_eff / n_raw if n_raw > 0 else np.nan),
        top1=float(ws[-1]), top10=float(ws[-10:].sum()),
        M_tail=int(min(0.2 * lw.size, 3.0 * np.sqrt(lw.size))),
        smoothed=bool(r.get("psis_smoothed", False)))


def _flags(d):
    flags = []

    if d["khat"] >= 0.7:
        flags.append("UNRELIABLE")
    elif d["khat"] >= 0.5:
        flags.append("marginal")

    if d["inflation"] > 2.0:
        flags.append("infl>2x")

    if d["top1"] > 0.2:
        flags.append("top1>20%")

    if d["khat_jitter"] > 0.15:
        flags.append("unstable-fit")

    return ",".join(flags) or "-"


def _print_table(recs):


    hdr = (f"{'event':<20}{'khat':>7}{'+-jit':>7}{'n_eff':>8}{'raw':>8}"
           f"{'infl':>6}{'top1%':>7}{'top10%':>8}{'M':>5}{'flag':>18}")

    print("\n" + hdr)
    print("-" * len(hdr))


    for d in sorted(recs, key=lambda d: -d["khat"]):
        print(f"{d['event']:<20}{d['khat']:>7.2f}{d['khat_jitter']:>7.2f}"
              f"{d['n_eff']:>8.0f}{d['n_eff_raw']:>8.0f}{d['inflation']:>6.1f}"
              f"{100*d['top1']:>7.1f}{100*d['top10']:>8.1f}{d['M_tail']:>5d}"
              f"{_flags(d):>18}")

    kh = np.array([d["khat"] for d in recs])
    print(f"\nkhat: median={np.median(kh):.2f}  "
          f">=0.5: {np.mean(kh >= 0.5)*100:.0f}%  "
          f">=0.7: {np.mean(kh >= 0.7)*100:.0f}%  (N={len(kh)})")


def _reliability_figure(recs, outfile, notitles):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    kh = np.array([d["khat"] for d in recs])
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    ax = axes[0, 0]
    ax.hist(kh, bins=20, alpha=0.8, color="tab:blue")
    for thr, c, lbl in [(0.5, "orange", "0.5 (variance finite-ish)"),
                        (0.7, "red", "0.7 (PSIS unreliable)")]:
        ax.axvline(thr, color=c, ls="--", label=lbl)
    ax.set_xlabel(r"$\hat{k}$")
    ax.set_ylabel("events")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    if not notitles:
        ax.set_title(r"$\hat{k}$ distribution")



    ax = axes[0, 1]
    infl = np.array([d["inflation"] for d in recs])
    ax.scatter(kh, infl, s=25, alpha=0.8)
    ax.axhline(1.0, color="k", lw=0.8)
    ax.axvline(0.7, color="red", ls="--", lw=0.8)
    ax.set_xlabel(r"$\hat{k}$")
    ax.set_ylabel("n_eff / n_eff_raw")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    if not notitles:
        ax.set_title("Smoothing inflation (bias proxy)")



    ax = axes[1, 0]
    t1 = np.array([d["top1"] for d in recs]) * 100
    eff = np.array([100 * d["n_eff_raw"] / d["n"] for d in recs])
    ax.scatter(t1, eff, s=25, alpha=0.8)
    ax.set_xlabel("top-1 weight fraction (%)")
    ax.set_ylabel("raw Kish efficiency (%)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    if not notitles:
        ax.set_title("Weight concentration vs raw efficiency")



    ax = axes[1, 1]
    jit = np.array([d["khat_jitter"] for d in recs])
    ax.scatter(kh, jit, s=25, alpha=0.8)
    ax.axhline(0.15, color="red", ls="--", lw=0.8, label="jitter 0.15")
    ax.set_xlabel(r"$\hat{k}$")
    ax.set_ylabel(r"$\hat{k}$ subsample std")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    if not notitles:
        ax.set_title(r"$\hat{k}$ fit stability (80% subsets)")



    fig.tight_layout()
    for ext in (".png", ".pdf"):
        fig.savefig(outfile + ext, dpi=150, bbox_inches="tight")
    plt.close(fig)


    print(f"Saved {outfile}.png/.pdf")


def psis_reliability_report(out_dir, outfile, notitles=False):
    """Everything to check BEFORE trusting PSIS-smoothed efficiencies:
      - khat distribution vs the 0.5 / 0.7 thresholds
      - smoothing inflation: n_eff vs n_eff_raw
      - weight concentration: top-1 / top-10 weight fraction
      - tail sample count used for the Pareto fit
      - khat stability: refit on random 80% subsets (khat jitter)

    Reads the per-event {event}_data.pkl files in out_dir, prints a table and
    saves a 4-panel figure. matplotlib is imported lazily so importing this
    module stays cheap.

    Reliability caveat: khat is a tail fit. When only a few tens of samples
    carry weight it is noisy (and can go negative with a heavy empirical tail),
    so a reliable/unreliable flag should be gated on n_eff, not khat alone.
    """

    recs = []

    for pkl in sorted(glob.glob(os.path.join(out_dir, "*_data.pkl"))): # finds all the relevant pickles
        try:
            rec = _event_record(pkl) # gets the relevant info from the pickles
            if rec is not None:
                recs.append(rec) # adds the results to main list if not empty
        except Exception as e:
            print(f"  skip {os.path.basename(pkl)}: {e}")
    
    if not recs:
        print("psis_reliability_report: no usable events") # #badbadnotgood
        return
    _print_table(recs)
    _reliability_figure(recs, outfile, notitles)