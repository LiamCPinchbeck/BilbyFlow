"""
bilbyflow.plotting.summary — efficiency summary figures from the per-event
{event}_data.pkl files written by the reweighting scripts.

Events are ordered by efficiency (most efficient left). Three efficiency
layers: full (PSIS/Kish ESS of the full IS weights), prior (ESS of prior-only
weights, recomputed), total (two-stage equal-weight yield). The full layer
draws PSIS/Kish and empirical two-stage together.

The PSIS reliability report lives in diagnostics.psis; the CLI wrapper that
calls all of these lives in scripts/plot_summary.py.

FONTS: this module does NOT touch text.usetex. The caller decides, via
rcParams or a matplotlibrc. If the local TeX toolchain is broken, either set
text.usetex=False in the entry script, or export with
`plot_summary --csv-only` and render elsewhere.
"""

import glob
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt

__all__ = [
    "ess_pct", 
    "load_from_pkls", "load_from_txt", 

    "efficiency_stats",
    "plot_sorted", "plot_theoretical_only", "plot_two_stage",

    "load_sharpness", "plot_sharpness", 

    "load_two_stage",
    "LAYER_FUNCS", "LAYER_LABEL",
]


def ess_pct(logw):
    """n_eff/n (Kish ESS) from log-weights, in %."""

    logw = np.asarray(logw, dtype=float).ravel()
    logw = logw[np.isfinite(logw)]

    if logw.size == 0:
        return 0.0

    logw = logw - logw.max()
    w = np.exp(logw)

    return 100.0 * (w.sum() ** 2) / (np.sum(w ** 2) * len(w))






####################################################################################
####################################################################################
# ---------------------------- layer efficiency extractors ----------------------------
####################################################################################
####################################################################################


PRIOR_LOGW_KEYS = ("log_prior_weights", "logw_prior", "prior_log_weights")
LOG_PRIOR_KEYS = ("log_prior", "logpi", "log_pi")
LOG_Q_KEYS = ("log_draw_prob", "log_q", "logq", "log_proposal")


def _find(cached, keys):
    r = cached.get("result", {})
    for k in keys:
        for src in (r, cached):
            if isinstance(src, dict) and k in src:
                return np.asarray(src[k], dtype=float)
    return None


def prior_layer_efficiency(cached):
    """ESS% of the prior-only weights w = pi(theta)/q(theta|d).

    Looks at efficiency of the first (internal) stage of reweighting.
    """

    logw = _find(cached, PRIOR_LOGW_KEYS)
    if logw is None:
        lp, lq = _find(cached, LOG_PRIOR_KEYS), _find(cached, LOG_Q_KEYS)
        if lp is not None and lq is not None:
            logw = lp - lq

    if logw is None:
        r = cached.get("result", {})
        raise KeyError("no prior log-weight components found; result keys: "
                       f"{sorted(r.keys()) if isinstance(r, dict) else '?'}")

    return ess_pct(logw)


def full_layer_efficiency(cached):
    """PSIS/Kish effective ESS (smoothed if k-hat < 0.7)."""

    return float(cached["result"]["efficiency"])


def total_layer_efficiency(cached):
    """Two-stage equal-weight efficiency (eff_total)."""

    r = cached.get("result", {})

    return float(r["eff_total"]) if "eff_total" in r else full_layer_efficiency(cached)



LAYER_FUNCS = {
    "full": full_layer_efficiency,
    "prior": prior_layer_efficiency,
    "total": total_layer_efficiency}

LAYER_LABEL = {
    "full": "Reweighting efficiency (%)",
    "prior": "Prior-layer efficiency (%)",
    "total": "Two-stage efficiency (%)"}


# title text without the trailing " (%)" stuff
LAYER_TITLE = {k: v.rsplit(" (", 1)[0] for k, v in LAYER_LABEL.items()}







####################################################################################
####################################################################################
# ----------------------------    loaders    ----------------------------
####################################################################################
####################################################################################

def load_from_pkls(out_dir, layer):
    """[(event, primary_eff, empirical|None, psis|None, is_two_stage,
         t_stage2|None, t_reweight_total|None)].

    The two timings are appended, so callers that only index r[0..4] are
    unaffected."""


    eff_fn = LAYER_FUNCS[layer]

    rows, vq_h, vq_l = [], [], []
    for pkl in sorted(glob.glob(os.path.join(out_dir, "*_data.pkl"))):
        try:
            with open(pkl, "rb") as f:
                cached = pickle.load(f)

            r = cached["result"]

            
            name = r.get("event", os.path.basename(pkl).replace("_data.pkl", ""))
            vq = r.get("noise_var_q") or {}
            if vq:
                vq_h.append(float(vq.get("H1", np.nan)))
                vq_l.append(float(vq.get("L1", np.nan)))

            
            primary = eff_fn(cached)

            empirical = float(r["eff_total"]) if "eff_total" in r else None
            psis_eff = float(r["efficiency"]) if "efficiency" in r else None
        
            is_two = "eff_stage2" in r and r.get("eff_stage2") != 100.0
        
            t_stage2 = (float(r["t_stage2"])
                        if r.get("t_stage2") is not None else None)
            t_total = (float(r["t_reweight_total"])
                       if r.get("t_reweight_total") is not None else None)
        
            rows.append((str(name), primary, empirical, psis_eff, is_two,
                         t_stage2, t_total))
        
        except Exception as e:
            print(f"  skip {os.path.basename(pkl)}: {e}")

    if vq_h or vq_l:
        # pooled over both detectors
        pooled = np.concatenate([np.asarray(vq_h, dtype=float),
                                 np.asarray(vq_l, dtype=float)])
    
        mh, ml, mp = np.nanmean(vq_h), np.nanmean(vq_l), np.nanmean(pooled)
    
        slope = np.sqrt(0.939 / mp) if np.isfinite(mp) and mp > 0 else np.nan
    
        print(f"  var_q: H1 mean={mh:.3f}, L1 mean={ml:.3f} "
              f"(baseline ~0.94; implied sigma-slope ~ {slope:.2f})")
    return rows


def load_from_txt(filename):
    """Fallback: parse event + efficiency from a summary.txt table."""

    rows = []
    with open(filename) as f:
        lines = f.readlines()

    
    header = next((i for i, l in enumerate(lines)
                   if "Event" in l and "Eff" in l), None)

    if header is None:
        header = next((i for i, l in enumerate(lines) if "PSIS%" in l), None)
    if header is None:
        raise ValueError("Could not find a table header in summary.txt")


    for line in lines[header + 1:]:

        line = line.strip()

        if not line or line.startswith(("Mean", "Model", "-", "PSIS", "Empirical",
                                        "Marginalized", "Noise", "Template",
                                        "Samples")):
            continue
    
        parts = line.split()
        if len(parts) >= 3:

            try:
                rows.append((parts[0], float(parts[1]), float(parts[2]),
                             float(parts[1]), False, None, None))
                continue
            except ValueError:
                pass
    
        if len(parts) >= 2:
            try:
                rows.append((parts[0], float(parts[1]), None, None,
                             False, None, None))
            except ValueError:
                pass
    return rows


def load_sharpness(out_dir):
    """[(event, efficiency, sigma_ln_Mc, total_sharpness)]."""


    rows = []
    for pkl in sorted(glob.glob(os.path.join(out_dir, "*_data.pkl"))):
    
        try:
            with open(pkl, "rb") as f:
                c = pickle.load(f)

            s, names, r = c["npe_samples"], c["param_names"], c["result"]
            s = np.asarray(s, dtype=float)

            mi = list(names).index("chirp_mass")

            mc = s[:, mi]
            mc = mc[np.isfinite(mc) & (mc > 0)]
            if mc.size < 2:
                continue

            frac_sig_Mc = float(np.std(np.log(mc)))

            sharp = -float(np.sum([np.log(np.std(s[:, j]) + 1e-30)
                                   for j in range(s.shape[1])]))
            
            rows.append((str(r.get("event", "?")), float(r["efficiency"]),
                         frac_sig_Mc, sharp))
            
        except Exception as e:
            print(f"  sharpness skip {os.path.basename(pkl)}: {e}")
        
    return rows


def load_two_stage(out_dir):
    """[(event, eff_stage1, eff_stage2, eff_total, t_stage2, t_total)]."""

    rows = []
    for pkl in sorted(glob.glob(os.path.join(out_dir, "*_data.pkl"))):
    
        try:
            with open(pkl, "rb") as f:
                r = pickle.load(f)["result"]
            if "eff_stage1" not in r:
                continue

            rows.append((str(r.get("event", "?")), float(r["eff_stage1"]),
                         float(r["eff_stage2"]), float(r["eff_total"]),
                         float(r.get("t_stage2", np.nan)),
                         float(r.get("t_reweight_total", np.nan))))
        except Exception as e:
            print(f"  two-stage skip {os.path.basename(pkl)}: {e}")

    return rows



####################################################################################
####################################################################################
# ----------------------------         stats        ----------------------------
####################################################################################
####################################################################################

def efficiency_stats(effs):
    """Robust to empty input and to non-finite entries."""

    effs = np.asarray(effs, dtype=float)
    effs = effs[np.isfinite(effs)]

    if effs.size == 0:
        return dict(n=0, mean=np.nan, median=np.nan, min=np.nan, max=np.nan,
                    pct_below_1=np.nan, pct_below_05=np.nan)


    return dict(
        n=int(effs.size), 
        mean=float(np.mean(effs)), median=float(np.median(effs)), 
        min=float(np.min(effs)), max=float(np.max(effs)),
        pct_below_1=100.0 * float(np.mean(effs < 1.0)), # mostly use this one
        pct_below_05=100.0 * float(np.mean(effs < 0.5)))


# Used to use, but was rarely useful, as you'd either look at ~40 events with known names
    # or ~300 in which case you don't have names and don't wanna see them
# def _fig_width(n, per_event=0.25, lo=10.0, hi=40.0):
#     """Width grows with event count but stays inside the renderer's limits.
#     0.25 in/event x 800 events x 150 dpi = 30000 px, past Agg's practical
#     ceiling -- clamp it."""

#     return float(np.clip(per_event * max(n, 1), lo, hi))


_RANK_SWITCH = 50   # above this many events: fractional-rank axis, no names

def _event_axis(ax, n, events,
                xlabel="Event (sorted by efficiency: most \u2192 least)"):
    """Bar positions + width for a sorted-by-efficiency axis. Up to
    _RANK_SWITCH events: named ticks at integer positions. Above: bars at
    fractional rank in [0, 1] (name labels are unreadable there anyway, and
    the axis becomes run-size independent)."""

    if n <= _RANK_SWITCH:
        x = np.arange(n)
        ax.set_xticks(x)
        ax.set_xticklabels(events, rotation=90, fontsize=8)
        ax.set_xlim(-0.6, n - 0.4)
        ax.set_xlabel(xlabel, size=12)
        return x, 1.0

    x = (np.arange(n) + 0.5) / n

    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Fractional rank (most --> least efficient)", size=12)

    return x, 1.0 / n


def _save(fig, stem, quiet=False):
    """Write PNG + PDF from a stem, with or without an extension."""

    stem = os.path.splitext(stem)[0]

    for ext in (".png", ".pdf"):
        fig.savefig(stem + ext, dpi=150, bbox_inches="tight")
    plt.close(fig)

    if not quiet:
        print(f"Saved {stem}.png/.pdf")
    return stem

####################################################################################
####################################################################################
# ----------------------------      plots      ----------------------------
####################################################################################
####################################################################################


def plot_sorted(rows, outfile, layer, log_y=False, notitles=False):
    """Efficiency by event, most efficient first. On the full layer the Kish
    ESS and the empirical two-stage yield are overlaid."""

    rows = [r for r in rows if r[1] is not None and np.isfinite(r[1])]
    if not rows:
        print("  plot_sorted: no finite efficiencies, skipped")
        return

    stem = os.path.splitext(outfile)[0] + f".{int(notitles)}"

    rows = sorted(rows, key=lambda r: r[1], reverse=True)
    events = [r[0] for r in rows]
    effs_primary = np.array([r[1] for r in rows], dtype=float)
    has_empirical = any(len(r) > 2 and r[2] is not None for r in rows)
    has_psis = any(len(r) > 3 and r[3] is not None for r in rows)
    show_both = has_empirical and has_psis and layer == "full"
    mode = ("two-stage" if any(len(r) > 4 and r[4] for r in rows)
            else "single-stage")

    s = efficiency_stats(effs_primary)

    fig, ax = plt.subplots(figsize=(12, 5.5))

    x, bw = _event_axis(ax, len(events), events)

    s_emp = None
    if show_both:
        effs_psis = np.array([r[3] if r[3] is not None else r[1] for r in rows],
                             dtype=float)
        effs_emp = np.array([r[2] if r[2] is not None else np.nan for r in rows],
                            dtype=float)
        s_emp = efficiency_stats(effs_emp)


        ax.bar(x, effs_psis, width=bw, alpha=0.45, color="tab:blue",
               label=f"Kish ESS (mean {s['mean']:.2f}%)")
        ax.bar(x, np.nan_to_num(effs_emp), width=bw, alpha=0.7,
               color="tab:green",
               label=f"Rejection yield, {mode} (mean {s_emp['mean']:.2f}%)")

    
        if not notitles:
            ax.axhline(s["mean"], color="tab:blue", ls="--", lw=1.0, alpha=0.5)
            ax.axhline(s_emp["mean"], color="tab:green", ls="--", lw=1.0,
                       alpha=0.5)
    else:

        ax.bar(x, effs_primary, width=bw, alpha=0.85, color="tab:blue")

    
        if not notitles:
            ax.axhline(s["mean"], color="red", ls="--", lw=1.3,
                       label=f"Mean: {s['mean']:.2f}%")
            ax.axhline(s["median"], color="purple", ls="--", lw=1.3,
                       label=f"Median: {s['median']:.2f}%")
            ax.axhline(0.5, color="gray", ls=":", lw=1.2,
                       label="0.5% efficiency")
        
    ax.axhline(1.0, color="tab:orange", ls="dashed", lw=2.5,
               label="1% efficiency")


    ax.set_ylabel(LAYER_LABEL[layer], size=12)

    if not notitles:
        scale = "log scale" if log_y else "linear scale"
        title = f"{LAYER_TITLE[layer]} by event ({scale})\n"
        if show_both:
            title += (f"PSIS mean {s['mean']:.2f}% | median {s['median']:.2f}%\n"
                      f"empirical mean {s_emp['mean']:.2f}% | "
                      f"median {s_emp['median']:.2f}%\n"
                      f"<1%: {s['pct_below_1']:.2f}% | "
                      f"{s_emp['pct_below_1']:.2f}% (N={s['n']})")
        else:
            title += (f"mean {s['mean']:.2f}% | median {s['median']:.2f}% | "
                      f"<1%: {s['pct_below_1']:.1f}% | "
                      f"<0.5%: {s['pct_below_05']:.1f}% (N={s['n']})")
        ax.set_title(title, fontsize=11)

    if log_y:
        ax.set_yscale("log")
        nonpos = int(np.sum(effs_primary <= 0))
        if nonpos:
            print(f"  note: {nonpos} event(s) have efficiency <= 0; "
                  f"omitted from log bars")
    
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, stem)

    msg = (f"  mean={s['mean']:.2f}%  median={s['median']:.2f}%  "
           f"min={s['min']:.4f}%")
    if s_emp is not None:
        msg += (f"  | empirical mean={s_emp['mean']:.2f}%  "
                f"median={s_emp['median']:.2f}%  min={s_emp['min']:.4f}%")
    msg += (f"  <1%={s['pct_below_1']:.1f}%  <0.5%={s['pct_below_05']:.1f}%  "
            f"N={s['n']}")

    print(msg)



def plot_theoretical_only(rows, outfile, log_y=True, notitles=False):
    """PSIS/Kish efficiency only, sorted, no empirical bars."""

    rows = [r for r in rows
            if len(r) > 3 and r[3] is not None and np.isfinite(r[3])]


    if not rows:
        print("  plot_theoretical_only: no PSIS efficiencies, skipped")
        return


    rows = sorted(rows, key=lambda r: r[3], reverse=True)
    events = [r[0] for r in rows]
    effs = np.array([r[3] for r in rows], dtype=float)


    s = efficiency_stats(effs)

    fig, ax = plt.subplots(figsize=(12, 5.5))

    x, bw = _event_axis(ax, len(events), events)
    ax.bar(x, effs, width=bw, alpha=0.85, color="tab:blue")
    ax.axhline(1.0, color="tab:orange", ls="dashed", lw=2.5,
               label="1% efficiency")
    if not notitles:
        ax.axhline(s["mean"], color="red", ls="--", lw=1.3,
                   label=f"Mean: {s['mean']:.2f}%")
        ax.axhline(s["median"], color="purple", ls="--", lw=1.3,
                   label=f"Median: {s['median']:.2f}%")
        ax.set_title("Theoretical (PSIS/Kish) efficiency by event\n"
                     f"mean {s['mean']:.2f}% | median {s['median']:.2f}% | "
                     f"<1%: {s['pct_below_1']:.1f}% (N={s['n']})", fontsize=11)

    
    ax.set_ylabel("PSIS/Kish reweighting efficiency (%)", size=12)
    if log_y:
        ax.set_yscale("log")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, outfile)

# informative when the flow is over-covering/too-uncertain/under-confident
    # and how that relates to SNR in particular
def plot_sharpness(rows, outfile):
    """Efficiency against two measures of posterior sharpness."""
    from scipy.stats import spearmanr # lazy import because I'm lazy

    rows = [r for r in rows if np.isfinite(r[1]) and r[1] > 0]
    if len(rows) < 5:
        print("  plot_sharpness: fewer than 5 usable events, skipped")
        return
    
    _, eff, fsm, sh = zip(*rows)
    eff = np.asarray(eff, dtype=float)
    fsm = np.asarray(fsm, dtype=float)
    sh = np.asarray(sh, dtype=float)


    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for ax, xv, lab, logx in [
            (axes[0], fsm, r"$\sigma(\ln M_c)$ (raw flow)", True),
            (axes[1], sh, r"total sharpness $-\sum_j \ln\sigma_j$", False)]:
        ok = np.isfinite(xv) & np.isfinite(eff) & (eff > 0)
        if logx:
            ok &= xv > 0
        ax.scatter(xv[ok], eff[ok], s=25, alpha=0.8)
        if logx:
            ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(lab)
        ax.set_ylabel("Efficiency (%)")
        if spearmanr is not None and ok.sum() >= 5:
            rho, p = spearmanr(xv[ok], eff[ok])
            ax.set_title(rf"Spearman $\rho$={rho:+.2f}, p={p:.1e}")
        ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    _save(fig, outfile)


# Not used very often now, should consider deprecating
def plot_two_stage(out_dir, outfile, notitles=False):
    """Stage-1 vs final efficiency, plus stage-2 acceptance."""

    rows = load_two_stage(out_dir)

    if len(rows) < 2:
        print("  plot_two_stage: fewer than 2 two-stage events, skipped")
        return

    rows.sort(key=lambda r: r[3], reverse=True)
    events = [r[0] for r in rows]

    e1 = np.array([r[1] for r in rows], dtype=float)
    e2 = np.array([r[2] for r in rows], dtype=float)
    et = np.array([r[3] for r in rows], dtype=float)
    t2 = np.array([r[4] for r in rows], dtype=float)

    fig, axes = plt.subplots(
        1, 2, figsize=(12, 5.5),
        gridspec_kw={"width_ratios": [2, 1]})

    ax = axes[0]

    x, bw = _event_axis(ax, len(events), events, xlabel="Event")
    
    ax.bar(x, et, width=bw, alpha=0.85, color="tab:green",
           label="Final (equal-weight HM)")
    ax.bar(x, e1, width=bw, alpha=0.4, color="tab:blue", label="Stage 1 (2,2)")

    ax.axhline(np.nanmean(et), color="red", ls="--", lw=1.2,
               label=f"Mean total: {np.nanmean(et):.2f}%")


    ax.set_ylabel("Efficiency (%)")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    if not notitles:
        ax.set_title(f"Two-stage efficiency (N={len(events)})")



    ax2 = axes[1]
    _event_axis(ax2, len(events), events, xlabel="Event")
    ax2.bar(x, e2, width=bw, alpha=0.85, color="tab:orange")
    ax2.axhline(np.nanmean(e2), color="red", ls="--", lw=1.2,
                label=f"Mean: {np.nanmean(e2):.1f}%")

    ax2.set_ylabel("Stage-2 acceptance (%)")
    ax2.legend(fontsize=8)
    ax2.grid(True, axis="y", alpha=0.3)
    if not notitles:
        ax2.set_title("HM correction cost")

    fig.tight_layout()
    stem = _save(fig, outfile, quiet=True)


    print(f"Saved {stem}.png/.pdf ({len(rows)} events)")
    print(f"  stage2 eff: mean={np.nanmean(e2):.1f}%, min={np.nanmin(e2):.1f}%")
    if np.isfinite(t2).any():
        print(f"  stage2 time: mean={np.nanmean(t2):.1f}s, "
              f"total={np.nansum(t2):.0f}s")