#!/usr/bin/env python
"""
scripts/plot_summary.py — efficiency plots from the per-event
{event}_data.pkl files written by a reweighting run.

One series   -> the original sorted bar charts (behaviour unchanged).
Many series  -> OVERLAY plots: sorted-efficiency curves, survival (CDF)
                curves, and the theoretical-vs-empirical comparison, all
                populations on shared axes.

A "series" is one (directory, name-pattern) pair. Supply several patterns
with --groups, several directories as extra positional paths, or both (the
series are then the cartesian product).

Usage:
    # one series: original bar charts
    python -m bilbyflow.scripts.plot_summary $OUT --filter 'GW*' --label real

    # overlay three populations from one run
    python -m bilbyflow.scripts.plot_summary $OUT \\
        --groups 'GW*' 'INJ_*' 'INJr_*' \\
        --labels 'On-Source' 'Gauss Injections' 'Off-Segment' \\
        --label allpops --notitles

    # overlay one population across two models
    python -m bilbyflow.scripts.plot_summary $OUT_A $OUT_B \\
        --groups 'INJ_*' --label modelcmp

    # matched parameter box (params from the pkls or --truth-dir)
    python -m bilbyflow.scripts.plot_summary $OUT --groups 'GW*' 'INJ_*' \\
        --select q:0.5:1.0 Mc:25:55 rho:8:20 --truth-dir synth_data
"""

import argparse
import fnmatch
import os

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from ..plotting.summary import (
    load_from_pkls, load_from_txt, load_sharpness,
    plot_sorted, plot_theoretical_only, plot_sharpness, plot_two_stage,
)
from ..diagnostics.psis import psis_reliability_report

# Enforce after all bilbyflow imports
matplotlib.rcParams['text.usetex'] = False


_COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
           "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]


# ------              row helpers.           ------------------------------------------------
# load_from_pkls row layout: (name, primary_eff, empirical, psis, two_stage?)

def _col(rows, i):
    v = np.array([r[i] if (len(r) > i and r[i] is not None) else np.nan
                  for r in rows], dtype=np.float64)
    return v[np.isfinite(v)]


def _stats_line(rows):
    e = _col(rows, 1)
    if len(e) == 0:
        return "no data"
    out = (f"mean {e.mean():.2f}%  med {np.median(e):.2f}%  "
           f"min {e.min():.3f}%  N={len(e)}")
    tt = _col(rows, 6)
    t2 = _col(rows, 5)
    if len(tt):
        out += f"  |  median reweight {np.median(tt):.1f}s"
    if len(t2):
        out += f"  median stage2 {np.median(t2):.1f}s"
    return out


def _save(fig, outfile):
    for ext in ("png", "pdf"):
        fig.savefig(f"{outfile}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {outfile}.png/.pdf")


# ---------       overlay plots      -----------------------------------------------------------

def plot_sorted_overlay(series, outfile, log_y=False, notitles=False,
                        col=1, ylabel="Kish ESS %"):
    """Sorted efficiency vs NORMALISED rank — the single-run bar chart drawn
    as a curve, so runs with different event counts share one axis."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    drawn = 0
    for rows, lbl, c in series:
        e = _col(rows, col)
        if len(e) < 2:
            continue
        e = np.sort(e)[::-1]
        rank = (np.arange(1, len(e) + 1) - 0.5) / len(e)
        ax.plot(rank, e, color=c, lw=2.2,
                label=f"{lbl} (n={len(e)}, med {np.median(e):.2f}%)")
        ax.axhline(np.median(e), color=c, ls=":", lw=1, alpha=0.45)
        drawn += 1
    if not drawn:
        plt.close(fig)
        return
    ax.axhline(1.0, color="k", ls="--", lw=1.6, alpha=0.7, label="1% efficiency")
    ax.axhline(0.5, color="gray", ls=":", lw=1.2, alpha=0.7, label="0.5%")
    if log_y:
        ax.set_yscale("log")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Normalised rank (most --> least efficient)", size=12)
    ax.set_ylabel(ylabel, size=12)
    if not notitles:
        ax.set_title(f"{ylabel} by rank ({'log' if log_y else 'linear'} scale)",
                     fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, outfile)


def plot_survival_overlay(series, outfile, notitles=False):
    """Fraction of events at or above an efficiency threshold."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for j, (col, title) in enumerate([(1, "Kish ESS"),
                                      (2, "Empirical (two-stage)")]):
        drawn = 0
        for rows, lbl, c in series:
            e = _col(rows, col)
            e = e[e > 0]
            if len(e) < 2:
                continue
            xs = np.sort(e)[::-1]
            ys = np.arange(1, len(xs) + 1) / len(xs)
            axes[j].step(xs, ys, where="post", color=c, lw=2.2,
                         label=f"{lbl} (n={len(xs)})")
            drawn += 1
        axes[j].set_xscale("log")
        axes[j].set_xlabel(f"{title} efficiency %")
        axes[j].set_ylabel("Fraction of events > x")
        axes[j].set_ylim(0, 1.05)
        for thr in (0.5, 1.0, 5.0, 10.0):
            axes[j].axvline(thr, color="gray", ls=":", lw=0.8, alpha=0.45)
        if not notitles:
            axes[j].set_title(f"{title} survival")
        if drawn:
            axes[j].legend(fontsize=9)
        axes[j].grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, outfile)


def plot_theoretical_overlay(series, outfile, notitles=False):
    """Theoretical (Kish) vs empirical yield: sorted curves for both, plus the
    per-event over-statement factor."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    any_ratio = False
    for rows, lbl, c in series:
        k = _col(rows, 3) if len(_col(rows, 3)) else _col(rows, 1)
        e = _col(rows, 2)
        if len(k) >= 2:
            ks = np.sort(k)[::-1]
            axes[0].plot((np.arange(1, len(ks) + 1) - .5) / len(ks), ks,
                         color=c, lw=2.2, label=f"{lbl} Kish (n={len(ks)})")
        if len(e) >= 2:
            es = np.sort(e)[::-1]
            axes[0].plot((np.arange(1, len(es) + 1) - .5) / len(es), es,
                         color=c, lw=1.6, ls="--", label=f"{lbl} empirical")
        kk = np.array([r[3] if (len(r) > 3 and r[3] is not None) else r[1]
                       for r in rows], dtype=np.float64)
        ee = np.array([r[2] if (len(r) > 2 and r[2] is not None) else np.nan
                       for r in rows], dtype=np.float64)
        ok = np.isfinite(kk) & np.isfinite(ee) & (ee > 0)
        if ok.sum() >= 3:
            ratio = kk[ok] / ee[ok]
            axes[1].hist(ratio, bins=np.logspace(0, 2.5, 30), alpha=0.45,
                         color=c, label=f"{lbl} (med {np.median(ratio):.1f}x)")
            any_ratio = True
    axes[0].set_yscale("log")
    axes[0].set_xlim(0, 1)
    axes[0].set_xlabel("Normalised rank")
    axes[0].set_ylabel("Efficiency %")
    axes[0].axhline(1.0, color="k", ls="--", lw=1.4, alpha=0.6)
    if not notitles:
        axes[0].set_title("Theoretical (Kish, solid) vs empirical (dashed)")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xscale("log")
    axes[1].set_xlabel("Kish / empirical (over-statement factor)")
    axes[1].set_ylabel("Events")
    if not notitles:
        axes[1].set_title("How much Kish over-states the deliverable yield")
    if any_ratio:
        axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, outfile)


# -------------------     CSV export      ------------------------------------------------

def _write_csv(path, header, rows):
    import csv
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"Wrote {path}  ({len(rows)} rows)")


def export_csv(series, stem, layer):
    """Three tidy CSVs covering everything the plots draw.

    {stem}_events.csv    one row per event: series, name, efficiencies, rank
    {stem}_curves.csv    the sorted/survival curves, ready to plot directly
    {stem}_summary.csv   one row per series: N, mean/median/min, quantiles,
                         fraction below thresholds, Kish/empirical ratio
    """
    # per-event long form
    ev_rows = []
    for rows, lbl, _c in series:
        eff = np.array([r[1] if r[1] is not None else np.nan for r in rows],
                       dtype=np.float64)
        order = np.argsort(-np.where(np.isfinite(eff), eff, -np.inf))
        rank_of = {int(j): i for i, j in enumerate(order)}
        for i, r in enumerate(rows):
            name = str(r[0])
            primary = r[1] if len(r) > 1 else None
            empirical = r[2] if len(r) > 2 else None
            psis = r[3] if len(r) > 3 else None
            two_stage = bool(r[4]) if len(r) > 4 else False
            t_stage2 = r[5] if len(r) > 5 else None
            t_total = r[6] if len(r) > 6 else None
            ratio = (psis / empirical if (psis is not None and empirical
                                          not in (None, 0)) else "")
            n = len(rows)
            ev_rows.append([lbl, name, layer,
                            "" if primary is None else f"{primary:.6g}",
                            "" if empirical is None else f"{empirical:.6g}",
                            "" if psis is None else f"{psis:.6g}",
                            ratio if ratio == "" else f"{ratio:.6g}",
                            int(two_stage),
                            rank_of[i] + 1,
                            f"{(rank_of[i] + 0.5) / n:.6f}",
                            "" if t_stage2 is None or not np.isfinite(t_stage2)
                            else f"{t_stage2:.6g}",
                            "" if t_total is None or not np.isfinite(t_total)
                            else f"{t_total:.6g}"])
    _write_csv(f"{stem}_events.csv",
               ["series", "event", "layer", "efficiency_primary",
                "efficiency_empirical", "efficiency_psis",
                "kish_over_empirical", "two_stage", "rank",
                "normalised_rank", "t_stage2_s", "t_reweight_total_s"],
               ev_rows)

    # curves: sorted + survival, both columns, per series
    cur_rows = []
    for rows, lbl, _c in series:
        for col, tag in ((1, "primary"), (2, "empirical"), (3, "psis")):
            v = _col(rows, col)
            if len(v) < 2:
                continue
            vs = np.sort(v)[::-1]
            n = len(vs)
            for i, val in enumerate(vs):
                cur_rows.append([lbl, tag, i + 1,
                                 f"{(i + 0.5) / n:.6f}",     # normalised rank
                                 f"{val:.6g}",
                                 f"{(i + 1) / n:.6f}"])      # survival fraction
    _write_csv(f"{stem}_curves.csv",
               ["series", "quantity", "rank", "normalised_rank",
                "efficiency", "fraction_at_or_above"],
               cur_rows)

    # per-series summary 
    sum_rows = []
    for rows, lbl, _c in series:
        e = _col(rows, 1)
        emp = _col(rows, 2)
        psis = _col(rows, 3)
        kk = np.array([r[3] if (len(r) > 3 and r[3] is not None) else r[1]
                       for r in rows], dtype=np.float64)
        ee = np.array([r[2] if (len(r) > 2 and r[2] is not None) else np.nan
                       for r in rows], dtype=np.float64)
        ok = np.isfinite(kk) & np.isfinite(ee) & (ee > 0)
        ratio_med = np.median(kk[ok] / ee[ok]) if ok.sum() else np.nan

        t2 = _col(rows, 5)
        tt = _col(rows, 6)

        def q(v, p):
            return f"{np.percentile(v, p):.6g}" if len(v) else ""
        sum_rows.append([
            lbl, layer, len(rows),
            f"{e.mean():.6g}" if len(e) else "",
            f"{np.median(e):.6g}" if len(e) else "",
            f"{e.min():.6g}" if len(e) else "",
            f"{e.max():.6g}" if len(e) else "",
            q(e, 5), q(e, 25), q(e, 75), q(e, 95),
            f"{np.mean(e < 1.0) * 100:.4g}" if len(e) else "",
            f"{np.mean(e < 0.5) * 100:.4g}" if len(e) else "",
            f"{emp.mean():.6g}" if len(emp) else "",
            f"{np.median(emp):.6g}" if len(emp) else "",
            f"{psis.mean():.6g}" if len(psis) else "",
            f"{np.median(psis):.6g}" if len(psis) else "",
            f"{ratio_med:.6g}" if np.isfinite(ratio_med) else "",
            f"{np.median(t2):.6g}" if len(t2) else "",
            f"{np.mean(t2):.6g}" if len(t2) else "",
            f"{np.median(tt):.6g}" if len(tt) else "",
            f"{np.mean(tt):.6g}" if len(tt) else "",
            f"{np.sum(tt):.6g}" if len(tt) else "",
        ])
    _write_csv(f"{stem}_summary.csv",
               ["series", "layer", "n_events", "mean", "median", "min", "max",
                "p05", "p25", "p75", "p95", "pct_below_1", "pct_below_0p5",
                "empirical_mean", "empirical_median",
                "psis_mean", "psis_median", "median_kish_over_empirical",
                "t_stage2_median_s", "t_stage2_mean_s",
                "t_reweight_median_s", "t_reweight_mean_s",
                "t_reweight_total_s"],
               sum_rows)


def export_extras_csv(path, stem):
    """Two-stage breakdown and sharpness for ONE directory -- the datasets
    behind the panels that have no multi-series overlay."""
    try:
        from ..plotting.summary import load_two_stage
        ts = load_two_stage(path)
        if ts:
            _write_csv(f"{stem}_two_stage.csv",
                       ["event", "eff_stage1", "eff_stage2", "eff_total",
                        "t_stage2_s", "t_reweight_total_s"],
                       [[e, f"{a:.6g}", f"{b:.6g}", f"{c:.6g}",
                         "" if not np.isfinite(d) else f"{d:.6g}",
                         "" if not np.isfinite(t) else f"{t:.6g}"]
                        for e, a, b, c, d, t in ts])
    except ImportError:
        print("  (two-stage CSV needs load_two_stage in plotting.summary "
              "-- apply the module patch)")
    except Exception as e:
        print(f"  two-stage CSV skipped: {e}")
    try:
        from ..plotting.summary import load_sharpness
        sh = load_sharpness(path)
        if sh:
            _write_csv(f"{stem}_sharpness.csv",
                       ["event", "efficiency", "sigma_ln_Mc", "total_sharpness"],
                       [[str(e), f"{eff:.6g}", f"{f:.6g}", f"{sp:.6g}"]
                        for e, eff, f, sp in sh])
    except Exception as e:
        print(f"  sharpness CSV skipped: {e}")



############################################################################################################
############################################################################################################
#
#                                         MAIN 
#
############################################################################################################
############################################################################################################


def main():
    parser = argparse.ArgumentParser(
        description="Efficiency plots; overlays when more than one series is "
                    "requested.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+",
                        help="reweight output dir(s), or one summary.txt with --from-txt")
    parser.add_argument("--layer", choices=("full", "prior", "total"), default="full",
                        help="full = full-IS efficiency; prior = ESS of prior-only "
                             "weights (logpi - logq), recomputed from the pkls; "
                             "total = two-stage equal-weight yield")
    parser.add_argument("--from-txt", action="store_true",
                        help="Parse an existing summary.txt instead of *_data.pkl "
                             "files (full layer only)")
    parser.add_argument("--outdir", type=str, default=None,
                        help="Where to write the figures (default: alongside the first input)")
    parser.add_argument("--notitles", type=int, nargs="?", const=1, default=0,
                        help="0 = titles, 1 = no titles. Bare --notitles means 1.")
    parser.add_argument("--filter", type=str, default=None,
                        help="single fnmatch pattern on event name (one series)")
    parser.add_argument("--groups", nargs="+", default=None,
                        help="several fnmatch patterns -> one series each, overlaid")
    parser.add_argument("--labels", nargs="+", default=None,
                        help="display labels, one per group")
    parser.add_argument("--label", type=str, default=None,
                        help="suffix for the output filenames")
    parser.add_argument("--select", nargs="+", default=None, metavar="KEY:LO:HI",
                        help="joint parameter cuts applied to every series")
    parser.add_argument("--truth-dir", type=str, default=None,
                        help="directory with INJ_*_truth.pkl (params for --select)")
    parser.add_argument("--max-events", type=int, default=None,
                        help="randomly subsample each series to at most this "
                             "many events (applied AFTER --filter/--select)")
    parser.add_argument("--match-n", action="store_true",
                        help="subsample every series to the size of the "
                             "SMALLEST one, so populations are compared at "
                             "equal N. Combines with --max-events (the "
                             "tighter of the two wins).")
    parser.add_argument("--csv", action="store_true",
                        help="write the plotted data as CSV instead of / as "
                             "well as figures (see --csv-only)")
    parser.add_argument("--csv-only", action="store_true",
                        help="write CSVs and skip all plotting entirely "
                             "(no matplotlib rendering, so no TeX needed)")
    parser.add_argument("--subsample-seed", type=int, default=0,
                        help="RNG seed for --max-events / --match-n (default 0, "
                             "so a re-run reproduces the same subset)")
    args = parser.parse_args()
    no_titles = bool(args.notitles)

    patterns = args.groups or ([args.filter] if args.filter else [None])
    if args.labels and len(args.labels) != len(patterns):
        raise SystemExit(f"--labels ({len(args.labels)}) must match the number "
                         f"of patterns ({len(patterns)})")
    plabels = args.labels or [
        ((p.replace("*", "").replace("?", "").strip("_") or "all") if p else "all")
        for p in patterns]

    # optional parameter cuts, shared by every series
    allowed_by_dir = {}
    if args.select:
        if args.from_txt:
            raise SystemExit("--select needs the pkls, not --from-txt.")
        from .diagnose_real_vs_synth import load_events, load_truth_dir
        cuts = []
        for spec in args.select:
            try:
                key, lo, hi = spec.split(":")
                cuts.append((key, float(lo), float(hi)))
            except ValueError:
                raise SystemExit(f"bad --select spec '{spec}', expected KEY:LO:HI")
        truth_idx = load_truth_dir(args.truth_dir)
        for path in args.paths:
            ok = set()
            for e in load_events(path, truth_idx=truth_idx):
                if all(np.isfinite(e.get(k, np.nan)) and lo <= e[k] <= hi
                       for k, lo, hi in cuts):
                    ok.add(e["name"])
            allowed_by_dir[path] = ok
            print(f"--select: {os.path.basename(os.path.normpath(path))} "
                  f"keeps {len(ok)} events")

    series = []
    for pat, plab in zip(patterns, plabels):
        for path in args.paths:
            if args.from_txt:
                rows = load_from_txt(path)
            else:
                if not os.path.isdir(path):
                    raise SystemExit(f"{path} is not a directory. Pass the "
                                     f"reweight output dir, or a summary.txt "
                                     f"with --from-txt.")
                rows = load_from_pkls(path, args.layer)
            n0 = len(rows)
            if pat:
                rows = [r for r in rows if fnmatch.fnmatch(str(r[0]), pat)]
            if args.select:
                rows = [r for r in rows if str(r[0]) in allowed_by_dir[path]]
            if not rows:
                print(f"  [skip] {plab} @ {os.path.basename(os.path.normpath(path))}: "
                      f"0 of {n0}")
                continue
            if len(args.paths) == 1:
                lbl = plab
            elif len(patterns) == 1:
                lbl = os.path.basename(os.path.normpath(path))[:28]
            else:
                lbl = f"{plab} | {os.path.basename(os.path.normpath(path))[:16]}"
            series.append([rows, lbl, _COLORS[len(series) % len(_COLORS)], n0])

    if not series:
        raise SystemExit("No events found.")

    # optional subsampling to a common / capped event count (e.g. 518, and 512 --> 500)
    cap = args.max_events
    if args.match_n:
        smallest = min(len(entry[0]) for entry in series)
        cap = smallest if cap is None else min(cap, smallest)
        print(f"--match-n: smallest series has {smallest} events")
    if cap is not None:
        rng = np.random.default_rng(args.subsample_seed)
        for entry in series:
            rows = entry[0]
            if len(rows) > cap:
                idx = np.sort(rng.choice(len(rows), size=cap, replace=False))
                entry[0] = [rows[i] for i in idx]

    for rows, lbl, _c, n0 in series:
        print(f"  {lbl}: {len(rows)}/{n0} events  |  {_stats_line(rows)}")

    series = [(rows, lbl, c) for rows, lbl, c, _n0 in series]
    if not any(len(rows) for rows, _l, _c in series):
        raise SystemExit("No events left after subsampling.")

    out_dir = args.outdir or (os.path.dirname(os.path.abspath(args.paths[0]))
                              if args.from_txt else args.paths[0])
    os.makedirs(out_dir, exist_ok=True)
    label = args.label

    subtag = ""
    if args.match_n or args.max_events is not None:
        n_kept = max(len(rows) for rows, _l, _c in series)
        subtag = f"_n{n_kept}s{args.subsample_seed}"

    if args.csv or args.csv_only:
        csv_stem = os.path.join(
            out_dir, ("overlay" if len(series) > 1 else "efficiency")
            + f"_{args.layer}" + (f"_{label}" if label else "") + subtag)
        export_csv(series, csv_stem, args.layer)
        if len(args.paths) == 1 and not args.from_txt:
            export_extras_csv(args.paths[0], csv_stem)
        if args.csv_only:
            print("\n--csv-only: no figures rendered.")
            return

    if len(series) == 1:
        stem = (f"efficiency_sorted_{args.layer}"
                + (f"_{label}" if label else "") + subtag)
        rows = series[0][0]
        plot_sorted(rows, os.path.join(out_dir, f"{stem}_linear.png"),
                    args.layer, log_y=False, notitles=no_titles)
        plot_sorted(rows, os.path.join(out_dir, f"{stem}_log.png"),
                    args.layer, log_y=True, notitles=no_titles)
        plot_theoretical_only(rows, os.path.join(out_dir, f"{stem}_theoretical"),
                              log_y=True, notitles=no_titles)
    else:
        ov = os.path.join(out_dir, f"overlay_{args.layer}"
                          + (f"_{label}" if label else "") + subtag)
        plot_sorted_overlay(series, f"{ov}_sorted_linear", log_y=False,
                            notitles=no_titles)
        plot_sorted_overlay(series, f"{ov}_sorted_log", log_y=True,
                            notitles=no_titles)
        plot_survival_overlay(series, f"{ov}_survival", notitles=no_titles)
        plot_theoretical_overlay(series, f"{ov}_theoretical", notitles=no_titles)

    # per-directory extras (single directory only)
    if not args.from_txt and len(args.paths) == 1:
        path = args.paths[0]
        try:
            srows = load_sharpness(path)
            if len(srows) >= 5:
                plot_sharpness(srows,
                               os.path.join(out_dir, "efficiency_vs_sharpness.png"))
                print(f"Saved efficiency_vs_sharpness.png ({len(srows)} events)")
        except Exception as e:
            print(f"  sharpness panel skipped: {e}")
        try:
            plot_two_stage(path, os.path.join(out_dir, "two_stage_breakdown"),
                           notitles=no_titles)
        except Exception as e:
            print(f"  two-stage panel skipped: {e}")
        try:
            psis_reliability_report(path,
                                    os.path.join(out_dir, "psis_reliability"),
                                    notitles=no_titles)
        except Exception as e:
            print(f"  psis reliability report skipped: {e}")


if __name__ == "__main__":
    main()