"""
bilbyflow.plotting.injections — efficiency plots for injection campaigns.

Rows are (label, efficiency, dL, rho_opt); loaders handle a single run
(summary.pkl / event_*_data.pkl / summary.txt) or a batch parent directory
(every summary.pkl found recursively is combined). Shares efficiency_stats
with plotting.summary.
"""

import glob
import os
import pickle

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from .summary import efficiency_stats

__all__ = ["resolve_and_load", "plot_sorted_injections", "plot_snr_panel"]


# ---------------------          row extraction            -----------------------------

def _label(r, fallback_idx):
    """Globally-unique event_uid ('<seed>_<i>') when present, so injections
    from different array runs stay distinct when combined."""

    uid = r.get("event_uid")
    if uid is not None:
        return str(uid)

    ev = r.get("event", fallback_idx)
    return str(int(ev)) if isinstance(ev, (int, np.integer)) else str(ev)


def _dL_of(r):

    ip = r.get("injection_params")

    if isinstance(ip, dict) and "luminosity_distance" in ip:
        return float(ip["luminosity_distance"])

    return float("nan")


def _rho_of(r):

    rho2 = r.get("rho2_opt")
    if rho2 is not None and rho2 > 0:
        return float(np.sqrt(rho2))

    return float("nan")


def _rows_from_results(results):

    out = []
    for i, r in enumerate(results):
    
        try:
            eff = float(r.get("eff_total", r["efficiency"]))
            out.append((_label(r, i), eff, _dL_of(r), _rho_of(r)))
        except (KeyError, TypeError, ValueError):
            pass
    return out


##########################################################################################
# ----------------------------   loaders ------------------------------------------

def load_summary_pkl(path):

    with open(path, "rb") as f:
        results = pickle.load(f)

    return _rows_from_results(results)


def load_event_pkl(path):

    with open(path, "rb") as f:
        cached = pickle.load(f)

    r = dict(cached["result"])
    r.setdefault("event_uid", cached.get("event_uid"))
    r.setdefault("injection_params", cached.get("injection_params"))

    return _rows_from_results([r])


def load_injections_from_txt(filename):

    rows = []
    with open(filename) as f:
        lines = f.readlines()
    
    header = next((i for i, l in enumerate(lines) if "Event" in l and
                   ("Eff" in l or "PSIS%" in l)), None)

    if header is None:
        raise ValueError("Could not find a table header in summary.txt")



    for line in lines[header + 1:]:
        line = line.strip()
    
        if not line or line.startswith(("Mean", "Model", "Seed", "Run", "PSD",
                                        "Samples", "Events", "Proposal",
                                        "geocent", "-", "PSIS", "Empirical",
                                        "Min", "Marginalized", "Noise",
                                        "Template")):
            continue

        parts = line.split()

        if len(parts) >= 3:
            try:
                rows.append((str(parts[0]), float(parts[2]),
                             float("nan"), float("nan")))
                continue
            except ValueError:
                pass



        if len(parts) >= 2:
            try:
                rows.append((str(parts[0]), float(parts[1]),
                             float("nan"), float("nan")))
            except ValueError:
                pass

    return rows


def resolve_and_load(path, from_txt=False, prefer_events=False):
    """(rows, default_outdir); combines all runs found under a directory.
    prefer_events forces the event-pkl path (summary.pkl rows lack dL)."""


    if from_txt:
        return load_injections_from_txt(path), os.path.dirname(os.path.abspath(path))

    if os.path.isfile(path):
        if path.endswith(".pkl"):
            return load_summary_pkl(path), os.path.dirname(os.path.abspath(path))
        if path.endswith(".txt"):
            return load_injections_from_txt(path), os.path.dirname(os.path.abspath(path))
        raise SystemExit(f"Unrecognised file type: {path}")


    if not os.path.isdir(path):
        raise SystemExit(f"{path} not found")



    if not prefer_events:
        summaries = sorted(glob.glob(os.path.join(path, "**", "summary.pkl"),
                                     recursive=True))
        if summaries:
            rows, n_runs = [], 0
            for sp in summaries:
                try:
                    r = load_summary_pkl(sp)
                    rows.extend(r)
                    n_runs += 1
                    print(f"  + {len(r):3d} injections from "
                          f"{os.path.relpath(sp, path)}")
                except Exception as e:
                    print(f"  skip {sp}: {e}")
            print(f"Combined {len(rows)} injections from {n_runs} run(s)")
            return rows, path



    ev_files = sorted(glob.glob(os.path.join(path, "**", "event_*_data.pkl"),
                                recursive=True))


    if ev_files:
        rows = []
        for fp in ev_files:
            try:
                rows.extend(load_event_pkl(fp))
            except Exception as e:
                print(f"  skip {os.path.basename(fp)}: {e}")
        print(f"Combined {len(rows)} injections from {len(ev_files)} event pkl(s)")
        return rows, path



    st = os.path.join(path, "summary.txt")
    if os.path.exists(st):
        print("Reading summary.txt")
        return load_injections_from_txt(st), path



    raise SystemExit(f"No summary.pkl, event_*_data.pkl, or summary.txt under {path}")

##################################################################################################
##################################################################################################
# --------------------------              plotting                --------------------------------
##################################################################################################
##################################################################################################



def plot_sorted_injections(rows, outfile, log_y=False, dl_max=None, dl_min=None):
    """Efficiency bars sorted most -> least; rank axis for large N."""


    if dl_max is not None or dl_min is not None:
        lo = dl_min if dl_min is not None else -np.inf
        hi = dl_max if dl_max is not None else np.inf
        n_drop = sum(1 for r in rows if not np.isfinite(r[2]))
        if n_drop:
            print(f"  warning: {n_drop} injection(s) had no dL (summary-only) "
                  f"-> excluded by the dL cut")
        rows = [r for r in rows if np.isfinite(r[2]) and lo <= r[2] <= hi]
    if not rows:
        raise SystemExit("No injections survive the dL cut.")


    rows = sorted(rows, key=lambda r: r[1], reverse=True)
    labels = [r[0] for r in rows]
    effs = np.array([r[1] for r in rows])
    s = efficiency_stats(effs)
    n = s["n"]



    fig, ax = plt.subplots(figsize=(max(10, 0.03 * n), 5.5))
    x = np.arange(n)
    ax.bar(x, effs, alpha=0.85, color="tab:green")

    ax.axhline(s["mean"], color="red", ls="--", lw=1.3, label=f"Mean: {s['mean']:.2f}%")
    ax.axhline(s["median"], color="purple", ls="--", lw=1.3,
               label=f"Median: {s['median']:.2f}%")
    ax.axhline(1.0, color="black", ls=":", lw=1.2, label="1% efficiency")
    ax.axhline(0.5, color="gray", ls=":", lw=1.2, label="0.5% efficiency")

    if n <= 60:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=90, fontsize=8)
        ax.set_xlabel("Injection (sorted by efficiency: most --> least)")
    else:
        ax.set_xlabel(f"Injection rank (sorted by efficiency: most --> least, N={n})")

    ax.set_xlim(-0.6, n - 0.4)
    ax.set_ylabel("Reweighting efficiency (%)")
    scale = "log scale" if log_y else "linear scale"

    ax.set_title(f"NPE reweighting efficiency by injection ({scale})\n"
                 f"mean {s['mean']:.2f}%  |  median {s['median']:.2f}%  |  "
                 f"min {s['min']:.2f}%  |  <1%: {s['pct_below_1']:.1f}%  |  "
                 f"<0.5%: {s['pct_below_05']:.1f}%  (N={n})", fontsize=11)

    if log_y:
        ax.set_yscale("log")
        nonpos = int(np.sum(effs <= 0))
        if nonpos:
            print(f"  note: {nonpos} injection(s) have efficiency <= 0; "
                  f"omitted from the log-scale bars")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()


    fig.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.close(fig)



    print(f"Saved {outfile}")
    print(f"  mean={s['mean']:.2f}%  median={s['median']:.2f}%  "
          f"min={s['min']:.2f}%  <1%={s['pct_below_1']:.1f}%  "
          f"<0.5%={s['pct_below_05']:.1f}%  N={n}")


def plot_snr_panel(rows, out_dir):
    """Efficiency vs injected rho_opt scatter + SNR-binned console table."""

    rhos = np.array([r[3] for r in rows])
    effs = np.array([r[1] for r in rows])
    has_rho = np.isfinite(rhos)

    if has_rho.sum() <= 5:
        return
    
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.scatter(rhos[has_rho], effs[has_rho], s=12, alpha=0.6)

    ax.set_xlabel(r"Injected $\rho$_opt")
    ax.set_ylabel("Efficiency (%)")
    ax.set_yscale("log")
    ax.set_title(f"Efficiency vs injected SNR (N={int(has_rho.sum())})\n"
                 f"median \rho = {np.nanmedian(rhos[has_rho]):.1f}")

    ax.axhline(1.0, color="red", ls="--", lw=0.8, alpha=0.5)

    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    snr_file = os.path.join(out_dir, "efficiency_vs_snr.png")
    
    fig.savefig(snr_file, dpi=150, bbox_inches="tight")
    plt.close(fig)


    print(f"Saved {snr_file}")
    for lo, hi in [(6, 8), (8, 10), (10, 12), (12, 15), (15, 25), (25, 100)]:
        m = has_rho & (rhos >= lo) & (rhos < hi)
        if m.sum() > 0:
            print(f"  \rho [{lo:>3d},{hi:>3d}): n={m.sum():>4d}  "
                  f"median_eff={np.median(effs[m]):.2f}%  "
                  f"mean_eff={np.mean(effs[m]):.2f}%")