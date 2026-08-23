"""
bilbyflow.plotting.corner — per-event corner overlays.

Raw flow (blue), reweighted (green), published (orange), MAP truth lines
(red). Reads the combined weight (log_weight_final) when present, falling
back to the stage-1 weight, so the green cloud is the HM posterior, not the
(2,2) screen.
"""
# A lot of stuff in this script that is a bit repetitive and should be updated later



import numpy as np
import matplotlib
matplotlib.rcParams["text.usetex"] = False
import matplotlib.pyplot as plt
from corner import corner

from .weights import final_log_weight


__all__ = [
    "plot_reweighted_vs_published",
    "plot_reweighted_npe_only",
    "plot_recovered_extrinsics_vs_published",
    "plot_recovered_extrinsics_vs_truth",
]

# _LEVELS gives 1 sigma, 2 sigma, and 3 sigma. 
    # Formula is 1-np.exp(-\sigma^2/2)
    # See https://corner.readthedocs.io/en/latest/pages/sigmas/ if not sure why
_LEVELS = (1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-4.5))


def _safe_range(col):

    lo, hi = float(np.percentile(col, 0.5)), float(np.percentile(col, 99.5))

    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-10:
        c = (lo + hi) / 2 if np.isfinite(lo) and np.isfinite(hi) else 0.0
        return (c - 1e-3, c + 1e-3)

    pad = 0.05 * (hi - lo)
    return (lo - pad, hi + pad)







# Often not a one to one, but is useful for pointing out biases in the NPE
def plot_reweighted_vs_published(npe_samples, result, published_array, truths,
                                 param_names, event_name, filename):

    log_w, valid = final_log_weight(result)

    if valid.sum() < 10:
        print(f"  Too few valid samples to plot for {event_name}")
        return

    lw_s = result.get("log_weight_smoothed")
    lw = lw_s if lw_s is not None else log_w[valid]

    weight = np.exp(lw - lw.max())
    weight = weight / weight.sum()

    valid_cols = [j for j, p in enumerate(param_names)
                  if not np.all(published_array[:, j] == 0)]

    if not valid_cols:
        print(f"  No overlapping parameters for {event_name}")
        return

    # Equal-weight HM output when there are enough distinct theta, or else the
    # stage-1 weighted cloud (contours need many more points than dimensions).
    idxf = result.get("idx_final")

    n_min = max(10 * len(valid_cols), 200)
    n_uniq = result.get("n_unique_final")


    if n_uniq is None and idxf is not None:
        n_uniq = int(np.unique(idxf).size)
    
    if ("log_weight_weighted" not in result and idxf is not None
            and n_uniq is not None and n_uniq >= n_min):
        npe_rw_plot = npe_samples[idxf][:, valid_cols]
        weight = None

    else:
        if idxf is not None:
            print(f"  stage-2 gave {len(idxf)} draws over {n_uniq} distinct theta "
                  f"(< {n_min} needed for {len(valid_cols)}-D corner), "
                  f"plotting stage-1 weighted samples instead")
        npe_rw_plot = npe_samples[valid][:, valid_cols]

    npe_plot = npe_samples[:, valid_cols]
    pub_plot, truths_plot = published_array[:, valid_cols], truths[valid_cols]
    names_plot = [param_names[j] for j in valid_cols]


    combined = np.vstack([npe_plot, pub_plot])
    ranges = [_safe_range(combined[:, j]) for j in range(len(valid_cols))]

    kw = dict(range=ranges, levels=_LEVELS, show_titles=False,
              quantiles=[0.05, 0.5, 0.95], hist_kwargs=dict(density=True))


    truth_list = [t if np.isfinite(t) else None for t in truths_plot]
    fig = corner(npe_plot, labels=names_plot, color="tab:blue",
                 truths=truth_list, truth_color="tab:red", **kw)

    if weight is not None:
        corner(npe_rw_plot, weights=weight, fig=fig, color="tab:green", **kw)
    else:
        corner(npe_rw_plot, fig=fig, color="tab:green", **kw)

    corner(pub_plot, fig=fig, color="tab:orange", **kw)



    fig.legend(handles=[
        plt.Line2D([], [], color="tab:blue", label=f"NPE raw (n={result['n_samples']})"),
        plt.Line2D([], [], color="tab:green",
                   label=f"NPE reweighted (n_eff={result['n_eff']:.0f}, "
                         f"eff={result['efficiency']:.1f}%)"),
        plt.Line2D([], [], color="tab:orange", label="Published"),
        plt.Line2D([], [], color="tab:red", ls="--", label="Published MAP"),
    ], loc="upper right", fontsize=11)


    fig.suptitle(rf"{event_name} (real data): NPE vs Reweighted vs Published "
                rf"| $\hat{{k}}$={result['khat']:.2f}", fontsize=13, y=1.02)
    fig.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"  Saved {filename}")







def plot_reweighted_npe_only(npe_samples, result, param_names, event_name, filename):

    log_w, valid = final_log_weight(result)
    if valid.sum() < 10:
        print(f"  Too few valid samples to plot for {event_name}")
        return
    
    lw_s = result.get("log_weight_smoothed")
    lw = lw_s if lw_s is not None else log_w[valid]

    weight = np.exp(lw - lw.max())
    weight = weight / weight.sum()


    ranges = [_safe_range(npe_samples[:, j]) for j in range(npe_samples.shape[1])]


    kw = dict(range=ranges, levels=_LEVELS, show_titles=False,
              quantiles=[0.05, 0.5, 0.95], hist_kwargs=dict(density=True))

    
    fig = corner(npe_samples, labels=param_names, color="tab:blue", **kw)

    corner(npe_samples[valid], weights=weight, fig=fig, color="tab:green", **kw)


    fig.legend(handles=[
        plt.Line2D([], [], color="tab:blue", label=f"NPE raw (n={result['n_samples']})"),
        plt.Line2D([], [], color="tab:green",
                   label=f"NPE reweighted (n_eff={result['n_eff']:.0f}, "
                         f"eff={result['efficiency']:.1f}%)"),
    ], loc="upper right", fontsize=11)


    fig.suptitle(rf"{event_name} (real data): NPE vs Reweighted | "
                 rf"$\hat{{k}}$={result['khat']:.2f}", fontsize=13, y=1.02)
    fig.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"  Saved {filename}")






def plot_recovered_extrinsics_vs_truth(result, injection_params, event_name,
                                       filename):
    """Injection runs: recovered (phase, psi, tc, dL) posteriors with the
    injected truths as corner truth lines (no published overlay)."""

    if not result.get("synthetic_phase", False):
        return
    
    log_w, valid = final_log_weight(result)

    spec = [("recovered_phase", "phase", "phase"),
            ("recovered_psi", "psi", "psi"),
            ("recovered_tc_offset", "tc offset [s]", "geocent_time"),
            ("recovered_dL", "dL [Mpc]", "luminosity_distance")]


    cols, names, truths = [], [], []
    for key, label, pkey in spec:
        if not result["marg_flags"].get(pkey, False):
            continue
        cols.append(result[key])
        names.append(label)
        t = float(injection_params.get(pkey, np.nan))
        truths.append(t - result["tc0"] if pkey == "geocent_time" else t)
    
    if not cols:
        return
    arr = np.vstack(cols).T
    ok = valid & np.all(np.isfinite(arr), axis=1)
    if ok.sum() < 10:
        return


    w = np.exp(log_w[ok] - log_w[ok].max())
    w /= w.sum()

    fig = corner(arr[ok], weights=w, labels=names, color="tab:green",
                 truths=truths, truth_color="tab:red", show_titles=True,
                 hist_kwargs=dict(density=True))

    fig.suptitle(f"{event_name}: recovered extrinsics (reweighted)", y=1.02)

    fig.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"  Saved {filename}")









def plot_recovered_extrinsics_vs_published(result, published, event_name, filename):
    """Recovered marginalised extrinsics (weighted) vs the published posterior."""

    if not result.get("synthetic_phase", False):
        return

    log_w, valid = final_log_weight(result)

    spec = [("recovered_phase", "phase", "phase"),
            ("recovered_psi", "psi", "psi"),
            ("recovered_tc_offset", "tc offset [s]", "geocent_time"),
            ("recovered_dL", "dL [Mpc]", "luminosity_distance")]

    cols, names, pub_cols = [], [], []
    for key, label, pkey in spec:
        if not result["marg_flags"].get(pkey, False):
            continue
        if published is None or pkey not in published or len(published[pkey]) == 0:
            continue
        cols.append(result[key])
        names.append(label)
        pv = np.asarray(published[pkey], dtype=float)
        if pkey == "geocent_time":
            pv = pv - result["tc0"]
        pub_cols.append(pv)
    
    if not cols:
        return

    arr = np.vstack(cols).T

    ok = valid & np.all(np.isfinite(arr), axis=1)
    if ok.sum() < 10:
        return

    n_pub = min(len(v) for v in pub_cols)
    pub = np.vstack([v[:n_pub] for v in pub_cols]).T

    w = np.exp(log_w[ok] - log_w[ok].max())
    w /= w.sum()

    combined = np.vstack([arr[ok], pub])

    ranges = [_safe_range(combined[:, j]) for j in range(arr.shape[1])]

    kw = dict(range=ranges, show_titles=False, hist_kwargs=dict(density=True))

    fig = corner(arr[ok], weights=w, labels=names, color="tab:green", **kw)

    corner(pub, fig=fig, color="tab:orange", **kw)

    fig.legend(handles=[
        plt.Line2D([], [], color="tab:green", label="Recovered (reweighted NPE)"),
        plt.Line2D([], [], color="tab:orange", label="Published")],
        loc="upper right", fontsize=11)

    fig.suptitle(f"{event_name}: recovered extrinsics vs published", y=1.02)
    fig.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {filename}")