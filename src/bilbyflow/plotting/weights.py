"""
bilbyflow.plotting.weights — weight diagnostics and the across-event summary.


plot_weight_diagnostics: per-event log-w histogram, normalized weights, and
    log-w vs log-L, using the combined weight (log_weight_final) when present.

plot_summary: efficiency and k-hat bars over all events; the median line is
    darkred/dotted.


"""

import numpy as np
import matplotlib
matplotlib.rcParams["text.usetex"] = False
import matplotlib.pyplot as plt

__all__ = ["final_log_weight", "plot_weight_diagnostics", "plot_summary"]


def final_log_weight(result):
    """THE weight-selection convention for all plotting: full-length weighted
    HM weights (log_weight_weighted) when present, else the equal-weight
    log_weight_final, else the stage-1 log_weight. Returns (log_w, finite
    mask). The order is load-bearing — it decides whether a plot shows the
    HM posterior or the (2,2) screen."""


    log_w = result.get("log_weight_weighted",
                       result.get("log_weight_final", result["log_weight"]))

    return log_w, np.isfinite(log_w)




def plot_weight_diagnostics(result, event_name, filename):

    log_w, valid = final_log_weight(result)
    if valid.sum() < 2:
        return

    lw = log_w[valid]


    fig, axes = plt.subplots(1, 3, figsize=(15, 4))



    axes[0].hist(lw, bins=50, density=True, alpha=0.8)
    axes[0].set_xlabel("log w")
    axes[0].set_title("Log weights")
    axes[0].grid(True, alpha=0.3)


    w = np.exp(lw - lw.max())

    axes[1].hist(w / w.sum(), bins=50, density=True, alpha=0.8, color="tab:orange")
    axes[1].set_xlabel("Normalised w")
    axes[1].set_title("Normalised weights")
    axes[1].grid(True, alpha=0.3)



    axes[2].scatter(result["log_likelihood"][valid], lw, s=1, alpha=0.3)
    axes[2].set_xlabel("log L (marginal)")
    axes[2].set_ylabel("log w")
    axes[2].set_title("Weights vs likelihood")
    axes[2].grid(True, alpha=0.3)



    fig.suptitle(f"{event_name}: eff={result['efficiency']:.1f}%, "
                 f"n_eff={result['n_eff']:.0f}/{result['n_valid']}, "
                 f"k\u0302={result['khat']:.2f}", fontsize=12)
    fig.tight_layout()

    fig.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"  Saved {filename}")


def plot_summary(results, filename):
    events = [r["event"] for r in results]
    effs = [r["efficiency"] for r in results]
    khats = [r["khat"] for r in results]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar([str(e) for e in events], effs, alpha=0.8, color="tab:blue")
    axes[0].axhline(np.mean(effs), color="red", ls="--",
                    label=f"Mean: {np.mean(effs):.1f}%")
    axes[0].axhline(np.median(effs), color="darkred", ls=":",
                    label=f"Median: {np.median(effs):.1f}%")
    axes[0].set_xlabel("Event")
    axes[0].set_ylabel("Efficiency (%)")
    axes[0].set_title("Reweighting Efficiency")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].tick_params(axis="x", rotation=45)
    axes[1].bar([str(e) for e in events], khats, alpha=0.8, color="tab:orange")
    axes[1].axhline(0.7, color="red", ls="--", label="k\u0302=0.7 (unreliable)")
    axes[1].set_xlabel("Event")
    axes[1].set_ylabel("k\u0302")
    axes[1].set_title("PSIS Diagnostic")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {filename}")