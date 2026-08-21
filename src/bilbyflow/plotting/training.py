"""
bilbyflow.plotting.training — training-curve plots.

_plot_losses draws the NLL panel (train, val full-prior, val stage-cap) with a
"last 50%" inset, 
plus optional stacked panels for the annealed aux MSE and 
the JEPA embedding-consistency MSE (only shown when those terms were actually
active, which is not often for JEPA atm, 21/08/2026). 

It is called periodically from training.trainer at each checkpoint,
and once at the end via the plot_losses summary-dict wrapper.

The training script writes val_cap ("stage cap") losses at every epoch and
val_full only every 5th epoch (NaN elsewhere) — matplotlib skips the NaNs, so
the full-prior curve renders as a dashed-through gappy line. Nothing here reads
best_state; these are pure diagnostics.

Having some issues with the val full every now and the, but the main val to watch
is the capped one anyways.
"""

import numpy as np
import matplotlib
matplotlib.rcParams["text.usetex"] = False
import matplotlib.pyplot as plt

__all__ = ["plot_losses", "_plot_losses"]




def _plot_losses(train_losses, val_losses, filename, val_cap_losses=None,
                 aux_losses=None, cons_losses=None):

    # written this way just to make the lines fit in a normal editor and not use \ (hate)
    has_aux = aux_losses is not None
    has_aux = has_aux and len(aux_losses) == len(train_losses)
    has_aux = has_aux and any(a > 0 for a in aux_losses if np.isfinite(a))

    has_cons = cons_losses is not None 
    has_cons = has_cons and len(cons_losses) == len(train_losses)
    has_cons = has_cons and any(c > 0 for c in cons_losses if np.isfinite(c))




    nrows = 1 + int(has_aux) + int(has_cons)


    fig, axes = plt.subplots(nrows, 1, figsize=(10, 5 * nrows), squeeze=False)


    ax = axes[0, 0]

    tl, vl = train_losses, val_losses

    ep = np.arange(1, len(tl) + 1)

    ax.plot(ep, tl, label="Train (NLL + lam*aux + lam_c*cons)", alpha=0.8)
    ax.plot(ep, vl, label="Val NLL (full prior)", alpha=0.8)


    if val_cap_losses is not None and len(val_cap_losses) == len(tl):
        ax.plot(ep, val_cap_losses, label="Val NLL (stage cap)", alpha=0.8)

    
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title(f"NPE Training ({len(tl)} epochs)")


    if len(tl) > 20:
        mid = len(tl) // 2
        ins = ax.inset_axes([0.5, 0.5, 0.45, 0.45])
        ins.plot(ep[mid:], tl[mid:], alpha=0.8)
        ins.plot(ep[mid:], vl[mid:], alpha=0.8)
        if val_cap_losses is not None and len(val_cap_losses) == len(tl):
            ins.plot(ep[mid:], val_cap_losses[mid:], alpha=0.8)
        ins.set_title("Last 50%", fontsize=9)
        ins.grid(True, alpha=0.3)

    row = 1

    if has_aux:
        ax2 = axes[row, 0]
        row += 1
        ax2.plot(ep, aux_losses, color="tab:purple", alpha=0.8)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("aux MSE (z-scored)")
        ax2.set_title("Auxiliary summary regression (detached probe)")
        ax2.grid(True, alpha=0.3)


    if has_cons:
        ax3 = axes[row, 0]
        ax3.plot(ep, cons_losses, color="tab:green", alpha=0.8)
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("consistency MSE")
        ax3.set_title("JEPA embedding consistency (real -> synthetic anchor)")
        ax3.grid(True, alpha=0.3)


    fig.tight_layout()
    fig.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)




def plot_losses(summary, filename):
    """Summary-dict entry point. Periodic per-checkpoint calls in
    training.trainer use the low-level _plot_losses(train, val, path, ...) directly."""

    return _plot_losses(
        summary["training_loss"], summary["validation_loss"], filename,
        val_cap_losses=summary.get("validation_loss_capped"),
        aux_losses=summary.get("aux_loss"),
        cons_losses=summary.get("cons_loss"))