"""
bilbyflow.scripts.utils — shared presentation helpers for the CLI scripts.

Inference-for-plotting, corner/PP figures, and the pre-training data
diagnostics. 

These import matplotlib and are deliberately NOT part of the
importable library surface (bilbyflow.plotting holds the reusable figure
code; 
these are script-side conveniences that glue posterior + standardiser
+ config together for quick looks making individual cli scripts simpler)

Understanding of these plots is not necessary to understand the main bit of
the code, they can just be helpful.
"""

import os

import numpy as np
import torch
from tqdm import tqdm
import matplotlib
matplotlib.rcParams["text.usetex"] = False
import matplotlib.pyplot as plt
from corner import corner

from ..coordinates.sky import samples_detector_to_radec
from ..coordinates.params import dL_to_physical
from ..data.dataset import generate_fixed_dataset
from ..nn.aux_head import AUX_NAMES, N_AUX

__all__ = ["run_inference", "plot_corner_fig", "pp_test",
           "plot_aux_diagnostics", "run_diagnostics"]




def run_inference(posterior, x_obs, std, cfg, param_names, n_samples=10_000,
                  tc=None):
    """Sample the flow at one x and return PHYSICAL parameters (ra/dec
    restored, dL/Mc un-logged)."""
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    x_n = std.normalise_x(x_obs).to(DEVICE)

    s = posterior.sample((n_samples,), x=x_n).cpu().numpy()
    s = s * std.theta_std.numpy() + std.theta_mean.numpy()
    s = dL_to_physical(s, param_names, cfg)

    event_tc = tc if tc is not None else float(cfg["ref_geocent_time"])

    return samples_detector_to_radec(s, param_names, tc=event_tc)


def plot_corner_fig(samples, truths, names, title, filename):

    fig = corner(samples, labels=names, truths=truths, truth_color="tab:red",
                 show_titles=True, title_kwargs=dict(fontsize=9),
                 quantiles=[0.05, 0.5, 0.95])
    fig.suptitle(title, fontsize=14, y=1.02)
    fig.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)



def pp_test(posterior, x_test, theta_test, std, names, filename,
            n_posterior=5000):
    """Quick end-of-training PP plot (pointwise binomial band only; the full
    simultaneous-band methodology lives in the diagnostics tooling).
    
    If unfamiliar this works as a self-calibration test but does not evaluate 
    accuracy in SBI settings. Just because the flow satisfies these curves 
    doesn't mean that it is 'correct'.
        - https://sbi.readthedocs.io/en/stable/advanced_tutorials/11_diagnostics_simulation_based_calibration.html

    It's also called a quantile-quantile plot.
        - https://en.wikipedia.org/wiki/Q%E2%80%93Q_plot
        
    """
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    nt, np_ = x_test.shape[0], theta_test.shape[1]

    xn = std.normalise_x(x_test)
    tn = std.normalise_theta(theta_test)

    pct = np.zeros((nt, np_))
    for i in tqdm(range(nt), desc="PP test"):
        s = posterior.sample((n_posterior,), x=xn[i].to(DEVICE)).cpu().numpy()
        for j in range(np_):
            pct[i, j] = np.mean(s[:, j] < tn[i, j].item())

    
    fig, ax = plt.subplots(figsize=(6, 6))

    cr = np.linspace(0, 1, 200)
    for j, n in enumerate(names):
        ax.plot(cr, [np.mean(pct[:, j] <= c) for c in cr], label=n)
    
    ax.plot([0, 1], [0, 1], "k--", label="ideal")

    ax.fill_between(cr, cr - 1.96 * np.sqrt(cr * (1 - cr) / nt),
                    cr + 1.96 * np.sqrt(cr * (1 - cr) / nt),
                    alpha=0.15, color="grey")

    
    ax.set_xlabel("Credible level")
    ax.set_ylabel("Expected coverage")
    ax.legend(fontsize=5, ncol=3)
    ax.set_title(f"PP test ({nt} injections)")

    fig.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_aux_diagnostics(dataset, std, diag_dir, n=2000):
    """Distributions of the aux targets (z-scored) — sanity that nothing is
    degenerate/constant before training on it."""

    if not getattr(dataset, "aux_supervision", False):
        return
    os.makedirs(diag_dir, exist_ok=True)

    _, _, _, aux = generate_fixed_dataset(dataset, n, return_aux=True)

    aux = aux.numpy()
    z = ((torch.from_numpy(aux) - std.aux_mean) / std.aux_std).numpy()

    ncol = 5
    nrow = int(np.ceil(N_AUX / ncol))

    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.6 * nrow))

    for j in range(N_AUX):
        a = axes.flat[j]
        a.hist(z[:, j], bins=60, density=True, alpha=0.8, color="tab:purple")
        a.set_title(AUX_NAMES[j], fontsize=9)
        a.grid(True, alpha=0.3)
    for j in range(N_AUX, nrow * ncol):
        axes.flat[j].axis("off")
    
    fig.suptitle("Aux targets (z-scored)", y=1.01)
    fig.tight_layout()

    fig.savefig(f"{diag_dir}/aux_targets.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"  Saved {diag_dir}/aux_targets.png")


def run_diagnostics(x_val, theta_val, std, cfg, diag_dir, dataset=None):
    """Per-channel + PSD-context data diagnostics on the fixed val set."""
    from scipy.signal.windows import tukey as _tukey

    print("\n=== Data Diagnostics ===")

    os.makedirs(diag_dir, exist_ok=True)

    dur, sr, fmin = (float(cfg["duration"]), int(cfg["sampling_frequency"]),
                     float(cfg["f_min"]))
    
    n_fd_full = int(dur * sr / 2) + 1
    n_td = 2 * (n_fd_full - 1)
    freq_array = np.arange(n_fd_full) / dur

    n_fd_masked = int(np.sum(freq_array >= fmin))
    per_det = 2 * n_fd_masked + n_td
    freq_masked = freq_array[freq_array >= fmin]
    time_array = np.arange(n_td) / sr

    print(f"  x={x_val.shape} | per_det={per_det} | expected={2*per_det}")
    print(f"  nan={torch.isnan(x_val).any()} inf={torch.isinf(x_val).any()}")

    strain_dim = 2 * per_det
    psd_cond = (dataset is not None
                and getattr(dataset, "psd_conditioning", False)
                and getattr(dataset, "psd_dim", 0) > 0)

    if psd_cond:
        fmask = np.asarray(dataset.freq_mask)

        n_band = int(fmask.sum())

        if x_val.shape[-1] < strain_dim + 2 * n_band:
            print("  WARNING: x has no PSD block, skipping PSD diagnostics")
            psd_cond = False


    def split_ch(xv):
        o, ch = 0, {}
        for d in ["H1", "L1"]:
            ch[f"Re_{d}"] = xv[o:o + n_fd_masked]; o += n_fd_masked
            ch[f"Im_{d}"] = xv[o:o + n_fd_masked]; o += n_fd_masked
            ch[f"TD_{d}"] = xv[o:o + n_td]; o += n_td
        return ch


    def split_psd(xv):
        o, out = strain_dim, {}
        for d in ["H1", "L1"]:
            out[d] = xv[o:o + n_fd_masked]; o += n_fd_masked
        return out


    xnp_all = x_val.numpy()
    re_h1 = xnp_all[:, :n_fd_masked]
    var_q = float(np.var(re_h1))
    lag1 = float(np.mean([np.corrcoef(r[:-1], r[1:])[0, 1]
                          for r in re_h1[:50]]))
    
    _wf = float(np.mean(_tukey(
        n_td, 2.0 * float(cfg.get("tukey_roll_off", 0.2)) / dur) ** 2))
    _mode = str(cfg.get("noise_source", "gaussian_whitened")).lower()
    _exp = 1.0 if _mode == "gaussian_whitened" else _wf
    
    print(f"  H1 Re(FD): per-quadrature var={var_q:.3f} "
          f"(expect ~{_exp:.3f} + signal, noise_source={_mode}), "
          f"mean lag-1 bin corr={lag1:.3f}")

    if psd_cond:
        ctx = xnp_all[:, strain_dim:strain_dim + 2 * n_band]
        frac_clip = float((np.abs(ctx) >= float(
            getattr(dataset, "psd_clip", 10.0)) - 1e-6).mean()) * 100
        print(f"  PSD context: mean={ctx.mean():+.3f} std={ctx.std():.3f} "
              f"at clip: {frac_clip:.2f}%")


    names = cfg["inferred_parameters"]

    rng = np.random.default_rng(123)
    idxs = rng.choice(len(x_val), size=min(5, len(x_val)), replace=False)
    xvn = std.normalise_x(x_val)
    nrows = 4 if psd_cond else 3

    freq_in = (freq_array[np.asarray(dataset.freq_mask)] if psd_cond else None)

    for idx in idxs:
        ch = split_ch(xvn[idx].numpy())
        p = theta_val[idx].numpy()
        ps = ", ".join([f"{names[i][:5]}={p[i]:.2f}"
                        for i in range(min(len(names), 9))])

        
        fig, ax = plt.subplots(nrows, 2, figsize=(14, 3.3 * nrows))

        for c, d in enumerate(["H1", "L1"]):
            ax[0, c].plot(freq_masked, ch[f"Re_{d}"], lw=0.5)
            ax[0, c].set_title(f"{d} Re(FD)")
            ax[1, c].plot(freq_masked, ch[f"Im_{d}"], lw=0.5)
            ax[1, c].set_title(f"{d} Im(FD)")
            ax[2, c].plot(time_array, ch[f"TD_{d}"], lw=0.5)
            ax[2, c].set_title(f"{d} TD")
            for a in ax[:3, c]:
                a.grid(True, alpha=0.3)
        
        if psd_cond:
            psd_s = split_psd(x_val[idx].numpy())
            for c, d in enumerate(["H1", "L1"]):
                ax[3, c].plot(freq_in, psd_s[d], lw=0.6, color="tab:purple")
                ax[3, c].axhline(0, color="k", lw=0.5, alpha=0.6)
                ax[3, c].set_title(f"{d} PSD context (standardised)")
                ax[3, c].grid(True, alpha=0.3)

        
        fig.suptitle(f"Sample {idx} (std) — {ps}", fontsize=10, y=1.005)
        fig.tight_layout()
        fig.savefig(f"{diag_dir}/sample_{idx}.png", dpi=150,
                    bbox_inches="tight")
        plt.close(fig)



    fig, ax = plt.subplots(2, 1, figsize=(14, 6))

    xnp = x_val.numpy()

    ax[0].semilogy(xnp.std(axis=0), lw=0.5); ax[0].set_title("Per-dim std")

    ax[1].plot(xnp.mean(axis=0), lw=0.5); ax[1].set_title("Per-dim mean")


    bounds = [n_fd_masked, 2 * n_fd_masked, per_det, per_det + n_fd_masked,
              per_det + 2 * n_fd_masked]

    if psd_cond:
        bounds += [strain_dim,
                   strain_dim + int(np.asarray(dataset.freq_mask).sum())]
    for a in ax:
        a.grid(True, alpha=0.3)
        for b in bounds:
            a.axvline(b, color="red", alpha=0.3, ls="--")
    
    fig.tight_layout()
    fig.savefig(f"{diag_dir}/data_stats.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved to {diag_dir}/")