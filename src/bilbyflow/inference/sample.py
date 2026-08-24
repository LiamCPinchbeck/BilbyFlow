"""
bilbyflow.inference.sample — batched NPE sampling with log-proposal density.

npe_sample_and_logprob draws from the trained flow on GPU (batched sampling +
batched log_prob), converts normalised theta back to physical parameters,
applies the ln-dL / ln-Mc Jacobians to the proposal density, maps the sky
coordinates back to (ra, dec), and optionally SIR-corrects the proposal from
the training prior to the target prior (the --prior-swap path) BEFORE any
likelihood evaluation.

Very centralized at the moment for ease of user use, but like many things in this
package at the moment, is subject to PhD-need-to-publish-spaghetti-fication^TM, so
will hopefully be simplified in the backend in later updates.

Returns (samples_phys, log_q, swap_info):
  samples_phys : (n, n_params) physical draws (ra/dec restored)
  log_q        : log proposal density in PHYSICAL coordinates (the reweighter's
                 log_draw_prob), including ln-dL / ln-Mc Jacobians and, when
                 prior-swapped, the SIR correction log_r - log_Zr.
  swap_info    : prior-swap ESS bookkeeping (empty dict when not swapping).

The prior-swap ratio also folds in the SNR training-loss weighting correction (if used)
(w(rho) = min(1 + rho^2/rho0^2, w_max)) analytically via the dL dependence, so
no extra waveform call is needed. The sky inverse comes from coordinates.sky.
"""

import time

import numpy as np
from scipy.special import logsumexp
import torch

from ..coordinates.sky import samples_detector_to_radec

__all__ = ["npe_sample_and_logprob"]


def npe_sample_and_logprob(posterior, x_npe, std, n_samples, cfg, tc=None,                            
                           proposal_noise=0.0, 
                           proposal_noise_k=16,
                           priors=None, prior_swap=True,
                           oversample=10, seed=0):
    """
    - proposal_noise (float)    --> if the flow is over-trained/doing under-coverage proposal noise could spread out,
                                    current problem (as of 21/08/2026 at least) is over-coverage so this is likely not
                                    the knob to turn here. If it does end up being an issue for you, it would probably
                                    be better to use a mixture of the flow architecture and the prior for any reweighting
                                    as it ensures a few more things
    - prior_swap (bool)         --> Flag for whether to correct for training vs physically motivated priors. Basically just
                                    always have this on unless you're doing diagnostics of some sort or another.
    """
    M = int(n_samples * oversample) if prior_swap else n_samples

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if next(posterior.parameters()).device.type != device:
        posterior.to(device)
    std = posterior.embedding.std

    _t0 = time.perf_counter()
    x_dev = torch.tensor(x_npe).unsqueeze(0).to(device)    # raw, not normalised

    BATCH = 5000
    with torch.no_grad():
        posterior.pin(x_dev)
        sample_chunks = []
        for start in range(0, M, BATCH):
            n_chunk = min(BATCH, M - start)
            sample_chunks.append(
                posterior.sample(n_chunk, x_dev).cpu())
        samples_norm = torch.cat(sample_chunks, dim=0)

        lq_chunks = []
        for start in range(0, M, BATCH):
            chunk = samples_norm[start:start+BATCH].to(device)
            lq_chunks.append(
                posterior.log_prob(chunk, x_dev).cpu().detach())
        log_q_norm = torch.cat(lq_chunks).numpy()
        posterior.unpin()
    # model stays resident on the GPU across events — the per-event


    if proposal_noise > 0.0:
        sigma = float(proposal_noise)
        k = int(proposal_noise_k)
        samples_norm = samples_norm + torch.randn_like(samples_norm) * sigma
        x_cpu = torch.tensor(x_npe).unsqueeze(0)
        lps = np.empty((k, samples_norm.shape[0]))
        with torch.no_grad():
            for j in range(k):
                shifted = samples_norm - torch.randn_like(samples_norm) * sigma
                for start in range(0, len(shifted), BATCH):
                    chunk = shifted[start:start + BATCH]
                    lps[j, start:start + len(chunk)] = posterior.log_prob(
                        chunk, x_cpu).detach().numpy()
        log_q_norm = logsumexp(lps, axis=0) - np.log(k)


    samples_phys = samples_norm.detach().cpu().numpy() * std.theta_std.numpy() + std.theta_mean.numpy()
    samples_phys = samples_detector_to_radec(
        samples_phys, cfg["inferred_parameters"],
        tc=float(tc) if tc is not None else float(cfg["ref_geocent_time"]))



    dL_log = bool(getattr(std, "dL_log", False)) or \
        str(cfg.get("dL_param", "linear")).lower() == "log"
    log_jac_dL = 0.0
    if dL_log and "luminosity_distance" in cfg["inferred_parameters"]:
        di = cfg["inferred_parameters"].index("luminosity_distance")
        ln_dL = samples_phys[:, di].copy()
        samples_phys[:, di] = np.exp(ln_dL)
        log_jac_dL = ln_dL



    Mc_log = bool(getattr(std, "Mc_log", False)) or \
        str(cfg.get("Mc_param", "linear")).lower() == "log"
    log_jac_Mc = 0.0
    if Mc_log and "chirp_mass" in cfg["inferred_parameters"]:
        mi = cfg["inferred_parameters"].index("chirp_mass")
        ln_Mc = samples_phys[:, mi].copy()
        samples_phys[:, mi] = np.exp(ln_Mc)
        log_jac_Mc = ln_Mc



    log_jacobian = torch.log(std.theta_std).sum().item()
    log_q = log_q_norm - log_jacobian - log_jac_dL - log_jac_Mc



    if prior_swap:
        log_r = np.zeros(len(samples_phys))
        if "luminosity_distance" in cfg["inferred_parameters"]:
            di = cfg["inferred_parameters"].index("luminosity_distance")
            dL = samples_phys[:, di]
            lo = float(cfg["priors"]["luminosity_distance"]["min"])
            hi = float(cfg["priors"]["luminosity_distance"]["max"])
            in_bounds = (dL >= lo) & (dL <= hi)

            alpha_target = 2                                  # PowerLaw(alpha=2)
            alpha_train = float(cfg.get("dL_train_alpha", 0))
            if alpha_train == -1:                             # log-uniform
                log_r += (alpha_target - (-1)) * np.log(dL)
            else:
                log_r += (alpha_target - alpha_train) * np.log(dL)

            # SNR training-weight correction, folded in via the dL dependence
            snr_rho0 = float(cfg.get("snr_weight_rho0", 0))
            snr_wmax = float(cfg.get("snr_weight_max", 1.0))
            if snr_rho0 > 0 and snr_wmax > 1.0:
                dL_ref = np.exp(0.5 * (np.log(lo) + np.log(hi)))
                rho_approx = snr_rho0 * (dL_ref / dL)
                w_snr = np.minimum(1.0 + (rho_approx / snr_rho0) ** 2, snr_wmax)
                log_r -= np.log(w_snr)

            log_r[~in_bounds] = -np.inf



        log_r -= log_r[np.isfinite(log_r)].max()
        w_r = np.exp(log_r)
        w_r /= w_r.sum()
        ess_swap = 1.0 / np.sum(w_r ** 2) / len(w_r) * 100
        rng_sir = np.random.default_rng(int(seed))
        idx = rng_sir.choice(len(w_r), size=n_samples, replace=True, p=w_r)
        log_Zr = np.log(np.mean(np.exp(log_r)))

        samples_phys = samples_phys[idx]
        log_q = log_q[idx] + log_r[idx] - log_Zr
        print(f"  prior-swap SIR: ESS={ess_swap:.1f}% of {M}, kept {n_samples}"
              + (f" (incl SNR-weight correction rho0={snr_rho0})" if snr_rho0 > 0 else ""))
        swap_info = dict(prior_swap_ess=float(ess_swap),
                         prior_swap_oversample=int(oversample),
                         prior_swap_n_draw=int(M))
    else:
        swap_info = {}



    print(f"Flow internal samples gen time: {time.perf_counter() - _t0:.3f}s")

    # model stays resident on the GPU across events — the per-event
    # .to("cpu") + empty_cache round trip costs ~1-2 s for nothing
    return samples_phys, log_q, swap_info