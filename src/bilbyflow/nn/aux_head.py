"""
bilbyflow.nn.aux_head — auxiliary supervision head (v4.2: 10 sliced targets).

A small MLP on a SLICE of the shared embedding regresses interpretable
summaries of the NOISELESS whitened signal. 

Purpose: 
shape the embedding early in training, when the NLL gradient is weak, 
so it does not collapse onto a low-rank subspace. 

The loss is annealed to zero within each curriculum stage and the
head is NEVER used at inference — best_state holds the embedding and flow
only (the head is stored under a separate checkpoint key), so the
reweighting path never imports this module.

Per detector I in {H1, L1}, from the noiseless whitened signal h_w:
    ln_rho_I    = ln sqrt(sum_k |h_w,k|^2)      optimal matched-filter SNR
    t_peak_I    = argmax_t |z_I(t)| / sr        arrival-time proxy; z = analytic TD
    ln_fcent_I  = ln[ sum f|h_w|^2 / sum|h_w|^2 ]  spectral centroid (chirp-mass proxy)

cross-detector:
    dt_HL        exact, from the sky draw (the flow's own sky coordinate)
    ln_rho_ratio ln(rho_H1 / rho_L1)            amplitude ratio
    cos_dphi, sin_dphi  cos/sin(phi_H1 - phi_L1)  phase difference

The absolute peak phases phi_H1, phi_L1 are computed internally only to
form the difference; they are not targets.

--------------------------------------------------------------------------
Scope Note. 
compute_aux_summaries builds targets from the whitened FD signal per detector; 
it lives here because it defines what the head regresses. 

The Standardiser owns aux z-scoring (aux_mean/aux_std) and check_aux_stats;
those live with the standardiser in bilbyflow.data.standardiser. 

This module is import-light on purpose: numpy + torch only.
--------------------------------------------------------------------------
"""

import numpy as np
import torch
import torch.nn as nn

__all__ = [
    "AUX_NAMES", "N_AUX",
    "AuxHead",
    "compute_aux_summaries",
    "aux_anneal_lambda",
]


# ── target set (v4.2: 10 dims; order is load-bearing — targets built positionally) ──

AUX_NAMES = ["ln_rho_H1", "ln_rho_L1",
             "t_peak_H1", "t_peak_L1",
             "ln_fcent_H1", "ln_fcent_L1",
             "dt_HL",
             "ln_rho_ratio", 
             "cos_dphi", "sin_dphi"]
N_AUX = len(AUX_NAMES)


# ── head ─────────────────────────────────────────────────────────────────────

class AuxHead(nn.Module):
    """Small MLP regressing the z-scored aux summaries from a SLICE of the
    shared embedding features (first ``in_dim`` dims). LayerNorm keeps it
    batch-size agnostic. Training-only scaffold.

        in_dim is aux_k = min(cfg["aux_n_channels"], feat_dim).

    NOTE (22/08/2026): the embeddings now fuse the PSD encoding BEFORE their
    head, so every context dim mixes strain and PSD information — slicing
    [0:aux_k] no longer isolates strain features, and aux gradients reach the
    PSD encoder. Previously the context was [strain(512) || psd_enc(64)] and
    the slice was strain-only. If the old confinement is wanted back, expose
    the pre-fusion strain features from the embedding and tap those instead.
    """
    def __init__(self, in_dim, n_aux=N_AUX, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden),
            nn.ELU(), nn.Linear(hidden, n_aux))

    def forward(self, f):
        return self.net(f)


# ── targets ──────────────────────────────────────────────────────────────────

def _analytic_td(sig_w_full, n_td):
    """One-sided whitened FD signal -> complex analytic TD signal z(t)
    (one complex iFFT; overall scaling is irrelevant for argmax/angle)."""
    spec = np.zeros(n_td, dtype=complex)
    n_freq = sig_w_full.shape[0]
    spec[:n_freq] = sig_w_full
    spec[1:n_td // 2] *= 2.0            # analytic-signal doubling (not DC/Nyquist)
    return np.fft.ifft(spec)


def compute_aux_summaries(sig_w_by_det, freq_array, freq_mask, n_td, sr, dt_HL):
    """10-dim aux target vector from the NOISELESS whitened signals.

    v3.1 convention: whitened noise has per-quadrature variance 1, so
    rho = sqrt(sum |h_w|^2) is the optimal matched-filter SNR up to the
    convention constant (absorbed by z-scoring). Absolute peak phases are
    computed internally only to form the physical H1-L1 difference; phi_c
    is a marginalised uniform nuisance and unidentifiable from x, so the
    absolute phase is not a target.

    sig_w_by_det : {"H1": h_w_full, "L1": h_w_full}  full-grid whitened FD signal
    freq_array   : full frequency grid
    freq_mask    : in-band boolean mask over the full grid
    n_td, sr     : TD length and sampling rate
    dt_HL        : exact Hanford-Livingston delay from the sky draw
    Returns      : float32 array in AUX_NAMES order.
    """
    out, phis = {}, {}
    for det in ["H1", "L1"]:
        hw = sig_w_by_det[det]
        hwm = hw[freq_mask]
        p = np.abs(hwm) ** 2
        ptot = float(p.sum())
        rho = np.sqrt(max(ptot, 1e-30))
        fcent = float((freq_array[freq_mask] * p).sum() / max(ptot, 1e-30))
        z = _analytic_td(hw, n_td)
        k = int(np.argmax(np.abs(z)))
        phis[det] = float(np.angle(z[k]))
        out[f"ln_rho_{det}"] = np.log(rho)
        out[f"t_peak_{det}"] = k / sr
        out[f"ln_fcent_{det}"] = np.log(max(fcent, 1e-3))
    dphi = phis["H1"] - phis["L1"]
    out["dt_HL"] = float(dt_HL)
    out["ln_rho_ratio"] = out["ln_rho_H1"] - out["ln_rho_L1"]
    out["cos_dphi"] = np.cos(dphi)
    out["sin_dphi"] = np.sin(dphi)
    return np.array([out[n] for n in AUX_NAMES], dtype=np.float32)


# ── anneal schedule ──────────────────────────────────────────────────────────

def aux_anneal_lambda(epoch_in_stage, stage_epochs, lam0, anneal_frac):
    """Per-stage aux weight: lambda0 linearly annealed to 0 by
    anneal_frac * stage_epochs, then held at 0.

    Matches the training driver:
        n_anneal = max(1, int(anneal_frac * stage_epochs))
        lam = lam0 * max(0.0, 1.0 - epoch_in_stage / n_anneal)
    """
    if lam0 <= 0.0:
        return 0.0
    n_anneal = max(1, int(anneal_frac * int(stage_epochs)))
    return float(lam0 * max(0.0, 1.0 - epoch_in_stage / n_anneal))