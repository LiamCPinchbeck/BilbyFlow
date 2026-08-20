"""
bilbyflow.training.losses — the separable pieces of the training objective.

As part of the code we allow for the use of a VICReg-like and SNR loss regularization thingies
(yes very technical).
This script was primarily made just to keep track of this regularization and basic dropout.


In the case with the VICReg the full loss becomes
L = NLL + lambda(e)*aux_MSE + lam_c*consistency + VICReg 
which is assembled in training.trainer, 
because the aux and JEPA terms need the live FeatureCache and per-row bookkeeping. 

The self-contained pieces below are factored out so they can be unit-tested and reused 
(as we try to do with most features):
  * snr_weights_from_aux   — SNR-tilt weights w for the NLL, from aux ln_rho.
  * snr_weights_from_amp    — same weights recovered from the val-set amp block.
  * vicreg_penalty          — variance + covariance regulariser on the features.
  * set_dropout_p           — final-stage sharpening helper.
"""

import torch

__all__ = [
    "snr_weights_from_aux", "snr_weights_from_amp",
    "vicreg_penalty", "set_dropout_p",
]

# Done some A/B testing with this, and it is unlikely to help training.
    # leaving it in for later possible modification and version consistency.
def snr_weights_from_aux(aux_raw, rho0, w_max=10.0):
    """Per-example NLL weight w = 1 + rho_net^2 / rho0^2, mean-normalised and
    capped. aux_raw[:, 0:2] = ln_rho_{H1, L1} (the first two aux targets).
    Returns ones when rho0 <= 0."""
    n = aux_raw.shape[0]
    if rho0 <= 0.0:
        return torch.ones(n)
    rho2 = torch.exp(2 * aux_raw[:, 0]) + torch.exp(2 * aux_raw[:, 1])
    w = 1.0 + rho2 / rho0 ** 2
    return (w / w.mean()).clamp(max=float(w_max))


def snr_weights_from_amp(x_raw, amp_dim, rho0, w_max=10.0):
    """Same SNR tilt recovered from the RAW (pre-normalise) amp block of a
    fixed val set: ln_pow ~ ln(noise_floor + rho^2), floor = median power.
    Returns None when the weighting is off or there is no amp block."""
    if rho0 <= 0.0 or amp_dim == 0:
        return None
    amp = x_raw[:, -amp_dim:]
    pow_H, pow_L = torch.exp(amp[:, 0]), torch.exp(amp[:, 1])
    floor_H, floor_L = pow_H.median(), pow_L.median()
    rho2 = (pow_H - floor_H).clamp(min=0) + (pow_L - floor_L).clamp(min=0)
    w = 1.0 + rho2 / rho0 ** 2
    return (w / w.mean()).clamp(max=float(w_max))


def vicreg_penalty(features, gamma, lam_var, lam_cov):
    """VICReg variance + covariance terms on a feature block (the strain slice
    of the shared context). Same graph as the NLL, so it adds to one backward.
    Returns (penalty, v_loss.detach(), c_loss.detach())."""
    f = features
    fc = f - f.mean(dim=0)
    std_f = torch.sqrt(fc.var(dim=0) + 1e-4)
    v_loss = torch.relu(gamma - std_f).mean()
    cov = (fc.T @ fc) / (f.shape[0] - 1)
    c_loss = (cov.pow(2).sum() - cov.diagonal().pow(2).sum()) / f.shape[1]
    penalty = lam_var * v_loss + lam_cov * c_loss
    return penalty, v_loss.detach(), c_loss.detach()


def set_dropout_p(module, p):
    """Set every nn.Dropout probability in `module` (final-stage sharpening).
    Returns the number of modules touched."""
    n = 0
    for m in module.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = float(p)
            n += 1
    return n