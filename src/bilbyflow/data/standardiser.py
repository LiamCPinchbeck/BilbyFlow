"""
bilbyflow.data.standardiser — input/target standardisation.

Owns z-scoring for x (strain block + trailing amp block), theta, and the aux
targets. 
Persisted to standardiser.pkl and reloaded by both training and the
reweighting scripts; the reweighter reads only theta_* and the log-coordinate
flags, never the aux fields.

Works as a kind of coordinate-system for the embedding's understanding of the data.
"""

import numpy as np
import torch

from .canonical import AMP_NAMES, N_AMP
from ..nn.aux_head import AUX_NAMES, N_AUX

__all__ = ["Standardiser", "check_aux_stats", "check_amp_stats"]


class Standardiser:
    def __init__(self, x_sample, theta_sample, strain_dim=None,
                 psd_log_mean=None, psd_log_std=None, psd_conditioning=False,
                 dL_log=False, dL_index=None, aux_sample=None,
                 Mc_log=False, Mc_index=None, amp_dim=0):
        
        self.dL_log = bool(dL_log)
        self.dL_index = dL_index
        self.Mc_log = bool(Mc_log)
        self.Mc_index = Mc_index


        self.amp_dim = int(amp_dim)

        self.psd_conditioning = bool(psd_conditioning)

        self.strain_dim = int(strain_dim) if strain_dim is not None else int(x_sample.shape[-1])


        xs = x_sample[..., :self.strain_dim]
        self.x_mean = xs.mean()
        self.x_std = xs.std().clamp(min=1e-10)

    
        if self.amp_dim > 0:
            xa = x_sample[..., -self.amp_dim:]
            self.amp_mean = xa.mean(dim=0)
            self.amp_std = xa.std(dim=0).clamp(min=1e-10)
            self.amp_names = list(AMP_NAMES)
        else:
            self.amp_mean = None
            self.amp_std = None
            self.amp_names = None


        self.theta_mean = theta_sample.mean(dim=0)
        self.theta_std = theta_sample.std(dim=0).clamp(min=1e-10)

        self.psd_log_mean = psd_log_mean
        self.psd_log_std = psd_log_std

        if aux_sample is not None:
            self.aux_mean = aux_sample.mean(dim=0)
            self.aux_std = aux_sample.std(dim=0).clamp(min=1e-10)
            self.aux_names = list(AUX_NAMES)
        else:
            self.aux_mean = None
            self.aux_std = None
            self.aux_names = None

    def update_x(self, x_sample):
        xs = x_sample[..., :getattr(self, "strain_dim", x_sample.shape[-1])]
        self.x_mean = xs.mean()
        self.x_std = xs.std().clamp(min=1e-10)

    def update_theta(self, theta_sample):
        self.theta_mean = theta_sample.mean(dim=0)
        self.theta_std = theta_sample.std(dim=0).clamp(min=1e-10)


    def normalise_x(self, x):
        sd = getattr(self, "strain_dim", x.shape[-1])
        if x.shape[-1] <= sd:
            return (x - self.x_mean) / self.x_std
        strain = (x[..., :sd] - self.x_mean) / self.x_std
        ad = int(getattr(self, "amp_dim", 0) or 0)
        if ad > 0 and x.shape[-1] > sd + ad:
            mid = x[..., sd:x.shape[-1] - ad]                 # PSD ctx (pre-standardised)
            ampz = (x[..., -ad:] - self.amp_mean) / self.amp_std
            return torch.cat([strain, mid, ampz], dim=-1)
        return torch.cat([strain, x[..., sd:]], dim=-1)

    def normalise_theta(self, theta):
        return (theta - self.theta_mean) / self.theta_std



    def unnormalise_theta(self, t):
        return t * self.theta_std + self.theta_mean

    def normalise_aux(self, aux):
        return (aux - self.aux_mean) / self.aux_std


    def get_normalised_prior_bounds(self, prior_low, prior_high):
        lo = (prior_low - self.theta_mean) / self.theta_std
        hi = (prior_high - self.theta_mean) / self.theta_std
        return torch.min(lo, hi), torch.max(lo, hi)


def check_aux_stats(std):
    """Fail fast if the standardiser's aux stats don't match this code's AUX_NAMES."""

    am = getattr(std, "aux_mean", None)
    if am is None:
        raise SystemExit("aux_supervision=True but standardiser.pkl has no aux "
                         "statistics -- regenerate it (do not reuse a pre-v4.1 pickle).")

    names = getattr(std, "aux_names", None)

    if len(am) != N_AUX or (names is not None and list(names) != list(AUX_NAMES)):
        raise SystemExit(f"standardiser aux stats ({len(am)} targets: {names}) do not "
                         f"match this code ({N_AUX}: {AUX_NAMES}) -- regenerate "
                         "standardiser.pkl (v4.2 dropped the absolute-phase targets).")


def check_amp_stats(std, cfg):
    """Fail fast on a stale standardiser when amp_context is toggled (x dim
    changed -- val/test pickles must be regenerated too)."""

    want = N_AMP if bool(cfg.get("amp_context", False)) else 0
    have = int(getattr(std, "amp_dim", 0) or 0)

    if want != have:
        raise SystemExit(f"amp_context={'ON' if want else 'off'} but "
                         f"standardiser.pkl has amp_dim={have} -- regenerate "
                         "standardiser.pkl AND val/test pickles (x dim changed).")