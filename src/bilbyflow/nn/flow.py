"""
bilbyflow.nn.flow — the conditional flow: context assembly, the shared-feature
tap, and (re)construction of the density estimator.

This module consolidates code that was previously duplicated across the
training script and the reweighting script:

  * FeatureCache        — training had it; reweighting had a cut-down _FeatureCache.
  * PSDConditionedEmbedding — identical two-branch context embedding.
  * build_density_estimator     — training-time build (from a live dataset).
  * reconstruct_from_checkpoint — inference-time build (from cfg + standardiser,
                                  loading best_state with the fd_branch<->fd_stem
                                  key remap for older checkpoints).

Both build paths now share ONE FeatureCache and ONE PSDConditionedEmbedding,
so the aux head, the JEPA slice, and the reweighter all see the same context
layout: [strain_emb (embedding_output_dim) || psd_enc || amp].
"""

import numpy as np
import torch
import torch.nn as nn

from sbi.utils import BoxUniform
from sbi.neural_nets import posterior_nn
from sbi.inference.posteriors import DirectPosterior

from .embedding import build_resnet_embedding
from ..io.config import grid_quantities

__all__ = [
    "FeatureCache", "PSDConditionedEmbedding", "build_embedding_net",
    "build_density_estimator", "reconstruct_from_checkpoint",
]


# ── MLP fallback embedding (embedding_type == "mlp") ─────────────────────────

def build_embedding_net(input_dim, layer_widths):
    layers, dims = [], [input_dim] + list(layer_widths)
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        layers.append(nn.ELU())
    return nn.Sequential(*layers)


# ── shared-feature tap ───────────────────────────────────────────────────────

class FeatureCache(nn.Module):
    """Wrapper caching the embedding's last output so the aux head
    (and JEPA consistency) can read the SHARED features computed inside
    de.log_prob -- one embedding forward per batch, no second pass.

    The reweighting path does not read .features; it only needs the wrapper so
    the checkpoint's ".net." key prefix matches.
    """
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.features = None

    def forward(self, x):
        f = self.net(x)
        self.features = f
        return f


# ── PSD-conditioned two-branch context ───────────────────────────────────────

class PSDConditionedEmbedding(nn.Module):
    """Context layout: [strain_embedding || psd_encoder || amp]. The aux slice
    (first aux_n_channels dims) and the JEPA strain slice are both inside the
    strain embedding; the amp block sits at the END (raw pass-through, already
    z-scored)."""
    def __init__(self, strain_embedding, strain_dim, psd_dim, cfg, amp_dim=0):
        super().__init__()

        self.strain_embedding = strain_embedding
        self.strain_dim = int(strain_dim)
        self.psd_dim = int(psd_dim)
        self.amp_dim = int(amp_dim)

        hidden = list(cfg.get("psd_encoder_hidden", [512, 256, 128]))

        out = int(cfg.get("psd_encoder_out", 64))

        dims = [self.psd_dim] + hidden + [out]

        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.LayerNorm(dims[i + 1]))
                layers.append(nn.ELU())
    
        self.psd_encoder = nn.Sequential(*layers)


    def forward(self, x):
        strain = x[..., :self.strain_dim]
        ad = int(getattr(self, "amp_dim", 0) or 0)     # old pickles lack the attr
        if ad > 0:
            psd = x[..., self.strain_dim:-ad]
            amp = x[..., -ad:]
            return torch.cat([self.strain_embedding(strain),
                              self.psd_encoder(psd), amp], dim=-1)
        psd = x[..., self.strain_dim:]
        return torch.cat([self.strain_embedding(strain), self.psd_encoder(psd)], dim=-1)


# ── strain-embedding dispatch (shared by both build paths) ───────────────────

def _strain_embed(dim, cfg):
    if cfg.get("embedding_type", "mlp") != "mlp":
        return build_resnet_embedding(dim, cfg)
    return build_embedding_net(dim, cfg["embedding_layers"])


def _assemble_embedding(cfg, strain_dim, psd_dim, amp_dim, input_dim):
    """Build the (possibly PSD-conditioned) embedding, wrapped in FeatureCache."""
    if bool(cfg.get("psd_conditioning", False)):
        strain_emb = _strain_embed(int(strain_dim), cfg)
        emb = PSDConditionedEmbedding(strain_emb, strain_dim, psd_dim, cfg,
                                      amp_dim=int(amp_dim))
    else:
        if int(amp_dim) > 0:
            raise ValueError("amp_context requires psd_conditioning")
        emb = _strain_embed(int(input_dim), cfg)
    return FeatureCache(emb)


# ── training-time build (from a live dataset) ────────────────────────────────

def build_density_estimator(dataset, std, cfg, DEVICE):
    """Returns (density_estimator, feature_cache). The FeatureCache wraps the
    FULL embedding so the aux head sees exactly the context the flow conditions
    on."""

    sample_x, sample_theta, _, _, _ = dataset[0]
    input_dim = sample_x.shape[0]


    cache = _assemble_embedding(
        cfg, dataset.strain_dim, dataset.psd_dim,
        int(getattr(dataset, "amp_dim", 0)), input_dim)



    flow_kwargs = dict(
        num_transforms=int(cfg["num_transforms"]),
        hidden_features=cfg["hidden_features"],
        num_bins=int(cfg.get("num_bins", 16)),
        z_score_x="none", z_score_theta="none",
    )


    flow_dropout = float(cfg.get("flow_dropout", 0.0))
    if flow_dropout > 0:
        flow_kwargs["dropout_probability"] = flow_dropout



    build_fn = posterior_nn("nsf", embedding_net=cache, **flow_kwargs)


    sample_theta_norm = std.normalise_theta(sample_theta.unsqueeze(0))
    sample_x_norm = std.normalise_x(sample_x.unsqueeze(0))


    cache.eval()   # sbi probes with batch-1; conv BatchNorm rejects batch-1 in train mode
    density_estimator = build_fn(sample_theta_norm, sample_x_norm)
    cache.train()


    return density_estimator.to(DEVICE), cache


# ── inference-time build (from cfg + standardiser + checkpoint) ──────────────

def reconstruct_from_checkpoint(ckpt_path, cfg, std):
    """Rebuild the DirectPosterior from a training checkpoint's best_state
    (flow-only, reweighting-compatible). Handles the fd_branch<->fd_stem key
    remap for checkpoints saved before the attribute rename."""

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "best_state" not in ckpt:
        raise RuntimeError(f"No best_state in {ckpt_path}. Keys: {list(ckpt.keys())}")

    g = grid_quantities(cfg)

    per_det = 2 * g["n_masked"] + g["n_td"]
    strain_dim = 2 * per_det
    n_params = len(cfg["inferred_parameters"])


    psd_cond = (getattr(std, "psd_conditioning", False)
                and getattr(std, "psd_log_mean", None) is not None)
    amp_dim = int(getattr(std, "amp_dim", 0) or 0)


    if psd_cond:
        psd_dim = 2 * g["n_masked"]
        input_dim = strain_dim + psd_dim + amp_dim
    else:
        if amp_dim > 0:
            raise RuntimeError("standardiser has amp_dim>0 but PSD conditioning off")
        psd_dim = 0
        input_dim = strain_dim



    embedding = _assemble_embedding(cfg, strain_dim, psd_dim, amp_dim, input_dim)



    flow_kwargs = dict(
        num_transforms=int(cfg["num_transforms"]),
        hidden_features=cfg["hidden_features"],
        z_score_x="none", z_score_theta="none",
    )


    if cfg.get("num_bins"):
        flow_kwargs["num_bins"] = int(cfg["num_bins"])
    build_fn = posterior_nn("nsf", embedding_net=embedding, **flow_kwargs)



    dummy_x = std.normalise_x(torch.zeros(1, input_dim))
    dummy_theta = std.normalise_theta(torch.zeros(1, n_params))
    embedding.eval()
    density_estimator = build_fn(dummy_theta, dummy_x)



    def _load(sd):
        miss, unexp = density_estimator.load_state_dict(sd, strict=False)
        bad_miss = [k for k in miss if not k.endswith("num_batches_tracked")]
        bad_unexp = [k for k in unexp if not k.endswith("num_batches_tracked")]
        return bad_miss, bad_unexp

    state = ckpt["best_state"]
    bad_miss, bad_unexp = _load(state)


    if bad_miss or bad_unexp:
        remapped = {k.replace(".fd_branch.", ".fd_stem.").replace(".td_branch.", ".td_stem."): v
                    for k, v in state.items()}
        bad_miss, bad_unexp = _load(remapped)
        if bad_miss or bad_unexp:
            raise RuntimeError(
                "Reconstructed model does not match the checkpoint.\n"
                f"  missing: {bad_miss[:6]}{' ...' if len(bad_miss) > 6 else ''}\n"
                f"  unexpected: {bad_unexp[:6]}{' ...' if len(bad_unexp) > 6 else ''}")

        
    density_estimator.eval()

    print(f"  Loaded best_state from epoch {ckpt['epoch']} "
          f"(best_val_loss={ckpt['best_val_loss']:.4f})")

    prior = _make_normalised_prior(cfg, std)

    return DirectPosterior(posterior_estimator=density_estimator, prior=prior)


def _make_normalised_prior(cfg, std):
    """BoxUniform over the analysis prior in normalised theta coordinates,
    with the sky reparam (ra->dt_HL, dec->phi_det) and log-coordinate edges."""

    from ..coordinates.sky import MAX_DT_HL   # provided by coordinates/sky.py (sky_coord_utils_b)


    lows = [cfg["priors"][p]["min"] for p in cfg["inferred_parameters"]]
    highs = [cfg["priors"][p]["max"] for p in cfg["inferred_parameters"]]
    prior_low = torch.tensor(lows, dtype=torch.float32)
    prior_high = torch.tensor(highs, dtype=torch.float32)



    if (bool(getattr(std, "dL_log", False))
            or str(cfg.get("dL_param", "linear")).lower() == "log") \
            and "luminosity_distance" in cfg["inferred_parameters"]:
        di = cfg["inferred_parameters"].index("luminosity_distance")
        prior_low[di] = float(np.log(float(prior_low[di])))
        prior_high[di] = float(np.log(float(prior_high[di])))



    if (bool(getattr(std, "Mc_log", False))
            or str(cfg.get("Mc_param", "linear")).lower() == "log") \
            and "chirp_mass" in cfg["inferred_parameters"]:
        mi = cfg["inferred_parameters"].index("chirp_mass")
        prior_low[mi] = float(np.log(float(prior_low[mi])))
        prior_high[mi] = float(np.log(float(prior_high[mi])))



    if "ra" in cfg["inferred_parameters"] and "dec" in cfg["inferred_parameters"]:
        ra_idx = cfg["inferred_parameters"].index("ra")
        dec_idx = cfg["inferred_parameters"].index("dec")
        prior_low[ra_idx] = -MAX_DT_HL
        prior_high[ra_idx] = MAX_DT_HL
        prior_low[dec_idx] = 0.0
        prior_high[dec_idx] = 2 * np.pi



    lo, hi = std.get_normalised_prior_bounds(prior_low, prior_high)
    return BoxUniform(low=lo, high=hi)