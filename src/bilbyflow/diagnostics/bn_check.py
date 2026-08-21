"""
bilbyflow.diagnostics.bn_check — BatchNorm sanity for the embedding.

The embedding net carries BatchNorm layers (conv/ResNet stem, currently 21/08/2026); 
the flow transforms do not. 


Two failure modes this script tries to catch:

  1. train-vs-eval log_prob gap. 
        If the BN running statistics are stale or
     mis-calibrated for the deployment input distribution, de.log_prob gives a
     different answer in train() mode (per-batch stats) than in eval() mode
     (running stats). A large gap means eval-time densities — the ones the
     reweighter uses — are not the ones the loss optimised. custom_train_npe
     recalibrates BN at the end for exactly this reason; this check verifies it
     took.

  2. degenerate running_var. 
        A running_var driven to ~0 (or non-finite) in any
     BN layer can make that channel's eval-time output explode; 
     running_mean/var summaries should show it before it corrupts inference.

Neither routine trains or mutates weights (running_var inspection is read-only;
the log_prob gap toggles train/eval but restores the original mode).
"""

import numpy as np
import torch

__all__ = ["bn_layers", "bn_running_stats_report", "train_eval_logprob_gap"]


def bn_layers(module):
    """All BatchNorm1d/2d submodules of `module`, as (name, layer) pairs."""

    return [(n, m) for n, m in module.named_modules()
            if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d))]




def bn_running_stats_report(de, verbose=True):
    """Summarises every BN layer's running_mean / running_var. 
    
    Flags layers whose running_var has collapsed toward zero or gone non-finite (eval-time
    blow-up risk). 
    
    Returns a list of per-layer dicts.
    """


    embed = getattr(de, "embedding_net", de)

    rows = []
    for name, m in bn_layers(embed):
        rv = m.running_var.detach().cpu().numpy() if m.running_var is not None else None
        rm = m.running_mean.detach().cpu().numpy() if m.running_mean is not None else None
    
        if rv is None:
            rows.append(dict(layer=name, note="track_running_stats=False"))
            continue

        vmin = float(np.min(rv))

        finite = bool(np.all(np.isfinite(rv)) and np.all(np.isfinite(rm)))

        bad = (not finite) or vmin < 1e-6

        rows.append(dict(layer=name, var_min=vmin, var_max=float(np.max(rv)),
                         mean_absmax=float(np.max(np.abs(rm))),
                         finite=finite, degenerate=bad))
    
        if verbose:
            flag = "  ** DEGENERATE" if bad else ""
            print(f"  [bn] {name:40s} var[min,max]=[{vmin:.2e},{np.max(rv):.2e}] "
                  f"|mean|max={np.max(np.abs(rm)):.2e}{flag}")
    
    if verbose:
        n_bad = sum(r.get("degenerate", False) for r in rows)
        print(f"  [bn] {len(rows)} BN layers, {n_bad} degenerate")
    
    return rows


def train_eval_logprob_gap(de, x_norm, theta_norm, device="cpu", chunk=512):
    """Mean |log_prob_train - log_prob_eval| over the given NORMALISED inputs.

    A well-calibrated model has a small gap: per-batch BN stats (train mode)
    and running stats (eval mode) agree on the deployment distribution. 
    
    A large gap means eval-time densities differ from what training optimised 
    — rerun the BN recalibration (training.trainer.recalibrate_bn / the tail of
    custom_train_npe) with representative inputs.

    Returns (mean_abs_gap, mean_train_nll, mean_eval_nll). 
    Restores whatever train/eval mode `de` was in.
    """


    was_training = de.training

    de.to(device)
    x_norm = x_norm.to(device)
    theta_norm = theta_norm.to(device)


    def _lp():
        out = []
        for i in range(0, len(x_norm), chunk):
            lp = de.log_prob(theta_norm[i:i+chunk], condition=x_norm[i:i+chunk])
            out.append(lp.reshape(-1).detach().cpu())
        return torch.cat(out)


    with torch.no_grad():
        de.train()
        lp_tr = _lp()
        de.eval()
        lp_ev = _lp()


    if was_training:
        de.train()
    else:
        de.eval()

    gap = float((lp_tr - lp_ev).abs().mean())

    return gap, float(-lp_tr.mean()), float(-lp_ev.mean())