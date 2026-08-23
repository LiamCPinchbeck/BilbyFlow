"""
bilbyflow.training.checkpoint — does checkpointing.

best_state is the WHOLE NPE state dict (embedding + flow), so a checkpoint is
self-sufficient for the reweighting path. To resurrect one:

    embedding = Conv1dResNetEmbedding(cfg, std)     # same cfg + standardiser
    flow      = NSF(theta_dim, embedding.context_dim, ...)   # same kwargs
    npe       = NPE(embedding, flow)
    npe.load_state_dict(torch.load(path)["best_state"])

The aux head is stored separately (aux_head_state_dict) and is never needed at
inference. aux_names / aux_n_channels are recorded so a resumed run can decide
whether the saved aux head is still compatible.
"""

import torch

from ..nn.aux_head import AUX_NAMES

__all__ = ["save_checkpoint", "load_checkpoint"]


def save_checkpoint(path, epoch, npe, optimiser, scheduler,
                    best_val_loss, best_state, train_losses, val_losses, train_times,
                    stage_idx=0, val_cap_losses=None, aux_head=None,
                    aux_losses=None, cons_losses=None, aux_n_channels=None):
    torch.save({
        "epoch": epoch,
        "stage_idx": stage_idx,
        "model_state_dict": npe.state_dict(),
        "optimiser_state_dict": optimiser.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_loss": best_val_loss,
        "best_state": best_state,                      # whole NPE (embedding + flow)
        "aux_head_state_dict": (aux_head.state_dict() if aux_head is not None else None),
        "aux_n_channels": aux_n_channels,
        "aux_names": list(AUX_NAMES),
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_cap_losses": val_cap_losses if val_cap_losses is not None else [],
        "aux_losses": aux_losses if aux_losses is not None else [],
        "cons_losses": cons_losses if cons_losses is not None else [],
        "train_times": train_times,
    }, path)


def load_checkpoint(path, device):
    return torch.load(path, map_location=device, weights_only=False)