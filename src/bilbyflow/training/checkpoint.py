"""
bilbyflow.training.checkpoint — does checkpointing.

best_state is FLOW-ONLY (to save space there is no aux head, no embedding-consistency state), 
so a checkpoint SHOULD BE (user experience may vary) directly loadable by the reweighting path via
nn.flow.reconstruct_from_checkpoint. aux_names / aux_n_channels are recorded
so a resumed run can decide whether the saved aux head is still compatible.
"""

import torch

from ..nn.aux_head import AUX_NAMES

__all__ = ["save_checkpoint", "load_checkpoint"]


def save_checkpoint(path, epoch, density_estimator, optimiser, scheduler,
                    best_val_loss, best_state, train_losses, val_losses, train_times,
                    stage_idx=0, val_cap_losses=None, aux_head=None,
                    aux_losses=None, cons_losses=None, aux_n_channels=None):
    torch.save({
        "epoch": epoch,
        "stage_idx": stage_idx,
        "model_state_dict": density_estimator.state_dict(),
        "optimiser_state_dict": optimiser.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_loss": best_val_loss,
        "best_state": best_state,                      # flow-only (reweighting-compatible)
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