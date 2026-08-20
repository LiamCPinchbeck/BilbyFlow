"""
bilbyflow.nn — neural network components.

Public surface:
    build_resnet_embedding, Conv1dEmbedding, Conv1dResNetEmbedding, ResNetEmbedding
    AuxHead, AUX_NAMES, N_AUX, compute_aux_summaries, aux_anneal_lambda

flow.py (build_density_estimator, FeatureCache, PSDConditionedEmbedding,
reconstruct_from_checkpoint) is intended to live alongside these but is not
part of this split.
"""

from .embedding import (
    build_resnet_embedding,
    Conv1dEmbedding,
    Conv1dResNetEmbedding,
    ResNetEmbedding,
)
from .aux_head import (
    AuxHead,
    AUX_NAMES,
    N_AUX,
    compute_aux_summaries,
    aux_anneal_lambda,
)

__all__ = [
    "build_resnet_embedding",
    "Conv1dEmbedding",
    "Conv1dResNetEmbedding",
    "ResNetEmbedding",
    "AuxHead",
    "AUX_NAMES",
    "N_AUX",
    "compute_aux_summaries",
    "aux_anneal_lambda",
]