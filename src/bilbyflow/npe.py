"""
bilbyflow.npe - an NPE model: an embedding + a flow.

    npe = NPE(embedding, flow)

    npe.embed(x)                  # raw data      -> context
    npe.flow_log_prob(theta, ctx) # context       -> log q(theta|ctx)
    npe.flow_sample(n, ctx)       # context       -> samples
    npe.log_prob(theta, x)        # raw data      -> log q(theta|x)
    npe.sample(n, x)              # raw data      -> samples

The embedding owns standardisation (see nn.embedding.StrainEmbedding), so
`x` here is RAW data - the model is the whole path from strain to posterior.
theta is in the flow's own (normalised) coordinates.

An nn.Module: npe.parameters() is everything, so the trainer optimises the
embedding and the flow together.
"""

import torch
import torch.nn as nn

__all__ = ["NPE"]


class NPE(nn.Module):

    def __init__(self, embedding, flow):
        super().__init__()
        ctx = getattr(embedding, "context_dim", embedding.output_dim)
        if flow.context_dim != ctx:
            raise ValueError(
                f"context mismatch: {type(embedding).__name__} outputs {ctx} "
                f"(embed {embedding.output_dim} + amp "
                f"{getattr(embedding, 'amp_dim', 0)}), "
                f"{type(flow).__name__} expects {flow.context_dim}")

        self.embedding = embedding
        self.flow = flow
        self._pinned = None

    # ── the five methods ─────────────────────────────────────────────────────

    def embed(self, x):
        """Raw data -> context vector. Returns the pinned context if set."""

        if self._pinned is not None:
            # Send the pinned context to whatever device x is on
            return self._pinned.to(x.device).expand(x.shape[0], -1)
        return self.embedding(x)

    def flow_log_prob(self, theta, context):
        if context.shape[0] == 1 and theta.shape[0] > 1:
            context = context.expand(theta.shape[0], -1)   # one x, many theta
        return self.flow.log_prob(theta, context)
    
    def flow_sample(self, n, context):
        return self.flow.sample(n, context)

    def log_prob(self, theta, x):
        return self.flow_log_prob(theta, self.embed(x))

    def sample(self, n, x):
        return self.flow_sample(n, self.embed(x))

    # ── training + inference conveniences ────────────────────────────────────

    def loss(self, theta, x):
        """Training objective: negative log-likelihood of theta given x."""
        return -self.log_prob(theta, x)

    def pin(self, x):
        """Embed x once; embed() then reuses it. One event = one embedding
        forward, however many times you sample or evaluate."""
        with torch.no_grad():
            self._pinned = self.embedding(x)

    def unpin(self):
        self._pinned = None

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path, map_location="cpu"):
        """Load weights into this (already constructed) model — save() stores
        weights only, so rebuild the embedding and flow with the same cfg,
        standardiser and constructor arguments first, then call load()."""
        self.load_state_dict(torch.load(path, map_location=map_location))
        return self