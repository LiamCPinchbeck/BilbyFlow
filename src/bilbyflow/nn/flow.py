"""
bilbyflow.nn.flow — the flow: context in, density out.

A flow never sees the raw data. It sees the embedding's output vector and
models p(theta | context). The built-ins are thin wrappers around zuko, which
supplies the architectures; the interface is unchanged from the nflows
version, so nothing else in the package needs editing.

    flow = NSF(theta_dim=12, context_dim=576, num_transforms=64,
               hidden_features=512, num_bins=24)
    flow = MAF(theta_dim=12, context_dim=576)

    flow.log_prob(theta, context)   # (B, T), (B, D) -> (B,)
    flow.sample(n, context)         # (1, D) -> (n, T)

Extra keywords go straight to zuko, so anything it supports is reachable
without touching this file:  NSF(..., residual=True), SOSPF(..., degree=4).
Head over to the Zuko documentation on the implementations 
https://zuko.readthedocs.io/stable/api/zuko.flows.html

For a custom architecture, subclass ConditionalFlow and implement log_prob
and _draw; sample() wraps _draw with the prior-box rejection, so do not
override it.

    class MyFlow(ConditionalFlow):
        def __init__(self, theta_dim, context_dim, **kw):
            super().__init__(theta_dim, context_dim)
        def log_prob(self, theta, context): ...
        def _draw(self, n, context): ...       # sample() wraps this

`bounds` (low, high) in the flow's own coordinates makes sample() reject
draws outside the box — what DirectPosterior(reject_outside_prior=True) used
to do. Pass it to the constructor, or call set_bounds() later once the
standardiser exists.

SAMPLING SPEED. zuko's flows are fully autoregressive by default: the inverse
(sampling) costs one network pass per parameter. The built-ins therefore
default to passes=2, i.e. coupling transforms, which invert in a single pass —
the structure the sbi/nflows model used. passes=None restores full
autoregression: better density estimation per transform, ~theta_dim times
slower to sample.

zuko's spline transforms are defined on [-5, 5] and features outside that box
pass through untransformed (there is no tail_bound knob; the argument is
accepted and ignored for signature compatibility). theta is standardised, so
this is fine — do not feed raw physical parameters to a spline flow.
"""

import torch
import torch.nn as nn

__all__ = ["ConditionalFlow", "NSF", "NCSF", "MAF", "NICE",
           "SOSPF", "GF"]


class ConditionalFlow(nn.Module):
    """THE flow interface. 
    - theta_dim = number of inferred parameters,
    - context_dim = width of the embedding's output. 
    
    Subclasses implement log_prob(theta, context) and _draw(n, context); 
    sample(), bounds handling and prior-box rejection are inherited."""

    def __init__(self, theta_dim, context_dim, bounds=None):
        super().__init__()
        self.theta_dim = int(theta_dim)
        self.context_dim = int(context_dim)
        self.prior_low = self.prior_high = None
        if bounds is not None:
            self.set_bounds(*bounds)

    def set_bounds(self, low, high):
        """(Re)set the prior box. Callable after construction, so a flow can
        be built before the standardiser exists."""
        for name, val in (("prior_low", low), ("prior_high", high)):
            if name in self._buffers:
                del self._buffers[name]
            elif hasattr(self, name):
                delattr(self, name)
            self.register_buffer(name, torch.as_tensor(val).float())

    def in_bounds(self, theta):
        """Boolean mask of draws inside the prior box (all True if unset)."""
        if self.prior_low is None:
            return torch.ones(theta.shape[0], dtype=torch.bool,
                              device=theta.device)
        return ((theta >= self.prior_low) & (theta <= self.prior_high)).all(-1)

    # -- the two methods a subclass provides ---------------------------------

    def log_prob(self, theta, context):
        raise NotImplementedError

    def _draw(self, n, context):
        raise NotImplementedError

    # -- sampling with prior-box rejection -----------------------------------

    def sample(self, n, context):
        """n draws. With `bounds` set and a single context, draws outside the
        box are rejected and replaced (max 20 rounds)."""
        n = int(n)
        out = self._draw(n, context)
        if self.prior_low is None or context.shape[0] != 1:
            return out

        keep = [out[self.in_bounds(out)]]
        have, drawn = keep[0].shape[0], n
        for _ in range(20):
            if have >= n:
                break
            # size the next draw from the acceptance rate seen so far, with
            # 2x headroom; a low rate then costs one big draw, not 20 small
            rate = max(have / max(drawn, 1), 1e-3)
            batch = min(int(2 * (n - have) / rate) + 64, 1_000_000)
            extra = self._draw(batch, context)
            drawn += batch
            extra = extra[self.in_bounds(extra)]
            keep.append(extra)
            have += extra.shape[0]

        out = torch.cat(keep, dim=0)
        if out.shape[0] < n:
            raise RuntimeError(
                f"{type(self).__name__}: only {out.shape[0]}/{n} draws inside "
                f"the prior box after 20 rounds ({out.shape[0]}/{drawn} = "
                f"{100 * out.shape[0] / max(drawn, 1):.2g}% accepted) — the "
                f"flow is proposing far outside the analysis prior")
        return out[:n]




# ── zuko-backed built-ins ────────────────────────────────────────────────────

def _act_with_dropout(p, base=nn.ELU):
    """Activation factory that appends Dropout(p).

    zuko's conditioner MLPs take no `dropout` argument, but they do take an
    `activation` FACTORY (called as activation()), so dropout goes in this
    way. Real nn.Dropout modules end up in the graph, so
    training.losses.set_dropout_p can still retune them for the final
    curriculum stage.
    """
    class _ActDrop(nn.Sequential):
        def __init__(self):
            super().__init__(base(), nn.Dropout(p))
    return _ActDrop


def _zuko_net(name, theta_dim, context_dim, num_transforms, hidden_features,
              num_blocks, passes, num_bins, kwargs, has_passes=True):
    """Build a zuko flow, mapping our config names onto zuko's."""
    import zuko

    hidden = (list(hidden_features)
              if isinstance(hidden_features, (list, tuple))
              else [int(hidden_features)] * int(num_blocks))

    kw = dict(features=int(theta_dim), context=int(context_dim),
              transforms=int(num_transforms), hidden_features=hidden)
    if passes is not None and has_passes:
        kw["passes"] = int(passes)                 # 2 = coupling, fast inverse
    if num_bins is not None and name in ("NSF", "NCSF"):
        kw["bins"] = int(num_bins)                 # spline flows only
    kw.update(kwargs)                              # caller wins
    return getattr(zuko.flows, name)(**kw)


class _ZukoFlow(ConditionalFlow):
    """Shared body for the built-ins. Subclasses set ZUKO."""

    ZUKO = None
    HAS_PASSES = True     # False for architectures with no autoregressive
                          # /coupling switch (NICE is always coupling; GF just doesn't  
                          # have the idea of it) — without this zuko would forward `passes` to
                          # the MLP builder and nn.Linear would reject it.

    def __init__(self, theta_dim, context_dim, num_transforms=64,
                 hidden_features=512, num_bins=None, dropout=None,
                 num_blocks=2, passes=2, tail_bound=None, bounds=None,
                 **kwargs):
        super().__init__(theta_dim, context_dim, bounds=bounds)
                # zuko's MaskedMLP has no `dropout` argument, so dropout is injected
        # through the `activation` factory instead (see _act_with_dropout).
        # Dropout is active in train mode only, so eval-time log_prob — the
        # density the reweighter uses — stays deterministic.
        if dropout:
            kwargs.setdefault("activation", _act_with_dropout(float(dropout)))
        self.net = _zuko_net(self.ZUKO, theta_dim, context_dim, num_transforms,
                        hidden_features, num_blocks, passes, num_bins,
                        kwargs, has_passes=self.HAS_PASSES)


    def log_prob(self, theta, context):
        return self.net(context).log_prob(theta)

    def rsample_and_log_prob(self, n, context):
        """Reparameterised draws with their densities in one pass — the
        primitive for IS-variance / chi^2 objectives. Returns (theta, log_q)."""
        return self.net(context).rsample_and_log_prob((int(n),))

    def _draw(self, n, context):
        if context.shape[0] == 1:                        # one event, n draws
            return self.net(context.squeeze(0)).sample((int(n),))
        if int(n) == 1:                                  # a batch, one each
            return self.net(context).sample()
        return self.net(context).sample((int(n),))       # (n, B, theta)


class NSF(_ZukoFlow):
    """Neural _spline_ flow — monotonic rational-quadratic splines on [-5, 5].
    The production architecture. num_bins maps to zuko's `bins`."""
    ZUKO = "NSF"


class NCSF(_ZukoFlow):
    """Neural _circular_ spline flow, for angles on [-pi, pi). Relevant if the
    periodic parameters are ever modelled in native coordinates."""
    ZUKO = "NCSF"


class MAF(_ZukoFlow):
    """Masked autoregressive flow — affine transforms, masked-MLP conditioner.
    Cheaper per transform than NSF; need more transforms for the same expressivity though."""
    ZUKO = "MAF"


class NICE(_ZukoFlow):
    """Additive coupling flow (volume preserving). Typical baseline.
    Always coupling, so it has no `passes` argument."""
    ZUKO = "NICE"
    HAS_PASSES = False


class SOSPF(_ZukoFlow):
    """Sum-of-squares polynomial flow. zuko kwargs: degree, polynomials."""
    ZUKO = "SOSPF"


class GF(_ZukoFlow):
    """Gaussianization flow — rotations composed with monotonic maps;
    no autoregressive/coupling switch, so no `passes` argument."""
    ZUKO = "GF"
    HAS_PASSES = False