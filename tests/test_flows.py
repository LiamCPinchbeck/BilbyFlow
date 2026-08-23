"""Every flow constructs, evaluates densities, and samples with the right
shapes — including the one-context-many-theta case the reweighter uses."""

import pytest
import torch

from bilbyflow.nn.flow import (ConditionalFlow, NSF, NCSF, MAF, NICE,
                               SOSPF, GF)

FLOWS = [NSF, NCSF, MAF, NICE, SOSPF, GF]
CTX = 32


@pytest.mark.parametrize("cls", FLOWS, ids=lambda c: c.__name__)
def test_log_prob_shape(cls, theta_dim):
    flow = cls(theta_dim, CTX, num_transforms=2, hidden_features=16,
               num_bins=8, passes=2, tail_bound=5.0, dropout=0.05)
    theta, ctx = torch.randn(5, theta_dim), torch.randn(5, CTX)
    lp = flow.log_prob(theta, ctx)
    assert lp.shape == (5,)
    assert torch.isfinite(lp).all()


@pytest.mark.parametrize("cls", FLOWS, ids=lambda c: c.__name__)
def test_sample_shape_single_context(cls, theta_dim):
    """One event, many draws — the reweighting pattern."""
    flow = cls(theta_dim, CTX, num_transforms=2, hidden_features=16)
    with torch.no_grad():
        s = flow.sample(7, torch.randn(1, CTX))
    assert s.shape == (7, theta_dim)


@pytest.mark.parametrize("cls", FLOWS, ids=lambda c: c.__name__)
def test_sample_batch_one_each(cls, theta_dim):
    """A batch of contexts, one draw each."""
    flow = cls(theta_dim, CTX, num_transforms=2, hidden_features=16)
    with torch.no_grad():
        s = flow.sample(1, torch.randn(4, CTX))
    assert s.shape == (4, theta_dim)


def test_hidden_features_list_and_int(theta_dim):
    """Both config forms build the same net: an int + num_blocks, or a list."""
    a = NSF(theta_dim, CTX, hidden_features=16, num_blocks=3)
    b = NSF(theta_dim, CTX, hidden_features=[16, 16, 16])
    assert (sum(p.numel() for p in a.parameters())
            == sum(p.numel() for p in b.parameters()))


def test_zuko_kwargs_forwarded(theta_dim):
    """num_bins maps to zuko's `bins`; extra kwargs reach zuko untouched."""
    coarse = NSF(theta_dim, CTX, num_transforms=2, hidden_features=16,
                 num_bins=4)
    fine = NSF(theta_dim, CTX, num_transforms=2, hidden_features=16,
               num_bins=32)
    n_coarse = sum(p.numel() for p in coarse.parameters())
    n_fine = sum(p.numel() for p in fine.parameters())
    assert n_fine > n_coarse, "num_bins did not reach zuko"


def test_bins_ignored_by_non_spline(theta_dim):
    """MAF has no bins; passing num_bins must not raise."""
    MAF(theta_dim, CTX, num_transforms=2, hidden_features=16, num_bins=24)


@pytest.mark.parametrize("cls", FLOWS, ids=lambda c: c.__name__)
def test_accepts_standard_kwargs(cls, theta_dim):
    """Every built-in must tolerate the full config kwarg set, whether or not
    its architecture uses each one — the wrapper is responsible for dropping
    what zuko would reject (bins on non-splines, passes on NICE/GF)."""
    cls(theta_dim, CTX, num_transforms=2, hidden_features=16,
        num_bins=8, passes=2, tail_bound=5.0, dropout=0.0)

def test_tail_bound_accepted_and_ignored(theta_dim):
    NSF(theta_dim, CTX, num_transforms=2, hidden_features=16, tail_bound=5.0)


def test_prior_box_rejection(theta_dim):
    """With bounds set, every returned draw is inside the box. The box is
    wide because an UNTRAINED flow is ~N(0,1): a +-0.5 box in 6-D accepts
    ~0.3% of draws, which is not a realistic test of the mechanism."""
    lo = torch.full((theta_dim,), -3.0)
    hi = torch.full((theta_dim,), 3.0)
    flow = NSF(theta_dim, CTX, num_transforms=2, hidden_features=16,
               bounds=(lo, hi))
    with torch.no_grad():
        s = flow.sample(50, torch.randn(1, CTX))
    assert s.shape == (50, theta_dim)
    assert ((s >= lo) & (s <= hi)).all()

def test_impossible_box_raises(theta_dim):
    """A box the flow cannot populate must fail loudly, not return short."""
    lo = torch.full((theta_dim,), 9.0)
    hi = torch.full((theta_dim,), 10.0)
    flow = NSF(theta_dim, CTX, num_transforms=2, hidden_features=16,
               bounds=(lo, hi))
    with torch.no_grad(), pytest.raises(RuntimeError, match="prior box"):
        flow.sample(50, torch.randn(1, CTX))

def test_set_bounds_after_construction(theta_dim):
    """The trainer fills the box in once the standardiser exists."""
    flow = NSF(theta_dim, CTX, num_transforms=2, hidden_features=16)
    assert flow.prior_low is None
    lo = torch.full((theta_dim,), -1.0)
    hi = torch.full((theta_dim,), 1.0)
    flow.set_bounds(lo, hi)
    assert flow.prior_low is not None
    flow.set_bounds(lo * 2, hi * 2)                # idempotent, no error
    assert torch.allclose(flow.prior_high, hi * 2)
    with torch.no_grad():
        s = flow.sample(20, torch.randn(1, CTX))
    assert ((s >= lo * 2) & (s <= hi * 2)).all()


def test_bounds_follow_device_move(theta_dim):
    """prior_low/high are buffers, so .to() moves them with the model."""
    flow = NSF(theta_dim, CTX, num_transforms=2, hidden_features=16,
               bounds=(torch.zeros(theta_dim), torch.ones(theta_dim)))
    assert "prior_low" in dict(flow.named_buffers())


def test_custom_flow_contract(theta_dim):
    """Subclass Flow, implement log_prob and _draw; sample() is inherited."""

    class Gaussian(ConditionalFlow):
        def __init__(self, theta_dim, context_dim, **kw):
            super().__init__(theta_dim, context_dim, **kw)
            self.mu = torch.nn.Linear(context_dim, theta_dim)

        def log_prob(self, theta, context):
            d = torch.distributions.Normal(self.mu(context), 1.0)
            return d.log_prob(theta).sum(-1)

        def _draw(self, n, context):
            mu = self.mu(context)
            if context.shape[0] == 1:
                return mu + torch.randn(n, self.theta_dim)
            return mu + torch.randn_like(mu)

    flow = Gaussian(theta_dim, CTX)
    assert flow.log_prob(torch.randn(3, theta_dim),
                         torch.randn(3, CTX)).shape == (3,)
    assert flow.sample(9, torch.randn(1, CTX)).shape == (9, theta_dim)