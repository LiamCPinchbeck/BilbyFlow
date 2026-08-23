"""The mix-and-match matrix: every embedding against every flow, end to end.

This is the test that would have caught the context_dim / output_dim
mismatches, the amp-widening bug, and the one-x-many-theta broadcast."""

import itertools

import pytest
import torch

from bilbyflow.nn.embedding import (Conv1dEmbedding, Conv1dResNetEmbedding,
                                    ResNetEmbedding, FDPSDEmbedding,
                                    FiLMEmbedding, MLPEmbedding)
from bilbyflow.nn.flow import NSF, NCSF, MAF, NICE, SOSPF, GF
from bilbyflow.npe import NPE

EMBEDDINGS = [Conv1dEmbedding, Conv1dResNetEmbedding, ResNetEmbedding,
              FDPSDEmbedding, FiLMEmbedding, MLPEmbedding]
FLOWS = [NSF, NCSF, MAF, NICE, SOSPF, GF]
PAIRS = list(itertools.product(EMBEDDINGS, FLOWS))


def _build(emb_cls, flow_cls, cfg, theta_dim, bounds=None):
    emb = emb_cls(cfg)
    flow = flow_cls(theta_dim, emb.context_dim,
                    num_transforms=int(cfg["num_transforms"]),
                    hidden_features=int(cfg["hidden_features"]),
                    num_bins=int(cfg["num_bins"]), bounds=bounds)
    return NPE(emb, flow)


@pytest.mark.parametrize("emb_cls,flow_cls", PAIRS,
                         ids=[f"{e.__name__}-{f.__name__}" for e, f in PAIRS])
def test_pair_constructs(emb_cls, flow_cls, cfg, theta_dim):
    npe = _build(emb_cls, flow_cls, cfg, theta_dim)
    assert npe.flow.context_dim == npe.embedding.context_dim


@pytest.mark.parametrize("emb_cls,flow_cls", PAIRS,
                         ids=[f"{e.__name__}-{f.__name__}" for e, f in PAIRS])
def test_pair_log_prob_and_sample(emb_cls, flow_cls, cfg, theta_dim):
    npe = _build(emb_cls, flow_cls, cfg, theta_dim)
    npe.eval()
    x = torch.randn(4, npe.embedding.sl["_total"])
    theta = torch.randn(4, theta_dim)
    with torch.no_grad():
        assert npe.log_prob(theta, x).shape == (4,)
        assert npe.sample(1, x).shape == (4, theta_dim)
        # the reweighting pattern: one event, many draws
        s = npe.sample(6, x[:1])
    assert s.shape == (6, theta_dim)


def test_context_mismatch_raises(cfg, theta_dim):
    """A flow built for the wrong width must fail at construction, not at the
    first forward."""
    emb = Conv1dEmbedding(cfg)
    flow = NSF(theta_dim, emb.context_dim + 5, num_transforms=2,
               hidden_features=16)
    with pytest.raises(ValueError, match="context mismatch"):
        NPE(emb, flow)


def test_one_x_many_theta(cfg, theta_dim):
    """log_prob broadcasts a single context over N thetas — the main
    reweighting call."""
    npe = _build(Conv1dResNetEmbedding, NSF, cfg, theta_dim)
    npe.eval()
    x = torch.randn(1, npe.embedding.sl["_total"])
    theta = torch.randn(32, theta_dim)
    with torch.no_grad():
        lp = npe.log_prob(theta, x)
    assert lp.shape == (32,)


def test_pin_reuses_context(cfg, theta_dim):
    """pin() embeds once; results must match the unpinned path."""
    npe = _build(Conv1dEmbedding, NSF, cfg, theta_dim)
    npe.eval()
    x = torch.randn(1, npe.embedding.sl["_total"])
    theta = torch.randn(16, theta_dim)
    with torch.no_grad():
        ref = npe.log_prob(theta, x)
        npe.pin(x)
        pinned = npe.log_prob(theta, x)
        npe.unpin()
        after = npe.log_prob(theta, x)
    assert torch.allclose(ref, pinned, atol=1e-5)
    assert torch.allclose(ref, after, atol=1e-5)


def test_loss_is_finite_and_backprops(cfg, theta_dim):
    npe = _build(Conv1dEmbedding, NSF, cfg, theta_dim)
    x = torch.randn(4, npe.embedding.sl["_total"])
    theta = torch.randn(4, theta_dim)
    loss = npe.loss(theta, x).mean()
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in npe.parameters() if p.grad is not None]
    assert grads, "no gradients reached the model"
    assert any(g.abs().sum() > 0 for g in grads)


def test_save_load_roundtrip(cfg, theta_dim, tmp_path):
    """save() stores weights; a freshly built model must load them and agree."""
    npe = _build(Conv1dEmbedding, NSF, cfg, theta_dim)
    npe.eval()
    x = torch.randn(2, npe.embedding.sl["_total"])
    theta = torch.randn(2, theta_dim)
    with torch.no_grad():
        before = npe.log_prob(theta, x)

    path = tmp_path / "npe.pt"
    npe.save(path)

    cfg2 = dict(cfg)
    cfg2.pop("x_blocks", None)
    other = _build(Conv1dEmbedding, NSF, cfg2, theta_dim)
    other.load(path)
    other.eval()
    with torch.no_grad():
        after = other.log_prob(theta, x)
    assert torch.allclose(before, after, atol=1e-6)


def test_parameters_cover_both_halves(cfg, theta_dim):
    npe = _build(Conv1dEmbedding, NSF, cfg, theta_dim)
    names = [n for n, _ in npe.named_parameters()]
    assert any(n.startswith("embedding.") for n in names)
    assert any(n.startswith("flow.") for n in names)