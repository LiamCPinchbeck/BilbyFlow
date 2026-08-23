"""Every embedding constructs, unpacks the right blocks, and returns
(B, context_dim) — including the batch-1 case."""

import pytest
import torch

from bilbyflow.nn.embedding import (StrainEmbedding, Conv1dEmbedding,
                                    Conv1dResNetEmbedding, ResNetEmbedding,
                                    FDPSDEmbedding, FiLMEmbedding,
                                    MLPEmbedding)

EMBEDDINGS = [Conv1dEmbedding, Conv1dResNetEmbedding, ResNetEmbedding,
              FDPSDEmbedding, FiLMEmbedding, MLPEmbedding]


def _x(emb, batch=3):
    return torch.randn(batch, emb.sl["_total"])


@pytest.mark.parametrize("cls", EMBEDDINGS, ids=lambda c: c.__name__)
def test_constructs_and_shapes(cls, cfg):
    emb = cls(cfg)
    emb.eval()
    with torch.no_grad():
        out = emb(_x(emb))
    assert out.shape == (3, emb.context_dim)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("cls", EMBEDDINGS, ids=lambda c: c.__name__)
def test_batch_one(cls, cfg):
    """A single sample must work in eval mode — BatchNorm rejects batch-1 in
    train mode, and inference always uses batch 1."""
    emb = cls(cfg)
    emb.eval()
    with torch.no_grad():
        out = emb(_x(emb, batch=1))
    assert out.shape == (1, emb.context_dim)


@pytest.mark.parametrize("cls", EMBEDDINGS, ids=lambda c: c.__name__)
def test_declares_blocks(cls, cfg):
    """The embedding declares which views it reads; the packing is fixed."""
    emb = cls(cfg)
    assert "fd" in emb._blocks
    assert emb.sl["_total"] == emb.strain_dim + emb.psd_dim + emb.amp_dim


@pytest.mark.parametrize("cls", EMBEDDINGS, ids=lambda c: c.__name__)
def test_wrong_width_raises(cls, cfg):
    emb = cls(cfg)
    with pytest.raises(ValueError, match="width"):
        emb(torch.randn(2, emb.sl["_total"] + 7))


@pytest.mark.parametrize("cls", [Conv1dEmbedding, Conv1dResNetEmbedding,
                                 ResNetEmbedding, FiLMEmbedding,
                                 MLPEmbedding], ids=lambda c: c.__name__)
def test_without_psd(cls, cfg_no_psd):
    """psd_conditioning: false drops the block; embed() must cope."""
    emb = cls(cfg_no_psd)
    assert "psd" not in emb._blocks
    emb.eval()
    with torch.no_grad():
        out = emb(_x(emb))
    assert out.shape == (3, emb.context_dim)


def test_fdpsd_needs_psd(cfg_no_psd):
    with pytest.raises(ValueError):
        FDPSDEmbedding(cfg_no_psd)


@pytest.mark.parametrize("cls", EMBEDDINGS, ids=lambda c: c.__name__)
def test_amp_context(cls, cfg_amp):
    """amp widens the context by amp_dim and is appended, not embedded."""
    emb = cls(cfg_amp)
    assert emb.amp_dim > 0
    assert emb.context_dim == emb.output_dim + emb.amp_dim
    emb.eval()
    with torch.no_grad():
        out = emb(_x(emb))
    assert out.shape == (3, emb.context_dim)


def test_standardise_is_identity_without_std(cfg):
    emb = Conv1dEmbedding(cfg)
    x = torch.randn(2, emb.sl["_total"])
    assert torch.equal(emb.standardise(x), x)


def test_forward_std_skips_standardisation(cfg):
    """forward_raw == forward_std when there is no standardiser."""
    emb = Conv1dEmbedding(cfg)
    emb.eval()
    x = torch.randn(2, emb.sl["_total"])
    with torch.no_grad():
        assert torch.allclose(emb.forward_raw(x), emb.forward_std(x))


def test_unpack_shapes(cfg):
    emb = Conv1dResNetEmbedding(cfg)
    blocks = emb.unpack(torch.randn(4, emb.sl["_total"]))
    assert blocks["fd"].shape == (4, 4, emb.n_masked)
    assert blocks["td"].shape == (4, 2, emb.n_td)
    assert blocks["psd"].shape == (4, 2, emb.n_masked)


def test_custom_embedding_contract(cfg):
    """The documented way to write one: declare blocks, implement embed."""

    class MyEmbedding(StrainEmbedding):
        blocks = ("fd", "psd")
        output_dim = 16

        def __init__(self, cfg, std=None, output_dim=None):
            super().__init__(cfg, std, output_dim)
            self.net = torch.nn.Linear(6 * self.n_masked, self.output_dim)

        def embed(self, fd, psd):
            return self.net(torch.cat([fd, psd], dim=1).flatten(1))

    emb = MyEmbedding(cfg)
    with torch.no_grad():
        out = emb(_x(emb))
    assert out.shape == (3, 16)