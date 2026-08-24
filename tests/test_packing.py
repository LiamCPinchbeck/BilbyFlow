"""THE invariant: data.canonical packs x, nn.embedding unpacks it, and the
two must agree channel for channel.

A disagreement here is the worst failure mode in the package: the widths
still match, nothing raises, and the network is handed time-domain samples
in a frequency-domain channel. It trains, it converges, the posterior is
wrong. Every other test in this suite would pass.
"""

import numpy as np
import pytest
import torch

from bilbyflow.data.canonical import (canonical_grid, canonical_tukey,
                                      canonical_td_norm, build_x_strain,
                                      build_x_full, canonical_bn)
from bilbyflow.nn.embedding import (Conv1dEmbedding, Conv1dResNetEmbedding,
                                    ResNetEmbedding, FDPSDEmbedding,
                                    FiLMEmbedding, MLPEmbedding)

EMBEDDINGS = [Conv1dEmbedding, Conv1dResNetEmbedding, ResNetEmbedding,
              FDPSDEmbedding, FiLMEmbedding, MLPEmbedding]

# one distinct constant per channel, in PACKING order
_TAGS = {("H1", "re"): 1.0, ("H1", "im"): 2.0, ("H1", "td"): 3.0,
         ("L1", "re"): 4.0, ("L1", "im"): 5.0, ("L1", "td"): 6.0}


def _tagged_x_strain(g):
    """A strain vector whose every channel is a distinct constant, packed by
    hand in exactly the order canonical.build_x_strain uses."""
    nm, ntd = g["n_masked"], g["n_td"]
    parts = []
    for det in ("H1", "L1"):
        parts += [np.full(nm, _TAGS[(det, "re")], dtype=np.float32),
                  np.full(nm, _TAGS[(det, "im")], dtype=np.float32),
                  np.full(ntd, _TAGS[(det, "td")], dtype=np.float32)]
    return np.concatenate(parts)


class _FakeStd:
    """Standardiser stand-in: PSD context on, everything else identity."""
    def __init__(self, g, amp_dim=0):
        self.psd_conditioning = True
        self.psd_log_mean = np.zeros(g["n_fd_full"])
        self.psd_log_std = np.ones(g["n_fd_full"])
        self.amp_dim = amp_dim


# -- channel order ------------------------------------------------------------

@pytest.mark.parametrize("cls", EMBEDDINGS, ids=lambda c: c.__name__)
def test_unpack_recovers_packing_order(cls, cfg):
    """Each unpacked view must hold the constant its channel was packed with.
    Detects any reordering on either side."""
    g = canonical_grid(cfg)
    x = _tagged_x_strain(g)

    emb = cls(cfg)
    # pad with a zero PSD block so the width matches what this embedding reads
    if emb.psd_dim:
        x = np.concatenate([x, np.zeros(emb.psd_dim, dtype=np.float32)])
    blocks = emb.unpack(torch.from_numpy(x).unsqueeze(0))

    if "fd" in blocks:
        fd = blocks["fd"][0]
        assert torch.allclose(fd[0], torch.tensor(_TAGS[("H1", "re")]))
        assert torch.allclose(fd[1], torch.tensor(_TAGS[("H1", "im")]))
        assert torch.allclose(fd[2], torch.tensor(_TAGS[("L1", "re")]))
        assert torch.allclose(fd[3], torch.tensor(_TAGS[("L1", "im")]))
    if "td" in blocks:
        td = blocks["td"][0]
        assert torch.allclose(td[0], torch.tensor(_TAGS[("H1", "td")]))
        assert torch.allclose(td[1], torch.tensor(_TAGS[("L1", "td")]))


def test_build_x_strain_matches_hand_packing(cfg):
    """canonical.build_x_strain must lay channels out in the order the
    embedding assumes ([Re, Im, TD] x [H1, L1])."""
    g = canonical_grid(cfg)
    nm, ntd = g["n_masked"], g["n_td"]
    fm = g["freq_mask"]

    # whitened FD whose masked Re/Im are known constants per detector
    sig, noise = {}, {}
    for det, (re, im) in (("H1", (1.0, 2.0)), ("L1", (4.0, 5.0))):
        v = np.zeros(g["n_fd_full"], dtype=complex)
        v[fm] = re + 1j * im
        sig[det] = v
        noise[det] = np.zeros(g["n_fd_full"], dtype=complex)

    x = build_x_strain(sig, noise, fm, ntd, canonical_td_norm(g))
    per = 2 * nm + ntd
    assert np.allclose(x[:nm], 1.0)                    # H1 Re
    assert np.allclose(x[nm:2 * nm], 2.0)              # H1 Im
    assert np.allclose(x[per:per + nm], 4.0)           # L1 Re
    assert np.allclose(x[per + nm:per + 2 * nm], 5.0)  # L1 Im


# -- widths -------------------------------------------------------------------

@pytest.mark.parametrize("psd_on", [True, False])
@pytest.mark.parametrize("amp_on", [True, False])
def test_packed_width_matches_embedding(cfg, psd_on, amp_on):
    """len(build_x_full(...)) must equal what the embedding expects, for
    every combination of the two optional blocks."""
    if amp_on and not psd_on:
        pytest.skip("amp_context requires psd_conditioning")
    c = dict(cfg, psd_conditioning=psd_on, amp_context=amp_on)
    g = canonical_grid(c)

    std = _FakeStd(g, amp_dim=5 if amp_on else 0)
    std.psd_conditioning = psd_on
    if not psd_on:
        std.psd_log_mean = None
    det_psds = {d: np.full(g["n_fd_full"], 1e-42) for d in ("H1", "L1")}

    x = build_x_full(_tagged_x_strain(g), det_psds, std, c, g)
    emb = Conv1dEmbedding(c)
    assert len(x) == emb.sl["_total"], (
        f"canonical packed {len(x)}, {type(emb).__name__} expects "
        f"{emb.sl['_total']} (psd_on={psd_on}, amp_on={amp_on})")


@pytest.mark.parametrize("cls", EMBEDDINGS, ids=lambda c: c.__name__)
def test_strain_dim_formula(cls, cfg):
    """strain_dim = 2 detectors x (2 x n_masked + n_td)."""
    emb = cls(cfg)
    g = canonical_grid(cfg)
    assert emb.strain_dim == 2 * (2 * g["n_masked"] + g["n_td"])
    assert emb.n_masked == g["n_masked"]
    assert emb.n_td == g["n_td"]


def test_full_pipeline_x_feeds_embedding(cfg):
    """End of the chain: a vector built by build_x_full must run through an
    embedding without a width error and give a finite context."""
    g = canonical_grid(cfg)
    std = _FakeStd(g)
    det_psds = {d: np.full(g["n_fd_full"], 1e-42) for d in ("H1", "L1")}
    x = build_x_full(_tagged_x_strain(g), det_psds, std, cfg, g)

    emb = Conv1dResNetEmbedding(cfg)
    emb.eval()
    with torch.no_grad():
        out = emb(torch.from_numpy(x).unsqueeze(0))
    assert out.shape == (1, emb.context_dim)
    assert torch.isfinite(out).all()