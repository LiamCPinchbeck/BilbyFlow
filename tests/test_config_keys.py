"""Config keys must DO something.

A few keys have changed throughout the package development and not
loudly errored. This script asserts the config kwargs change the model.
Cheapest general guard against a swallowed kwarg.
"""

import pytest
import torch

from bilbyflow.nn.embedding import Conv1dEmbedding, Conv1dResNetEmbedding
from bilbyflow.nn.flow import NSF, MAF

CTX = 32


def _n(model):
    return sum(p.numel() for p in model.parameters())


# ------------ flow keys -----------------------------------------------------

def test_num_transforms_reaches_the_flow(theta_dim):
    small = NSF(theta_dim, CTX, num_transforms=2, hidden_features=16)
    big = NSF(theta_dim, CTX, num_transforms=8, hidden_features=16)
    assert _n(big) > _n(small), "num_transforms did not reach zuko"


def test_hidden_features_reaches_the_flow(theta_dim):
    narrow = NSF(theta_dim, CTX, num_transforms=2, hidden_features=16)
    wide = NSF(theta_dim, CTX, num_transforms=2, hidden_features=128)
    assert _n(wide) > _n(narrow), "hidden_features did not reach zuko"


def test_num_bins_reaches_spline_flows(theta_dim):
    coarse = NSF(theta_dim, CTX, num_transforms=2, hidden_features=16,
                 num_bins=4)
    fine = NSF(theta_dim, CTX, num_transforms=2, hidden_features=16,
               num_bins=32)
    assert _n(fine) > _n(coarse), "num_bins did not reach zuko as `bins`"


def test_flow_dropout_creates_dropout_modules(theta_dim):
    """dropout goes in through zuko's `activation` factory (MaskedMLP has no
    dropout argument), so it is easy to break silently. Real nn.Dropout
    modules must appear, or set_dropout_p has nothing to retune either."""
    off = NSF(theta_dim, CTX, num_transforms=2, hidden_features=16)
    on = NSF(theta_dim, CTX, num_transforms=2, hidden_features=16,
             dropout=0.1)
    n_off = sum(isinstance(m, torch.nn.Dropout) for m in off.modules())
    n_on = sum(isinstance(m, torch.nn.Dropout) for m in on.modules())
    assert n_on > n_off, "flow_dropout produced no Dropout modules"


def test_set_dropout_p_finds_flow_dropout(theta_dim):
    """The final-stage override (final_stage_flow_dropout) must be able to
    retune what flow_dropout created."""
    from bilbyflow.training.losses import set_dropout_p
    flow = NSF(theta_dim, CTX, num_transforms=2, hidden_features=16,
               dropout=0.1)
    n = set_dropout_p(flow, 0.0)
    assert n > 0, "set_dropout_p found no Dropout modules in the flow"
    for m in flow.modules():
        if isinstance(m, torch.nn.Dropout):
            assert m.p == 0.0


def test_passes_changes_the_flow(theta_dim):
    """passes=2 (coupling) and full autoregression must build DIFFERENT nets
    — this is a ~theta_dim factor in sampling speed."""
    coupling = NSF(theta_dim, CTX, num_transforms=2, hidden_features=16,
                   passes=2)
    autoreg = NSF(theta_dim, CTX, num_transforms=2, hidden_features=16,
                  passes=None)
    assert _n(coupling) != _n(autoreg) or \
        str(coupling.net) != str(autoreg.net), "passes had no effect"


# ------------------ embedding keys -------------------------------------

def test_psd_conditioning_changes_the_embedding(cfg, cfg_no_psd):
    on = Conv1dEmbedding(cfg)
    off = Conv1dEmbedding(cfg_no_psd)
    assert "psd" in on._blocks and "psd" not in off._blocks
    assert _n(on) > _n(off), "psd_conditioning added no parameters"
    assert on.sl["_total"] > off.sl["_total"], "psd block not in x"


def test_embedding_output_dim_reaches_the_head(cfg):
    small = Conv1dEmbedding(dict(cfg, embedding_output_dim=16))
    big = Conv1dEmbedding(dict(cfg, embedding_output_dim=64))
    assert small.output_dim == 16 and big.output_dim == 64
    assert _n(big) > _n(small)


def test_conv1d_channels_reach_the_stem(cfg):
    thin = Conv1dEmbedding(dict(cfg, conv1d_channels=[4, 8]))
    thick = Conv1dEmbedding(dict(cfg, conv1d_channels=[16, 32]))
    assert _n(thick) > _n(thin), "conv1d_channels did not reach the stem"


def test_psd_encoder_out_widens_the_head(cfg):
    a = Conv1dResNetEmbedding(dict(cfg, psd_encoder_out=8))
    b = Conv1dResNetEmbedding(dict(cfg, psd_encoder_out=64))
    assert _n(b) > _n(a), "psd_encoder_out did not reach the PSD MLP"


def test_amp_context_widens_the_context(cfg, cfg_amp):
    off = Conv1dEmbedding(cfg)
    on = Conv1dEmbedding(cfg_amp)
    assert off.amp_dim == 0 and on.amp_dim == 5
    assert on.context_dim == on.output_dim + 5
    assert on.sl["_total"] == off.sl["_total"] + 5


def test_backbone_key_is_honoured(cfg):
    r18 = Conv1dResNetEmbedding(dict(cfg, conv1d_resnet_backbone="resnet18"))
    r34 = Conv1dResNetEmbedding(dict(cfg, conv1d_resnet_backbone="resnet34"))
    assert _n(r34) > _n(r18), "conv1d_resnet_backbone ignored"


def test_unknown_backbone_raises(cfg):
    with pytest.raises(ValueError, match="[Uu]nknown"):
        Conv1dResNetEmbedding(dict(cfg, conv1d_resnet_backbone="resnet99"))


# -------------- keys that should NOT silently vanish ---------------

def test_bins_on_non_spline_is_dropped_not_forwarded(theta_dim):
    """MAF has no `bins`; the wrapper must drop it rather than let zuko
    forward it into a Linear."""
    MAF(theta_dim, CTX, num_transforms=2, hidden_features=16, num_bins=24)


def test_full_config_kwarg_set_is_tolerated(theta_dim):
    """Every key the config may carry, at once, on the production flow."""
    NSF(theta_dim, CTX, num_transforms=2, hidden_features=16, num_bins=8,
        dropout=0.05, num_blocks=2, passes=2, tail_bound=5.0)