"""Shared fixtures. The configs here are TINY — these tests check that things
construct and produce the right shapes, not that they train well."""

import pytest


@pytest.fixture
def cfg():
    """A minimal config: 0.25 s at 256 Hz -> n_masked=17, n_td=64, so a full
    forward pass costs milliseconds."""
    return dict(
        duration=0.25,
        sampling_frequency=256,
        f_min=20.0,
        inferred_parameters=["chirp_mass", "mass_ratio",
                             "luminosity_distance", "theta_jn", "ra", "dec"],
        psd_conditioning=True,
        amp_context=False,
        embedding_output_dim=32,
        conv1d_channels=[8, 16],
        conv1d_kernel=5,
        conv1d_dropout=0.0,
        conv1d_resnet_stem_out=8,
        conv1d_resnet_backbone="resnet18",
        psd_encoder_hidden=[32, 16],
        psd_encoder_out=8,
        embedding_layers=[32, 32],
        num_transforms=2,
        hidden_features=16,
        num_bins=4,
        flow_dropout=0.0,
    )


@pytest.fixture
def cfg_no_psd(cfg):
    out = dict(cfg)
    out["psd_conditioning"] = False
    out.pop("x_blocks", None)
    return out


@pytest.fixture
def cfg_amp(cfg):
    out = dict(cfg)
    out["amp_context"] = True
    out.pop("x_blocks", None)
    return out


@pytest.fixture
def theta_dim(cfg):
    return len(cfg["inferred_parameters"])