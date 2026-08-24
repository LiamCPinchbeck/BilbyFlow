"""Tests transforms that _should_ invert exactly. 

Mostly related to sky parameter reparameterizations, data standardization and general whitening.
"""

import numpy as np
import pytest
import torch

from bilbyflow.coordinates.sky import (MAX_DT_HL, radec_to_detector,
                                       detector_to_radec,
                                       samples_detector_to_radec)


def _wrap(d):
    """Angular difference in (-pi, pi]."""
    
    return np.angle(np.exp(1j * d))


# ------ sky reparameterisation ---------------

def test_sky_roundtrip():
    """(ra, dec) -> (dt_HL, phi_det) -> (ra, dec), 200 isotropic points."""
    rng = np.random.default_rng(0)
    ra = rng.uniform(0, 2 * np.pi, 200)
    dec = np.arcsin(rng.uniform(-1, 1, 200))
    tc = 1126259462.4

    err_ra = err_dec = 0.0
    for r, d in zip(ra, dec):
        dt, phi = radec_to_detector(r, d, tc)
        r2, d2 = detector_to_radec(dt, phi, tc)
        err_ra = max(err_ra, abs(_wrap(r2 - r)))
        err_dec = max(err_dec, abs(d2 - d))
    
    assert err_ra < 1e-9, f"ra round-trip error {err_ra:.2e}"
    assert err_dec < 1e-9, f"dec round-trip error {err_dec:.2e}"


def test_dt_within_light_travel_time():
    """|dt_HL| can never exceed the H1-L1 light travel time; the prior box in
    the flow is exactly [-MAX_DT_HL, +MAX_DT_HL]."""
    rng = np.random.default_rng(1)
    tc = 1126259462.4 # yes this is hard-coded, sorry
    for r, d in zip(rng.uniform(0, 2 * np.pi, 500),
                    np.arcsin(rng.uniform(-1, 1, 500))):
        dt, _ = radec_to_detector(r, d, tc)
        assert abs(dt) <= MAX_DT_HL * (1 + 1e-12), f"|dt|={abs(dt)} > MAX_DT_HL"


def test_max_dt_is_margin_free():
    """MAX_DT_HL must be the bare light travel time — the trained models saw
    no safety margin, and a margin shifts the normalised prior box."""
    c = 299792458.0
    assert 0.0090 < MAX_DT_HL < 0.0105, MAX_DT_HL   # ~10.01 ms for H1-L1
    assert MAX_DT_HL * c > 2.9e6                     # ~3000 km baseline


def test_samples_detector_to_radec_matches_scalar():
    """The vectorised sample converter must agree with the scalar transform
    it is meant to batch."""
    rng = np.random.default_rng(2)

    n = 50
    names = ["chirp_mass", "ra", "dec"]
    tc = 1126259462.4
    dt = rng.uniform(-MAX_DT_HL, MAX_DT_HL, n)
    phi = rng.uniform(0, 2 * np.pi, n)

    samples = np.zeros((n, 3))
    samples[:, 0] = 30.0
    samples[:, 1] = dt
    samples[:, 2] = phi

    out = samples_detector_to_radec(samples.copy(), names, tc=tc)
    for i in range(n):
        ra, dec = detector_to_radec(dt[i], phi[i], tc)
        assert abs(_wrap(out[i, 1] - ra)) < 1e-9
        assert abs(out[i, 2] - dec) < 1e-9
    assert np.allclose(out[:, 0], 30.0), "non-sky columns were modified"


# ------ standardiser -------------------------------------------

def _std(cfg, x_width, n_theta):
    from bilbyflow.data.standardiser import Standardiser
    torch.manual_seed(0)
    x = torch.randn(64, x_width)
    theta = torch.randn(64, n_theta) * 3.0 + 7.0
    return Standardiser(x, theta, strain_dim=x_width)


def test_theta_normalise_roundtrip(cfg, theta_dim):

    std = _std(cfg, 32, theta_dim)
    theta = torch.randn(20, theta_dim) * 5.0 - 2.0
    back = std.unnormalise_theta(std.normalise_theta(theta))

    assert torch.allclose(theta, back, atol=1e-5)


def test_normalised_prior_bounds_are_ordered(cfg, theta_dim):
    """get_normalised_prior_bounds must return (low, high) with low <= high
    even when theta_std flips a sign somewhere."""
    
    std = _std(cfg, 32, theta_dim)
    lo = torch.full((theta_dim,), -1.0)
    hi = torch.full((theta_dim,), 3.0)
    nlo, nhi = std.get_normalised_prior_bounds(lo, hi)

    assert (nlo <= nhi).all()


def test_normalise_x_leaves_psd_block_untouched(cfg):
    """The PSD context is z-scored per-bin when it is built, so the
    standardiser must pass it through unchanged."""
    from bilbyflow.data.standardiser import Standardiser
    torch.manual_seed(0)
    strain_dim, psd_dim = 32, 8
    x = torch.randn(64, strain_dim + psd_dim)
    std = Standardiser(x, torch.randn(64, 4), strain_dim=strain_dim,
                       psd_conditioning=True)
    probe = torch.randn(3, strain_dim + psd_dim)
    out = std.normalise_x(probe)

    assert torch.allclose(out[:, strain_dim:], probe[:, strain_dim:])
    assert not torch.allclose(out[:, :strain_dim], probe[:, :strain_dim])


# --------    whitening    -----------------------------------------------------

def test_whiten_recolour_roundtrip(cfg):
    """whiten_fd then multiply by bn recovers the input at valid bins."""
    from bilbyflow.data.canonical import (canonical_grid, canonical_bn,
                                          canonical_valid_mask, whiten_fd)
    g = canonical_grid(cfg)
    rng = np.random.default_rng(3)
    psd = np.full(g["n_fd_full"], 1e-42)
    psd[~g["freq_mask"]] = np.inf
    bn = canonical_bn(psd, g["df"])
    valid = canonical_valid_mask(bn)

    fd = (rng.normal(size=g["n_fd_full"])
          + 1j * rng.normal(size=g["n_fd_full"])) * 1e-21
    w = whiten_fd(fd, bn)

    assert np.allclose(w[valid] * bn[valid], fd[valid])
    assert np.all(w[~valid] == 0), "whiten_fd left non-zero invalid bins"


def test_window_fd_zeroes_out_of_band(cfg):
    """window_fd must mask BEFORE windowing (the D1 fix): out-of-band power
    in the input must not leak back in."""
    from bilbyflow.data.canonical import (canonical_grid, canonical_tukey,
                                          window_fd)
    g = canonical_grid(cfg)
    tukey = canonical_tukey(cfg, g["n_td"])
    fd = np.ones(g["n_fd_full"], dtype=complex) * 1e-20
    out = window_fd(fd, g["freq_mask"], tukey, g["n_td"])

    # windowing spreads power, but the out-of-band input must be gone first:
    # compare against the same call with the out-of-band bins already zeroed
    fd_masked = fd.copy()
    fd_masked[~g["freq_mask"]] = 0.0

    ref = window_fd(fd_masked, g["freq_mask"], tukey, g["n_td"])
    assert np.allclose(out, ref)