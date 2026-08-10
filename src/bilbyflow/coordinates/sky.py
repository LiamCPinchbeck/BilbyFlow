"""
sky.py — Detector-frame sky parametrisation for the H1-L1 network.

The flow infers (dt_HL, φ_det) instead of (ra, dec):
  dt_HL   : H1-L1 arrival-time delay (physically encoded in the strain)
  φ_det   : azimuth about the H1-L1 baseline

These are Earth-fixed coordinates, independent of the absolute epoch.
GMST matters only when converting back to celestial (ra, dec): pass the
observation's tc to detector_to_radec / samples_detector_to_radec.
"""

import numpy as np
import bilby

C_SI = 299792458.0

# Precompute the H1-L1 baseline in ECEF
_H1 = bilby.gw.detector.get_empty_interferometer("H1")
_L1 = bilby.gw.detector.get_empty_interferometer("L1")
_BASELINE_ECEF = _H1.geometry.vertex - _L1.geometry.vertex
MAX_DT_HL = np.linalg.norm(_BASELINE_ECEF) / C_SI


def _earth_rotation_matrix(gmst):
    """Rotation matrix from ECEF to celestial frame at given GMST."""
    cg, sg = np.cos(gmst), np.sin(gmst)
    return np.array([[cg, -sg, 0.0],
                     [sg,  cg, 0.0],
                     [0.0, 0.0, 1.0]])


def _detector_frame(tc):
    """Orthonormal frame: z along the celestial-frame H1-L1 baseline at
    GMST(tc), x toward celestial north's projection. Computed once per tc."""
    gmst = bilby.gw.utils.greenwich_mean_sidereal_time(tc)
    baseline = _earth_rotation_matrix(gmst) @ _BASELINE_ECEF

    z_hat = baseline / np.linalg.norm(baseline)
    north = np.array([0.0, 0.0, 1.0])
    x_hat = north - np.dot(north, z_hat) * z_hat
    if np.linalg.norm(x_hat) < 1e-10:
        x_hat = np.array([1.0, 0.0, 0.0])
    x_hat /= np.linalg.norm(x_hat)
    y_hat = np.cross(z_hat, x_hat)
    y_hat /= np.linalg.norm(y_hat)

    return x_hat, y_hat, z_hat


# ── Scalar conversions ──────────────────────────────────────────────────────

def radec_to_detector(ra, dec, tc):
    """Celestial (ra, dec) → detector-frame (dt_HL, φ_det) at time tc."""
    n = np.array([np.cos(dec) * np.cos(ra),
                  np.cos(dec) * np.sin(ra),
                  np.sin(dec)])
    x_hat, y_hat, z_hat = _detector_frame(tc)
    phi_det = np.mod(np.arctan2(np.dot(n, y_hat), np.dot(n, x_hat)), 2 * np.pi)
    dt_HL = -np.dot(n, z_hat) * MAX_DT_HL
    return float(dt_HL), float(phi_det)


def detector_to_radec(dt_HL, phi_det, tc):
    """Detector-frame (dt_HL, φ_det) → celestial (ra, dec) at time tc."""
    nz = np.clip(-dt_HL / MAX_DT_HL, -1.0, 1.0)
    rho = np.sqrt(1.0 - nz ** 2)
    nx = rho * np.cos(phi_det)
    ny = rho * np.sin(phi_det)

    x_hat, y_hat, z_hat = _detector_frame(tc)
    n = nx * x_hat + ny * y_hat + nz * z_hat

    ra = np.mod(np.arctan2(n[1], n[0]), 2 * np.pi)
    dec = np.arcsin(np.clip(n[2], -1.0, 1.0))
    return float(ra), float(dec)


# ── Vectorised batch conversion ─────────────────────────────────────────────

def samples_detector_to_radec(samples, param_names, tc):
    """Convert NPE samples from detector frame to (ra, dec) — vectorised.

    The frame depends only on tc, so it's computed once; per-sample work
    is pure array algebra (~1000× faster than a Python loop).

    Assumes the 'ra' column stores dt_HL and 'dec' stores φ_det. Pass the
    observation's tc (trigger time for real events, ref_geocent_time for
    training-frame injections).

    Parameters
    ----------
    samples : ndarray (N, n_params)
    param_names : list[str]
    tc : float

    Returns
    -------
    ndarray (N, n_params) with ra/dec columns replaced.
    """
    if "ra" not in param_names or "dec" not in param_names:
        return samples

    ra_idx = param_names.index("ra")
    dec_idx = param_names.index("dec")

    dt_HL = np.asarray(samples[:, ra_idx], dtype=np.float64)
    phi_det = np.asarray(samples[:, dec_idx], dtype=np.float64)

    x_hat, y_hat, z_hat = _detector_frame(tc)

    nz = np.clip(-dt_HL / MAX_DT_HL, -1.0, 1.0)
    rho = np.sqrt(1.0 - nz ** 2)
    nx = rho * np.cos(phi_det)
    ny = rho * np.sin(phi_det)

    n = (nx[:, None] * x_hat[None, :]
         + ny[:, None] * y_hat[None, :]
         + nz[:, None] * z_hat[None, :])

    out = samples.copy()
    out[:, ra_idx] = np.mod(np.arctan2(n[:, 1], n[:, 0]), 2 * np.pi)
    out[:, dec_idx] = np.arcsin(np.clip(n[:, 2], -1.0, 1.0))
    return out