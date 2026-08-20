"""
bilbyflow.coordinates.sky — detector-frame sky reparameterisation.

This is the package home of the symbols that previously lived in
sky_coords_reparam_conditional_PSD_b.py, with conventions IDENTICAL to that
module (the trained models, sky banks, and cached samples all assume them).

Downstream stuff imports from here:

  * MAX_DT_HL                : max |Hanford-Livingston| geometric time delay
                               [s]; the (ra -> dt_HL) prior half-width. NO
                               safety margin — it must equal the trained
                               models' constant exactly.

  * radec_to_detector(ra, dec, tc) -> (dt_HL, phi_det)
                               forward map (used when building the sky bank).

  * detector_to_radec(dt_HL, phi_det, tc) -> (ra, dec)
                               scalar inverse.

  * samples_detector_to_radec(samples, param_names, tc) -> samples
                               vectorised inverse (used post-sampling to put
                               ra/dec back into physical sky coordinates for
                               corner plots and the reweighting likelihood).

WHY REPARAMETERISE.
Training the flow directly in (ra, dec) is hard (look at the posteriors for
any mildly localised event): the detector network only constrains the
arrival-time delay dt_HL and a detector-frame azimuth phi_det, and the
(ra, dec) posterior is multi-modal (mirror sky positions). The flow learns
dt_HL (bounded by the light travel time between detectors) and phi_det (an
azimuth in [0, 2pi)) instead; the physical (ra, dec) are recovered afterward.
So the `ra` slot in theta actually carries dt_HL and `dec` carries phi_det
during training/inference, and only samples_detector_to_radec turns them
back. Not the best coding practice, made a todo to rework in later updates.

GMST HANDLING:
  * (dt_HL, phi_det) are EARTH-FIXED coordinates of the source direction (the
    detector frame rotates with the Earth). 
    They are what the strain encodes, independent of the absolute epoch.
  * A GMST change is a rigid rotation about Earth's axis. With ra/dec drawn
    ISOTROPICALLY, the induced distribution of every Earth-frame quantity
    (fp, fc, dt_H, dt_L, dt_HL, phi_det) is IDENTICAL at any fixed GMST —
    randomizing a sidereal offset in the sky bank would be a
    distribution-level no-op (verified by Monte Carlo). The bank's
    sidereal_offset = 0 is NOT a training-distribution deficiency.
  * The GMST matters only when converting (dt_HL, phi_det) back to celestial
    (ra, dec): pass the tc of the OBSERVATION — ref_geocent_time for
    training-frame injections, the trigger time for real events. It does NOT
    "cancel statistically", it is simply the frame of the inversion.

CONVENTIONS (DO NOT CHANGE ((for now)) ):
  * baseline  = vertex_H1 - vertex_L1  (H1 minus L1, ECEF metres)
  * detector frame at GMST(tc): z along the celestial-frame baseline,
    x toward celestial north's projection, y = z cross x
  * dt_HL     = -n_z * MAX_DT_HL   (n = source unit vector in that frame)
  * phi_det   = atan2(n_y, n_x) mod 2pi
"""

import numpy as np

__all__ = ["MAX_DT_HL", "H1_L1_LIGHT_TRAVEL_TIME", "radec_to_detector",
           "detector_to_radec", "samples_detector_to_radec"]


# ── detector geometry (from bilby's H1/L1 vertices) ──────────────────────────

_C_SI = 299792458.0  # m/s


def _hl_baseline():
    """H1 - L1 baseline vector in geocentric metres (ECEF)."""
    import bilby
    h1 = bilby.gw.detector.get_empty_interferometer("H1")
    l1 = bilby.gw.detector.get_empty_interferometer("L1")
    return np.asarray(h1.geometry.vertex - l1.geometry.vertex,
                      dtype=np.float64)


try:
    _baseline_ecef = _hl_baseline()
    H1_L1_LIGHT_TRAVEL_TIME = float(np.linalg.norm(_baseline_ecef) / _C_SI)
except Exception:   # bilby unavailable at import time (e.g. docs build)
    _baseline_ecef = None
    H1_L1_LIGHT_TRAVEL_TIME = 0.010002567302556083   # bilby's H1-L1 value [s]

# NO margin: MAX_DT_HL enters the dt_HL <-> n_z scaling, so any factor here
# rescales the coordinate relative to what the models were trained with.
MAX_DT_HL = H1_L1_LIGHT_TRAVEL_TIME


def _gmst(tc):
    """Greenwich Mean Sidereal Time (rad) at GPS tc, via bilby/lal."""
    from bilby.gw.utils import greenwich_mean_sidereal_time
    return float(greenwich_mean_sidereal_time(float(tc)))


def _earth_rotation_matrix(gmst):
    cg, sg = np.cos(gmst), np.sin(gmst)
    return np.array([[cg, -sg, 0.0],
                     [sg,  cg, 0.0],
                     [0.0, 0.0, 1.0]])


def _detector_frame(tc):
    """Orthonormal frame with z along the (celestial-frame) H1-L1 baseline at
    GMST(tc) and x toward celestial north's projection. Depends on tc only
    through the GMST, so compute it ONCE per conversion call."""
    if _baseline_ecef is None:
        raise RuntimeError("bilby detector geometry unavailable; cannot build "
                           "the sky reparameterisation. Install bilby.")
    gmst = _gmst(tc)
    baseline_celestial = _earth_rotation_matrix(gmst) @ _baseline_ecef
    z_hat = baseline_celestial / np.linalg.norm(baseline_celestial)
    north = np.array([0.0, 0.0, 1.0])
    x_hat = north - np.dot(north, z_hat) * z_hat
    if np.linalg.norm(x_hat) < 1e-10:
        x_hat = np.array([1.0, 0.0, 0.0])
    x_hat /= np.linalg.norm(x_hat)
    y_hat = np.cross(z_hat, x_hat)
    y_hat /= np.linalg.norm(y_hat)
    return x_hat, y_hat, z_hat


# ── (ra, dec) -> (dt_HL, phi_det) ───────────────────────────────────

def radec_to_detector(ra, dec, tc):
    """Celestial (ra, dec) at time tc -> Earth-frame (dt_HL, phi_det)."""
    n = np.array([np.cos(dec) * np.cos(ra),
                  np.cos(dec) * np.sin(ra),
                  np.sin(dec)])
    
    x_hat, y_hat, z_hat = _detector_frame(tc)

    nx = np.dot(n, x_hat)
    ny = np.dot(n, y_hat)
    nz = np.dot(n, z_hat)

    phi_det = np.mod(np.arctan2(ny, nx), 2 * np.pi)

    dt_HL = -nz * MAX_DT_HL

    return float(dt_HL), float(phi_det)


# ── detector-frame -> physical (ra, dec) ────────────────────────────

def detector_to_radec(dt_HL, phi_det, tc):
    """Earth-frame (dt_HL, phi_det) -> celestial (ra, dec) at time tc.
    tc must be the GMST epoch of the OBSERVATION (see module docstring)."""
    nz = np.clip(-dt_HL / MAX_DT_HL, -1.0, 1.0)

    rho = np.sqrt(1.0 - nz ** 2)
    nx = rho * np.cos(phi_det)
    ny = rho * np.sin(phi_det)
    x_hat, y_hat, z_hat = _detector_frame(tc)
    n = nx * x_hat + ny * y_hat + nz * z_hat

    ra = np.mod(np.arctan2(n[1], n[0]), 2 * np.pi)
    dec = np.arcsin(np.clip(n[2], -1.0, 1.0))

    return float(ra), float(dec)


def samples_detector_to_radec(samples, param_names, tc):
    """
    Convert NPE samples from detector frame back to (ra, dec) — VECTORIZED
    (the frame depends only on tc, so it is computed once; the per-sample
    work is pure array algebra).

    Assumes the 'ra' column stores dt_HL and the 'dec' column stores phi_det.
    Pass the OBSERVATION's tc: ref_geocent_time for training-frame
    injections, the event trigger time for real data. No-op when ra/dec are
    not both inferred; other columns are untouched.

    Parameters
    ----------
    samples : np.ndarray (N, n_params)
    param_names : list[str]
    tc : float

    Returns
    -------
    samples_out : np.ndarray (N, n_params) with the ra/dec replaced
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