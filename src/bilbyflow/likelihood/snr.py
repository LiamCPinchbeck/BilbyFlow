"""
bilbyflow.likelihood.snr — SNR bookkeeping using the windowed analysis inner
product (4 df / PSD over the in-band, finite-PSD bins). 

Tried to replicate what Bilby natively does as much as possible. Should be 
pretty simple but nonetheless.
"""

import numpy as np

__all__ = ["rho_opt_mf", "optimal_snr_squared"]


def optimal_snr_squared(ifos, wfg, params):
    """rho_opt^2 = <h|h> summed over detectors (same inner product as
    rho_opt_mf; data is not touched)."""
    return rho_opt_mf(ifos, wfg, params)[0] ** 2


def rho_opt_mf(ifos, wfg, params):
    """(rho_opt, rho_mf): rho_opt = sqrt(<h|h>), rho_mf = <d|h>/sqrt(<h|h>).
    Uses the same 4 df / PSD weighting as the GravitationalWaveTransient
    likelihood, restricted to finite-PSD in-band bins."""
    pf = {k: float(v) for k, v in params.items()}
    pols = wfg.frequency_domain_strain(pf)
    hh = dh = 0.0
    for ifo in ifos:
        h = ifo.get_detector_response(pols, pf)
        d = ifo.strain_data.frequency_domain_strain
        psd = ifo.power_spectral_density_array
        df = 1.0 / ifo.strain_data.duration
        m = np.isfinite(psd) & (psd > 0) & ifo.frequency_mask
        w = 4.0 * df / psd[m]
        hh += float(np.sum(w * np.abs(h[m]) ** 2))
        dh += float(np.sum(w * np.real(np.conj(d[m]) * h[m])))
    rho_opt = np.sqrt(hh)
    return float(rho_opt), float(dh / rho_opt if rho_opt > 0 else np.nan)