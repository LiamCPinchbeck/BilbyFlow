"""
bilbyflow.inference.injections — synthetic-event construction for efficiency
analyses.

create_injection_and_npe_input projects a parameter draw through the SAME
whitening path as training (data.canonical / data.noise) and returns both the
NPE input vector and the bilby IFOs holding the unwhitened data, so the
reweighting target is exactly the injected data. 

Noise comes from 
- the PSD bank (gaussian modes) or 
- real segments (data.noise.draw_real_noise_pair, with full-file or off-source PSD scope).

draw_injection_params wraps the prior draw with the optimal-SNR gate
(--min-snr / --max-snr) and the nuisance-parameter fill.
"""

import numpy as np
import bilby

from ..io.config import grid_quantities, window_quantities, td_norm_from
from ..likelihood.waveform import make_waveform_generator
from ..likelihood.snr import optimal_snr_squared
from ..data.noise import injection_to_x, draw_real_noise_pair

__all__ = ["SIDEREAL_DAY", "create_injection_and_npe_input",
           "draw_injection_params"]

SIDEREAL_DAY = 86164.0905  # seconds, if you couldn't guess that...

_FILL_RANGES = {"phase": (0.0, 2 * np.pi), "psi": (0.0, np.pi),
                "phi_12": (0.0, 2 * np.pi), "phi_jl": (0.0, 2 * np.pi)}
_ZERO_FILL = ["chirp_mass", "mass_ratio", "luminosity_distance", "theta_jn",
              "ra", "dec", "geocent_time", "a_1", "a_2", "tilt_1", "tilt_2"]


def create_injection_and_npe_input(params, cfg, psd_bank,
                                   noise_source="gaussian_whitened",
                                   noise_bank=None, noise_index=None,
                                   ref_tc_override=None, psd_scope="full-file"):
    """Synthesise one injection. Returns
    (x_strain, ifos, era_used, det_psds, noise_var_q)."""

    g = grid_quantities(cfg)
    freq_array = g["freq_array"]
    n_fd_full, n_td = g["n_fd_full"], g["n_td"]
    sr, duration, f_min = g["sr"], g["duration"], g["f_min"]

    tukey_window, _ = window_quantities(cfg, n_td)
    td_norm = td_norm_from(g)



    det_noise_td = None
    if noise_source == "real":

        if noise_index is not None:
            det_noise_td, det_psds, era_used = draw_real_noise_pair(
                noise_index, cfg, g, psd_scope=psd_scope)


        elif noise_bank is not None:
            seg_idx = np.random.randint(noise_bank["noise_H1"].shape[0])
            det_noise_td = {"H1": noise_bank["noise_H1"][seg_idx],
                            "L1": noise_bank["noise_L1"][seg_idx]}
            if "psd_H1" in noise_bank and "psd_L1" in noise_bank:
                det_psds = {"H1": np.asarray(noise_bank["psd_H1"][seg_idx], dtype=np.float64),
                            "L1": np.asarray(noise_bank["psd_L1"][seg_idx], dtype=np.float64)}
            else:
                pidx = np.random.randint(psd_bank["psd_H1"].shape[0])
                det_psds = {"H1": psd_bank["psd_H1"][pidx].astype(np.float64),
                            "L1": psd_bank["psd_L1"][pidx].astype(np.float64)}
            era_used = str(noise_bank["era"][seg_idx]) if "era" in noise_bank else "real_segment"

        else:
            raise ValueError("noise_source='real' needs a noise index or noise bank.")

    else:
        psd_idx = np.random.randint(psd_bank["psd_H1"].shape[0])
        det_psds = {"H1": psd_bank["psd_H1"][psd_idx].astype(np.float64),
                    "L1": psd_bank["psd_L1"][psd_idx].astype(np.float64)}
        era_used = str(psd_bank["era"][psd_idx])



    ref_tc = float(cfg["ref_geocent_time"]) if ref_tc_override is None else float(ref_tc_override)
    start_time = ref_tc - duration / 2.0
    params_f = {k: float(v) for k, v in params.items()}



    ifos = bilby.gw.detector.InterferometerList(["H1", "L1"])
    wfg = make_waveform_generator(cfg, start_time=start_time)
    pols = wfg.frequency_domain_strain(params_f)



    signal_fd_by_det = {}
    for ifo in ifos:
        ifo.minimum_frequency = f_min
        ifo.set_strain_data_from_frequency_domain_strain(
            np.zeros(n_fd_full, dtype=complex),
            sampling_frequency=float(sr), duration=float(duration),
            start_time=float(start_time))
        ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(
            frequency_array=freq_array, psd_array=det_psds[ifo.name])
        signal_fd_by_det[ifo.name] = ifo.get_detector_response(pols, params_f)

    x_npe, _sig_w, total_fd_by_det, noise_var_q = injection_to_x(
        signal_fd_by_det, det_psds, cfg, g, tukey_window, td_norm,
        noise_kind=noise_source, noise_segments=det_noise_td)

    for ifo in ifos:
        ifo.set_strain_data_from_frequency_domain_strain(
            total_fd_by_det[ifo.name],
            sampling_frequency=float(sr), duration=float(duration),
            start_time=float(start_time))

    return x_npe, ifos, era_used, det_psds, noise_var_q


def draw_injection_params(inj_priors, cfg, psd_bank, noise_source, noise_bank,
                          noise_index, inj_ref, psd_scope,
                          min_snr=8.0, max_snr=None, approximant=None,
                          window_template=False, max_tries=1000):
    """Draw injection params + synthesise data until the optimal-SNR gate
    passes. Returns (params, x_npe, ifos, era_used, det_psds, noise_var_q,
    rho_inj) or None after max_tries."""

    g = grid_quantities(cfg)
    tukey_window, _ = window_quantities(cfg, g["n_td"])


    for _try in range(max_tries):
        cand = inj_priors.sample()
        cand["geocent_time"] = inj_ref  
        for p, (lo, hi) in _FILL_RANGES.items(): 
            if p not in cand:
                cand[p] = float(np.random.uniform(lo, hi))
        for p in _ZERO_FILL:
            cand.setdefault(p, 0.0)

    
        try:
            x_npe, ifos, era_used, det_psds, noise_var_q = \
                create_injection_and_npe_input(
                    cand, cfg, psd_bank, noise_source=noise_source,
                    noise_bank=noise_bank, noise_index=noise_index,
                    ref_tc_override=inj_ref, psd_scope=psd_scope)
        except Exception as e:
            print(f"  Error creating injection: {e}")
            continue


        if min_snr <= 0 and max_snr is None:
            return cand, x_npe, ifos, era_used, det_psds, noise_var_q, np.nan

        wfg_chk = make_waveform_generator(
            cfg, start_time=ifos[0].strain_data.start_time,
            approximant=approximant, windowed=window_template, g=g,
            tukey_window=tukey_window)
    
        rho_inj = float(np.sqrt(optimal_snr_squared(ifos, wfg_chk, cand)))
    
        if rho_inj < min_snr:
            continue
    
        if max_snr is not None and rho_inj > max_snr:
            continue
    
        print(f"  injected rho_opt = {rho_inj:.1f} (draw {_try + 1})")
    
        return cand, x_npe, ifos, era_used, det_psds, noise_var_q, rho_inj
    return None