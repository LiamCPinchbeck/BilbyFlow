"""
bilbyflow.io.strain — real-event strain loading and analysis-IFO construction.

Given per-event .npy strain (stored as [t0, dt, samples]) and a directory of
off-source noise segments, this module:

  * find_events_from_data   — discover (name, gps) pairs from *_gps.npy files.
  * get_event_psd           — segment-matched median-Welch PSD per detector,
                              ASD-floored, on the analysis frequency grid.
  * fetch_real_strain_and_build_ifos — window + whiten the on-source data into
                              the NPE input x (training-matched ordering) AND
                              set up bilby InterferometerList objects carrying
                              the unwhitened FD data + event PSD, ready for the
                              exact likelihood.

GWOSC_TRIGGER_TIMES holds the canonical trigger GPS times used to centre the
analysis segment (overriding the coarse GPS in the data files when known).

Whitening/windowing/PSD estimation go through data.canonical
(real_strain_to_x_training_order, welch_psd_floored) so the real-event path
is byte-identical to training and the PSD recipe matches the noise-segment
bank; the grid comes from io.config.grid_quantities.

data.canonical is imported lazily inside functions to break the
data.canonical -> io.config -> io.__init__ -> io.strain -> data.canonical
cycle. Will try to un-spaghetti-fy later.
"""

import os
import glob

import numpy as np
import bilby

from .config import grid_quantities, window_quantities, td_norm_from

__all__ = [
    "GWOSC_TRIGGER_TIMES", "find_events_from_data",
    "get_event_psd", "fetch_real_strain_and_build_ifos",
]


GWOSC_TRIGGER_TIMES = {
    "GW150914": 1126259462.4,
    "GW190521_074359": 1242459857.5,
}


def find_events_from_data(data_dir, event_names=None):
    """(name, gps) for every event with H1 + L1 strain files present."""

    events = []
    for gps_file in sorted(glob.glob(os.path.join(data_dir, "*_gps.npy"))):
        name = os.path.basename(gps_file).replace("_gps.npy", "")

        if event_names and name not in event_names:
            continue

        h1 = os.path.join(data_dir, f"{name}_H1.npy")
        l1 = os.path.join(data_dir, f"{name}_L1.npy")

        if not (os.path.exists(h1) and os.path.exists(l1)):
            print(f"  skip {name}: missing H1 or L1 strain file")
            continue

        gps = float(np.load(gps_file)[0])
        events.append((name, gps))
    return events


def get_event_psd(event_name, cfg, noise_data_dir):
    """Segment-matched median-Welch PSD per detector on the analysis grid.

    Scans noise_data_dir/<era>/<event>_{H1,L1}_noise.npy and routes the
    estimation through canonical.welch_psd_floored (median Welch, no f_min
    inf-mask, in-band ASD floor at median/asd_floor_factor — identical to
    the training noise-segment bank's per-file PSDs). Returns
    (det_psds, era_dir)."""
    from ..data.canonical import welch_psd_floored

    g = grid_quantities(cfg)
    sr = g["sr"]
    floor_factor = float(cfg.get("psd_bank", {}).get("asd_floor_factor", 5.0))

    for era_dir in sorted(os.listdir(noise_data_dir)):
        era_path = os.path.join(noise_data_dir, era_dir)
        if not os.path.isdir(era_path):
            continue

        h1_file = os.path.join(era_path, f"{event_name}_H1_noise.npy")
        l1_file = os.path.join(era_path, f"{event_name}_L1_noise.npy")


        if not (os.path.exists(h1_file) and os.path.exists(l1_file)):
            continue


        det_psds = {}
        for det, fp in [("H1", h1_file), ("L1", l1_file)]:
            stored = np.load(fp, allow_pickle=True)
            t0, dt = float(stored[0]), float(stored[1])
            strain = stored[2].astype(np.float64)
            sr_actual = int(round(1.0 / dt))

            if sr_actual != sr:
                print(f"  WARNING: {det} noise has sr={sr_actual}, "
                      f"expected {sr}")
                continue

            print(f"  {det}: t0={t0:.1f}, length={len(strain) / sr:.0f}s")

            det_psds[det] = welch_psd_floored(strain, cfg, g,
                                              floor_factor=floor_factor)
            
        print(f"  PSD from {era_dir} noise segments "
              f"(tukey median Welch, floored)")
        return det_psds, era_dir

    raise FileNotFoundError(f"No noise files found for {event_name} "
                            f"in {noise_data_dir}")


def _load_on_source(datafile, on_source_start, duration, sr):
    """One detector's on-source TD segment: mask to the analysis window,
    clean non-finite samples, pad/trim to the exact length."""

    stored = np.load(datafile, allow_pickle=True)
    t0, dt = float(stored[0]), float(stored[1])
    strain_vals = stored[2].astype(np.float64)

    times = t0 + dt * np.arange(len(strain_vals))

    on_mask = (times >= on_source_start) & (times < on_source_start + duration)
    on_source = strain_vals[on_mask]

    if not np.all(np.isfinite(on_source)):
        on_source = np.nan_to_num(on_source, nan=0.0, posinf=0.0, neginf=0.0)
    
    expected = int(duration * sr)

    if len(on_source) < expected:
        return np.pad(on_source, (0, expected - len(on_source)))

    return on_source[:expected]


def fetch_real_strain_and_build_ifos(event_name, gps_time, cfg, data_dir,
                                     noise_data_dir):
    """Window + whiten the on-source strain into x and build the analysis IFOs.

    Returns (x_npe, ifos, det_psds, det_var_q):
      x_npe    : NPE input in training-matched channel order
      ifos     : bilby InterferometerList with the unwhitened FD data + PSD
      det_psds : per-detector event PSD on the analysis grid
      det_var_q: per-detector whitened-quadrature variance (windowing sanity)
    """

    g = grid_quantities(cfg)
    freq_array = g["freq_array"]
    n_td, sr, duration, f_min = g["n_td"], g["sr"], g["duration"], g["f_min"]

    tukey_window, window_factor = window_quantities(cfg, n_td)
    td_norm = td_norm_from(g, window_factor)

    det_psds, era_used = get_event_psd(event_name, cfg, noise_data_dir)
    print(f"  Using PSD from era: {era_used}")
    on_source_start = gps_time - duration / 2.0

    ifos = bilby.gw.detector.InterferometerList(["H1", "L1"])
    on_source_td = {}
    for ifo in ifos:
        ifo.minimum_frequency = f_min
        datafile = os.path.join(data_dir, f"{event_name}_{ifo.name}.npy")
        if not os.path.exists(datafile):
            raise FileNotFoundError(f"No strain file at {datafile}")
        on_source_td[ifo.name] = _load_on_source(datafile, on_source_start,
                                                 duration, sr)

    # training-matched windowing (mask-first via window_fd, inside canonical)
    from ..data.canonical import real_strain_to_x_training_order
    x_npe, whitened_fd_by_det, total_fd_by_det, det_var_q = \
        real_strain_to_x_training_order(
            on_source_td, det_psds, cfg, g, tukey_window, td_norm, sr)


    for ifo in ifos:
        ifo.set_strain_data_from_frequency_domain_strain(
            total_fd_by_det[ifo.name],
            sampling_frequency=float(sr), duration=float(duration),
            start_time=float(on_source_start))
        ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(
            frequency_array=freq_array, psd_array=det_psds[ifo.name])

    for det, vq in det_var_q.items():
        print(f"  {det}: var_q={vq:.3f} (baseline ~0.94)")

    return x_npe, ifos, det_psds, det_var_q