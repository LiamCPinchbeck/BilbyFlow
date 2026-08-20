"""
bilbyflow.data.banks — precomputed banks feeding OnTheFlyGWDataset particularly for training.
The waveform, sky directional characteristics and noise profile are treated independently, 
then combined during runtime. Hope is that this leads to a much higher effective trainin set
size than if they were all used simultaneously.

  precompute_waveforms          — parallel FD waveform bank at D_REF (intrinsic
                                  only; sky/distance applied on the fly).
  precompute_sky_bank           — antenna responses, time delays, dt_HL/phi_det
                                  reparam, distance draws.
  precompute_noise_segment_bank — real detector-noise segments + per-file PSDs
                                  for noise_source="real" and JEPA consistency.
  load_or_compute               — just a pickle-cache helper.

Training-side PROPOSAL RESHAPING (dL/Mc PowerLaw pads) IS APPLIED HERE, in the
banks; the physical prior in inference.priors is unchanged, and the importance
weight w = L pi_phys / q should correct for the difference.

Depends on coordinates.sky.radec_to_detector and inference.priors.make_prior_dict.

Different to convention to Bilby atm, but will make it in line in future updates,
however much that displeases me. 
"""

import os
import pickle
import numpy as np
import bilby
from bilby.core.utils.random import seed as bilby_seed
from tqdm import tqdm

from ..coordinates.sky import radec_to_detector
from ..inference.priors import make_prior_dict

__all__ = [
    "D_REF", "precompute_waveforms", 
    "precompute_sky_bank",
    "precompute_noise_segment_bank", 
    "load_or_compute",
]

# Making it an explicit constant for readability
D_REF = 1.0  # reference distance (Mpc) for stored waveforms

# just a handy wraper for the waveform generator really
def _bank_waveform_generator(cfg):
    """Bank-side generator: no start_time, reference_frequency = f_min.
    (Distinct from likelihood.waveform.make_waveform_generator, which sets
    start_time for the analysis IFOs.)"""
    return bilby.gw.WaveformGenerator(
        duration=float(cfg["duration"]),
        sampling_frequency=int(cfg["sampling_frequency"]),
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
        waveform_arguments=dict(
            waveform_approximant=cfg["waveform_approximant"],
            reference_frequency=float(cfg["f_min"])))


# ── waveforms (parallelization) ─────────────────────────────────────────────────────
# not saying you can't do this all single-threaded, but would not recommend.

_WFG = None

def _wf_worker_init(cfg):
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS"):
        os.environ[v] = "1"
    global _WFG
    bilby.core.utils.setup_logger(log_level="WARNING")
    _WFG = _bank_waveform_generator(cfg)


# Should do the chunking of the input parameters
def _wf_worker_chunk(params_chunk):
    ref_tc = _wf_worker_chunk._ref_tc
    hps, hcs, stored_list = [], [], []
    for sampled in params_chunk:
        params = dict(sampled)
        params["geocent_time"] = ref_tc
        params["ra"] = 0.0
        params["dec"] = 0.0
        params["luminosity_distance"] = 1.0      # D_REF, rescaled on the fly
        try:
            pols = _WFG.frequency_domain_strain(params)
            hp, hc = pols["plus"], pols["cross"]
        except Exception:
            continue
        if np.any(np.isnan(hp)) or np.any(np.isnan(hc)):
            continue
        hps.append(hp.astype(np.complex64))
        hcs.append(hc.astype(np.complex64))
        stored_list.append({p: float(sampled[p]) for p in sampled})
    return hps, hcs, stored_list


# You will notice quite a few print statements. It is because I didn't want to have
    # to deal with any bugs related to the Bilby logger. Will fix in future updates
# TODO: Remove print statements and move them into a Bilby logger
def precompute_waveforms(cfg, seed_val=42):
    import multiprocessing as _mp
    import time as _time

    bilby_seed(seed_val)
    bilby.core.utils.setup_logger(log_level="WARNING")
    ref_tc = float(cfg["ref_geocent_time"])

    priors = make_prior_dict(cfg)
    for p in ("geocent_time", "ra", "dec", "luminosity_distance"):
        priors.pop(p, None)

    # training-side Mc pad / proposal reshape (physical pi unchanged)
    _pr = cfg["priors"]["chirp_mass"]

    # Motivated by a paper "Don't Cut Corners", better to slightly pad the 
        # parameter space to avoid edge effects and give more diverse training
    _pad = ((cfg.get("training_prior_pad", {}) or {}).get("chirp_mass", {}) or {})


    mc_min = float(_pad.get("min", _pr["min"]))
    mc_max = float(_pad.get("max", _pr["max"]))

    # Typically left as None at the moment, but can be changed
    _a = cfg.get("Mc_train_alpha", None)


    if _a is not None:
        from bilby.core.prior import PowerLaw
        priors["chirp_mass"] = PowerLaw(alpha=float(_a), minimum=mc_min,
                                        maximum=mc_max, name="chirp_mass")
        print(f"  [waveform bank] Mc proposal PowerLaw(alpha={_a}) on [{mc_min}, {mc_max}]")
    elif (mc_min, mc_max) != (float(_pr["min"]), float(_pr["max"])):
        priors["chirp_mass"] = bilby.gw.prior.UniformInComponentsChirpMass(
            minimum=mc_min, maximum=mc_max, name="chirp_mass")
        print(f"  [waveform bank] Mc pad [{mc_min}, {mc_max}]")

    n = int(cfg["n_waveforms"])

    print(f"  drawing {n} intrinsic parameter sets (seed {seed_val})...")

    draws = priors.sample(n)

    keys = list(draws.keys())
    param_list = [{k: float(draws[k][i]) for k in keys} for i in range(n)]
    del draws


    n_workers = int(cfg.get("n_waveform_workers", 0)) or \
        int(os.environ.get("SLURM_CPUS_PER_TASK", 0)) or (os.cpu_count() or 1)
    n_workers = max(1, min(n_workers, os.cpu_count() or 1))

    h_plus_all, h_cross_all, intrinsic_params = [], [], []

    # For profiling purposes
    _t0 = _time.time()

    if n_workers == 1:
        _wf_worker_init(cfg)
        _wf_worker_chunk._ref_tc = ref_tc
        for i in tqdm(range(0, n, 512), desc="Pre-computing waveforms"):
            hps, hcs, st = _wf_worker_chunk(param_list[i:i + 512])
            h_plus_all += hps
            h_cross_all += hcs
            intrinsic_params += st
    else:
        chunk = 512
        chunks = [param_list[i:i + chunk] for i in range(0, n, chunk)]
        ctx = _mp.get_context("fork")
        _wf_worker_chunk._ref_tc = ref_tc
        print(f"  generating on {n_workers} workers ({len(chunks)} chunks of {chunk})...")
        with ctx.Pool(processes=n_workers, initializer=_wf_worker_init,
                      initargs=(cfg,)) as pool:
            for hps, hcs, st in tqdm(pool.imap(_wf_worker_chunk, chunks),
                                     total=len(chunks), desc="Pre-computing waveforms"):
                h_plus_all += hps
                h_cross_all += hcs
                intrinsic_params += st

    # For profiling purposes
    dt = _time.time() - _t0

    n_ok = len(h_plus_all)

    print(f"Pre-computed {n_ok} waveforms in {dt/60:.1f} min "
          f"({n_ok/max(dt,1e-9):.0f}/s, {n - n_ok} failed/NaN)")

    return h_plus_all, h_cross_all, intrinsic_params


# ── sky bank ─────────────────────────────────────────────────────────────────

def precompute_sky_bank(cfg, n_sky=1_000_000):

    # Currently fixed to H1/L1, future updates will hopefully allow specification on behalf of the user
        # which detectors are used (H1+L1, H1+L1+V1, L1+V1, etc)
    detectors = [bilby.gw.detector.get_empty_interferometer("H1"),
                 bilby.gw.detector.get_empty_interferometer("L1")]

    
    ref_tc = float(cfg["ref_geocent_time"]) # typically GW150914 because why not
    priors = make_prior_dict(cfg)

    # training-side dL proposal reshape / pad (physical pi unchanged)
    dL_train = priors["luminosity_distance"]

    # usually set to -1 at the moment
    _a = cfg.get("dL_train_alpha", None)
    _pr = cfg["priors"]["luminosity_distance"]

    # see above on padding
    _pad = ((cfg.get("training_prior_pad", {}) or {}).get("luminosity_distance", {}) or {})

    # first paper used 100Mpc to 5200Mpc
    d_min = float(_pad.get("min", _pr["min"]))
    d_max = float(_pad.get("max", _pr["max"]))

    if _a is not None or (d_min, d_max) != (float(_pr["min"]), float(_pr["max"])):
        from bilby.core.prior import PowerLaw
        alpha_eff = float(_a) if _a is not None else 2.0
        dL_train = PowerLaw(alpha=alpha_eff, minimum=d_min, maximum=d_max,
                            name="luminosity_distance")
        print(f"  [sky bank] dL proposal PowerLaw(alpha={alpha_eff}) on [{d_min}, {d_max}]")


    sky_bank = {}
    for det in ["H1", "L1"]: # again, H1/L1 is fixed atm. Will change later
        for key in ("fp", "fc", "dt"):
            sky_bank[f"{key}_{det}"] = np.empty(n_sky, dtype=np.float64)


    for key in ("psi", "dL", "tc_offset", "dt_HL", "phi_det"):
        sky_bank[key] = np.empty(n_sky, dtype=np.float64)


    sky_bank["sidereal_offset"] = np.zeros(n_sky, dtype=np.float64)

    for i in tqdm(range(n_sky), desc="Precomputing sky bank"):
        ra = priors["ra"].sample()
        dec = priors["dec"].sample()
        psi = priors["psi"].sample()
        dL = dL_train.sample()
        tc_offset = priors["geocent_time"].sample() - ref_tc


        for det, name in zip(detectors, ["H1", "L1"]):
            sky_bank[f"fp_{name}"][i] = det.antenna_response(ra, dec, ref_tc, psi, "plus")
            sky_bank[f"fc_{name}"][i] = det.antenna_response(ra, dec, ref_tc, psi, "cross")
            sky_bank[f"dt_{name}"][i] = det.time_delay_from_geocenter(ra, dec, ref_tc)

        dt_HL, phi_det = radec_to_detector(ra, dec, ref_tc)
        sky_bank["dL"][i] = dL
        sky_bank["tc_offset"][i] = tc_offset
        sky_bank["dt_HL"][i] = dt_HL
        sky_bank["phi_det"][i] = phi_det
        sky_bank["psi"][i] = psi
    return sky_bank


# ── real-noise segment bank ──────────────────────────────────────────────────

def precompute_noise_segment_bank(cfg, noise_data_dir):
    """Real detector-noise segments + per-file PSDs for noise_source="real"
    and embedding consistency. Each segment records its file's PSD row so
    training whitens it with the SAME PSD the real path would use."""
    import glob
    from scipy.signal import welch, butter, sosfiltfilt
    from scipy.interpolate import interp1d

    bank_cfg = cfg.get("noise_segment_bank", {}) or {}
    n_segments = int(bank_cfg.get("n_segments", 50000))
    eras_filter = bank_cfg.get("eras", None)
    f_hp = float(bank_cfg.get("highpass_fc", 15.0))
    edge_margin_s = float(bank_cfg.get("edge_margin_s", 4.0))
    asd_floor_factor = float(cfg.get("psd_bank", {}).get("asd_floor_factor", 5.0))

    duration = float(cfg["duration"])
    sr = int(cfg["sampling_frequency"])
    f_min = float(cfg["f_min"])
    n_fd_full = int(duration * sr / 2) + 1
    n_td = 2 * (n_fd_full - 1)
    freq_array = np.arange(n_fd_full) / duration
    in_band = freq_array >= f_min
    nperseg = n_td
    roll_off = float(cfg.get("tukey_roll_off", 0.2))
    welch_window = ("tukey", 2.0 * roll_off / duration)
    sos = butter(8, f_hp, btype="highpass", fs=sr, output="sos")
    margin = int(edge_margin_s * sr)

    era_dirs = sorted(d for d in os.listdir(noise_data_dir)
                      if os.path.isdir(os.path.join(noise_data_dir, d))
                      and (eras_filter is None or d in eras_filter))
    if not era_dirs:
        raise RuntimeError(f"No era directories in {noise_data_dir} (filter {eras_filter})")
    print(f"Building real-noise segment bank from {noise_data_dir}")
    print(f"  n_segments={n_segments}/det | highpass {f_hp} Hz | eras {era_dirs}")

    file_strain = {"H1": [], "L1": []}
    file_psd = {"H1": [], "L1": []}
    file_era = {"H1": [], "L1": []}
    for era in era_dirs:
        for det in ["H1", "L1"]:
            for fp_ in sorted(glob.glob(os.path.join(noise_data_dir, era, f"*_{det}_noise.npy"))):
                try:
                    stored = np.load(fp_, allow_pickle=True)
                    dt_s = float(stored[1])
                    strain = stored[2].astype(np.float64)
                    if int(round(1.0 / dt_s)) != sr:
                        print(f"  skip {os.path.basename(fp_)}: sr mismatch")
                        continue
                    if not np.all(np.isfinite(strain)):
                        print(f"  skip {os.path.basename(fp_)}: non-finite")
                        continue
                    if len(strain) < n_td + 2 * margin + 1:
                        print(f"  skip {os.path.basename(fp_)}: too short")
                        continue
                    strain = sosfiltfilt(sos, strain)
                    fw, pw = welch(strain, fs=sr, nperseg=nperseg,
                                   noverlap=nperseg // 2, window=welch_window,
                                   detrend="linear", average="median")
                    psd = interp1d(fw, pw, bounds_error=False,
                                   fill_value=np.inf)(freq_array).astype(np.float64)
                    psd[~np.isfinite(psd) | (psd <= 0)] = np.inf
                    asd = np.sqrt(psd)
                    finite = np.isfinite(asd) & (asd > 0)
                    med = np.nanmedian(np.where(finite & in_band, asd, np.nan))
                    floor = med / asd_floor_factor
                    asd = np.where(finite & (asd < floor) & in_band, floor, asd)
                    file_strain[det].append(strain.astype(np.float32))
                    file_psd[det].append(asd ** 2)
                    file_era[det].append(era)
                except Exception as e:
                    print(f"  WARNING {fp_}: {e}")
        print(f"  {era}: {sum(1 for e in file_era['H1'] if e == era)} H1, "
              f"{sum(1 for e in file_era['L1'] if e == era)} L1 files")

    eras_ok = sorted(set(file_era["H1"]) & set(file_era["L1"]))
    if not eras_ok:
        raise RuntimeError("no era has usable noise files for BOTH detectors")

    rng = np.random.default_rng(31415)
    out = {"eras": eras_ok, "n_td": n_td, "freq_array": freq_array.astype(np.float64)}
    for det in ["H1", "L1"]:
        eras_d = np.asarray(file_era[det], dtype=object)
        idx_by_era = {e: np.flatnonzero(eras_d == e) for e in eras_ok}
        per_era = n_segments // len(eras_ok)
        n_total = per_era * len(eras_ok) + (n_segments % len(eras_ok))
        segs = np.empty((n_total, n_td), dtype=np.float32)
        rows = np.empty(n_total, dtype=np.int32)
        elab = np.empty(n_total, dtype=object)
        k = 0
        for ei, e in enumerate(eras_ok):
            n_this = per_era + (1 if ei < n_segments % len(eras_ok) else 0)
            files = idx_by_era[e]
            for _ in range(n_this):
                fi = int(files[rng.integers(len(files))])
                s = file_strain[det][fi]
                start = int(rng.integers(margin, len(s) - n_td - margin))
                segs[k] = s[start:start + n_td]
                rows[k] = fi
                elab[k] = e
                k += 1
        out[f"segments_{det}"] = segs
        out[f"psd_row_{det}"] = rows
        out[f"era_{det}"] = elab
        out[f"psds_{det}"] = np.asarray(file_psd[det], dtype=np.float64)
        print(f"  {det}: {n_total} segments from {len(file_strain[det])} files")
    return out


def load_or_compute(path, compute_fn, *args, **kwargs):
    """Load from pickle if cached, else compute and save."""
    if os.path.exists(path):
        print(f"Loading cached {os.path.basename(path)}...")
        with open(path, "rb") as f:
            return pickle.load(f)
    result = compute_fn(*args, **kwargs)
    with open(path, "wb") as f:
        pickle.dump(result, f)
    return result