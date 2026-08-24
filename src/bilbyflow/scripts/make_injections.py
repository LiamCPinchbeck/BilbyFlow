#!/usr/bin/env python
r"""
make_injections.py -- pure-bilby synthetic events in the REAL-DATA format.

Generates injections using ONLY bilby (waveforms, projection, noise, SNR) and
writes them as the same .npy files the real-data pipeline consumes:

    <data-dir>/INJ###_H1.npy, INJ###_L1.npy, INJ###_gps.npy
    <noise-data-dir>/SYNTH/INJ###_H1_noise.npy, INJ###_L1_noise.npy
    <data-dir>/INJ###_truth.pkl          (injected params + bilby SNRs)

File layout matches the real loaders: np.array([t0, dt, strain], dtype=object).

Then run the UNMODIFIED real-data script on them:

    python -m bilbyflow.scripts.reweight_real $MODEL \
        --data-dir <data-dir> --noise-data-dir <noise-data-dir> \
        --events INJ000 INJ001 ... \
        <exact same flags as your real-event runs>

No BilbyFlow / data_prep_canonical machinery is used here. The only non-bilby
input is optional: --psd-bank-path (a pickle of PSD arrays) so the noise is
coloured like O1-O3; omit it for bilby's aLIGO design PSD.

Usage:
    python make_bilby_injections.py <npe_dir> \
        --data-dir synth_data --noise-data-dir synth_noise \
        [--psd-bank-path $MODEL/psd_bank.pkl] [--n-events 10] \
        [--min-snr 8] [--max-snr 20] [--seed 1] [--noise-duration 512] \
        [--dl-max 4500]
"""
import argparse
import os
import pickle

import numpy as np
import yaml
import bilby
from bilby.core.utils.random import seed as bilby_seed


def load_cfg(npe_dir):
    p = npe_dir if npe_dir.endswith(".yaml") else os.path.join(npe_dir, "config.yaml")
    with open(p) as f:
        return yaml.safe_load(f)


def build_priors(cfg, gps):
    """Injection priors from cfg['priors'], standard bilby distributions.
    Mirrors the analysis prior; dL optionally capped via --dl-max later."""
    from bilby.core.prior import PriorDict, Uniform, Cosine, Sine, PowerLaw
    pr = cfg["priors"]
    P = PriorDict()
    P["chirp_mass"] = bilby.gw.prior.UniformInComponentsChirpMass(
        minimum=pr["chirp_mass"]["min"], maximum=pr["chirp_mass"]["max"])
    P["mass_ratio"] = bilby.gw.prior.UniformInComponentsMassRatio(
        minimum=pr["mass_ratio"]["min"], maximum=pr["mass_ratio"]["max"])
    P["luminosity_distance"] = PowerLaw(
        alpha=2, minimum=pr["luminosity_distance"]["min"],
        maximum=pr["luminosity_distance"]["max"])
    P["dec"] = Cosine()
    P["ra"] = Uniform(minimum=0, maximum=2 * np.pi, boundary="periodic")
    P["theta_jn"] = Sine()
    P["psi"] = Uniform(minimum=0, maximum=np.pi, boundary="periodic")
    P["phase"] = Uniform(minimum=0, maximum=2 * np.pi, boundary="periodic")
    for s in ("a_1", "a_2"):
        P[s] = Uniform(minimum=pr[s]["min"], maximum=pr[s]["max"])
    for t in ("tilt_1", "tilt_2"):
        P[t] = Sine()
    for ph in ("phi_12", "phi_jl"):
        P[ph] = Uniform(minimum=0, maximum=2 * np.pi, boundary="periodic")
    P["geocent_time"] = float(gps)          # trigger at segment centre, like real
    return P


def psd_for(det, cfg, bank, rng):
    """(frequency_array, psd_array) — from the bank pickle if given, else
    bilby's default aLIGO PSD."""
    duration = float(cfg["duration"])
    sr = int(cfg["sampling_frequency"])
    f_min = float(cfg["f_min"])
    n_fd = int(duration * sr / 2) + 1
    freq = np.arange(n_fd) / duration
    if bank is not None:
        idx = int(rng.integers(bank[f"psd_{det}"].shape[0]))
        psd = np.asarray(bank[f"psd_{det}"][idx], dtype=np.float64).copy()
    else:
        ifo = bilby.gw.detector.get_empty_interferometer(det)
        psd = ifo.power_spectral_density.power_spectral_density_interpolated(freq)
        psd = np.asarray(psd, dtype=np.float64)
    # noise generation needs finite PSD everywhere: extend flat below f_min
    good = np.isfinite(psd) & (psd > 0)
    usable = psd[good & (freq >= f_min)]
    if usable.size == 0:
        raise ValueError(f"{det}: PSD has no usable in-band bins "
                         f"(all inf/zero above {f_min} Hz)")
    anchor = usable[0]
    psd[~good] = anchor
    psd[freq < f_min] = anchor
    return freq, psd


def coloured_noise_td(det, freq, psd, duration, sr, start_time):
    """Pure bilby: coloured Gaussian TD noise of the given duration."""
    ifo = bilby.gw.detector.get_empty_interferometer(det)
    ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(
        frequency_array=freq, psd_array=psd)
    ifo.set_strain_data_from_power_spectral_density(
        sampling_frequency=sr, duration=duration, start_time=start_time)
    return np.asarray(ifo.strain_data.time_domain_strain, dtype=np.float64), ifo


def save_strain(path, t0, dt, strain):
    np.save(path, np.array([t0, dt, strain], dtype=object), allow_pickle=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npe_dir")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--noise-data-dir", required=True)
    ap.add_argument("--psd-bank-path", default=None,
                    help="optional: colour noise with a training-era PSD "
                         "(plain pickle of arrays; omit for aLIGO design)")
    ap.add_argument("--n-events", type=int, default=10)
    ap.add_argument("--min-snr", type=float, default=8.0)
    ap.add_argument("--max-snr", type=float, default=None)
    ap.add_argument("--dl-max", type=float, default=None,
                    help="cap injection dL [Mpc] (loud test population)")
    ap.add_argument("--noise-duration", type=float, default=512.0,
                    help="length [s] of the off-source noise file for the "
                         "real path's Welch PSD estimate")
    ap.add_argument("--gps", type=float, default=None,
                    help="trigger GPS; default cfg ref_geocent_time")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--run-tag", default=None,
                    help="identifier for this batch; default s{seed}_{YYmmdd_HHMM}. "
                         "Events are named INJ_{tag}_###, so the reweighting "
                         "script can select the whole batch with "
                         "--events $(ls <data-dir> | grep INJ_{tag} ...)")
    args = ap.parse_args()

    import time as _time
    tag = args.run_tag or f"s{args.seed}_{_time.strftime('%y%m%d_%H%M')}"

    bilby.core.utils.setup_logger(log_level="WARNING")
    bilby_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    cfg = load_cfg(args.npe_dir)
    duration = float(cfg["duration"])
    sr = int(cfg["sampling_frequency"])
    f_min = float(cfg["f_min"])
    gps = float(args.gps) if args.gps is not None else float(cfg["ref_geocent_time"])

    bank = None
    if args.psd_bank_path:
        with open(args.psd_bank_path, "rb") as f:
            bank = pickle.load(f)
        print(f"noise coloured from PSD bank ({bank['psd_H1'].shape[0]} PSDs)")
    else:
        print("noise coloured from bilby aLIGO design PSD "
              "(pass --psd-bank-path for training-era PSDs)")

    os.makedirs(args.data_dir, exist_ok=True)
    era_dir = os.path.join(args.noise_data_dir, "SYNTH")
    os.makedirs(era_dir, exist_ok=True)

    wfg = bilby.gw.WaveformGenerator(
        duration=duration, sampling_frequency=sr,
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
        waveform_arguments=dict(
            waveform_approximant=cfg["waveform_approximant"],
            reference_frequency=f_min, minimum_frequency=f_min))

    priors = build_priors(cfg, gps)
    if args.dl_max is not None:
        priors["luminosity_distance"] = bilby.core.prior.Uniform(
            minimum=float(cfg["priors"]["luminosity_distance"]["min"]),
            maximum=float(args.dl_max), name="luminosity_distance")

    on_start = gps - duration / 2.0
    made = 0
    tries = 0
    while made < args.n_events and tries < 200 * args.n_events:
        tries += 1
        params = priors.sample()
        params["geocent_time"] = gps

        # one PSD per detector per event (era-realistic when bank given)
        psds = {d: psd_for(d, cfg, bank, rng) for d in ("H1", "L1")}

        # -- on-source: bilby noise + bilby injection --
        ifos = []
        ok = True
        net_snr2 = 0.0
        for det in ("H1", "L1"):
            freq, psd = psds[det]
            _, ifo = coloured_noise_td(det, freq, psd, duration, sr, on_start)
            ifo.minimum_frequency = f_min
            try:
                ifo.inject_signal(parameters=dict(params),
                                  waveform_generator=wfg)
            except Exception as e:
                print(f"  inject failed ({e}), redrawing")
                ok = False
                break
            net_snr2 += float(np.abs(ifo.meta_data["optimal_SNR"]) ** 2)
            ifos.append(ifo)
        if not ok:
            continue
        rho = float(np.sqrt(net_snr2))
        if rho < args.min_snr:
            continue
        if args.max_snr is not None and rho > args.max_snr:
            continue

        name = f"INJ_{tag}_{made:03d}"
        dt = 1.0 / sr
        for ifo in ifos:
            strain = np.asarray(ifo.strain_data.time_domain_strain,
                                dtype=np.float64)
            save_strain(os.path.join(args.data_dir, f"{name}_{ifo.name}.npy"),
                        on_start, dt, strain)

        # -- off-source noise file (independent realisation, same PSDs) --
        for det in ("H1", "L1"):
            freq, psd = psds[det]
            noise, _ = coloured_noise_td(det, freq, psd, args.noise_duration,
                                         sr, on_start - args.noise_duration - 8)
            save_strain(os.path.join(era_dir, f"{name}_{det}_noise.npy"),
                        on_start - args.noise_duration - 8, dt, noise)

        np.save(os.path.join(args.data_dir, f"{name}_gps.npy"),
                np.array([gps]))
        with open(os.path.join(args.data_dir, f"{name}_truth.pkl"), "wb") as f:
            pickle.dump(dict(params={k: float(v) for k, v in params.items()},
                             rho_opt_net=rho,
                             per_det_snr={i.name: float(np.abs(
                                 i.meta_data["optimal_SNR"])) for i in ifos},
                             gps=gps, psd_source=("bank" if bank else "aligo"),
                             seed=args.seed), f)
        print(f"{name}: rho={rho:.1f} Mc={params['chirp_mass']:.1f} "
              f"q={params['mass_ratio']:.2f} dL={params['luminosity_distance']:.0f} "
              f"(draw {tries})")
        made += 1

    if made == 0:
        raise SystemExit(
            f"no injections passed the gate after {tries} draws "
            f"(min_snr={args.min_snr}, max_snr={args.max_snr}, "
            f"dl_max={args.dl_max}). Loosen the SNR window or lower dl_max.")
    if made < args.n_events:
        print(f"WARNING: only {made}/{args.n_events} events after {tries} "
              f"draws -- the SNR gate is tight for this dL range")

    events = [f"INJ_{tag}_{i:03d}" for i in range(made)]
    events_file = os.path.join(args.data_dir, f"events_{tag}.txt")
    with open(events_file, "w") as f:
        f.write("\n".join(events) + "\n")

    print(f"\n{made} events -> {args.data_dir} (+ noise in {era_dir})")
    print(f"batch tag: {tag}")
    print(f"event list: {events_file}")
    print("\nrun the UNMODIFIED real-data script on the whole batch:")
    print(f"""  python -m bilbyflow.scripts.reweight_real $MODEL \\
      --data-dir {args.data_dir} --noise-data-dir {args.noise_data_dir}\\
      --events $(cat {events_file}) \\
      <same flags as your real-event runs>""")


if __name__ == "__main__":
    main()