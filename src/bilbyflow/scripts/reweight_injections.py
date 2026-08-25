"""
bilbyflow-reweight-injections — NPE reweighting efficiency on injections.

    python -m bilbyflow.scripts.reweight_injections <npe_dir>
        --psd-bank-path /path/to/psd_bank.pkl
        [--n-events 20] [--n-samples 5000] [--npool 16]
        [--seed 1234 | --random-seed] [--use-checkpoint]
        [--vary-reference-time [--ref-time-window SECONDS]]
        [--noise-source gaussian_whitened|gaussian_physical|real]
        [--no-synthetic-phase] [--do-single-stage] [--n-phase-basis 9]
        [--window-template pols|off] [--sky-prior detector-uniform|isotropic]
        [--injection-prior physical|training|nearby] [--min-snr 8]
        [--psd-scope full-file|off-source] [--prior-swap]

Port of sk_psd_cond_reweight_efficiency.py (v4.3 fixes + v4.5 psd-scope).
Draws injections (SNR-gated), synthesises data through the training whitening
path, reweights to the exact HM-marginalised likelihood (two-stage by default,
single-stage with --do-single-stage), and writes per-event pickles + plots.

All numerics live in the package: inference.injections (synthesis + SNR gate),
inference.priors (injection prior + sky convention), inference.reweight
(shared reweight_event with injection extras), likelihood.waveform
(WindowedWaveformGenerator), data.noise (real segments + off-source PSD),
inference.sample (FIX-4 convolved-proposal density). This script is
orchestration + I/O only.

Note vs the standalone: the package synthetic-extrinsic evaluator is the pure
v5 (FFT basis) — --phase-basis is accepted but the mode path / bilby
self-check are unavailable; the batch self-check reports refine_delta only.
"""

import argparse
import os
import pickle
import secrets
import time

import numpy as np
import matplotlib
matplotlib.rcParams["text.usetex"] = False
import torch

import bilby
from bilby.core.utils.random import seed as bilby_seed

from bilbyflow.io.config import load_config, grid_quantities
from bilbyflow.data.canonical import build_x_full
from bilbyflow.data.noise import build_noise_index
from bilbyflow.inference.priors import make_prior_dict, make_injection_priors
from bilbyflow.inference.injections import (SIDEREAL_DAY, draw_injection_params)
from bilbyflow.inference.sample import npe_sample_and_logprob
from bilbyflow.inference.reweight import reweight_event, marg_flags, MARGABLE
from bilbyflow.plotting.corner import (plot_reweighted_npe_only,
                               plot_recovered_extrinsics_vs_truth)
from bilbyflow.plotting.weights import plot_weight_diagnostics, plot_summary
from .reweight_real import load_trained_posterior
from bilbyflow.data.standardiser import Standardiser  # noqa: F401  (pickle compat: standardiser.pkl records __main__.Standardiser)

PIPELINE_TAG = ("v4.5: psd-scope-selectable + noise-var-q + psis-smoothing "
                "+ synthetic-extrinsics-v5")


def main():
    parser = argparse.ArgumentParser(
        description="NPE reweighting efficiency on injections")
    parser.add_argument("npe_dir", type=str)
    parser.add_argument("--psd-bank-path", type=str, required=True)
    parser.add_argument("--n-events", type=int, default=20)
    parser.add_argument("--n-samples", type=int, default=5000)
    parser.add_argument("--npool", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--random-seed", action="store_true")
    parser.add_argument("--use-checkpoint", action="store_true")
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--standardiser-path", type=str, default=None)
    parser.add_argument("--proposal-noise", type=float, default=0.0)
    parser.add_argument("--proposal-noise-k", type=int, default=16,
                        help="MC draws for the convolved-proposal density (FIX-4)")
    parser.add_argument("--nuisance-fill", choices=["truth", "zero"], default="truth")
    parser.add_argument("--no-assert", action="store_true")
    parser.add_argument("--vary-reference-time", action="store_true")
    parser.add_argument("--ref-time-window", type=float, default=SIDEREAL_DAY)
    parser.add_argument("--noise-source",
                        choices=["gaussian_whitened", "gaussian_physical", "real"],
                        default=None)
    parser.add_argument("--noise-bank-path", type=str, default=None)
    parser.add_argument("--noise-data-dir", type=str, default=None)
    parser.add_argument("--waveform-approximant", type=str, default=None)
    parser.add_argument("--injection-prior",
                        choices=["physical", "training", "nearby"],
                        default="physical",
                        help="physical: population-weighted efficiency; "
                             "training: raw-flow PP calibration (target prior "
                             "stays physical, mean NOT population-weighted); "
                             "nearby: dL uniform up to --max-inj-dL")
    parser.add_argument("--max-inj-dL", type=float, default=5000.0)
    parser.add_argument("--window-template", choices=["pols", "off"], default="off",
                        help="pols: likelihood template windowed like the "
                             "injection (FIX-1, recommended)")
    parser.add_argument("--sky-prior", choices=["detector-uniform", "isotropic"],
                        default="detector-uniform")
    # synthetic extrinsics
    parser.add_argument("--no-synthetic-phase", action="store_true")
    parser.add_argument("--do-single-stage", action="store_true")
    parser.add_argument("--n-phase-basis", type=int, default=9)
    parser.add_argument("--positive-harmonics", action="store_true")
    parser.add_argument("--phase-basis", choices=["auto", "modes", "fft"],
                        default="fft", help="package evaluator is FFT-only")
    parser.add_argument("--n-phi", type=int, default=64)
    parser.add_argument("--n-psi", type=int, default=32)
    parser.add_argument("--refine-fac", type=int, default=8)
    parser.add_argument("--dt-res", type=float, default=2.5e-5)
    parser.add_argument("--n-selfcheck", type=int, default=4)
    parser.add_argument("--psd-scope", choices=["off-source", "full-file"],
                        default="full-file")
    parser.add_argument("--min-snr", type=float, default=8.0)
    parser.add_argument("--max-snr", type=float, default=None)
    parser.add_argument("--prior-swap", action="store_true")
    parser.add_argument("--prior-swap-oversample", type=int, default=10)
    args = parser.parse_args()



    seed = secrets.randbelow(2 ** 31 - 1) if args.random_seed else int(args.seed)
    print(f"  Seed: {seed}{' (random)' if args.random_seed else ''}")
    bilby_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    bilby.core.utils.setup_logger(log_level="WARNING")

    cfg = load_config(os.path.join(args.npe_dir, "config.yaml"))
    g = grid_quantities(cfg)
    param_names = cfg["inferred_parameters"]
    flags = marg_flags(cfg)
    synth = (not args.no_synthetic_phase) and any(flags.values())
    do_two_stage = not args.do_single_stage
    window_template = (args.window_template == "pols")

    print(f"  [{PIPELINE_TAG}]")
    print(f"  inferred ({len(param_names)}): {param_names}")
    print(f"  marginalized in target: {[p for p in MARGABLE if flags[p]]}"
          + ("" if synth else "  (synthetic marg OFF)"))
    print(f"  template windowing: {args.window_template}"
          + ("" if window_template else "  ** unwindowed template: expect dL/Mc bias **"))
    print(f"  sky prior: {args.sky_prior}")
    print(f"  stage mode: {'two-stage' if do_two_stage else 'single-stage'}")

    noise_source = args.noise_source or cfg.get("noise_source", "gaussian_whitened")
    print(f"  noise_source: {noise_source}"
          + ("" if args.noise_source else "  (from training cfg)"))

    
    approximant = args.waveform_approximant or cfg["waveform_approximant"]
    print(f"  likelihood approximant: {approximant}")

    sp_kwargs = dict(n_basis=args.n_phase_basis, n_phi=args.n_phi,
                     n_psi=args.n_psi, refine_fac=args.refine_fac,
                     dt_res=args.dt_res, phase_basis=args.phase_basis,
                     positive_harmonics=args.positive_harmonics)

    timestamp = time.strftime("%Y%m%d_%H%M%S")


    uid = secrets.token_hex(3) # unique id


    run_name = f"run_{timestamp}_seed{seed}_{uid}"
    reftag = "_varyGMST" if args.vary_reference_time else ""
    sptag = f"_synthext{args.n_phase_basis}" if synth else ""
    wtag = "" if window_template else "_nowin"
    sktag = "" if args.sky_prior == "detector-uniform" else "_isosky"
    injtag = "" if args.injection_prior == "physical" else f"_{args.injection_prior}prior"
    sctag = "" if args.psd_scope == "full-file" else "_offsrc"
    stagetag = "" if do_two_stage else "_1stage"


    print(f"  injection prior: {args.injection_prior}"
          + ("" if args.injection_prior == "physical"
             else "  ** mean efficiency NOT population-weighted **"))
    parent = (f"{args.npe_dir}/reweighting_injections_eff_psis_"
              f"{args.proposal_noise}_{noise_source}_{approximant}"
              f"{reftag}{sptag}{wtag}{sktag}{injtag}{sctag}{stagetag}")
    rw_dir = f"{parent}/{run_name}"
    os.makedirs(rw_dir, exist_ok=True)
    print(f"  Run folder: {rw_dir}")

    print("Loading NPE model...")
    posterior, std = load_trained_posterior(
        args.npe_dir, cfg, checkpoint_path=args.checkpoint_path,
        standardiser_path=args.standardiser_path,
        use_checkpoint=args.use_checkpoint)

    print(f"Loading PSD bank from {args.psd_bank_path}...")
    with open(args.psd_bank_path, "rb") as f:
        psd_bank = pickle.load(f)
    print(f"  PSD bank: {psd_bank['psd_H1'].shape[0]} PSDs, "
          f"eras={np.unique(psd_bank['era'])}")

    noise_bank = None
    noise_index = None
    if noise_source == "real":
        eras_filter = cfg.get("psd_bank", {}).get("eras", None)
        if args.noise_data_dir:
            noise_index = build_noise_index(args.noise_data_dir, eras_filter)
        elif args.noise_bank_path:
            with open(args.noise_bank_path, "rb") as f:
                noise_bank = pickle.load(f)
        else:
            raise SystemExit("--noise-source real requires --noise-data-dir "
                             "or --noise-bank-path.")

    ncpu = int(os.environ.get("SLURM_CPUS_PER_TASK", "16"))
    all_results = []
    for i in range(args.n_events):
        print(f"\n{'=' * 60}\nEVENT {i}\n{'=' * 60}")

        ref_tc = float(cfg["ref_geocent_time"])
        delta = (float(np.random.uniform(0.0, args.ref_time_window))
                 if args.vary_reference_time else 0.0)
        inj_ref = ref_tc + delta
        # target prior (IS weights): ALWAYS physical; injection prior per flag
        priors = make_prior_dict(cfg, tc_gps=inj_ref)

        inj_priors = make_injection_priors(cfg, inj_ref, args.injection_prior,
                                           max_inj_dL=args.max_inj_dL)

        drawn = draw_injection_params(
            inj_priors, cfg, psd_bank, noise_source, noise_bank, noise_index,
            inj_ref, args.psd_scope, min_snr=args.min_snr, max_snr=args.max_snr,
            approximant=approximant, window_template=window_template)
        
        if drawn is None:
            print(f"  ** no draw passing the SNR gate in 1000 tries, skipping **")
            continue
        injection_params, x_npe, ifos, era_used, det_psds, noise_var_q, rho_inj = drawn

        tc_offset = float(injection_params["geocent_time"]) - inj_ref
        print(f"  Mc={injection_params['chirp_mass']:.1f}, "
              f"q={injection_params['mass_ratio']:.2f}, "
              f"dL={injection_params['luminosity_distance']:.0f} Mpc, "
              f"tc_offset={tc_offset:.4f}s"
              + (f", GMST-shift={delta:.0f}s" if args.vary_reference_time else ""))
        print("  whitened-noise var_q: " + ", ".join(
            f"{d}={v:.3f}" for d, v in noise_var_q.items())
            + "  (gaussian_physical baseline ~0.94)")

        strain_dim = int(getattr(std, "strain_dim", x_npe.shape[0]))
        x_strain = x_npe[:strain_dim]
        x_npe = build_x_full(x_strain, det_psds, std, cfg, g)

        ratio = x_strain.std() / std.x_std.item()
        print(f"  x_npe strain std: {x_strain.std():.4f} "
              f"(training x_std: {std.x_std:.4f}, ratio: {ratio:.3f})")
        if not args.no_assert:
            assert 0.7 < ratio < 1.4, f"x_npe/x_std={ratio:.3g}: convention mismatch."

        print(f"  Running NPE ({args.n_samples} samples)...")

        torch.set_num_threads(ncpu)
        npe_samples, log_draw_prob, swap_info = npe_sample_and_logprob(
            posterior, x_npe, std, args.n_samples, cfg,
            tc=inj_ref, proposal_noise=args.proposal_noise,
            proposal_noise_k=args.proposal_noise_k,
            prior_swap=args.prior_swap, oversample=args.prior_swap_oversample,
            seed=seed + i)
        
        valid_draw = np.isfinite(log_draw_prob)
        if (~valid_draw).sum() > 0:
            npe_samples = npe_samples[valid_draw]
            log_draw_prob = log_draw_prob[valid_draw]
        torch.set_num_threads(1)   # workers get 1 BLAS thread each

        print("  Reweighting...")
        t0 = time.perf_counter()
        result = reweight_event(
            npe_samples, log_draw_prob, ifos, cfg, priors,
            npool=args.npool, tc_gps=inj_ref, approximant=approximant,
            synthetic_phase=synth, sp_kwargs=sp_kwargs,
            do_two_stage=do_two_stage,
            injection_params=injection_params,
            nuisance_reference=args.nuisance_fill,
            sky_prior=args.sky_prior, n_selfcheck=args.n_selfcheck,
            event_seed=seed + i, window_template=window_template)

        
        t_reweight = time.perf_counter() - t0

        torch.set_num_threads(ncpu) # Have had some issues with thread allocation

        print(f"  reweighting time: {t_reweight:.1f}s")

        result["event"] = i
        result["event_uid"] = f"{seed}_{i}"
        result["seed"] = seed
        result["noise_source"] = noise_source
        result["approximant"] = approximant
        result["gmst_shift"] = delta
        result["noise_var_q"] = noise_var_q
        result["psd_scope"] = args.psd_scope
        result["t_reweight_total"] = t_reweight
        result.update(swap_info)

        print(f"  n_eff: {result['n_eff']:.0f} / {result['n_valid']} "
              f"({result['efficiency']:.1f}%)")
        print(f"  PSIS k hat: {result['khat']:.2f}")
        if "eff_total" in result:
            print(f"  stage: eff1={result['eff_stage1']:.1f}% "
                  f"eff2={result['eff_stage2']:.1f}% "
                  f"total={result['eff_total']:.2f}% "
                  f"({len(result['idx_final'])} final samples, "
                  f"stage2 {result['t_stage2']:.1f}s)")

        all_results.append(result)
        plot_reweighted_npe_only(npe_samples, result, param_names, f"event {i}",
                                 f"{rw_dir}/reweighted_event_{i}.png")

        
        plot_recovered_extrinsics_vs_truth(
            result, injection_params, f"event {i}",
            f"{rw_dir}/recovered_extrinsics_event_{i}.png")

        
        plot_weight_diagnostics(result, f"event {i}",
                                f"{rw_dir}/weights_event_{i}.png")

        with open(f"{rw_dir}/event_{i}_data.pkl", "wb") as f:
            pickle.dump(dict(npe_samples=npe_samples, log_draw_prob=log_draw_prob,
                             injection_params=injection_params, era_used=era_used,
                             result=result, seed=seed, noise_source=noise_source,
                             approximant=approximant, ref_geocent_used=inj_ref,
                             event_uid=result["event_uid"],
                             injection_prior=args.injection_prior,
                             psd_scope=args.psd_scope,
                             rho_opt=float(np.sqrt(result.get("rho2_opt", 0))),
                             gmst_shift=delta), f)

    if all_results:
        print(f"\n{'=' * 60}\nSUMMARY\n{'=' * 60}")
        print(f"{'Event':>6} {'PSIS%':>8} {'2stg%':>8} {'n_eff':>8} "
              f"{'k hat':>7} {'deficit':>9} {'var_q H/L':>12}")
        for r in all_results:
            deficit = r["rho2_opt"] / 2 - r["log_l_truth"]
            vq = r.get("noise_var_q", {})
            eff_2s = r.get("eff_total", r["efficiency"])
            n_final = len(r["idx_final"]) if "idx_final" in r else int(r["n_eff"])
            print(f"{r['event']:>6d} {r['efficiency']:>8.2f} {eff_2s:>8.2f} "
                  f"{n_final:>8d} {r['khat']:>7.2f} {deficit:>+9.1f} "
                  f"{vq.get('H1', float('nan')):>5.2f}/{vq.get('L1', float('nan')):<5.2f}")
        psis_mean = np.mean([r["efficiency"] for r in all_results])
        emp_mean = np.mean([r.get("eff_total", r["efficiency"]) for r in all_results])
        min_eff = np.min([r["efficiency"] for r in all_results])


        print(f"\nPSIS mean: {psis_mean:.2f}%  |  Empirical mean: {emp_mean:.2f}%")
        print(f"Min efficiency: {min_eff:.3f}%")


        plot_summary(all_results, f"{rw_dir}/efficiency_summary.png")


        with open(f"{rw_dir}/summary.pkl", "wb") as f:
            pickle.dump(all_results, f)
        with open(f"{rw_dir}/summary.txt", "w") as f:
            f.write(f"[{PIPELINE_TAG}]\nModel: {args.npe_dir}\nSeed: {seed}\n")
            f.write(f"Marginalized: {[p for p in MARGABLE if flags[p]]} (synth={synth})\n")
            f.write(f"Noise: {noise_source} | approximant: {approximant}\n")
            f.write(f"Template windowing: {args.window_template} | sky prior: "
                    f"{args.sky_prior} | psd scope: {args.psd_scope} | "
                    f"stage: {'two' if do_two_stage else 'single'}\n")
            f.write(f"Samples/event: {args.n_samples} | events: {args.n_events}\n\n")
            f.write(f"{'Event':>6} {'PSIS%':>8} {'2stg%':>8} {'k hat':>7}\n")
            for r in all_results:
                eff_2s = r.get("eff_total", r["efficiency"])
                f.write(f"{r['event']:>6d} {r['efficiency']:>8.2f} "
                        f"{eff_2s:>8.2f} {r['khat']:>7.2f}\n")
            f.write(f"\nPSIS mean: {psis_mean:.2f}%\n")
            f.write(f"Empirical mean: {emp_mean:.2f}%\n")
    print(f"\nAll outputs -> {rw_dir}/")


if __name__ == "__main__":
    main()