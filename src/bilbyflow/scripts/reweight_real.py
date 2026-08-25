"""
bilbyflow-reweight-real — reweight NPE posteriors on real GW event strain.

    python -m bilbyflow.scripts.reweight_real <npe_dir> [<samples_dir>]
        --data-dir /path/to/gwosc_data --noise-data-dir /path/to/noise_data
        [--events GW150914 ...] [--n-samples 5000] [--npool 16]
        [--no-synthetic-phase] [--n-phase-basis 9] [--n-phi 64] [--n-psi 32]
        [--refine-fac 8] [--dt-res 2.5e-5] [--do-single-stage] [--prior-swap]

For each event: load the whitened strain into the NPE input, draw from the
trained flow (optionally prior-swapped), reweight to the exact
(higher-mode-marginalised) likelihood via the two-stage scheme, and write a
per-event pickle plus corner / weight-diagnostic / summary plots. Published PE
is overlaid when a matching samples file is found.

All numerics live in the package: io.config / io.strain / io.samples,
inference.sample / inference.reweight, coordinates.*, data.canonical, plotting.* and
diagnostics.psis. This script is orchestration + I/O only.
"""

import os
import glob
import pickle
import argparse

import numpy as np
import matplotlib
matplotlib.rcParams["text.usetex"] = False
import matplotlib.pyplot as plt
import torch

import bilby
from bilby.core.utils.random import seed as bilby_seed

from bilbyflow.io.config import load_config, grid_quantities
from bilbyflow.io.strain import (GWOSC_TRIGGER_TIMES, find_events_from_data,
                         fetch_real_strain_and_build_ifos)
from bilbyflow.io.samples import (find_sample_files, load_published_samples,
                          extract_map_params, published_samples_to_array)
from bilbyflow.data.canonical import build_x_full
from bilbyflow.inference.priors import make_prior_dict
from bilbyflow.inference.sample import npe_sample_and_logprob
from bilbyflow.inference.reweight import reweight_event, is_geocent_inferred, marg_flags, MARGABLE
from bilbyflow.plotting.corner import (plot_reweighted_vs_published, plot_reweighted_npe_only,
                               plot_recovered_extrinsics_vs_published)
from bilbyflow.plotting.weights import plot_weight_diagnostics, plot_summary
from bilbyflow.data.standardiser import Standardiser
from .utils import build_npe

PIPELINE_TAG = "v4.0: hp15 + window-first + log-dL + synthetic-extrinsics(phase,psi,tc)"


def load_trained_posterior(output_dir, cfg, checkpoint_path=None,
                           standardiser_path=None, use_checkpoint=False):
    """Load the standardiser + rebuild the DirectPosterior."""
    if checkpoint_path:
        model_path = checkpoint_path
    elif use_checkpoint:
        model_path = os.path.join(output_dir, "checkpoint.pt")
    else:
        model_path = os.path.join(output_dir, "posterior.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    print(f"  Model:        {model_path}")

    std_file = standardiser_path or os.path.join(output_dir, "standardiser.pkl")
    if not os.path.exists(std_file):
        raise FileNotFoundError(f"Standardiser not found: {std_file}")
    with open(std_file, "rb") as f:
        std = pickle.load(f)
    print("theta_mean:", std.theta_mean.numpy())
    print("theta_std: ", std.theta_std.numpy())

    raw = torch.load(model_path, map_location="cpu", weights_only=False)
    if isinstance(raw, dict):
        print("  Detected training checkpoint -- reconstructing from best_state...")
        npe, _ = build_npe(cfg, std)
        state = torch.load(model_path, map_location="cpu",
                           weights_only=False)
        npe.load_state_dict(state.get("best_state", state))
        npe.eval()
        posterior = npe
    else:
        posterior = raw
        if hasattr(posterior, "_device"):
            posterior._device = "cpu"
        if hasattr(posterior, "posterior_estimator"):
            posterior.posterior_estimator = posterior.posterior_estimator.to("cpu")
    return posterior, std


def main():
    parser = argparse.ArgumentParser(description="Reweight NPE posteriors on real GW strain vs published")
    parser.add_argument("npe_dir")
    parser.add_argument("samples_dir", nargs="?", default=None)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--noise-data-dir", type=str, required=True)
    parser.add_argument("--events", nargs="+", default=None)
    parser.add_argument("--n-samples", type=int, default=5000)
    parser.add_argument("--npool", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-checkpoint", action="store_true")
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--standardiser-path", type=str, default=None)
    parser.add_argument("--proposal-noise", type=float, default=0.0)
    parser.add_argument("--no-assert", action="store_true")
    parser.add_argument("--force-regenerate", action="store_true")
    parser.add_argument("--waveform-approximant", type=str, default=None)
    parser.add_argument("--x-rescale", type=float, default=1.0)
    parser.add_argument("--no-synthetic-phase", action="store_true")
    parser.add_argument("--n-phase-basis", type=int, default=9)
    parser.add_argument("--n-phi", type=int, default=64)
    parser.add_argument("--n-psi", type=int, default=32)
    parser.add_argument("--refine-fac", type=int, default=8)
    parser.add_argument("--dt-res", type=float, default=2.5e-5)
    parser.add_argument("--phase-basis", choices=["auto", "modes", "fft"], default="auto")
    parser.add_argument("--positive-harmonics", action="store_true")
    parser.add_argument("--prior-swap", action="store_true",
                        help="SIR-correct the flow proposal from the training prior "
                             "to the target prior BEFORE likelihood evaluation")
    parser.add_argument("--prior-oversample", type=int, default=5)
    parser.add_argument("--do-single-stage", action="store_true",
                        help="single-stage HM instead of two-stage reweighting")
    args = parser.parse_args()

    plt.rcParams["text.usetex"] = False
    plt.rcParams["font.family"] = "sans-serif"
    bilby_seed(args.seed)
    torch.manual_seed(args.seed)
    bilby.core.utils.setup_logger(log_level="WARNING")

    cfg = load_config(args.npe_dir)
    param_names = cfg["inferred_parameters"]
    geocent_inferred = is_geocent_inferred(cfg)
    flags = marg_flags(cfg)
    synth = (not args.no_synthetic_phase) and any(flags.values())
    do_two_stage = not args.do_single_stage
    print(f"  inferred ({len(param_names)}): {param_names}")
    print(f"  marginalized in target: {[p for p in MARGABLE if flags[p]]}"
          + ("" if synth else "  (synthetic marg OFF)"))

    approximant = args.waveform_approximant or cfg["waveform_approximant"]
    default_approx = cfg["waveform_approximant"]
    print(f"  likelihood approximant: {approximant}")
    _base = (f"{args.npe_dir}/reweight_real_data" if approximant == default_approx
             else f"{args.npe_dir}/reweight_real_data_{approximant}")
    if synth:
        _base = f"{_base}_synthext_ts_{args.n_phase_basis}"
    if args.prior_swap:
        _base = f"{_base}_priorswap{args.prior_oversample}"
    out_dir = _base if args.x_rescale == 1.0 else f"{_base}_eff_psis_{args.x_rescale:.4f}"
    os.makedirs(out_dir, exist_ok=True)

    sp_kwargs = dict(n_basis=args.n_phase_basis, n_phi=args.n_phi,
                     n_psi=args.n_psi, refine_fac=args.refine_fac,
                     dt_res=args.dt_res, phase_basis=args.phase_basis,
                     positive_harmonics=args.positive_harmonics)

    print("Loading NPE model...")
    posterior, std = load_trained_posterior(
        args.npe_dir, cfg, checkpoint_path=args.checkpoint_path,
        standardiser_path=args.standardiser_path, use_checkpoint=args.use_checkpoint)
    psd_cond = getattr(std, "psd_conditioning", False) and getattr(std, "psd_log_mean", None) is not None
    print(f"  PSD conditioning: {'ON' if psd_cond else 'off'}")

    g = grid_quantities(cfg)
    event_list = find_events_from_data(args.data_dir, args.events)
    print(f"Found {len(event_list)} events: {[e for e, _ in event_list]}")

    sample_files = {}
    if args.samples_dir and os.path.isdir(args.samples_dir):
        sample_files = find_sample_files(args.samples_dir, [e for e, _ in event_list])
        print(f"Published PE for {len(sample_files)}/{len(event_list)} events")

    all_results = []
    for event_name, event_gps_time in event_list:
        print(f"\n{'='*60}\n{event_name}\n{'='*60}")

        cache_pkl = f"{out_dir}/{event_name}_data.pkl"
        if os.path.exists(cache_pkl) and not args.force_regenerate:
            try:
                with open(cache_pkl, "rb") as f:
                    cached = pickle.load(f)
                result = cached["result"]
                result["event"] = event_name
                npe_samples = cached["npe_samples"]
                published_array = cached.get("published_array")
                truths = cached.get("truths")
                published = cached.get("published")
                print(f"  CACHED: eff={result['efficiency']:.1f}%, "
                      f"n_eff={result['n_eff']:.0f}, k hat={result['khat']:.2f}")
                if published_array is not None and truths is not None:
                    plot_reweighted_vs_published(
                        npe_samples, result, published_array, truths, param_names,
                        event_name, f"{out_dir}/{event_name}_reweighted_vs_published.png")
                else:
                    plot_reweighted_npe_only(
                        npe_samples, result, param_names,
                        event_name, f"{out_dir}/{event_name}_reweighted.png")
                plot_recovered_extrinsics_vs_published(
                    result, published, event_name,
                    f"{out_dir}/{event_name}_recovered_extrinsics.png")
                plot_weight_diagnostics(
                    result, event_name, f"{out_dir}/{event_name}_weight_diagnostics.png")
                all_results.append(result)
                continue
            except Exception as e:
                print(f"  cache load failed ({e}), regenerating")

        if event_name in GWOSC_TRIGGER_TIMES:
            event_gps_time = GWOSC_TRIGGER_TIMES[event_name]
            print(f"  GWOSC trigger time: {event_gps_time:.4f}")
        else:
            print(f"  GPS from data: {event_gps_time:.4f}")

        published = None
        map_params = None
        has_published = event_name in sample_files
        if has_published:
            try:
                published = load_published_samples(sample_files[event_name])
                map_params = extract_map_params(published)
                print(f"  MAP: Mc={map_params.get('chirp_mass', 0):.1f}, "
                      f"q={map_params.get('mass_ratio', 0):.2f}, "
                      f"dL={map_params.get('luminosity_distance', 0):.0f} Mpc")
            except Exception as e:
                print(f"  Warning: could not load published samples: {e}")
                published, has_published = None, False
        else:
            print("  No published PE samples -- reweighting without overlay")

        priors = make_prior_dict(cfg, tc_gps=event_gps_time)

        print("  Loading and whitening real strain...")
        try:
            x_npe, ifos, det_psds, det_var_q = fetch_real_strain_and_build_ifos(
                event_name, event_gps_time, cfg, data_dir=args.data_dir,
                noise_data_dir=args.noise_data_dir)
        except Exception as e:
            print(f"  Error loading strain: {e}")
            continue

        strain_dim = int(getattr(std, "strain_dim", x_npe.shape[0]))
        x_strain = x_npe[:strain_dim]
        raw_ratio = x_strain.std() / std.x_std.item()
        if args.x_rescale != 1.0:
            x_strain = (x_strain * np.float32(args.x_rescale)).astype(np.float32)
        x_npe = build_x_full(x_strain, det_psds, std, cfg, g)

        eff_ratio = raw_ratio * args.x_rescale
        print(f"  x_npe strain std: {x_strain.std():.4f} (training x_std: {std.x_std:.4f}, "
              f"ratio: {eff_ratio:.3f})")
        if not args.no_assert:
            assert 0.7 < eff_ratio < 1.5, f"x_npe/x_std={eff_ratio:.3g}: OOD input."

        print(f"  Running NPE ({args.n_samples} samples)...")
        n_threads = torch.get_num_threads()
        torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", 16)))

        npe_samples, log_draw_prob, swap_info = npe_sample_and_logprob(
            posterior, x_npe, std, args.n_samples, cfg, tc=event_gps_time,
            proposal_noise=args.proposal_noise, priors=priors,
            prior_swap=args.prior_swap, oversample=args.prior_oversample,
            seed=args.seed + int(event_gps_time) % 100000)
        torch.set_num_threads(n_threads)   # restore BEFORE reweight_event forks its pool

        print("  Reweighting...")
        result = reweight_event(npe_samples, log_draw_prob, ifos, cfg, priors,
                                npool=args.npool, tc_gps=event_gps_time,
                                map_params=map_params, approximant=approximant,
                                synthetic_phase=synth, sp_kwargs=sp_kwargs,
                                do_two_stage=do_two_stage)
        result.update(swap_info)
        result["event"] = event_name
        result["approximant"] = approximant
        result["noise_var_q"] = det_var_q

        print(f"  n_eff: {result['n_eff']:.0f} / {result['n_valid']} ({result['efficiency']:.1f}%)")
        print(f"  PSIS k hat: {result['khat']:.2f}")
        if "eff_total" in result:
            print(f"  2-stage: eff1={result['eff_stage1']:.1f}% "
                  f"eff2={result['eff_stage2']:.1f}% total={result['eff_total']:.2f}% "
                  f"({len(result['idx_final'])} equal-weight samples, "
                  f"stage2 {result['t_stage2']:.1f}s)")

        if geocent_inferred and "geocent_time" in param_names:
            gt_idx = param_names.index("geocent_time")
            npe_samples[:, gt_idx] -= (event_gps_time - float(cfg["ref_geocent_time"]))

        published_array = None
        truths = None
        if has_published and published is not None:
            published_array = published_samples_to_array(published, param_names,
                                                         event_tc=event_gps_time, cfg=cfg)
            truths = np.array([(0.0 if geocent_inferred else map_params.get(p, 0.0))
                               if p == "geocent_time" else map_params.get(p, 0.0)
                               for p in param_names], dtype=np.float32)
            plot_reweighted_vs_published(npe_samples, result, published_array, truths,
                                         param_names, event_name,
                                         f"{out_dir}/{event_name}_reweighted_vs_published.png")
        else:
            plot_reweighted_npe_only(npe_samples, result, param_names,
                                     event_name, f"{out_dir}/{event_name}_reweighted.png")

        plot_recovered_extrinsics_vs_published(
            result, published, event_name,
            f"{out_dir}/{event_name}_recovered_extrinsics.png")
        plot_weight_diagnostics(result, event_name, f"{out_dir}/{event_name}_weight_diagnostics.png")
        all_results.append(result)

        with open(f"{out_dir}/{event_name}_data.pkl", "wb") as f:
            pickle.dump(dict(npe_samples=npe_samples, log_draw_prob=log_draw_prob,
                             map_params=map_params, event_gps_time=event_gps_time,
                             published_array=published_array, truths=truths,
                             published=published, param_names=param_names, result=result,
                             proposal_noise=args.proposal_noise, x_rescale=args.x_rescale,
                             geocent_inferred=geocent_inferred), f)

    if all_results:
        plot_summary(all_results, f"{out_dir}/summary.png")
        print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
        print(f"{'Event':>25} {'PSIS%':>8} {'2stg%':>8} {'n_eff':>8} {'k hat':>7}")
        for r in all_results:
            eff_2s = r.get("eff_total", r["efficiency"])
            n_final = len(r["idx_final"]) if "idx_final" in r else int(r["n_eff"])
            print(f"{str(r['event']):>25} {r['efficiency']:>8.2f} {eff_2s:>8.2f} "
                  f"{n_final:>8d} {r['khat']:>7.2f}")
        psis_mean = np.mean([r["efficiency"] for r in all_results])
        psis_median = np.median([r["efficiency"] for r in all_results])
        emp_mean = np.mean([r.get("eff_total", r["efficiency"]) for r in all_results])
        print(f"\nPSIS mean: {psis_mean:.2f}%  |PSIS median: {psis_median:.2f}%  |  "
              f"Empirical 2-stage mean: {emp_mean:.2f}%")
        with open(f"{out_dir}/summary.txt", "w") as f:
            f.write(f"[{PIPELINE_TAG}]\nModel: {args.npe_dir}\n")
            f.write(f"Marginalized: {[p for p in MARGABLE if flags[p]]} (synth={synth})\n")
            f.write(f"Samples/event: {args.n_samples}\n\n")
            f.write(f"{'Event':>25} {'PSIS%':>8} {'2stg%':>8} {'k hat':>7}\n")
            for r in all_results:
                eff_2s = r.get("eff_total", r["efficiency"])
                f.write(f"{str(r['event']):>25} {r['efficiency']:>8.2f} "
                        f"{eff_2s:>8.2f} {r['khat']:>7.2f}\n")
            f.write(f"\nPSIS mean: {psis_mean:.2f}%\n")
            f.write(f"Empirical 2-stage mean: {emp_mean:.2f}%\n")
    print(f"\nAll outputs -> {out_dir}/")


if __name__ == "__main__":
    main()