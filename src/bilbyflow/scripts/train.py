"""
bilbyflow-train — train an NPE model with on-the-fly extrinsic generation.

    python -m bilbyflow.scripts.train config.yaml
    python -m bilbyflow.scripts.train config.yaml --resume /path/to/run/

Only puts internal functions in correct ordering:
    - build/load the waveform, sky, PSD, and noise banks (cached via data.banks.load_or_compute)
    - construct the train + disjoint val/test datasets
    - fit (or reuse) the Standardiser
    - run data diagnostics
    - drive the curriculum via training.trainer.custom_train_npe
    - write posterior.pt / density_estimator.pt / summary.pkl and 
    - the example corner + PP plots.

Core numerics live in the package (data.*, nn.*, training.*, coordinates.*,
inference.priors, io.config). 

Presentation helpers (run_inference, plot_corner_fig, pp_test, run_diagnostics, plot_aux_diagnostics) live in
scripts.utils, shared by the CLI scripts.
"""

import os
import time
import pickle
import argparse

import numpy as np
import torch
import yaml

from ..io.config import (load_config, get_prior_bounds,
                         get_reference_detector_data)
from ..coordinates.sky import MAX_DT_HL, samples_detector_to_radec
from ..coordinates.params import dL_to_physical
from ..data.dataset import OnTheFlyGWDataset, generate_fixed_dataset
from ..data.standardiser import Standardiser, check_aux_stats, check_amp_stats
from ..data.banks import (precompute_waveforms, precompute_sky_bank,
                          precompute_noise_segment_bank, load_or_compute)
from ..data.psd import precompute_psd_bank_from_segments
from ..nn.aux_head import AUX_NAMES, N_AUX
from ..inference.priors import make_prior_dict
from ..training.trainer import custom_train_npe
from ..plotting.training import plot_losses
from .utils import (run_inference, plot_corner_fig, pp_test,
                    plot_aux_diagnostics, run_diagnostics)


DATA_FILES = [
    "waveforms.pkl", "sky_bank.pkl",
    "psd_bank.pkl", "noise_bank.pkl",
    "standardiser.pkl",
    "val_waveforms.pkl", "val_data.pkl", 
    "test_data.pkl",
    ]


# ── setup helpers ────────────────────────────────────────────────────────────



def _print_run_banner(cfg):
    try:
        _pr = make_prior_dict(cfg)
        fams = ", ".join(f"{k}:{type(_pr[k]).__name__}" for k in _pr)
        print(f"[priors] actual families: {fams}")
        if str(cfg.get("dL_param", "linear")).lower() == "log":
            print("[priors] dL coordinate: ln(dL)")
    except Exception as _e:
        print(f"[priors] could not summarise prior families: {_e}")
    
    if cfg.get("curriculum_stages"):
        caps = [s.get("dL_max") for s in cfg["curriculum_stages"]]
        print(f"[curriculum] {len(caps)} stages, dL caps {caps} Mpc")
    
    if cfg.get("aux_supervision", False):
        print(f"[aux] supervision ON: lambda={cfg.get('aux_lambda', 0.5)}, "
              f"anneal_frac={cfg.get('aux_anneal_frac', 0.7)}, "
              f"slice={cfg.get('aux_n_channels', 128)} ctx dims, "
              f"{N_AUX} targets")


def _make_out_dir(cfg, args):
    """Create the run directory; on resume, symlink the reused data files."""

    uid = cfg.get("unique_analysis_identifier", None)
    uid = (uid + "_") if uid is not None else ""
    tag = "resume_" if args.resume else ""
    OUT = time.strftime(
        f"{cfg['output_prefix']}_{uid}{tag}%m%d_%H%M%S_{int(time.time()*1000)}",
        )

    if os.path.isdir(OUT):
        OUT = OUT + ("_alpha" if not os.path.isdir(OUT + "_alpha") else "_beta")
    os.makedirs(OUT, exist_ok=False)

    if not args.resume:
        return OUT

    if args.no_fresh_data:
        reuse = set(DATA_FILES)
    else:
        reuse = set(cfg.get("reuse_files", []) or [])
        unknown = reuse - set(DATA_FILES)
        if unknown:
            raise ValueError(f"reuse_files has unrecognised entries "
                             f"{sorted(unknown)}, valid options: {DATA_FILES}")

    if "standardiser.pkl" in reuse and cfg.get("aux_supervision", False):
        print("  NOTE: reusing standardiser.pkl with aux_supervision ON "
              "requires matching aux stats (v4.2: 10 targets) -- an old "
              "pickle will fail fast at training start.")


    
    for fname in DATA_FILES:
        src = os.path.join(args.resume, fname)
    
        if fname in reuse:
            if os.path.exists(src):
                print(f"Linking {fname} from {args.resume} (reuse)")
                os.symlink(os.path.abspath(src), os.path.join(OUT, fname))
            else:
                print(f"  WARNING: reuse requested for {fname} but {src} missing")
        else:
            print(f"Skipping symlink for {fname} (will regenerate)")
    
    return OUT


def _val_waveforms(cfg):

    cfg_val = dict(cfg)
    cfg_val["n_waveforms"] = int(cfg.get("n_val_waveforms", 5000))

    return precompute_waveforms(cfg_val, seed_val=4242)



def _build_psd_bank(cfg, OUT, noise_data_dir):
    """PSD bank from real segments (cached), or None for the fixed bilby
    PSD. GP sampling (psd_gp_dir) is retired — fail loudly up front."""

    if cfg.get("psd_gp_dir"):
        raise SystemExit("psd_gp_dir (GP-sampled PSD bank) is retired; use "
                         "noise_data_dir with real segments instead.")

    if noise_data_dir is None:
        print("No noise_data_dir in config — using fixed bilby default PSD")
        return None

    psd_bank = load_or_compute(f"{OUT}/psd_bank.pkl",
                               precompute_psd_bank_from_segments,
                               cfg, noise_data_dir)

    print(f"  PSD bank: {psd_bank['psd_H1'].shape[0]} PSDs, "
          f"eras={np.unique(psd_bank['era'])}")

    return psd_bank


def _build_noise_bank(cfg, OUT, noise_data_dir):

    need = (str(cfg.get("noise_source", "gaussian_whitened")).lower() == "real"
            or bool(cfg.get("embed_consistency", False)))

    if not need:
        return None
    
    if noise_data_dir is None:
        raise ValueError("noise_source: real / embed_consistency requires "
                         "noise_data_dir")

    noise_bank = load_or_compute(f"{OUT}/noise_bank.pkl",
                                 precompute_noise_segment_bank,
                                 cfg, noise_data_dir)

    print(f"  noise bank: {len(noise_bank['era_H1'])} H1 + "
          f"{len(noise_bank['era_L1'])} L1 segments")
    return noise_bank


def _fit_standardiser(dataset, cfg, PARAM_NAMES):
    print("Computing standardisation (theta full range, + aux stats)...")

    dataset.set_dL_cap(None)
    xs, ts, _sids, auxs = generate_fixed_dataset(
        dataset, int(cfg["n_standardisation"]), return_aux=True)

    return Standardiser(
        xs, ts, strain_dim=dataset.strain_dim,
        psd_log_mean=dataset.psd_log_mean, psd_log_std=dataset.psd_log_std,
        psd_conditioning=dataset.psd_conditioning,
        dL_log=dataset.dL_log,
        dL_index=(PARAM_NAMES.index("luminosity_distance")
                  if "luminosity_distance" in PARAM_NAMES else None),
        Mc_log=dataset.Mc_log,
        Mc_index=(PARAM_NAMES.index("chirp_mass")
                  if "chirp_mass" in PARAM_NAMES else None),
        amp_dim=dataset.amp_dim,
        aux_sample=(auxs if cfg.get("aux_supervision", False) else None))


# --------------------------- main ------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to config.yaml")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--no-fresh-data", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    PARAM_NAMES = cfg["inferred_parameters"]
    PRIOR_LOW, PRIOR_HIGH = get_prior_bounds(cfg)
    _print_run_banner(cfg)

    if "ra" in PARAM_NAMES and "dec" in PARAM_NAMES:
        ra_idx, dec_idx = PARAM_NAMES.index("ra"), PARAM_NAMES.index("dec")
        PRIOR_LOW[ra_idx], PRIOR_HIGH[ra_idx] = -MAX_DT_HL, MAX_DT_HL
        PRIOR_LOW[dec_idx], PRIOR_HIGH[dec_idx] = 0.0, 2 * np.pi



    OUT = _make_out_dir(cfg, args)
    with open(f"{OUT}/config.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    ##########################################################################################
    ##########################################################################################
    #
    # Steps 1->4 are for setting up training priors, data, standardization and some diagnostic stuff
    # 
    ##########################################################################################
    ##########################################################################################


    ############################################################
    # 1. banks (each cached in OUT; resume symlinks land here too)
    h_plus, h_cross, intr_params = load_or_compute(
        f"{OUT}/waveforms.pkl", precompute_waveforms, cfg, seed_val=42)
    h_plus_v, h_cross_v, intr_params_v = load_or_compute(
        f"{OUT}/val_waveforms.pkl", _val_waveforms, cfg)

    sky_bank = load_or_compute(f"{OUT}/sky_bank.pkl", precompute_sky_bank,
                               cfg, n_sky=cfg.get("n_sky_bank", 100000))
    noise_data_dir = cfg.get("noise_data_dir", None)
    psd_bank = _build_psd_bank(cfg, OUT, noise_data_dir)
    noise_bank = _build_noise_bank(cfg, OUT, noise_data_dir)



    ############################################################
    # 2. datasets (train + disjoint-intrinsics val/test)
    ref_data = get_reference_detector_data(cfg)

    dataset = OnTheFlyGWDataset(h_plus, h_cross, intr_params, ref_data, cfg,
                                sky_bank, psd_bank=psd_bank,
                                noise_bank=noise_bank)
    val_dataset = OnTheFlyGWDataset(h_plus_v, h_cross_v, intr_params_v,
                                    ref_data, cfg, sky_bank,
                                    psd_bank=psd_bank, noise_bank=noise_bank)
    dataset.set_dL_cap(None)
    val_dataset.set_dL_cap(None)

    print(f"Dataset: {len(dataset)} training waveforms | "
          f"{len(val_dataset)} disjoint val/test waveforms | "
          f"dL range [{dataset.dL_all.min():.0f}, "
          f"{dataset.dL_all.max():.0f}] Mpc")



    ############################################################
    # 3. fixed val/test sets + standardiser (aux stats when aux on)
    x_val, theta_val, sid_val = load_or_compute(
        f"{OUT}/val_data.pkl", generate_fixed_dataset, val_dataset,
        int(cfg["n_val"]))
    x_test, theta_test, sid_test = load_or_compute(
        f"{OUT}/test_data.pkl", generate_fixed_dataset, val_dataset,
        int(cfg["n_test"]))

    std_path = f"{OUT}/standardiser.pkl"

    std = load_or_compute(std_path, 
                          _fit_standardiser, 
                          dataset, 
                          cfg,
                          PARAM_NAMES)

    
    if cfg.get("aux_supervision", False):
        check_aux_stats(std)
    
    check_amp_stats(std, cfg)

    print(f"  x: mean={std.x_mean:.4f}, std={std.x_std:.4f}")
    if getattr(std, "aux_mean", None) is not None:
        print(f"  aux: {N_AUX} targets, std range "
              f"[{float(std.aux_std.min()):.3g}, "
              f"{float(std.aux_std.max()):.3g}]")



    ############################################################
    # 4. sanity + diagnostics
    xc, _theta_c, _, aux_c, _ = dataset[0]
    xcn = std.normalise_x(xc)

    print(f"  sample x: [{xc.min():.3f}, {xc.max():.3f}] "
          f"nan={torch.isnan(xc).any()}")
    print(f"  sample x_norm: [{xcn.min():.3f}, {xcn.max():.3f}]")

    if cfg.get("aux_supervision", False):
        print(f"  sample aux: "
              f"{dict(zip(AUX_NAMES, [f'{v:.3g}' for v in aux_c.numpy()]))}")
    if not args.resume:
        run_diagnostics(x_val, theta_val, std, cfg, f"{OUT}/diagnostics",
                        dataset=dataset)
        plot_aux_diagnostics(dataset, std, f"{OUT}/diagnostics")


    ################################################################################################################
    ################################################################################################################
    ################################################################################################################
    ############
    ############                Step   5.      TRAIN
    ############
    ################################################################################################################
    ################################################################################################################
    ################################################################################################################

    print("\n=== Training NPE ===")
    posterior, de, summary = custom_train_npe(
        dataset, x_val, theta_val, std, cfg, PRIOR_LOW, PRIOR_HIGH,
        OUT=OUT, std_path=std_path, resume_dir=args.resume)
    torch.save(de.state_dict(), f"{OUT}/density_estimator.pt")
    torch.save(posterior, f"{OUT}/posterior.pt")
    with open(f"{OUT}/summary.pkl", "wb") as f:
        pickle.dump(summary, f)
    plot_losses(summary, f"{OUT}/loss_curves.png")



    ################################################################################################################
    ################################################################################################################
    # 6. example posteriors + PP test


    ref_tc = float(cfg["ref_geocent_time"])
    for lbl, xd, td, sd, pre in [("Test 0", x_test, theta_test, sid_test,
                                  "test"),
                                 ("Val 0", x_val, theta_val, sid_val, "val")]:
        tc_gps = ref_tc + float(sd[0])
        s = run_inference(posterior, xd[0], std, cfg, PARAM_NAMES, tc=tc_gps)
        truths = samples_detector_to_radec(
            dL_to_physical(td[0].numpy()[np.newaxis, :], PARAM_NAMES, cfg),
            PARAM_NAMES, tc=tc_gps)[0]
        plot_corner_fig(s, truths, PARAM_NAMES, lbl,
                        f"{OUT}/posterior_{pre}_0.png")


    # PP truths restricted to the ANALYSIS prior (the padded training
    # proposal covers regions the DirectPosterior box rejects).
    _tt = theta_test.numpy()
    _in = np.all((_tt >= PRIOR_LOW.numpy()) & (_tt <= PRIOR_HIGH.numpy()),
                 axis=1)
    _idx = torch.from_numpy(np.flatnonzero(_in)[:200].copy())
    print(f"PP test: {int(_in.sum())}/{len(_tt)} test injections inside "
          f"analysis prior")
    pp_test(posterior, x_test[_idx], theta_test[_idx], std, PARAM_NAMES,
            f"{OUT}/pp_test.png")

    print(f"\nAll outputs → {OUT}/")


if __name__ == "__main__":
    main()