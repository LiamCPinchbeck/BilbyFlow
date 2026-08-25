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
    - write npe.pt / summary.pkl and
    - the example corner + PP plots.

Core numerics live in the package (data.*, nn.*, training.*, coordinates.*,
inference.priors, io.config). 

Presentation helpers (run_inference, plot_corner_fig, pp_test, run_diagnostics, plot_aux_diagnostics) live in
scripts.utils, shared by the CLI scripts.
"""

import pickle
import argparse
 
import numpy as np
import torch
import yaml
 
from bilbyflow.io.config import load_config, get_reference_detector_data
from bilbyflow.coordinates.sky import samples_detector_to_radec
from bilbyflow.coordinates.params import dL_to_physical
from bilbyflow.data.dataset import OnTheFlyGWDataset, generate_fixed_dataset
from bilbyflow.data.standardiser import check_aux_stats, check_amp_stats
from bilbyflow.data.banks import (precompute_waveforms, precompute_sky_bank,
                          load_or_compute)
from bilbyflow.nn.aux_head import AUX_NAMES, N_AUX, AuxHead
from bilbyflow.training.trainer import custom_train_npe
from bilbyflow.plotting.training import plot_losses
from .utils import (run_inference, plot_corner_fig, pp_test,
                    plot_aux_diagnostics, run_diagnostics,
                    print_run_banner, make_out_dir, build_val_waveforms,
                    build_psd_bank, build_noise_bank, fit_standardiser,
                    build_npe)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to config.yaml")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--no-fresh-data", action="store_true")
    args = parser.parse_args()
 
    cfg = load_config(args.config)
    PARAM_NAMES = cfg["inferred_parameters"]
    print_run_banner(cfg)

    # Build the model first so the prior box comes back reparameterised; the
    # x packing is fixed (data.canonical.build_x_full), so nothing here has
    # to agree with the dataset beyond the config grid.
    npe, (PRIOR_LOW, PRIOR_HIGH) = build_npe(cfg)
    embedding = npe.embedding
    print(f"  embedding: {type(embedding).__name__}, reads "
          f"{embedding._blocks}, context {embedding.context_dim}")
    
    OUT = make_out_dir(cfg, args)
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
        f"{OUT}/val_waveforms.pkl", build_val_waveforms, cfg)
    sky_bank = load_or_compute(f"{OUT}/sky_bank.pkl", precompute_sky_bank,
                               cfg, n_sky=cfg.get("n_sky_bank", 100000))
    noise_data_dir = cfg.get("noise_data_dir", None)
    psd_bank = build_psd_bank(cfg, OUT, noise_data_dir)
    noise_bank = build_noise_bank(cfg, OUT, noise_data_dir)
 
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
    std = load_or_compute(std_path, fit_standardiser, dataset, cfg,
                          PARAM_NAMES)
    if cfg.get("aux_supervision", False):
        check_aux_stats(std)
    check_amp_stats(std, cfg)
    print(f"  x: mean={std.x_mean:.4f}, std={std.x_std:.4f}")
    if getattr(std, "aux_mean", None) is not None:
        print(f"  aux: {N_AUX} targets, std range "
              f"[{float(std.aux_std.min()):.3g}, "
              f"{float(std.aux_std.max()):.3g}]")

    embedding.std = std          # the embedding now standardises raw x itself

    # the flow was built before std existed, so its prior box is still empty;
    # fill it now that the normalised bounds can be computed
    npe.flow.set_bounds(*std.get_normalised_prior_bounds(PRIOR_LOW, PRIOR_HIGH))

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
    aux_head = None
    if cfg.get("aux_supervision", False):
        aux_k = min(int(cfg.get("aux_n_channels", 128)),
                    embedding.output_dim)
        aux_head = AuxHead(aux_k, N_AUX,
                           hidden=int(cfg.get("aux_head_hidden", 128)))

    npe, summary = custom_train_npe(
        npe, aux_head, dataset, x_val, theta_val, std, cfg, OUT,
        std_path=std_path, resume_dir=args.resume)

    npe.save(f"{OUT}/npe.pt")
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
        s = run_inference(npe, xd[0], cfg, PARAM_NAMES, tc=tc_gps)
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
    pp_test(npe, x_test[_idx], theta_test[_idx], std, PARAM_NAMES,
        f"{OUT}/pp_test.png")
 
    print(f"\nAll outputs → {OUT}/")
 
 
if __name__ == "__main__":
    main()