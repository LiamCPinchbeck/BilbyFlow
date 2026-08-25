"""
Example: the full training pipeline with a CUSTOM embedding.

    python -m bilbyflow.scripts.fd_psd_embedding config.yaml
    python -m bilbyflow.scripts.fd_psd_embedding config.yaml --resume RUN_DIR

Identical to scripts/train.py except for one line — the embedding class handed
to build_npe. Copy this file, swap in your own StrainEmbedding subclass, and
everything else (banks, datasets, standardiser, curriculum, diagnostics, PP
test) works unchanged.

THE EMBEDDING. FDPSDEmbedding reads frequency domain + PSD only, no time
domain: the six frequency-aligned channels [H1 Re, H1 Im, L1 Re, L1 Im,
psd_H1, psd_L1] feed ONE Conv1D-stem -> ResNet branch.

  * fd (B,4,n_masked) and psd (B,2,n_masked) live on the SAME frequency axis,
    so channel-concatenating them is exact per-bin fusion — the conv can learn
    "distrust this bin, there is a line here", which late fusion after global
    pooling cannot.
  * the TD channels are a deterministic irfft of the FD channels: no new
    information, only a different inductive bias. Dropping them makes this the
    TD ablation as well as the PSD-fusion experiment.
  * one branch instead of two, so roughly half the parameters of
    Conv1dResNetEmbedding.

Discussed at the Monash LIGO meeting, 20/08/2026.

NOTE ON x: the dataset always packs the full vector
[strain || psd || amp]; `blocks` selects which views embed() receives, not
what is built. Reading fewer blocks saves network compute, not data
generation.
"""

import pickle
import argparse

import numpy as np
import torch
import torch.nn as nn
import yaml

from bilbyflow.io.config import load_config, get_reference_detector_data
from bilbyflow.coordinates.sky import samples_detector_to_radec
from bilbyflow.coordinates.params import dL_to_physical
from bilbyflow.data.dataset import OnTheFlyGWDataset, generate_fixed_dataset
from bilbyflow.data.standardiser import check_aux_stats, check_amp_stats
from bilbyflow.data.banks import (precompute_waveforms, precompute_sky_bank,
                          load_or_compute)
from bilbyflow.nn.aux_head import AUX_NAMES, N_AUX, AuxHead
from bilbyflow.nn.embedding import StrainEmbedding, _Branch
from bilbyflow.training.trainer import custom_train_npe
from bilbyflow.plotting.training import plot_losses
from .utils import (run_inference, plot_corner_fig, pp_test,
                    plot_aux_diagnostics, run_diagnostics,
                    print_run_banner, make_out_dir, build_val_waveforms,
                    build_psd_bank, build_noise_bank, fit_standardiser,
                    build_npe)

##########################################################################################
##########################################################################################
#
#### --------         the custom embedding        -------------------
#
##########################################################################################
##########################################################################################

class FDPSDEmbedding(StrainEmbedding):
    """[FD (4ch) || PSD (2ch)] -> one Conv1D/ResNet branch -> MLP head."""

    blocks = ("fd", "psd")          # td is packed, but not handed to embed()

    def __init__(self, cfg, std=None, output_dim=None):
        super().__init__(cfg, std, output_dim)

        if not cfg.get("psd_conditioning", False):
            raise ValueError(
                "FDPSDEmbedding needs psd_conditioning: true — the PSD block "
                "is one of its input channels, not an optional extra.")

        self.branch = _Branch(
            6,                                                  # 4 FD + 2 PSD
            list(cfg.get("conv1d_channels", [32, 64, 128, 128])),
            int(cfg.get("conv1d_kernel", 7)),
            int(cfg.get("conv1d_resnet_stem_out", 64)),
            cfg.get("conv1d_resnet_backbone", "resnet18"),
            self.output_dim,
            self.n_masked,                                      # channel length
        )

        self.head = nn.Sequential(
            nn.Linear(self.output_dim, self.output_dim), nn.ELU(),
            nn.Dropout(float(cfg.get("conv1d_dropout", 0.1))),
            nn.Linear(self.output_dim, self.output_dim))

    def embed(self, fd, psd):
        # same frequency axis -> channel-concat is exact per-bin fusion
        return self.head(self.branch(torch.cat([fd, psd], dim=1)))







##########################################################################################
# --           main: the same pipeline as scripts/train.py              ------------------
##########################################################################################

def main():
    parser = argparse.ArgumentParser(
        description="Train an NPE model with the FD+PSD custom embedding")
    parser.add_argument("config", type=str, help="Path to config.yaml")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--no-fresh-data", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    PARAM_NAMES = cfg["inferred_parameters"]
    print_run_banner(cfg)

    # THE ONLY LINE THAT DIFFERS FROM scripts/train.py: the embedding class.
    npe, (PRIOR_LOW, PRIOR_HIGH) = build_npe(cfg,
                                             embedding_cls=FDPSDEmbedding)
    embedding = npe.embedding
    print(f"  embedding: {type(embedding).__name__}, reads "
          f"{embedding._blocks}, context {embedding.context_dim}, "
          f"{sum(p.numel() for p in embedding.parameters()) / 1e6:.1f}M params")

    OUT = make_out_dir(cfg, args)
    with open(f"{OUT}/config.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    # -- 1. banks (cached in OUT; resume symlinks land here too) --
    h_plus, h_cross, intr_params = load_or_compute(
        f"{OUT}/waveforms.pkl", precompute_waveforms, cfg, seed_val=42)
    h_plus_v, h_cross_v, intr_params_v = load_or_compute(
        f"{OUT}/val_waveforms.pkl", build_val_waveforms, cfg)
    sky_bank = load_or_compute(f"{OUT}/sky_bank.pkl", precompute_sky_bank,
                               cfg, n_sky=cfg.get("n_sky_bank", 100000))
    noise_data_dir = cfg.get("noise_data_dir", None)
    psd_bank = build_psd_bank(cfg, OUT, noise_data_dir)
    noise_bank = build_noise_bank(cfg, OUT, noise_data_dir)

    # -- 2. datasets (train + disjoint-intrinsics val/test) --
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

    # -- 3. fixed val/test sets + standardiser --
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

    embedding.std = std          # the embedding standardises raw x itself
    npe.flow.set_bounds(*std.get_normalised_prior_bounds(PRIOR_LOW,
                                                         PRIOR_HIGH))

    # -- 4. sanity + diagnostics --
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

    # -- 5. train --
    print("\n=== Training NPE (FD+PSD embedding) ===")
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

    # -- 6. example posteriors + PP test --
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
    # proposal covers regions the flow's prior box rejects).
    _tt = theta_test.numpy()
    _in = np.all((_tt >= PRIOR_LOW.numpy()) & (_tt <= PRIOR_HIGH.numpy()),
                 axis=1)
    _idx = torch.from_numpy(np.flatnonzero(_in)[:200].copy())
    print(f"PP test: {int(_in.sum())}/{len(_tt)} test injections inside "
          f"analysis prior")
    pp_test(npe, x_test[_idx], theta_test[_idx], std, PARAM_NAMES,
            f"{OUT}/pp_test.png")

    print(f"\nAll outputs → {OUT}/")
    print("REWEIGHTING: this run's config records embedding_type from the "
          "config, NOT this class — pass embedding_cls=FDPSDEmbedding to "
          "build_npe when rebuilding the checkpoint.")


if __name__ == "__main__":
    main()