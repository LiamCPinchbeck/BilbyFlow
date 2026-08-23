"""
bilbyflow.training.trainer — training driver.

The main file that contains methods to train the flow architecture.

The NPE/flow/embedding is assumed to have ALREADY been constructed by the
caller. This module is responsible only for training that supplied NPE.

Three main entry points:
  * custom_train_npe   — curriculum driver using an externally supplied NPE,
                         iterates the stages, does the final BN recalibration,
                         and returns (npe, estimator, summary).
  * train_one_stage    — one curriculum stage in place on the supplied NPE:
                         NLL + annealed aux MSE + JEPA consistency + VICReg,
                         early stopping / best_state on the deployable val_sel NLL.
  * recalibrate_bn     — reset the embedding's BatchNorm running stats from
                         synthetic signal+noise inputs; also called from
                         the reweighting path before inference.

IMPORTANT:
  The NPE must already contain the embedding network and density estimator /
  posterior estimator. This module does NOT build them.

The separable loss pieces (SNR weights, VICReg, dropout) live in
training.losses;
the curriculum schedule in training.curriculum;
checkpoint I/O in training.checkpoint;
the flow architecture lives in nn.flow;
the aux head + targets in nn.aux_head;
loss-curve plotting in plotting.training.

recalibrate_bn imports make_prior_dict lazily (parity with the original).
Had some issues with recalibrate_bn so it is a noted "todo".
"""

import os
import time
import pickle

import numpy as np
import torch
import bilby

from ..data.canonical import canonical_grid  # noqa: F401  (parity with call sites)
from ..data.noise import injection_to_x
from ..data.canonical import build_x_full
from ..coordinates.sky import MAX_DT_HL
from ..nn.aux_head import AuxHead, N_AUX
from ..data.standardiser import check_aux_stats
from ..data.dataset import generate_fixed_dataset
from .curriculum import build_curriculum_stages
from .checkpoint import save_checkpoint, load_checkpoint
from .losses import snr_weights_from_amp, vicreg_penalty, set_dropout_p
from ..plotting.training import _plot_losses


__all__ = ["custom_train_npe", "train_one_stage"]


# -- one curriculum stage -----------------------------------------------------

def train_one_stage(
    npe, aux_head, dataset,
    x_val, theta_val, std, cfg,
    stage, stage_idx,
    OUT, DEVICE,
    global_train, global_val, global_val_cap,
    global_aux, global_cons, global_times,
    n_stages,
    aux_k=None,
):
    """Train one curriculum stage on an already-constructed NPE."""

    dataset.set_dL_cap(stage["dL_max"])

    use_cache = getattr(dataset, "cache_slots", 0) > 0
    if use_cache:
        dataset.refresh_clean_cache()

    restd = str(
        cfg.get("curriculum_restandardise", "x")
    ).lower()

    n_restd = int(
        cfg.get(
            "curriculum_restd_n",
            min(int(cfg["n_standardisation"]), 20000),
        )
    )

    if restd in ("x", "full"):
        xs, ts, _ = generate_fixed_dataset(dataset, n_restd)

        # The embedding owns x-standardisation, so keep std in sync
        # but do NOT standardise x before passing it to npe.
        std.update_x(xs)

        if restd == "full":
            std.update_theta(ts)

        del xs, ts

    x_val_model = x_val
    theta_val_norm = std.normalise_theta(theta_val)

    dL_idx = cfg["inferred_parameters"].index("luminosity_distance")

    _cap = float(stage["dL_max"])
    if str(cfg.get("dL_param", "linear")).lower() == "log":
        _cap = float(np.log(_cap))

    cap_mask = theta_val[:, dL_idx].numpy() <= _cap
    n_cap = int(cap_mask.sum())

    if n_cap >= 64 and n_cap < len(x_val):
        x_val_cap = x_val[cap_mask]  # RAW x
        theta_val_cap = std.normalise_theta(
            theta_val[cap_mask]
        )

    elif n_cap >= 64:
        x_val_cap = x_val_model
        theta_val_cap = theta_val_norm

    else:
        print(
            f"  [stage {stage_idx+1}] WARNING: "
            f"only {n_cap} capped val samples"
        )
        x_val_cap = x_val_model
        theta_val_cap = theta_val_norm

    # SNR weights for the FIXED val set, recovered from the amp block of x.
    rho0 = float(cfg.get("snr_weight_rho0", 0.0))
    w_max = float(cfg.get("snr_weight_max", 10.0))
    ad_std = int(getattr(std, "amp_dim", 0) or 0)

    w_val_cap = snr_weights_from_amp(
        x_val[cap_mask] if (n_cap >= 64 and n_cap < len(x_val)) else x_val,
        ad_std, rho0, w_max=w_max)

    # v4.1 aux-lambda schedule for this stage: linear anneal to 0 by
    # aux_anneal_frac * stage epochs (guides early optimisation, then releases)
    aux_requested = bool(
        cfg.get("aux_supervision", False)
    )

    if aux_requested and aux_head is None:
        raise AttributeError(
            "aux_supervision=True but no aux_head was supplied."
        )

    aux_on = aux_requested and aux_head is not None

    if aux_on:
        aux_head = aux_head.to(DEVICE)

        lam0 = float(
            cfg.get("aux_lambda", 0.5)
        )
        anneal_frac = float(
            cfg.get("aux_anneal_frac", 0.7)
        )
        n_anneal = max(
            1,
            int(
                anneal_frac * int(stage["epochs"])
            ),
        )
    else:
        lam0 = 0.0
        n_anneal = 1
    
    # v4.3: JEPA embedding consistency
    ec_on = bool(cfg.get("embed_consistency", False))
    lam_c = float(cfg.get("consistency_lambda", 0.05)) if ec_on else 0.0
    strain_emb_dim = int(
        npe.embedding.output_dim
    )

    # v4.7: VICReg
    vc_on = bool(cfg.get("vc_reg", False))
    vc_gamma = float(cfg.get("vc_gamma", 0.5))
    lam_v0 = float(cfg.get("vc_lambda_var", 0.1)) if vc_on else 0.0
    lam_cov0 = float(cfg.get("vc_lambda_cov", 0.01)) if vc_on else 0.0

    # v4.1 final-stage dropout override
    if stage_idx == n_stages - 1 and "final_stage_flow_dropout" in cfg:
        nd = set_dropout_p(npe.flow, float(cfg["final_stage_flow_dropout"]))
        print(f"  [stage {stage_idx+1}] FINAL stage: set {nd} Dropout modules "
              f"to p={float(cfg['final_stage_flow_dropout'])}")

    print(f"\n=== Curriculum stage {stage_idx+1}/{n_stages} ===")
    print(
        f"  dL_max={stage['dL_max']} Mpc | "
        f"eligible={dataset.eligible_count()}/{dataset.n_sky} "
        f"| x_std={float(std.x_std):.4f}"
    )

    print(
        f"  lr={stage['lr']:.2e} | "
        f"epochs={stage['epochs']} | "
        f"patience={stage['patience']}"
        + (
            f" | aux lambda0={lam0} anneal->0 by epoch {n_anneal} "
            f"(slice: first {aux_k} ctx dims)"
            if aux_on
            else ""
        )
    )

    weight_decay = float(cfg.get("weight_decay", 0.0))

    params = list(npe.parameters())

    if aux_on:
        params += list(aux_head.parameters())

    optimiser = torch.optim.AdamW(
        params,
        lr=float(stage["lr"]),
        weight_decay=weight_decay,
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=int(stage["t_max"]),
        eta_min=min(float(cfg.get("eta_min", 1e-5)), float(stage["lr"]) / 5.0))

    def _worker_init_fn(worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)

    eff_bs = int(cfg["batch_size"])
    micro_bs = int(cfg.get("micro_batch_size", eff_bs))
    accum = max(1, eff_bs // micro_bs)
    if eff_bs % micro_bs:
        raise ValueError(f"batch_size {eff_bs} not divisible by micro_batch_size {micro_bs}")
    print(f"  effective batch {eff_bs} = {accum} x micro {micro_bs}")

    # -- collate: x must remain RAW -----------------------------------------------

    def _collate_cons(batch):
        xs, ths, sids, auxs, cons = zip(*batch)

        aux_raw = torch.stack(auxs)

        if rho0 > 0.0:
            rho2 = (
                torch.exp(2 * aux_raw[:, 0])
                + torch.exp(2 * aux_raw[:, 1])
            )
            w = 1.0 + rho2 / rho0 ** 2
            w = (w / w.mean()).clamp(max=w_max)
        else:
            w = torch.ones(len(xs))

        keep = [
            i for i, c in enumerate(cons)
            if c.dim() == 2
        ]

        cons_stack = (
            torch.stack([cons[i] for i in keep])      # RAW x
            if keep
            else torch.zeros(0)
        )

        cons_idx = torch.tensor(
            keep,
            dtype=torch.long,
        )

        return (
            torch.stack(xs),
            std.normalise_theta(torch.stack(ths)),
            torch.stack(sids),
            (
                std.normalise_aux(aux_raw)
                if getattr(std, "aux_mean", None) is not None
                else aux_raw
            ),
            cons_stack,
            cons_idx,
            w,
        )

    
    def _make_loader():
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=micro_bs,
            shuffle=True,
            collate_fn=_collate_cons,
            num_workers=int(cfg.get("num_workers", 12)),
            pin_memory=True,
            drop_last=True,
            worker_init_fn=_worker_init_fn,
            persistent_workers=True,
            prefetch_factor=int(cfg.get("prefetch_factor", 4)),
        )

    dataloader = _make_loader()

    clip_norm = float(cfg["clip_max_norm"])
    checkpoint_interval = int(cfg.get("checkpoint_interval", 30))
    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    def _nll(x_n, t_n, w=None):
        out = []
        ch = int(cfg.get("val_eval_chunk", 512))

        for i in range(0, len(x_n), ch):
            xb = x_n[i:i + ch].to(
                DEVICE,
                non_blocking=True,
            )
            tb = t_n[i:i + ch].to(
                DEVICE,
                non_blocking=True,
            )

            context = npe.embed(xb)
            lp = npe.flow_log_prob(tb, context)

            out.append(
                lp.reshape(-1).cpu()
            )

        lp_all = torch.cat(out)

        if w is None:
            return -lp_all.mean().item()

        return -(w * lp_all).mean().item()

    for e in range(int(stage["epochs"])):
        epoch_start = time.perf_counter()

        if use_cache and e > 0 and e % dataset.cache_refresh_epochs == 0:
            dataset.refresh_clean_cache()
            # Persistent workers hold a forked copy of the cache,
            # so they have to be respawned.
            dataloader = _make_loader()

        lam = lam0 * max(0.0, 1.0 - e / n_anneal) if aux_on else 0.0

        n_cwarm = max(1, int(cfg.get("consistency_warmup_epochs", 5)))

        lam_c_e = lam_c * min(1.0, (e + 1) / n_cwarm) if ec_on else 0.0
        lam_v_e = lam_v0 * min(1.0, (e + 1) / n_cwarm) if vc_on else 0.0
        lam_cov_e = lam_cov0 * min(1.0, (e + 1) / n_cwarm) if vc_on else 0.0


        ########################################################################
        ########################################################################        
        #
        #    training mode + main forward pass 
        #
        ########################################################################
        ########################################################################

        npe.train()

        if aux_on:
            aux_head.train()

        epoch_losses = []
        epoch_aux = []
        epoch_cons = []
        epoch_nll_only = []
        epoch_gnll, epoch_gaux = [], []
        epoch_vc = []
        v_m = c_m = 0.0

        optimiser.zero_grad(set_to_none=True)
        micros_since_step = 0

        for (
            x_batch,
            theta_batch,
            _,
            aux_batch,
            cons_batch,
            cons_idx,
            w_batch,
        ) in dataloader:

            x_batch = x_batch.to(
                DEVICE,
                non_blocking=True,
            )
            theta_batch = theta_batch.to(
                DEVICE,
                non_blocking=True,
            )
            w_batch = w_batch.to(
                DEVICE,
                non_blocking=True,
            )

            # One embedding forward: raw x -> context.
            context = npe.embed(x_batch)

            # Flow sees only the context.
            log_prob = npe.flow_log_prob(
                theta_batch,
                context,
            )

            nll = -(w_batch * log_prob).mean()
            epoch_nll_only.append(nll.detach())

            vc_pen = 0.0

            if vc_on and (
                lam_v_e > 0.0
                or lam_cov_e > 0.0
            ):
                f = context[:, :strain_emb_dim]

                vc_pen, v_det, c_det = vicreg_penalty(
                    f,
                    vc_gamma,
                    lam_v_e,
                    lam_cov_e,
                )

                epoch_vc.append(
                    (v_det, c_det)
                )

            aux_term = 0.0

            if aux_on and lam > 0.0:
                aux_z = aux_batch.to(
                    DEVICE,
                    non_blocking=True,
                )

                aux_pred = aux_head(
                    context[:, :aux_k]
                )

                aux_mse = torch.nn.functional.mse_loss(
                    aux_pred,
                    aux_z,
                )

                aux_term = lam * aux_mse
                epoch_aux.append(
                    aux_mse.detach()
                )

            main_loss = (
                nll
                + vc_pen
                + aux_term
            )

            (main_loss / accum).backward()

            total = main_loss.detach()

            # JEPA block

            if (
                ec_on
                and lam_c_e > 0.0
                and cons_batch.dim() == 3
                and cons_batch.shape[0] > 0
            ):
                B, K, D = cons_batch.shape

                ks = int(
                    cfg.get(
                        "consistency_k_synth",
                        1,
                    )
                )
                kr = K - ks

                if kr < 1:
                    raise ValueError(
                        f"consistency stack has {K} rows with "
                        f"consistency_k_synth={ks}: the k_synth rows are "
                        f"no_grad ANCHORS, so >= 1 further row is needed or "
                        f"the term contributes no gradient.")

                anchors = []

                was_training = npe.embedding.training

                if bool(
                    cfg.get(
                        "consistency_eval_mode",
                        True,
                    )
                ):
                    npe.embedding.eval()

                try:
                    flat = cons_batch
                    chunk = int(
                        cfg.get(
                            "consistency_chunk",
                            1024,
                        )
                    )

                    # the main batch's own context is a synthetic draw:
                    # reuse it as anchor row 0 at zero extra cost
                    if bool(cfg.get("consistency_reuse_main", True)):
                        anchors.append(
                            context[cons_idx.to(DEVICE)]
                            .detach()[:, :strain_emb_dim]
                            .unsqueeze(1)
                        )

                    if ks > 0:
                        with torch.no_grad():
                            srows = flat[:, :ks].reshape(
                                B * ks,
                                D,
                            )

                            fs = torch.cat(
                                [
                                    npe.embedding(
                                        srows[i:i + chunk].to(DEVICE)
                                    )[:, :strain_emb_dim]
                                    for i in range(
                                        0,
                                        B * ks,
                                        chunk,
                                    )
                                ],
                                dim=0,
                            )

                        anchors.append(
                            fs.view(
                                B,
                                ks,
                                strain_emb_dim,
                            )
                        )

                    assert anchors, (
                        "need consistency_reuse_main "
                        "or k_synth >= 1"
                    )

                    anchor = torch.cat(
                        anchors,
                        dim=1,
                    ).mean(dim=1)

                    fr = npe.embedding(
                        flat[:, ks:].reshape(
                            B * kr,
                            D,
                        ).to(DEVICE)
                    )

                    fr = fr[:, :strain_emb_dim].view(
                        B,
                        kr,
                        strain_emb_dim,
                    )

                    if bool(
                        cfg.get(
                            "consistency_normalize",
                            True,
                        )
                    ):
                        fr_n = torch.nn.functional.normalize(
                            fr,
                            dim=-1,
                        )

                        an_n = torch.nn.functional.normalize(
                            anchor,
                            dim=-1,
                        ).unsqueeze(1)

                        cons_mse = (
                            (fr_n - an_n) ** 2
                        ).sum(-1).mean()

                    else:
                        cons_mse = torch.nn.functional.mse_loss(
                            fr,
                            anchor.unsqueeze(1).expand_as(fr),
                        )

                finally:
                    if was_training:
                        npe.embedding.train()

                skip_thr = float(
                    cfg.get(
                        "consistency_skip_above",
                        25.0,
                    )
                )

                gate = (
                    cons_mse.detach() < skip_thr
                ).float()

                (
                    lam_c_e
                    * gate
                    * cons_mse
                    / accum
                ).backward()

                total = (
                    total
                    + lam_c_e
                    * (gate * cons_mse).detach()
                )

                epoch_cons.append(
                    cons_mse.detach()
                )

            epoch_losses.append(total)
            micros_since_step += 1
            if micros_since_step == accum:
                torch.nn.utils.clip_grad_norm_(params, clip_norm)
                optimiser.step()
                optimiser.zero_grad(set_to_none=True)
                micros_since_step = 0

        if micros_since_step:                  # flush a trailing partial group
            torch.nn.utils.clip_grad_norm_(params, clip_norm)
            optimiser.step()
            optimiser.zero_grad(set_to_none=True)
        scheduler.step()

        if not epoch_losses:
            global_train.append(float("nan"))
            global_val.append(float("nan"))
            global_val_cap.append(float("nan"))
            global_aux.append(float("nan"))
            global_times.append(0.0)
            continue

        train_loss = float(torch.stack(epoch_losses).mean().cpu())
        train_nll = float(torch.stack(epoch_nll_only).mean().cpu())
        aux_loss = float(torch.stack(epoch_aux).mean().cpu()) if epoch_aux else 0.0

        # evaluation + best-state block

        npe.eval()

        with torch.no_grad():
            val_full = (
                _nll(x_val_model,theta_val_norm,)
                if e == 0 or (e + 1) % 5 == 0 else float("nan")
                )

            val_cap = _nll(x_val_cap, theta_val_cap)

            val_sel = _nll(x_val_cap, theta_val_cap, w=w_val_cap)

        global_train.append(train_loss)
        global_val.append(val_full)
        global_val_cap.append(val_cap)
        global_aux.append(aux_loss)
        cons_loss = float(torch.stack(epoch_cons).mean().cpu()) if epoch_cons else 0.0

        if epoch_vc:
            v_m = float(torch.stack([v for v, _ in epoch_vc]).mean().cpu())
            c_m = float(torch.stack([c for _, c in epoch_vc]).mean().cpu())

        global_cons.append(cons_loss)
        epoch_time = time.perf_counter() - epoch_start
        global_times.append(epoch_time)

        if val_sel < best_val_loss:
            best_val_loss = val_sel
            epochs_without_improvement = 0

            best_state = {
                k: v.detach().cpu().clone()
                for k, v in npe.state_dict().items()
            }
        else:
            epochs_without_improvement += 1

        if (e + 1) % 5 == 0 or e == 0:
            lr_now = scheduler.get_last_lr()[0]

            msg = (f"  [stage {stage_idx+1}] epoch {e+1:4d}/{stage['epochs']} | "
                   f"train={train_loss:.4f} | nll={train_nll:.4f} | "
                   f"val_cap={val_cap:.4f} | "
                   f"val_sel={val_sel:.4f} | val_full={val_full:.4f} | "
                   f"best_sel={best_val_loss:.4f} | ")

            if aux_on:
                msg += f"aux_mse={aux_loss:.4f} (lam={lam:.3f}) | "
            if ec_on and epoch_cons:
                msg += f"cons={cons_loss:.4f} | "
            if vc_on and epoch_vc:
                msg += f"vc(v={v_m:.3f},c={c_m:.4f}) | "
            msg += (f"lr={lr_now:.2e} | patience={epochs_without_improvement}/"
                    f"{stage['patience']} | time={epoch_time:.1f}s")
            if aux_on and epoch_gnll:
                r = np.mean(epoch_gaux) / max(np.mean(epoch_gnll), 1e-30)
                msg += (f" | |g_nll|={np.mean(epoch_gnll):.2e} "
                        f"|lam*g_aux|={np.mean(epoch_gaux):.2e} ratio={r:.2f}")

            print(msg)

        if len(global_train) % checkpoint_interval == 0:
            save_checkpoint(
                f"{OUT}/checkpoint.pt",
                len(global_train),
                npe,
                optimiser,
                scheduler,
                best_val_loss,
                best_state,
                global_train,
                global_val,
                global_times,
                stage_idx=stage_idx,
                val_cap_losses=global_val_cap,
                aux_head=aux_head if aux_on else None,
                aux_losses=global_aux,
                cons_losses=global_cons,
                aux_n_channels=aux_k,
            )

            _plot_losses(global_train, global_val, f"{OUT}/loss_curves.png",
                         val_cap_losses=global_val_cap, aux_losses=global_aux,
                         cons_losses=global_cons)
            print(f"Checkpoint saved ({len(global_train)} total epochs)")

        if epochs_without_improvement >= int(stage["patience"]):
            print(f"  [stage {stage_idx+1}] early stop at stage-epoch {e+1}")
            break

    save_checkpoint(
        f"{OUT}/checkpoint.pt",
        len(global_train),
        npe,
        optimiser,
        scheduler,
        best_val_loss,
        best_state,
        global_train,
        global_val,
        global_times,
        stage_idx=stage_idx,
        val_cap_losses=global_val_cap,
        aux_head=aux_head if aux_on else None,
        aux_losses=global_aux,
        cons_losses=global_cons,
        aux_n_channels=aux_k,
    )

    return best_state, best_val_loss


# -- curriculum training driver ----------------------------------------------

def custom_train_npe(npe, aux_head, dataset,
                     x_val, theta_val,
                     std, cfg, OUT,
                     std_path=None,
                     resume_dir=None,
                     ):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    torch.backends.cudnn.benchmark = True

    # Move the complete supplied NPE.
    npe = npe.to(DEVICE)

    if npe.embedding.std is not std:
        raise ValueError(
            "Trainer std and npe.embedding.std are different objects. "
            "Update the embedding's standardiser rather than standardising "
            "x in the trainer."
        )

    dL_full = float(np.max(dataset.dL_all))

    resume_lr = (
        cfg.get("resume_learning_rate", None)
        if resume_dir else None
    )

    stages = build_curriculum_stages(cfg, dL_full,
                                     resume_lr=resume_lr)

    aux_on = bool(cfg.get("aux_supervision", False) )

    if aux_on and aux_head is None:
        raise ValueError("aux_supervision=True but no aux_head was supplied.")

    if aux_head is not None:
        aux_head = aux_head.to(DEVICE)

    aux_k = None

    if aux_on:
        check_aux_stats(std)

        feat_dim = int(npe.embedding.output_dim)

        aux_k = min(int(cfg.get("aux_n_channels", 128)), feat_dim)
        in_dim = int(aux_head.net[0].in_features)
        if in_dim != aux_k:
            raise ValueError(
                f"aux_head expects {in_dim} input dims but the aux slice is "
                f"{aux_k} (= min(aux_n_channels, embedding.output_dim)). "
                f"Build it as AuxHead({aux_k}, ...).")
        print(
            f"  aux head: ctx dims [0:{aux_k}] of {feat_dim} "
            f"-> {N_AUX} summaries "
            f"(lambda0={cfg.get('aux_lambda', 0.5)}, "
            f"anneal_frac={cfg.get('aux_anneal_frac', 0.7)})"
        )

    global_train = []
    global_val = []
    global_val_cap = []
    global_aux = []
    global_cons = []
    global_times = []
    start_stage = 0
    carried_best_state = None

    # ---------------- resume ------------------------------

    if resume_dir:
        ckpt_path = os.path.join(resume_dir,"checkpoint.pt",)

        if os.path.exists(ckpt_path):
            ckpt = load_checkpoint(ckpt_path,
                                   DEVICE,
                                   )

            npe.load_state_dict(ckpt["model_state_dict"])

            if (aux_on and ckpt.get("aux_head_state_dict") is not None):
                from ..nn.aux_head import AUX_NAMES as _AUX_NAMES

                ck_names = ckpt.get("aux_names")
                ck_k = ckpt.get("aux_n_channels")

                if (ck_names is not None and list(ck_names) == list(_AUX_NAMES)
                    and ck_k == aux_k):
                    aux_head.load_state_dict(ckpt["aux_head_state_dict"])
                else:
                    print(f"  NOTE: saved aux head incompatible "
                          f"(names match: {list(ck_names or []) == list(_AUX_NAMES)}, "
                          f"k: {ck_k} vs {aux_k}) — reinitialised")

            carried_best_state = ckpt.get("best_state", None,)


            global_train = list(ckpt.get("train_losses",[]))
            global_val = list(ckpt.get("val_losses", []))
            global_val_cap = list(ckpt.get("val_cap_losses",[]))
            global_aux = list(ckpt.get("aux_losses", []))
            global_cons = list(ckpt.get("cons_losses", []))
            global_times = list(ckpt.get("train_times", []))


            start_stage = int(ckpt.get("stage_idx",0,))

        if carried_best_state is not None:
            npe.load_state_dict(carried_best_state)

    # -- training -------------------------------------------------------------

    print(
        f"\nTraining on {DEVICE} | "
        f"{len(dataset)} waveforms | "
        f"batch={cfg['batch_size']} | "
        f"{len(cfg['inferred_parameters'])} params | "
        f"{len(stages)} stage(s) | "
        f"aux={'ON' if aux_on else 'off'}"
    )

    for stage_idx in range(start_stage, len(stages)):
        if carried_best_state is not None:
            npe.load_state_dict(carried_best_state)

        best_state, best_val = train_one_stage(
            npe=npe,
            aux_head=aux_head,
            dataset=dataset,
            x_val=x_val,
            theta_val=theta_val,
            std=std,
            cfg=cfg,
            stage=stages[stage_idx],
            stage_idx=stage_idx,
            OUT=OUT,
            DEVICE=DEVICE,
            global_train=global_train,
            global_val=global_val,
            global_val_cap=global_val_cap,
            global_aux=global_aux,
            global_cons=global_cons,
            global_times=global_times,
            n_stages=len(stages),
            aux_k=aux_k,
        )

        if best_state is not None:
            npe.load_state_dict(best_state)

        carried_best_state = best_state if best_state is not None else carried_best_state
        if std_path is not None:
            with open(std_path, "wb") as f:
                pickle.dump(std, f)

        print(f"=== stage {stage_idx+1} done: best val={best_val:.4f} ===")

    # ------------------ restore best state ------------------------

    if carried_best_state is not None:
        npe.load_state_dict(
            carried_best_state
        )

    npe = npe.to(DEVICE)

    print(
        "Recalibrating BatchNorm running statistics..."
    )

    for m in npe.embedding.modules():
        if isinstance(
            m,
            (
                torch.nn.BatchNorm1d,
                torch.nn.BatchNorm2d,
            ),
        ):
            m.reset_running_stats()
            m.momentum = None

    dataset.set_dL_cap(None)

    npe.embedding.train()

    _bn_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=512,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        collate_fn=lambda b: torch.stack([item[0] for item in b]),
    )

    _bn_seen = 0

    with torch.no_grad():
        for _bn_batch in _bn_loader:
            _xb = _bn_batch.to(DEVICE, non_blocking=True)

            # RAW x -> embedding; no theta or flow evaluation needed.
            npe.embedding(_xb)

            _bn_seen += len(_xb)

            if _bn_seen >= 20000:
                break

    npe.eval()

    print(
        f"  BN recalibrated over {_bn_seen} samples"
    )


    print(f"\nDone: {len(global_train)} total epochs across {len(stages)} stage(s)")
    return (
        npe,
        dict(
            training_loss=global_train,
            validation_loss=global_val,
            validation_loss_capped=global_val_cap,
            aux_loss=global_aux,
            cons_loss=global_cons,
        ),
    )


# Deprecated as of 22/08/2026
# def recalibrate_bn(
#         npe, std,
#         cfg, g, psd_bank,
#         tukey_window, td_norm,
#         n_cal=5000, device="cpu"
#         ):
#     from ..data.banks import make_prior_dict

#     embed = npe.embedding

#     for m in embed.modules():
#         if isinstance(
#             m,
#             (
#                 torch.nn.BatchNorm1d,
#                 torch.nn.BatchNorm2d,
#             ),
#         ):
#             m.reset_running_stats()
#             m.momentum = None

#     embed.train().to(device)

#     priors = make_prior_dict(cfg)
#     n_psds = psd_bank["psd_H1"].shape[0]
#     wfg = bilby.gw.WaveformGenerator(
#         duration=float(cfg["duration"]),
#         sampling_frequency=int(cfg["sampling_frequency"]),
#         frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
#         parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
#         waveform_arguments=dict(
#             waveform_approximant=cfg["waveform_approximant"],
#             reference_frequency=float(cfg["f_min"])))

#     ifos = bilby.gw.detector.InterferometerList(["H1", "L1"])

#     ref_tc = float(cfg["ref_geocent_time"])
#     start_time = ref_tc - g["duration"] / 2

#     for ifo in ifos:
#         ifo.minimum_frequency = float(cfg["f_min"])

#         ifo.set_strain_data_from_frequency_domain_strain(
#             np.zeros(g["n_fd_full"], dtype=complex),
#             sampling_frequency=g["sr"], duration=g["duration"],
#             start_time=start_time)

#     n_ok = 0
#     with torch.no_grad():
#         batch_x = []
#         for _ in range(n_cal * 2):
#             if n_ok >= n_cal:
#                 break

#             pidx = np.random.randint(n_psds)
#             det_psds = {d: psd_bank[f"psd_{d}"][pidx].astype(np.float64)
#                         for d in ["H1", "L1"]}

#             params = priors.sample()
#             params["geocent_time"] = ref_tc
#             try:
#                 pols = wfg.frequency_domain_strain({k: float(v) for k, v in params.items()})
#                 if pols is None:
#                     continue
#             except Exception:
#                 continue

#             signal_fd = {}
#             for ifo in ifos:
#                 ifo.power_spectral_density = (
#                     bilby.gw.detector.PowerSpectralDensity(
#                     frequency_array=g["freq_array"], 
#                     psd_array=det_psds[ifo.name]))

#                 signal_fd[ifo.name] = ifo.get_detector_response(
#                     pols, {k: float(v) for k, v in params.items()})
#             try:
#                 x_strain, _, _, _ = injection_to_x(
#                     signal_fd, det_psds, cfg, g, tukey_window, td_norm,
#                     noise_kind=cfg.get("noise_source", "gaussian_physical"))
#                 x_full = build_x_full(x_strain, det_psds, std, cfg, g)
#                 batch_x.append(torch.tensor(x_full).unsqueeze(0))

#                 n_ok += 1

#             except Exception:
#                 continue

#             if len(batch_x) >= 64:
#                 embed(std.normalise_x(torch.cat(batch_x)).to(device))
#                 batch_x = []

#         if batch_x:
#             embed(std.normalise_x(torch.cat(batch_x)).to(device))

#     embed.eval().to("cpu")
#     print(f"  BN recalibrated on {n_ok} signal+noise inputs")