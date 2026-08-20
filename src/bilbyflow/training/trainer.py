"""
bilbyflow.training.trainer — training driver.

The main file that contains methods to train the flow architecture.
It's currently quite verbose and future updates will endeavour to split it up 
to more manageable bite-sized pieces.

Three main entry points:
  * custom_train_npe   — curriculum driver: builds the flow (and aux head),
                         iterates the stages, does the final BN recalibration,
                         and returns (posterior, de, summary).
  * train_one_stage    — one curriculum stage in place on `de`: NLL +
                         annealed aux MSE + JEPA consistency + VICReg, early
                         stopping / best_state on the deployable val_sel NLL.
                         We recommend not using the JEPA consistency terms in the current 
                         version (1.0.1). 
  * recalibrate_bn     — reset the embedding's BatchNorm running stats from
                         synthetic signal+noise inputs; also called from the
                         reweighting path before inference.

The separable loss pieces (SNR weights, VICReg, dropout) live in
training.losses; 
the curriculum schedule in training.curriculum; 
checkpoint I/O in training.checkpoint; 
the flow build + FeatureCache in nn.flow; 
the aux head + targets in nn.aux_head; 
loss-curve plotting in plotting.training.

recalibrate_bn imports make_prior_dict lazily (parity with the original). 
Had some issues with recalibrate_bn so it is a noted "todo"
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
from ..nn.flow import build_density_estimator
from ..nn.aux_head import AuxHead, N_AUX
from ..data.standardiser import check_aux_stats
from ..data.dataset import generate_fixed_dataset
from .curriculum import build_curriculum_stages
from .checkpoint import save_checkpoint, load_checkpoint
from .losses import snr_weights_from_amp, vicreg_penalty, set_dropout_p
from ..plotting.training import _plot_losses

from sbi.utils import BoxUniform
from sbi.inference.posteriors import DirectPosterior

__all__ = ["custom_train_npe", "train_one_stage", "recalibrate_bn"]


# ── one curriculum stage ─────────────────────────────────────────────────────

def train_one_stage(de, cache, aux_head, dataset, x_val, theta_val, std, cfg,
                    stage, stage_idx, OUT, DEVICE, global_train, global_val,
                    global_val_cap, global_aux, global_cons, global_times, n_stages,
                    aux_k=None):
    """Train one curriculum stage in place on `de` (+ aux_head).
    Loss: NLL + lambda(e) * MSE(aux_head(features[:, :aux_k]), aux_z), with
    lambda annealed to 0 within the stage (aux_anneal_frac). Only the first
    aux_k context dims receive aux gradient at the feature level (v4.2).
    Early stopping and best_state remain on val_sel NLL ONLY (the deployable
    objective)."""
    dataset.set_dL_cap(stage["dL_max"])
    use_cache = getattr(dataset, "cache_slots", 0) > 0
    if use_cache:
        dataset.refresh_clean_cache()      # BEFORE restd: restd samples the
                                           # cached distribution (cached path
                                           # activates once _cache_ready)

    restd = str(cfg.get("curriculum_restandardise", "x")).lower()
    n_restd = int(cfg.get("curriculum_restd_n", min(int(cfg["n_standardisation"]), 20000)))
    if restd in ("x", "full"):
        xs, ts, _ = generate_fixed_dataset(dataset, n_restd)
        std.update_x(xs)
        if restd == "full":
            std.update_theta(ts)
        del xs, ts

    x_val_norm = std.normalise_x(x_val)          # CPU, moved per-chunk in _nll
    theta_val_norm = std.normalise_theta(theta_val)

    dL_idx = cfg["inferred_parameters"].index("luminosity_distance")
    _cap = float(stage["dL_max"])
    if str(cfg.get("dL_param", "linear")).lower() == "log":
        _cap = float(np.log(_cap))
    cap_mask = theta_val[:, dL_idx].numpy() <= _cap
    n_cap = int(cap_mask.sum())
    if n_cap >= 64 and n_cap < len(x_val):
        x_val_cap = std.normalise_x(x_val[cap_mask])       # CPU, _nll chunks
        theta_val_cap = std.normalise_theta(theta_val[cap_mask])
    elif n_cap >= 64:
        x_val_cap, theta_val_cap = x_val_norm, theta_val_norm
    else:
        print(f"  [stage {stage_idx+1}] WARNING: only {n_cap} capped val samples")
        x_val_cap, theta_val_cap = x_val_norm, theta_val_norm

    # SNR weights for the FIXED val set, recovered from the amp block of x.
    rho0 = float(cfg.get("snr_weight_rho0", 0.0))
    w_max = float(cfg.get("snr_weight_max", 10.0))
    ad_std = int(getattr(std, "amp_dim", 0) or 0)
    w_val_cap = snr_weights_from_amp(
        x_val[cap_mask] if (n_cap >= 64 and n_cap < len(x_val)) else x_val,
        ad_std, rho0, w_max=w_max)

    # v4.1 aux-lambda schedule for this stage: linear anneal to 0 by
    # aux_anneal_frac * stage epochs (guides early optimisation, then releases)
    aux_on = bool(cfg.get("aux_supervision", False)) and aux_head is not None
    lam0 = float(cfg.get("aux_lambda", 0.5)) if aux_on else 0.0
    anneal_frac = float(cfg.get("aux_anneal_frac", 0.7))
    n_anneal = max(1, int(anneal_frac * int(stage["epochs"])))

    # v4.3: JEPA embedding consistency
    ec_on = bool(cfg.get("embed_consistency", False))
    lam_c = float(cfg.get("consistency_lambda", 0.05)) if ec_on else 0.0
    strain_emb_dim = int(cfg.get("embedding_output_dim", 512))

    # v4.7: VICReg
    vc_on = bool(cfg.get("vc_reg", False))
    vc_gamma = float(cfg.get("vc_gamma", 0.5))
    lam_v0 = float(cfg.get("vc_lambda_var", 0.1)) if vc_on else 0.0
    lam_cov0 = float(cfg.get("vc_lambda_cov", 0.01)) if vc_on else 0.0

    # v4.1 final-stage dropout override
    if stage_idx == n_stages - 1 and "final_stage_flow_dropout" in cfg:
        nd = set_dropout_p(de, float(cfg["final_stage_flow_dropout"]))
        print(f"  [stage {stage_idx+1}] FINAL stage: set {nd} Dropout modules "
              f"to p={float(cfg['final_stage_flow_dropout'])}")

    print(f"\n=== Curriculum stage {stage_idx+1}/{n_stages} ===")
    print(f"  dL_max={stage['dL_max']} Mpc | eligible={dataset.eligible_count()}/{dataset.n_sky} "
          f"| x_std={float(std.x_std):.4f}")
    print(f"  lr={stage['lr']:.2e} | epochs={stage['epochs']} | patience={stage['patience']}"
          + (f" | aux lambda0={lam0} anneal->0 by epoch {n_anneal} "
             f"(slice: first {aux_k} ctx dims)" if aux_on else ""))

    weight_decay = float(cfg.get("weight_decay", 0.0))
    params = list(de.parameters()) + (list(aux_head.parameters()) if aux_on else [])
    optimiser = torch.optim.AdamW(params, lr=float(stage["lr"]),
                                  weight_decay=weight_decay)

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

    def _collate_cons(batch):
        xs, ths, sids, auxs, cons = zip(*batch)
        aux_raw = torch.stack(auxs)
        # SNR-weighted NLL: w ~ 1 + rho_net^2/rho0^2 (aux[:,0:2] = ln_rho_{H1,L1})
        if rho0 > 0.0:
            rho2 = torch.exp(2 * aux_raw[:, 0]) + torch.exp(2 * aux_raw[:, 1])
            w = 1.0 + rho2 / rho0 ** 2
            w = (w / w.mean()).clamp(max=w_max)
        else:
            w = torch.ones(len(xs))
        keep = [i for i, c in enumerate(cons) if c.dim() == 2]
        cons_stack = (std.normalise_x(torch.stack([cons[i] for i in keep]))
                      if keep else torch.zeros(0))
        cons_idx = torch.tensor(keep, dtype=torch.long)
        return (std.normalise_x(torch.stack(xs)),
                std.normalise_theta(torch.stack(ths)),
                torch.stack(sids),
                std.normalise_aux(aux_raw) if getattr(std, "aux_mean", None) is not None
                else aux_raw,
                cons_stack, cons_idx, w)

    def _make_loader():
        return torch.utils.data.DataLoader(
            dataset, batch_size=micro_bs, shuffle=True,
            collate_fn=_collate_cons,
            num_workers=int(cfg.get("num_workers", 12)), pin_memory=True,
            drop_last=True, worker_init_fn=_worker_init_fn,
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
            lp = de.log_prob(t_n[i:i+ch].to(DEVICE), condition=x_n[i:i+ch].to(DEVICE))
            out.append(lp.reshape(-1).cpu())
        lp_all = torch.cat(out)
        if w is None:
            return -lp_all.mean().item()
        return -(w * lp_all).mean().item()

    for e in range(int(stage["epochs"])):
        epoch_start = time.perf_counter()

        if use_cache and e > 0 and e % dataset.cache_refresh_epochs == 0:
            dataset.refresh_clean_cache()
            dataloader = _make_loader()    # persistent workers hold a FORKED
                                           # copy of the cache -- must respawn

        lam = lam0 * max(0.0, 1.0 - e / n_anneal) if aux_on else 0.0
        n_cwarm = max(1, int(cfg.get("consistency_warmup_epochs", 5)))
        lam_c_e = lam_c * min(1.0, (e + 1) / n_cwarm) if ec_on else 0.0
        lam_v_e = lam_v0 * min(1.0, (e + 1) / n_cwarm) if vc_on else 0.0
        lam_cov_e = lam_cov0 * min(1.0, (e + 1) / n_cwarm) if vc_on else 0.0
        de.train()
        if aux_on:
            aux_head.train()
        epoch_losses, epoch_aux = [], []
        epoch_cons = []
        epoch_nll_only = []
        epoch_gnll, epoch_gaux = [], []
        epoch_vc = []
        v_m = c_m = 0.0

        optimiser.zero_grad(set_to_none=True)
        micros_since_step = 0
        for x_batch, theta_batch, _, aux_batch, cons_batch, cons_idx, w_batch in dataloader:
            x_batch = x_batch.to(DEVICE, non_blocking=True)
            theta_batch = theta_batch.to(DEVICE, non_blocking=True)
            w_batch = w_batch.to(DEVICE, non_blocking=True)

            log_prob = de.log_prob(theta_batch, condition=x_batch)
            nll = -(w_batch * log_prob).mean()
            epoch_nll_only.append(nll.detach())

            # v4.7: VICReg on the strain slice of the shared features. Same
            # graph as the NLL -> one backward, no extra forward.
            vc_pen = 0.0
            if vc_on and (lam_v_e > 0.0 or lam_cov_e > 0.0):
                f = cache.features[:, :strain_emb_dim]
                vc_pen, v_det, c_det = vicreg_penalty(f, vc_gamma, lam_v_e, lam_cov_e)
                epoch_vc.append((v_det, c_det))

            aux_term = 0.0
            if aux_on and lam > 0.0:
                aux_z = aux_batch.to(DEVICE, non_blocking=True)
                aux_pred = aux_head(cache.features[:, :aux_k])
                aux_mse = torch.nn.functional.mse_loss(aux_pred, aux_z)
                aux_term = lam * aux_mse
                epoch_aux.append(aux_mse.detach())
            ((nll + vc_pen + aux_term) / accum).backward()
            total = (nll + vc_pen + aux_term).detach() \
                if (torch.is_tensor(aux_term) or torch.is_tensor(vc_pen)) else nll.detach()

            # (2) v4.5 multi-draw JEPA -- separate graph, separate backward.
            # cache.features (main batch) survives the NLL backward: the GRAPH
            # is freed, the tensor data is not, .detach() below never needed it.
            if ec_on and lam_c_e > 0.0 and cons_batch.dim() == 3 and cons_batch.shape[0] > 0:
                B, K, D = cons_batch.shape
                ks = int(cfg.get("consistency_k_synth", 1))
                kr = K - ks
                assert kr >= 1, "consistency needs >= 1 real row"

                anchors = []
                was_training = cache.net.training
                if bool(cfg.get("consistency_eval_mode", True)):
                    cache.net.eval()      # freeze BN stats + dropout for cons rows
                try:
                    flat = cons_batch
                    chunk = int(cfg.get("consistency_chunk", 1024))
                    if ks > 0:
                        with torch.no_grad():          # anchors: fwd-only, cheap
                            srows = flat[:, :ks].reshape(B * ks, D)
                            fs = torch.cat(
                                [cache.net(srows[i:i + chunk].to(DEVICE))[:, :strain_emb_dim]
                                 for i in range(0, B * ks, chunk)], dim=0)
                        anchors.append(fs.view(B, ks, strain_emb_dim))
                    assert anchors, "need consistency_reuse_main or k_synth >= 1"
                    anchor = torch.cat(anchors, dim=1).mean(dim=1)      # (B, emb)

                    fr = cache.net(flat[:, ks:].reshape(B * kr, D).to(DEVICE))
                    fr = fr[:, :strain_emb_dim].view(B, kr, strain_emb_dim)
                    if bool(cfg.get("consistency_normalize", True)):
                        fr_n = torch.nn.functional.normalize(fr, dim=-1)
                        an_n = torch.nn.functional.normalize(anchor, dim=-1).unsqueeze(1)
                        cons_mse = ((fr_n - an_n) ** 2).sum(-1).mean()  # = 2(1-cos), in [0,4]
                    else:
                        cons_mse = torch.nn.functional.mse_loss(
                            fr, anchor.unsqueeze(1).expand_as(fr))
                finally:
                    if was_training:
                        cache.net.train()

                skip_thr = float(cfg.get("consistency_skip_above", 25.0))
                gate = (cons_mse.detach() < skip_thr).float()   # 0/1 on GPU, no sync
                (lam_c_e * gate * cons_mse / accum).backward()
                total = total + lam_c_e * (gate * cons_mse).detach()
                epoch_cons.append(cons_mse.detach())            # log RAW value: spikes visible

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

        de.eval()
        with torch.no_grad():
            val_full = (_nll(x_val_norm, theta_val_norm)
                        if (e == 0 or (e + 1) % 5 == 0) else float("nan"))
            val_cap = _nll(x_val_cap, theta_val_cap)              # unweighted, for the A/B
            val_sel = (_nll(x_val_cap, theta_val_cap, w=w_val_cap)
                       if w_val_cap is not None else val_cap)     # drives best_state

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
            best_state = {k: v.cpu().clone() for k, v in de.state_dict().items()}
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
            save_checkpoint(f"{OUT}/checkpoint.pt", len(global_train), de,
                            optimiser, scheduler, best_val_loss, best_state,
                            global_train, global_val, global_times,
                            stage_idx=stage_idx, val_cap_losses=global_val_cap,
                            aux_head=aux_head if aux_on else None,
                            aux_losses=global_aux, cons_losses=global_cons,
                            aux_n_channels=aux_k)
            _plot_losses(global_train, global_val, f"{OUT}/loss_curves.png",
                         val_cap_losses=global_val_cap, aux_losses=global_aux,
                         cons_losses=global_cons)
            print(f"  \u2713 Checkpoint saved ({len(global_train)} total epochs)")

        if epochs_without_improvement >= int(stage["patience"]):
            print(f"  [stage {stage_idx+1}] early stop at stage-epoch {e+1}")
            break

    save_checkpoint(f"{OUT}/checkpoint.pt", len(global_train), de,
                    optimiser, scheduler, best_val_loss, best_state,
                    global_train, global_val, global_times,
                    stage_idx=stage_idx, val_cap_losses=global_val_cap,
                    aux_head=aux_head if aux_on else None, aux_losses=global_aux,
                    cons_losses=global_cons,
                    aux_n_channels=aux_k)
    return best_state, best_val_loss


# ── curriculum training driver ──────────────────────────────────────────────

def custom_train_npe(dataset, x_val, theta_val, std, cfg, prior_low, prior_high,
                     OUT, std_path=None, resume_dir=None):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True

    dL_full = float(np.max(dataset.dL_all))
    resume_lr = cfg.get("resume_learning_rate", None) if resume_dir else None
    stages = build_curriculum_stages(cfg, dL_full, resume_lr=resume_lr)

    de, cache = build_density_estimator(dataset, std, cfg, DEVICE)

    aux_on = bool(cfg.get("aux_supervision", False))
    rho0 = float(cfg.get("snr_weight_rho0", 0.0))
    if rho0 > 0.0 and not aux_on:
        raise SystemExit("snr_weight_rho0 > 0 requires aux_supervision: true "
                         "(weights derive from aux ln_rho targets)")
    if rho0 > 0.0:
        print(f"  SNR-weighted NLL ON: w = 1 + rho^2/{rho0}^2, "
              f"cap {cfg.get('snr_weight_max', 10.0)} (train + val_sel)")
    aux_head = None
    aux_k = None
    if aux_on:
        check_aux_stats(std)
        with torch.no_grad():
            sample_x, _, _, _, _ = dataset[0]
            cache.eval()
            cache(std.normalise_x(sample_x.unsqueeze(0)).to(DEVICE))
            feat_dim = int(cache.features.shape[-1])
            cache.train()
        # v4.2: aux gradient restricted to the first aux_k context dims
        # (inside the strain embedding; PSD block is beyond the slice).
        aux_k = min(int(cfg.get("aux_n_channels", 128)), feat_dim)
        aux_head = AuxHead(aux_k, N_AUX,
                           hidden=int(cfg.get("aux_head_hidden", 128))).to(DEVICE)
        print(f"  aux head: ctx dims [0:{aux_k}] of {feat_dim} -> {N_AUX} summaries "
              f"(lambda0={cfg.get('aux_lambda', 0.5)}, "
              f"anneal_frac={cfg.get('aux_anneal_frac', 0.7)}), "
              f"{feat_dim - aux_k} ctx dims flow-only at the feature level")

    global_train, global_val, global_val_cap, global_aux, global_cons, global_times = \
        [], [], [], [], [], []
    start_stage = 0
    carried_best_state = None

    if resume_dir:
        ckpt_path = os.path.join(resume_dir, "checkpoint.pt")
        if os.path.exists(ckpt_path):
            ckpt = load_checkpoint(ckpt_path, DEVICE)
            de.load_state_dict(ckpt["model_state_dict"])
            if aux_on and ckpt.get("aux_head_state_dict") is not None:
                from ..nn.aux_head import AUX_NAMES as _AUX_NAMES
                ck_names = ckpt.get("aux_names")
                ck_k = ckpt.get("aux_n_channels")
                if (ck_names is not None and list(ck_names) == list(_AUX_NAMES)
                        and ck_k == aux_k):
                    aux_head.load_state_dict(ckpt["aux_head_state_dict"])
                else:
                    print(f"  NOTE: checkpoint aux head incompatible "
                          f"(targets {ck_names}, slice {ck_k}) -- fresh aux head.")
            carried_best_state = ckpt.get("best_state", None)
            global_train = list(ckpt.get("train_losses", []))
            global_val = list(ckpt.get("val_losses", []))
            global_val_cap = list(ckpt.get("val_cap_losses", []))
            global_aux = list(ckpt.get("aux_losses", []))
            global_cons = list(ckpt.get("cons_losses", []))
            global_times = list(ckpt.get("train_times", []))
            start_stage = int(ckpt.get("stage_idx", 0))
            print(f"Resumed from {ckpt_path}: stage {start_stage+1}/{len(stages)} "
                  f"({len(global_train)} epochs so far)")
            if carried_best_state is not None:
                de.load_state_dict(carried_best_state)
        else:
            print(f"  Warning: no checkpoint at {ckpt_path}, starting fresh")

    print(f"\nTraining on {DEVICE} | {len(dataset)} waveforms | "
          f"batch={cfg['batch_size']} | {len(cfg['inferred_parameters'])} params | "
          f"{len(stages)} stage(s) | aux={'ON' if aux_on else 'off'}")

    for stage_idx in range(start_stage, len(stages)):
        if carried_best_state is not None:
            de.load_state_dict(carried_best_state)
        best_state, best_val = train_one_stage(
            de, cache, aux_head, dataset, x_val, theta_val, std, cfg,
            stages[stage_idx], stage_idx, OUT, DEVICE,
            global_train, global_val, global_val_cap, global_aux, global_cons, global_times,
            len(stages), aux_k=aux_k)
        carried_best_state = best_state if best_state is not None else carried_best_state
        if std_path is not None:
            with open(std_path, "wb") as f:
                pickle.dump(std, f)
        print(f"=== stage {stage_idx+1} done: best val={best_val:.4f} ===")

    if carried_best_state is not None:
        de.load_state_dict(carried_best_state)
    de = de.to(DEVICE)

    print("Recalibrating BatchNorm running statistics...")
    for m in de.modules():
        if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
            m.reset_running_stats()
            m.momentum = None
    dataset.set_dL_cap(None)
    de.train()
    _bn_loader = torch.utils.data.DataLoader(
        dataset, batch_size=512, shuffle=True,
        num_workers=0, drop_last=True)
    _bn_seen = 0
    with torch.no_grad():
        for _bn_batch in _bn_loader:
            _xb = std.normalise_x(_bn_batch[0]).to(DEVICE)
            _tb = std.normalise_theta(_bn_batch[1]).to(DEVICE)
            de.log_prob(_tb, condition=_xb)
            _bn_seen += len(_xb)
            if _bn_seen >= 20000:
                break
    de.eval()
    carried_best_state = {k: v.cpu().clone() for k, v in de.state_dict().items()}
    print(f"  BN recalibrated over {_bn_seen} samples")

    if "ra" in cfg["inferred_parameters"] and "dec" in cfg["inferred_parameters"]:
        ra_idx = cfg["inferred_parameters"].index("ra")
        dec_idx = cfg["inferred_parameters"].index("dec")
        prior_low = prior_low.clone()
        prior_high = prior_high.clone()
        prior_low[ra_idx] = -MAX_DT_HL
        prior_high[ra_idx] = MAX_DT_HL
        prior_low[dec_idx] = 0.0
        prior_high[dec_idx] = 2 * np.pi

    lo, hi = std.get_normalised_prior_bounds(prior_low, prior_high)
    prior = BoxUniform(low=lo.to(DEVICE), high=hi.to(DEVICE))
    posterior = DirectPosterior(posterior_estimator=de, prior=prior)
    print(f"\nDone: {len(global_train)} total epochs across {len(stages)} stage(s)")
    return posterior, de, dict(training_loss=global_train, validation_loss=global_val,
                               validation_loss_capped=global_val_cap,
                               aux_loss=global_aux,
                               cons_loss=global_cons)

# TODO: Figure out why this actualy DEGRADES performance when applied post-training
def recalibrate_bn(posterior, std, cfg, g, psd_bank, tukey_window, td_norm,
                   n_cal=5000, device="cpu"):
    """Reset the embedding's BatchNorm running statistics from synthetic
    training-like (signal + noise) inputs, so eval-time BN matches the
    distribution the flow was trained on. BN lives in the embedding net, not
    the flow transforms."""
    from precompute_banks_b import make_prior_dict

    de = posterior.posterior_estimator
    embed = de.embedding_net
    for m in embed.modules():
        if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
            m.reset_running_stats()
            m.momentum = None
    embed.train().to(device)

    priors = make_prior_dict(cfg)
    n_psds = psd_bank["psd_H1"].shape[0]
    wfg = bilby.gw.WaveformGenerator(
        duration=float(cfg["duration"]),
        sampling_frequency=int(cfg["sampling_frequency"]),
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
        waveform_arguments=dict(
            waveform_approximant=cfg["waveform_approximant"],
            reference_frequency=float(cfg["f_min"])))

    ifos = bilby.gw.detector.InterferometerList(["H1", "L1"])
    ref_tc = float(cfg["ref_geocent_time"])
    start_time = ref_tc - g["duration"] / 2
    for ifo in ifos:
        ifo.minimum_frequency = float(cfg["f_min"])
        ifo.set_strain_data_from_frequency_domain_strain(
            np.zeros(g["n_fd_full"], dtype=complex),
            sampling_frequency=g["sr"], duration=g["duration"],
            start_time=start_time)

    n_ok = 0
    with torch.no_grad():
        batch_x = []
        for _ in range(n_cal * 2):
            if n_ok >= n_cal:
                break
            pidx = np.random.randint(n_psds)
            det_psds = {d: psd_bank[f"psd_{d}"][pidx].astype(np.float64)
                        for d in ["H1", "L1"]}
            params = priors.sample()
            params["geocent_time"] = ref_tc
            try:
                pols = wfg.frequency_domain_strain({k: float(v) for k, v in params.items()})
                if pols is None:
                    continue
            except Exception:
                continue
            signal_fd = {}
            for ifo in ifos:
                ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(
                    frequency_array=g["freq_array"], psd_array=det_psds[ifo.name])
                signal_fd[ifo.name] = ifo.get_detector_response(
                    pols, {k: float(v) for k, v in params.items()})
            try:
                x_strain, _, _, _ = injection_to_x(
                    signal_fd, det_psds, cfg, g, tukey_window, td_norm,
                    noise_kind=cfg.get("noise_source", "gaussian_physical"))
                x_full = build_x_full(x_strain, det_psds, std, cfg, g)
                batch_x.append(torch.tensor(x_full).unsqueeze(0))
                n_ok += 1
            except Exception:
                continue
            if len(batch_x) >= 64:
                embed(std.normalise_x(torch.cat(batch_x)).to(device))
                batch_x = []
        if batch_x:
            embed(std.normalise_x(torch.cat(batch_x)).to(device))

    embed.eval().to("cpu")
    print(f"  BN recalibrated on {n_ok} signal+noise inputs")