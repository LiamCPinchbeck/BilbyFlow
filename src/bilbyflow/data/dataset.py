"""
bilbyflow.data.dataset — on-the-fly extrinsic generation (v4.8).

OnTheFlyGWDataset takes the intrinsic-only waveform bank (stored at D_REF)
and, per __getitem__, draws a sky/PSD/noise realisation to synthesise the
whitened detector strain x, the flow target theta, the aux-supervision
targets, and (optionally) a JEPA embedding-consistency stack.

Whitening, windowing, PSD-context and amp-context all route through
bilbyflow.data.canonical so training and the reweighting path share one
definition. Aux targets route through bilbyflow.nn.aux_head. The sky
reparam (ra->dt_HL, dec->phi_det) is applied when building theta, matching
coordinates.sky.samples_detector_to_radec on the inference side.
"""

from typing import NamedTuple

import numpy as np
import torch
import torch.utils.data
from tqdm import tqdm

from .canonical import (
    window_fd, whitened_fd_to_channels, compute_amp_context,
    canonical_bn, canonical_psd_context, canonical_valid_mask, whiten_fd,
    AMP_NAMES, N_AMP,
)
from .noise import (
    noise_gaussian_physical, noise_real_segment, noise_gaussian_whitened,
)
from .banks import D_REF
from .dataset_cache import CleanCacheMixin
from .dataset_cons import ConsStackMixin
from ..nn.aux_head import AUX_NAMES, N_AUX, compute_aux_summaries

__all__ = ["OnTheFlyGWDataset", "generate_fixed_dataset"]


class SkyDraw(NamedTuple):
    idx: int
    dL: float
    tc_offset: float
    dt_HL: float
    phi_det: float
    psi: float
    sid: float
    tc: float
    start_time: float

#TODO: Not make a behemoth
class OnTheFlyGWDataset(CleanCacheMixin, ConsStackMixin,
                        torch.utils.data.Dataset):

    def __init__(self, h_plus_all, h_cross_all, intrinsic_params, ref_data,
                 cfg, sky_bank, psd_bank=None, noise_bank=None):
        self.h_plus = h_plus_all
        self.h_cross = h_cross_all
        self.intrinsic_params = intrinsic_params
        self.cfg = cfg
        self._init_grid(cfg, ref_data)
        self._init_targets(cfg)
        self._init_sky(sky_bank)
        self._init_noise(cfg, psd_bank, noise_bank)
        self._init_psd_conditioning(cfg)
        self._init_amp_and_aux(cfg)
        self._init_cache_and_cons(cfg, noise_bank)

    # ── init blocks ─────────────────────────────────────────────────────────

    def _init_grid(self, cfg, ref_data):
        from scipy.signal.windows import tukey
        self.freq_array = ref_data["freq_array"]
        self.freq_mask = ref_data["freq_mask"]
        self.asd = ref_data["asd"]
        self.df = ref_data["df"]
        self.n_freq = len(self.freq_array)
        self._n_masked = int(np.sum(self.freq_mask))
        self._freq_window_factor = self._n_masked / len(self.freq_mask)
        self.td_norm = np.sqrt(self._n_masked) / self._freq_window_factor
        self.n_td = 2 * (self.n_freq - 1)
        self.sr = int(cfg["sampling_frequency"])
        roll_off = float(cfg.get("tukey_roll_off", 0.2))
        self.tukey_window = tukey(self.n_td,
                                  alpha=2.0 * roll_off / float(cfg["duration"]))
        self.window_factor = float(np.mean(self.tukey_window ** 2))
        self.fft_c64 = bool(cfg.get("fft_complex64", False))
        self._cdtype = np.complex64 if self.fft_c64 else np.complex128
        self._win = (self.tukey_window.astype(np.float32) if self.fft_c64
                     else self.tukey_window)
        self.bilby_norm = self.asd * np.sqrt(4.0 * self.df)
        # grid dict for canonical_psd_context
        self._g = dict(freq_mask=self.freq_mask, n_fd_full=self.n_freq)

    def _init_targets(self, cfg):
        self.ref_tc = float(cfg["ref_geocent_time"])
        self.inferred = cfg["inferred_parameters"]
        self.dL_log = str(cfg.get("dL_param", "linear")).lower() == "log"
        self.Mc_log = str(cfg.get("Mc_param", "linear")).lower() == "log"

    def _init_sky(self, sky_bank):
        self.sky_bank = sky_bank
        self.n_sky = len(sky_bank["dL"])
        self.dL_all = np.asarray(sky_bank["dL"], dtype=np.float64)
        self._dL_full = float(np.max(self.dL_all))
        self._cap_cache = {}
        self.current_dL_max = self._dL_full
        self._eligible = np.arange(self.n_sky)

    def _init_noise(self, cfg, psd_bank, noise_bank):
        if psd_bank is not None:
            self.psd_bank_H1 = psd_bank["psd_H1"]
            self.psd_bank_L1 = psd_bank["psd_L1"]
            self.n_psd_bank = self.psd_bank_H1.shape[0]
            print(f"  PSD bank loaded: {self.n_psd_bank} PSDs per detector")
        else:
            self.psd_bank_H1 = None
            self.psd_bank_L1 = None
            self.n_psd_bank = 0

        self.noise_source = str(cfg.get("noise_source",
                                        "gaussian_whitened")).lower()
        if self.noise_source not in ("gaussian_whitened", "gaussian_physical",
                                     "real"):
            raise ValueError("noise_source must be gaussian_whitened|"
                             f"gaussian_physical|real, got {self.noise_source!r}")
        self.noise_bank = noise_bank
        self.real_noise_fraction = float(cfg.get("real_noise_fraction", 1.0))
        if not 0.0 <= self.real_noise_fraction <= 1.0:
            raise ValueError("real_noise_fraction must be in [0,1], "
                             f"got {self.real_noise_fraction}")
        self.real_noise_mix_source = str(
            cfg.get("real_noise_mix_source", "gaussian_whitened")).lower()
        if self.real_noise_mix_source not in ("gaussian_whitened",
                                              "gaussian_physical"):
            raise ValueError("real_noise_mix_source must be gaussian_whitened|"
                             f"gaussian_physical, got "
                             f"{self.real_noise_mix_source!r}")

        self.embed_consistency = bool(cfg.get("embed_consistency", False))
        self.noise_eras = None
        self._noise_idx = None

        if self.noise_source == "real" or self.embed_consistency:
            if noise_bank is None:
                raise ValueError("noise_source: real / embed_consistency "
                                 "requires the noise segment bank")
            if int(noise_bank["n_td"]) != self.n_td:
                raise ValueError(f"noise bank segment length "
                                 f"{noise_bank['n_td']} != analysis n_td "
                                 f"{self.n_td}")
            eH = np.asarray(noise_bank["era_H1"])
            eL = np.asarray(noise_bank["era_L1"])
            self.noise_eras = sorted(set(eH) & set(eL))
            self._noise_idx = {
                e: (np.flatnonzero(eH == e), np.flatnonzero(eL == e))
                for e in self.noise_eras}
            # v4.8: per-file segment pools for matched-PSD cons stacks
            self._seg_by_row = {
                det: {int(r): np.flatnonzero(
                        np.asarray(noise_bank[f"psd_row_{det}"]) == r)
                      for r in np.unique(noise_bank[f"psd_row_{det}"])}
                for det in ["H1", "L1"]}
            if self.noise_source == "real":
                # psd_row_* indices are only valid against the bank's own PSDs
                self.psd_bank_H1 = np.asarray(noise_bank["psds_H1"])
                self.psd_bank_L1 = np.asarray(noise_bank["psds_L1"])
                self.n_psd_bank = self.psd_bank_H1.shape[0]
                print(f"  noise_source=real "
                      f"(fraction={self.real_noise_fraction:.2f}, "
                      f"mix={self.real_noise_mix_source}): {len(eH)} H1 / "
                      f"{len(eL)} L1 segments, eras {self.noise_eras}, "
                      f"{self.n_psd_bank} file PSDs")
            else:
                print(f"  noise_source={self.noise_source} "
                      f"(+ consistency bank: {len(eH)} H1 / {len(eL)} L1 "
                      f"segments, eras {self.noise_eras})")
        else:
            print(f"  noise_source={self.noise_source}")

    def _init_psd_conditioning(self, cfg):
        self.psd_conditioning = bool(cfg.get("psd_conditioning", False))
        self.strain_dim = 2 * (2 * self._n_masked + self.n_td)
        self.psd_dim = 2 * self._n_masked if self.psd_conditioning else 0
        self.psd_clip = float(cfg.get("psd_context_clip", 10.0))
        self.psd_log_mean = None
        self.psd_log_std = None
        if not self.psd_conditioning:
            return
        if self.psd_bank_H1 is None:
            raise ValueError("psd_conditioning=True requires a PSD bank")
        fm = self.freq_mask
        with np.errstate(divide="ignore", invalid="ignore"):
            lH = 0.5 * np.log10(np.where(
                fm[None, :] & np.isfinite(self.psd_bank_H1)
                & (self.psd_bank_H1 > 0), self.psd_bank_H1, np.nan))
            lL = 0.5 * np.log10(np.where(
                fm[None, :] & np.isfinite(self.psd_bank_L1)
                & (self.psd_bank_L1 > 0), self.psd_bank_L1, np.nan))
        pooled = np.concatenate([lH, lL], axis=0)
        mu = np.nanmean(pooled, axis=0)
        sd = np.nanstd(pooled, axis=0)
        self.psd_log_mean = np.where(np.isfinite(mu), mu, 0.0).astype(np.float64)
        self.psd_log_std = np.where(np.isfinite(sd) & (sd > 1e-8), sd,
                                    1.0).astype(np.float64)
        print(f"  PSD conditioning ON: context dim={self.psd_dim}")

    def _init_amp_and_aux(self, cfg):
        self.amp_context = bool(cfg.get("amp_context", False))
        self.amp_dim = N_AMP if self.amp_context else 0
        if self.amp_context:
            if not self.psd_conditioning:
                raise ValueError("amp_context=True requires psd_conditioning "
                                 "(context assembled by "
                                 "PSDConditionedEmbedding)")
            print(f"  amp context ON: {N_AMP} observable summaries "
                  f"({', '.join(AMP_NAMES)}) appended to x")
        self.aux_supervision = bool(cfg.get("aux_supervision", False))
        if self.aux_supervision:
            print(f"  aux supervision ON: {N_AUX} targets "
                  f"({', '.join(AUX_NAMES)})")

    def _init_cache_and_cons(self, cfg, noise_bank):
        # v4.3: JEPA embedding consistency (real->synthetic anchor)
        self.k_synth = int(cfg.get("consistency_k_synth", 1))  # 0 = anchor
        self.k_real = int(cfg.get("consistency_k_real", 1))
        self.cons_frac = float(cfg.get("consistency_frac", 1.0))
        if self.embed_consistency and noise_bank is None:
            raise ValueError("embed_consistency=True requires the noise "
                             "segment bank (load-only, noise_source can stay "
                             "gaussian_physical)")
        # v4.6: clean-signal cache
        self.cache_slots = int(cfg.get("clean_cache_slots", 0))
        self.cache_refresh_epochs = max(
            1, int(cfg.get("cache_refresh_epochs", 10)))
        self._cache_ready = False
        if self.cache_slots > 0:
            if str(cfg.get("noise_source", "")).lower() != "gaussian_physical":
                raise ValueError("clean_cache_slots requires noise_source: "
                                 "gaussian_physical (real-noise MAIN draws "
                                 "are not cached, cons real rows are "
                                 "unaffected)")
            gb = (self.cache_slots * (16 if self.fft_c64 else 32)
                  * self.n_freq / 2 ** 30)
            print(f"  clean-signal cache ON: {self.cache_slots} slots, "
                  f"refresh every {self.cache_refresh_epochs} epochs "
                  f"(~{gb:.1f} GB)")

    # ── curriculum dL cap ───────────────────────────────────────────────────

    def set_dL_cap(self, dL_max):
        if dL_max is None or float(dL_max) >= self._dL_full:
            dL_max = self._dL_full
        dL_max = float(dL_max)
        self.current_dL_max = dL_max
        if dL_max not in self._cap_cache:
            elig = np.flatnonzero(self.dL_all <= dL_max)
            if len(elig) == 0:
                raise ValueError(f"dL cap {dL_max} excludes all sky-bank "
                                 f"entries")
            self._cap_cache[dL_max] = elig
        self._eligible = self._cap_cache[dL_max]

    def eligible_count(self):
        return len(self._eligible)

    # ── shared draw helpers ─────────────────────────────────────────────────

    def _bn(self, psd):
        """Whitening denominator via canonical_bn, cast for the FFT dtype."""
        bn = canonical_bn(psd, self.df)
        return bn.astype(np.float32) if self.fft_c64 else bn

    def _psd_ctx(self, sel_psd):
        """PSD context via canonical_psd_context (self carries the std
        attributes it reads: psd_conditioning, psd_log_mean, psd_log_std)."""
        if not self.psd_conditioning or sel_psd is None:
            return None
        return canonical_psd_context(sel_psd, self, self.cfg, self._g)

    def _draw_sky(self):
        i = int(self._eligible[np.random.randint(len(self._eligible))])
        sb = self.sky_bank
        dL = float(sb["dL"][i])
        tc_offset = float(sb["tc_offset"][i])
        sid = float(sb["sidereal_offset"][i])
        tc_gps = self.ref_tc + sid
        return SkyDraw(i, dL, tc_offset, float(sb["dt_HL"][i]),
                       float(sb["phi_det"][i]), float(sb["psi"][i]), sid,
                       tc_gps + tc_offset,
                       tc_gps - float(self.cfg["duration"]) / 2)

    def _scaled_waveforms(self, wf_idx, dL):
        hp = self.h_plus[wf_idx].astype(self._cdtype)
        hc = self.h_cross[wf_idx].astype(self._cdtype)
        scale = D_REF / dL
        return hp * scale, hc * scale

    def _project_and_whiten(self, hp_s, hc_s, sky, det, bn):
        """Project (h+, hx) onto one detector, window, whiten. Noiseless."""
        sb, i = self.sky_bank, sky.idx
        phase = np.exp(-2j * np.pi * self.freq_array *
                       (sky.tc - sky.start_time + sb[f"dt_{det}"][i])
                       ).astype(self._cdtype)
        signal_fd = (sb[f"fp_{det}"][i] * hp_s
                     + sb[f"fc_{det}"][i] * hc_s) * phase
        windowed = window_fd(signal_fd, self.freq_mask, self._win, self.n_td)
        return whiten_fd(windowed, bn)

    def _draw_noise_context(self):
        """(sel_psd, bn_by_det, seg, kind) for one uncached draw. seg is None
        unless this draw uses a real segment."""
        if self.noise_source == "real":
            use_real = np.random.random() < self.real_noise_fraction
            kind = "real" if use_real else self.real_noise_mix_source
            era = self.noise_eras[np.random.randint(len(self.noise_eras))]
            idxH, idxL = self._noise_idx[era]
            iH = int(idxH[np.random.randint(len(idxH))])
            iL = int(idxL[np.random.randint(len(idxL))])
            sel_psd = {
                "H1": self.psd_bank_H1[int(self.noise_bank["psd_row_H1"][iH])],
                "L1": self.psd_bank_L1[int(self.noise_bank["psd_row_L1"][iL])]}
            seg = ({"H1": self.noise_bank["segments_H1"][iH],
                    "L1": self.noise_bank["segments_L1"][iL]}
                   if use_real else None)
        elif self.psd_bank_H1 is not None:
            kind = self.noise_source
            psd_idx = np.random.randint(self.n_psd_bank)
            sel_psd = {"H1": self.psd_bank_H1[psd_idx],
                       "L1": self.psd_bank_L1[psd_idx]}
            seg = None
        else:
            kind = self.noise_source
            sel_psd = None
            seg = None
        if sel_psd is not None:
            bn_by_det = {d: self._bn(sel_psd[d]) for d in ("H1", "L1")}
        else:
            bn = (self.bilby_norm.astype(np.float32) if self.fft_c64
                  else self.bilby_norm)
            bn_by_det = {"H1": bn, "L1": bn}
        return sel_psd, bn_by_det, seg, kind

    def _append_context(self, x_strain, psd_ctx):
        """x = [strain || psd_ctx || amp]; amp is noise-realisation-dependent
        so it is computed from the final x_strain."""
        tail = []
        if psd_ctx is not None:
            tail.append(psd_ctx)
        if self.amp_context:
            tail.append(compute_amp_context(x_strain, self._n_masked,
                                            self.n_td))
        return (np.concatenate([x_strain] + tail).astype(np.float32)
                if tail else x_strain)

    def _theta_from_draw(self, intr, dL, dt_HL, phi_det, tc_offset, psi):
        """Flow target vector in the sky-reparam training coordinates."""
        theta_vals = []
        for p in self.inferred:
            if p == "luminosity_distance":
                theta_vals.append(np.log(dL) if self.dL_log else dL)
            elif p == "chirp_mass":
                theta_vals.append(np.log(intr[p]) if self.Mc_log else intr[p])
            elif p == "ra":
                theta_vals.append(dt_HL)
            elif p == "dec":
                theta_vals.append(phi_det)
            elif p == "geocent_time":
                theta_vals.append(tc_offset)
            elif p == "psi":
                theta_vals.append(psi)
            else:
                theta_vals.append(intr[p])
        return theta_vals

    def _aux(self, sig_w_by_det, dt_HL):
        if self.aux_supervision:
            return compute_aux_summaries(sig_w_by_det, self.freq_array,
                                         self.freq_mask, self.n_td, self.sr,
                                         dt_HL)
        return np.zeros(N_AUX, dtype=np.float32)

    # ── noise packing (shared by uncached, cached, and cons paths) ──────────

    def _pack(self, sig_w_by_det, bn_by_det, kind, seg=None):
        """Same signal + one noise realisation of `kind` -> flat strain
        vector, ordering [Re, Im, TD] x [H1, L1]. The realisations route
        through data.noise (the single definition of each noise model)."""
        parts = []
        for det in ["H1", "L1"]:
            bn = bn_by_det[det]
            valid = canonical_valid_mask(bn)
            if kind == "real":
                noise_w, _ = noise_real_segment(
                    seg[det], bn, valid, self.freq_mask, self._win,
                    self.n_td, self.sr,
                    td_dtype=np.float32 if self.fft_c64 else np.float64)
            elif kind == "gaussian_physical":
                noise_w, _ = noise_gaussian_physical(
                    bn, valid, self.freq_mask, self._win, self.n_td,
                    self.n_freq, cdtype=self._cdtype)
            else:  # gaussian_whitened
                noise_w = noise_gaussian_whitened(valid, self.n_freq)
            re_fd, im_fd, w_td, _ = whitened_fd_to_channels(
                sig_w_by_det[det] + noise_w, self.freq_mask, self.n_td,
                self.td_norm)
            parts.extend([re_fd, im_fd, w_td])
        return np.concatenate(parts)

    # ── length / item ───────────────────────────────────────────────────────

    def __len__(self):
        # v4.8: decouple epoch length from waveform-bank size; with the
        # clean-signal cache, one epoch should be ~one pass over the cache
        # (epoch_size ~= clean_cache_slots), not the full 1M bank.
        n = int(self.cfg.get("epoch_size", 0))
        return min(n, len(self.h_plus)) if n > 0 else len(self.h_plus)

    def __getitem__(self, idx):
        if self._cache_ready:                    # v4.6 cached path
            return self._getitem_cached(idx)

        sky = self._draw_sky()
        hp_s, hc_s = self._scaled_waveforms(idx, sky.dL)
        sel_psd, bn_by_det, seg, kind = self._draw_noise_context()

        sig_w_by_det = {d: self._project_and_whiten(hp_s, hc_s, sky, d,
                                                    bn_by_det[d])
                        for d in ("H1", "L1")}

        x_strain = self._pack(sig_w_by_det, bn_by_det, kind,
                              seg=seg).astype(np.float32)
        x = self._append_context(x_strain, self._psd_ctx(sel_psd))

        cons = self._maybe_cons_stack(sig_w_by_det, bn_by_det)

        intr = self.intrinsic_params[idx]
        theta_vals = self._theta_from_draw(intr, sky.dL, sky.dt_HL,
                                           sky.phi_det, sky.tc_offset,
                                           sky.psi)
        aux = self._aux(sig_w_by_det, sky.dt_HL)

        return (
            torch.from_numpy(x),
            torch.from_numpy(np.array(theta_vals, dtype=np.float32)),
            torch.tensor(sky.sid, dtype=torch.float32),
            torch.from_numpy(aux),
            cons,
        )


def generate_fixed_dataset(dataset, n, return_aux=False):
    """Materialise n (x, theta, sid[, aux]) draws into stacked tensors — used
    for the fixed val/test sets and for standardiser fitting."""
    xs, thetas, sidereal_offsets, auxs = [], [], [], []
    indices = np.random.choice(len(dataset), size=n, replace=n > len(dataset))
    for idx in tqdm(indices, desc="Generating fixed dataset"):
        x, theta, sid, aux, _ = dataset[int(idx)]
        xs.append(x)
        thetas.append(theta)
        sidereal_offsets.append(sid)
        auxs.append(aux)
    out = (torch.stack(xs), torch.stack(thetas), torch.stack(sidereal_offsets))
    return out + (torch.stack(auxs),) if return_aux else out