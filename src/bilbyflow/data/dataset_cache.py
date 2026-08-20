"""
bilbyflow.data.dataset_cache — v4.6 clean-signal cache (mixin).

The expensive noise-INDEPENDENT half of a draw (project + window + whiten,
theta, aux, psd_ctx) is precomputed into slots and refreshed every
cache_refresh_epochs; noise stays fresh EVERY epoch, so the anti-overfit
property is preserved. refresh_clean_cache() must be called from the MAIN
process and the DataLoader rebuilt afterwards (persistent workers fork the
cache).

Requires the host class to provide (all defined on OnTheFlyGWDataset):
  _draw_sky, _scaled_waveforms, _project_and_whiten, _bn, _psd_ctx,
  _theta_from_draw, _aux, _pack, _append_context, _maybe_cons_stack,
  plus the attributes cache_slots, n_freq, psd_dim, inferred, h_plus,
  n_psd_bank, psd_bank_H1/L1, psd_conditioning, _cdtype, current_dL_max.
"""

import os
import numpy as np
import torch

from ..nn.aux_head import N_AUX

__all__ = ["CleanCacheMixin"]

# module-level so the fork-based refresh pool can reach the dataset
_CACHE_REF = {}


def _cache_refresh_chunk(args):
    lo, hi, seed = args
    ds = _CACHE_REF["ds"]
    np.random.seed(seed)
    rows = [ds._draw_clean() for _ in range(lo, hi)]
    return lo, rows


class CleanCacheMixin:

    def _draw_clean(self):
        """One cache slot: draw (waveform, sky, PSD) and build everything
        noise-INDEPENDENT: sig_w per det, psd_ctx, theta, aux, sid."""

    
        wf_idx = np.random.randint(len(self.h_plus))
        intr = self.intrinsic_params[wf_idx]
        sky = self._draw_sky()


        hp_s, hc_s = self._scaled_waveforms(wf_idx, sky.dL)

        psd_idx = np.random.randint(self.n_psd_bank)


        sel_psd = {"H1": self.psd_bank_H1[psd_idx],
                   "L1": self.psd_bank_L1[psd_idx]}

        
        bn_by_det = {d: self._bn(sel_psd[d]) for d in ("H1", "L1")}


        sig_w_by_det = {
            d: self._project_and_whiten(hp_s, hc_s, sky, d,
                                        bn_by_det[d]).astype(self._cdtype)
            for d in ("H1", "L1")}


        psd_ctx = self._psd_ctx(sel_psd)
        theta_vals = self._theta_from_draw(intr, sky.dL, sky.dt_HL,
                                    sky.phi_det, sky.tc_offset,
                                    sky.psi)
        aux = self._aux(sig_w_by_det, sky.dt_HL)

        return (sig_w_by_det["H1"], sig_w_by_det["L1"], np.int32(psd_idx),
                psd_ctx, np.asarray(theta_vals, dtype=np.float32), aux,
                np.float32(sky.sid))



    def refresh_clean_cache(self, n_workers=None):
        """(Re)fill all slots. Call from the MAIN process, rebuild the
        DataLoader afterwards so persistent workers fork the new cache."""
        import multiprocessing as _mp
        import time as _time
        n = self.cache_slots

        if n <= 0:
            return

        if not self._cache_ready:
            self._c_sigH = np.empty((n, self.n_freq), dtype=self._cdtype)
            self._c_sigL = np.empty((n, self.n_freq), dtype=self._cdtype)
            self._c_psd = np.empty(n, dtype=np.int32)
            self._c_ctx = (np.empty((n, self.psd_dim), dtype=np.float32)
                           if self.psd_conditioning else None)
            self._c_theta = np.empty((n, len(self.inferred)),
                                     dtype=np.float32)
            self._c_aux = np.empty((n, N_AUX), dtype=np.float32)
            self._c_sid = np.empty(n, dtype=np.float32)

        if n_workers is None:
            n_workers = max(1, len(os.sched_getaffinity(0)) - 1)
        t0 = _time.time()
        chunk = 512
        base = np.random.randint(0, 2 ** 31 - 1)
        jobs = [(lo, min(lo + chunk, n), base + j)
                for j, lo in enumerate(range(0, n, chunk))]


        def _store(lo, rows):
            for k, r in enumerate(rows):
                s = lo + k
                self._c_sigH[s] = r[0]
                self._c_sigL[s] = r[1]
                self._c_psd[s] = r[2]
                if self._c_ctx is not None:
                    self._c_ctx[s] = r[3]
                self._c_theta[s] = r[4]
                self._c_aux[s] = r[5]
                self._c_sid[s] = r[6]


        if n_workers == 1:
            for job in jobs:
                np.random.seed(job[2])
                _store(job[0],
                       [self._draw_clean() for _ in range(job[0], job[1])])
        else:
            _CACHE_REF["ds"] = self
            ctx = _mp.get_context("fork")
            with ctx.Pool(processes=n_workers) as pool:
                for lo, rows in pool.imap_unordered(_cache_refresh_chunk,
                                                    jobs):
                    _store(lo, rows)
            _CACHE_REF.clear()

        self._cache_ready = True
        print(f"  [cache] refreshed {n} clean signals in "
              f"{_time.time() - t0:.0f}s ({n_workers} workers, "
              f"dL cap {self.current_dL_max:.0f})")



    def _getitem_cached(self, idx):
        """Cached slot + FRESH noise: main x + cons stack + cached labels."""


        s = idx % self.cache_slots
        sig_w_by_det = {"H1": self._c_sigH[s], "L1": self._c_sigL[s]}

        pidx = int(self._c_psd[s])

        sel = {"H1": self.psd_bank_H1[pidx], "L1": self.psd_bank_L1[pidx]}

        bn_by_det = {d: self._bn(sel[d]) for d in ("H1", "L1")}

        x_strain = self._pack(sig_w_by_det, bn_by_det,
                              "gaussian_physical").astype(np.float32)
        
        psd_ctx = self._c_ctx[s] if self._c_ctx is not None else None

        x = self._append_context(x_strain, psd_ctx)

        cons = self._maybe_cons_stack(sig_w_by_det, bn_by_det)

        return (torch.from_numpy(np.ascontiguousarray(x)),
                torch.from_numpy(self._c_theta[s].copy()),
                torch.tensor(float(self._c_sid[s]), dtype=torch.float32),
                torch.from_numpy(self._c_aux[s].copy()),
                cons)