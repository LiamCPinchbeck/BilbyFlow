"""
bilbyflow.data.dataset_cons — VICReg embedding-consistency stack (mixin).

Builds a matched-PSD stack of the SAME signal under different synthetic/real
noise realisations; the embedding is trained to map them to one point
(real -> synthetic anchor). 

Every row of a stack (synthetic AND real)
shares one correctly-matched PSD drawn per stack — same signal + same PSD,
only the noise realisation differs.

Requires the host class to provide (all defined on OnTheFlyGWDataset):
  _pack, _bn, _psd_ctx, plus the attributes embed_consistency, cons_frac,
  k_synth, k_real, noise_bank, noise_eras, _noise_idx, _seg_by_row,
  psd_conditioning, amp_context, _n_masked, n_td.
"""

import numpy as np
import torch

from .canonical import compute_amp_context

__all__ = ["ConsStackMixin"]


class ConsStackMixin:

    def _maybe_cons_stack(self, sig_w_by_det, bn_by_det):
        """torch.zeros(1) sentinel when this item carries no cons rows,
        else the (K, D) stack."""
        if not (self.embed_consistency
                and (self.cons_frac >= 1.0
                     or np.random.random() < self.cons_frac)):
            return torch.zeros(1)
        bn_m, ctx_m, pools = self._matched_cons_draw()
        sig_m = self._rewhiten(sig_w_by_det, bn_by_det, bn_m)
        return self._build_cons_stack(sig_m, bn_m, ctx_m, pools)

    def _matched_cons_draw(self):
        """One PSD-consistent cons stack context. Draw an era and one
        (H1, L1) file pair, return that pair's matched bn / psd_ctx plus the
        per-file segment index pools."""
        era = self.noise_eras[np.random.randint(len(self.noise_eras))]
        idxH, idxL = self._noise_idx[era]
        iH = int(idxH[np.random.randint(len(idxH))])
        iL = int(idxL[np.random.randint(len(idxL))])
        rowH = int(self.noise_bank["psd_row_H1"][iH])
        rowL = int(self.noise_bank["psd_row_L1"][iL])
        psd_m = {"H1": self.noise_bank["psds_H1"][rowH],
                 "L1": self.noise_bank["psds_L1"][rowL]}
        bn_m = {d: self._bn(psd_m[d]) for d in ("H1", "L1")}
        psd_ctx_m = self._psd_ctx(psd_m)
        pools = {"H1": self._seg_by_row["H1"][rowH],
                 "L1": self._seg_by_row["L1"][rowL]}
        return bn_m, psd_ctx_m, pools

    def _build_cons_stack(self, sig_m, bn_m, ctx_m, pools):
        """k_synth synthetic + k_real real rows of the SAME (matched-PSD)
        signal, each with its own noise realisation, packed + context-tailed
        into one (K, D) float32 tensor for the JEPA loss."""
        kinds = (["gaussian_physical"] * self.k_synth
                 + ["real"] * self.k_real)
        stack = []
        for kind in kinds:
            sg = None
            if kind == "real":
                sg = {d: self.noise_bank[f"segments_{d}"]
                         [int(pools[d][np.random.randint(len(pools[d]))])]
                      for d in ["H1", "L1"]}
            xs = self._pack(sig_m, bn_m, kind, seg=sg)
            xt = []
            if self.psd_conditioning:
                xt.append(ctx_m)
            if self.amp_context:
                xt.append(compute_amp_context(xs, self._n_masked, self.n_td))
            if xt:
                xs = np.concatenate([xs] + xt)
            stack.append(xs.astype(np.float32))
        return torch.from_numpy(np.stack(stack))

    def _rewhiten(self, sig_w_by_det, bn_old_by_det, bn_new_by_det):
        """Signal whitened by bn_old -> whitened by bn_new (valid bins
        only)."""
        out = {}
        for d in ["H1", "L1"]:
            bo, bn_ = bn_old_by_det[d], bn_new_by_det[d]
            v = (np.isfinite(bo) & (bo > 0)
                 & np.isfinite(bn_) & (bn_ > 0))
            sw = np.zeros_like(sig_w_by_det[d])
            sw[v] = sig_w_by_det[d][v] * bo[v] / bn_[v]
            out[d] = sw
        return out