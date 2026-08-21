"""
bilbyflow.inference.synthetic_phase — exact higher-mode marginalisation of
(phi_c, psi, t_c, d_L).

Handles the full multi-harmonic phi_c dependence of precessing HM waveforms
(XPHM, ell<=4) by constructing harmonic coefficients via a DFT over n_basis
phase-sampled waveforms, then gridding over (phi_c, psi, t_c) with local
refinement. dL is marginalised against its prior via a precomputed table.

Stage-2 evaluator for inference.two_stage. It also provides
marginal_loglr_pair(), which returns BOTH the full-HM marginal and a
(2,2)-collapse approximation from ONE waveform basis (the m=2 harmonic
coefficient is the sum over all harmonics), so the two-stage denominator
costs no extra waveform calls.

Conventions (bilby-matched):
  * inner product   4 Re sum_k df d*_k h_k / S_k
  * FFT time shift  e^{-2 pi i f dt}
  * whitened noise per-quadrature variance handled via the ifo PSD arrays

References:
  Thrane & Talbot 2019 (arXiv:1809.02293) Sec. 5
    --> Ref for this file is mostly this and the GW-likelihood relationships they 
        outline. 

  Dax+ 2023 (arXiv:2210.05686) 
    --> wanted to do the actual synthetic marginalization like in this code
        but it ended up being too expensive in the short term (time-wise)
    --> has a really cool way of generating semi-exact samples out of some of
        the marginals. 
    --> Currently we would have the discretized grid used for each of the parameters
        and could do ancestral sampling on them for those discrete values.
"""

import numpy as np
from scipy.special import logsumexp

__all__ = ["SyntheticExtrinsicLikelihood", "SyntheticPhaseLikelihood"]


class SyntheticExtrinsicLikelihood:

    def __init__(self, ifos, waveform_generator, priors,
                 marg_phase=True, marg_psi=True, marg_time=True, marg_dist=False,
                 tc0=None, n_basis=5, n_phi=64, n_psi=32, refine_fac=8,
                 dt_res=2.5e-5, d_ref=None, n_s=512,
                 delta_gate=30.0, delta_refine=6.0, max_refine_cells=128,
                 window_fn=None, positive_harmonics=True, t_window=None,
                 **_ignored):
        self.ifos = list(ifos)
        self.wfg = waveform_generator
        self.mphase = bool(marg_phase)
        self.mpsi = bool(marg_psi)
        self.mtime = bool(marg_time)
        self.mdist = bool(marg_dist)
        self.window_fn = window_fn
        self.n_basis = int(n_basis) if self.mphase else 1
        self.refine_fac = int(refine_fac)
        self.delta_gate = float(delta_gate)
        self.delta_refine = float(delta_refine)
        self.max_refine_cells = int(max_refine_cells)
        self.last_refine_delta = 0.0
        self.n_waveform_calls = 0

        # ── per-detector data ──
        fa = ifos[0].strain_data.frequency_array
        self.df = float(fa[1] - fa[0])
        self._det = []
        for ifo in ifos:
            m = np.asarray(ifo.frequency_mask, dtype=bool)
            psd = np.asarray(ifo.power_spectral_density_array, dtype=np.float64)[m]
            w = 4.0 * self.df / psd
            w[~np.isfinite(w)] = 0.0
            self._det.append(dict(
                mask=m, w=w,
                d=np.asarray(ifo.strain_data.frequency_domain_strain)[m],
                kbin=np.flatnonzero(m),
            ))
        self._n_full = len(fa)

        # ── harmonic labels ──
        self.phis_basis = 2.0 * np.pi * np.arange(self.n_basis) / self.n_basis
        if bool(positive_harmonics):
            lab = []
            for b in range(self.n_basis):
                ks = [k for k in (1, 2, 3, 4) if k % self.n_basis == b]
                if len(ks) > 1:
                    raise ValueError(f"n_basis={self.n_basis} aliases {ks}")
                lab.append(ks[0] if ks else b)
            self.m_vals = np.asarray(lab, dtype=int)
        else:
            self.m_vals = np.fft.fftfreq(self.n_basis, d=1.0 / self.n_basis).astype(int)

        # ── grids ──
        if self.mphase:
            self.phi_grid = 2.0 * np.pi * np.arange(int(n_phi)) / int(n_phi)
            self.dphi = 2.0 * np.pi / int(n_phi)
        if self.mpsi:
            self.psi_grid = np.pi * np.arange(int(n_psi)) / int(n_psi)
            self.dpsi = np.pi / int(n_psi)

        # ── t_c ──
        if self.mtime:
            if tc0 is None:
                raise ValueError("marg_time requires tc0")
            self.tc0 = float(tc0)
            t_pr = priors["geocent_time"]
            self.t_lo = float(t_pr.minimum) - self.tc0
            self.t_hi = float(t_pr.maximum) - self.tc0
            # optional narrowed evaluation window (from a time-support estimate).
            # Prior normalisation Tw stays the FULL prior width.
            g_lo, g_hi = (t_window if t_window is not None
                          else (self.t_lo, self.t_hi))
            self.Nfft = int(round(1.0 / (float(dt_res) * self.df)))
            dts = np.arange(self.Nfft) / (self.Nfft * self.df)
            dts[dts > 0.5 / self.df] -= 1.0 / self.df
            sel = (dts >= g_lo) & (dts <= g_hi)
            order = np.argsort(dts[sel])
            self.dt_idx = np.flatnonzero(sel)[order]
            self.dt_grid = dts[self.dt_idx]
            self.ddt = 1.0 / (self.Nfft * self.df)
            self.Tw = self.t_hi - self.t_lo
        else:
            self.tc0 = float(tc0) if tc0 is not None else 0.0
            self.dt_grid = np.zeros(1)
            self.ddt = self.Tw = 1.0

        # ── dL prior table ──
        d_pr = priors["luminosity_distance"]
        self.dL_min, self.dL_max = float(d_pr.minimum), float(d_pr.maximum)
        self.d_ref = float(d_ref) if d_ref is not None else self.dL_min
        if self.mdist:
            dl = np.exp(np.linspace(np.log(self.dL_min), np.log(self.dL_max), int(n_s)))
            self._s = self.d_ref / dl
            wgt = np.abs(np.gradient(dl)) * np.asarray(d_pr.prob(dl), dtype=np.float64)
            self._log_ws = np.log(np.maximum(wgt / wgt.sum(), 1e-300))
            self._build_dist_table()




    # ------------------ distance table ------------------------
    # only used if luminosity distance is treated as a nuisance parameter and not in
        # the flow
    def _build_dist_table(self):
        s = self._s
        self._r_grid = np.linspace(-3.0 * s.max(), 6.0 * s.max(), 1200)
        self._logH_grid = np.linspace(np.log(1e-4), np.log(4e6), 500)
        tab = np.empty((self._r_grid.size, self._logH_grid.size))
        rs = np.outer(self._r_grid, s)
        s2 = 0.5 * s ** 2
        for j, Hj in enumerate(np.exp(self._logH_grid)):
            ex = Hj * (rs - s2)
            mx = ex.max(axis=1, keepdims=True)
            tab[:, j] = mx[:, 0] + np.log(np.exp(ex - mx) @ np.exp(self._log_ws))
        self._dist_tab = tab




    def _dist_marg(self, D, H):
        H = np.maximum(H, 1e-4)
        r = np.clip(D / H, self._r_grid[0], self._r_grid[-1])
        lH = np.clip(np.log(H), self._logH_grid[0], self._logH_grid[-1])
        dr = self._r_grid[1] - self._r_grid[0]
        dH = self._logH_grid[1] - self._logH_grid[0]
        ir = np.clip(((r - self._r_grid[0]) / dr).astype(int), 0, self._r_grid.size - 2)
        iH = np.clip(((lH - self._logH_grid[0]) / dH).astype(int), 0, self._logH_grid.size - 2)
        fr = (r - self._r_grid[ir]) / dr
        fH = (lH - self._logH_grid[iH]) / dH
        t = self._dist_tab
        return ((1 - fr) * (1 - fH) * t[ir, iH] + fr * (1 - fH) * t[ir + 1, iH]
                + (1 - fr) * fH * t[ir, iH + 1] + fr * fH * t[ir + 1, iH + 1])



    def _point_loglr(self, D, H, s_fix=None):
        if self.mdist:
            return self._dist_marg(D, H)
        return s_fix * D - 0.5 * s_fix ** 2 * H




    # ── harmonic basis (n_basis waveform calls) ──────────────────────────────

    def _basis_fft(self, params):
        """n_basis calls at equally spaced phi -> DFT -> C_m(f) per (det, psi-basis)."""
        p = dict(params, luminosity_distance=self.d_ref)
        if self.mtime:
            p["geocent_time"] = self.tc0
        psis = (0.0, 0.25 * np.pi) if self.mpsi else (float(params["psi"]),)
        ndet = len(self._det)
        H = [[np.empty((self.n_basis, len(self._det[i]["d"])), dtype=complex)
              for i in range(ndet)] for _ in psis]
        for k in range(self.n_basis):
            p["phase"] = float(self.phis_basis[k]) if self.mphase else float(params["phase"])
            pols = self.wfg.frequency_domain_strain(p)
            self.n_waveform_calls += 1
            if self.window_fn is not None:
                pols = {key: self.window_fn(v) for key, v in pols.items()}
            for x, ps in enumerate(psis):
                p2 = dict(p, psi=float(ps))
                for i, ifo in enumerate(self.ifos):
                    H[x][i][k] = ifo.get_detector_response(pols, p2)[self._det[i]["mask"]]
        return [[np.fft.fft(Hxi, axis=0) / self.n_basis for Hxi in Hx] for Hx in H]




    # ── cache: Z_m(dt) and Gram G_{mm'} ──────────────────────────────────────

    def _cache(self, C):
        nX = len(C)
        ndet = len(self._det)
        Z = [[None] * ndet for _ in range(nX)]
        G = {}
        for i, det in enumerate(self._det):
            wd = det["w"] * np.conj(det["d"])
            for x in range(nX):
                a = C[x][i] * wd
                if self.mtime:
                    full = np.zeros((self.n_basis, self.Nfft), dtype=complex)
                    full[:, det["kbin"]] = a
                    Z[x][i] = np.fft.fft(full, axis=1)[:, self.dt_idx]
                else:
                    Z[x][i] = a.sum(axis=1)[:, None]
                for y in range(nX):
                    G[(x, y, i)] = (np.conj(C[x][i]) * det["w"]) @ C[y][i].T
        return Z, G




    # -------------------- D(phi,dt) and H(phi) from the cache -------------------------

    def _DH(self, Z, G, phis, dt_cols=None):
        A = np.exp(1j * np.outer(self.m_vals, phis))
        nX = len(Z)
        ndt = len(self.dt_grid) if dt_cols is None else len(dt_cols)
        R = [np.zeros((len(phis), ndt)) for _ in range(nX)]
        hh = np.zeros((nX, nX, len(phis)))
        for i in range(len(self._det)):
            for x in range(nX):
                Zi = Z[x][i] if dt_cols is None else Z[x][i][:, dt_cols]
                R[x] += np.real(A.T @ Zi)
                for y in range(x, nX):
                    hh[x, y] += np.real(np.sum(np.conj(A) * (G[(x, y, i)] @ A), axis=0))
        return R, hh




    @staticmethod
    def _H_of(hh, c, s):
        if hh.shape[0] == 2:
            return np.maximum(c[:, None] ** 2 * hh[0, 0][None, :] + s[:, None] ** 2 * hh[1, 1][None, :]
                              + 2 * c[:, None] * s[:, None] * hh[0, 1][None, :], 1e-30)
        return np.maximum(np.broadcast_to(hh[0, 0][None, :], (len(c), hh.shape[2])).copy(), 1e-30)



    ################################################################################################
    ################################################################################################
    # -------------------------------------- gridded marginal --------------------------------------
    ################################################################################################
    ################################################################################################

    def _marg_from_C(self, C, params):
        """Marginal logLR from pre-computed harmonic coefficients C."""
        Z, G = self._cache(C)
        s_fix = None if self.mdist else self.d_ref / float(params["luminosity_distance"])
        c, s = ((np.cos(2 * self.psi_grid), np.sin(2 * self.psi_grid)) if self.mpsi
                else (np.array([1.0]), np.array([0.0])))
        phis = self.phi_grid if self.mphase else np.array([float(params["phase"])])
        psi_vals = self.psi_grid if self.mpsi else np.array([float(params["psi"])])
        log_marg, _, _, self.last_refine_delta = self._grid_pass(
            Z, G, phis, psi_vals, c, s, s_fix)
        return log_marg



    def marginal_loglr(self, params, return_extrinsic=False, rng=None):
        C = self._basis_fft(params)
        log_marg = self._marg_from_C(C, params)
        if not return_extrinsic:
            return log_marg
        return log_marg, np.nan, np.nan, np.nan, np.nan



    def marginal_loglr_pair(self, params):
        """(logL_HM, logL_22approx) from ONE waveform basis.

        The (2,2)-approx denominator collapses all harmonics into the m=2 bin:
        C_22[m=2] = sum_m C[m], all others zero. This is exactly what the (2,2)
        analytic phase marginal assumes (full strain at phi=0, treated as a
        single e^{2 i phi} harmonic), evaluated on the same grids so numerics
        cancel in the ratio. Cost: n_basis waveform calls total.
        """
        C = self._basis_fft(params)
        log_hm = self._marg_from_C(C, params)              # full multi-harmonic

        j2 = int(np.flatnonzero(self.m_vals == 2)[0])
        C22 = [[np.zeros_like(Cxi) for Cxi in Cx] for Cx in C]
        for x in range(len(C)):
            for i in range(len(C[x])):
                C22[x][i][j2] = C[x][i].sum(axis=0)
        log_22 = self._marg_from_C(C22, params)

        return log_hm, log_22




    # ------------ grid pass ------------

    def _grid_pass(self, Z, G, phis, psi_vals, c, s, s_fix):
        nphi, ndt = len(phis), len(self.dt_grid)
        R, hh = self._DH(Z, G, phis)
        Hpp = self._H_of(hh, c, s)

        # profiled gate
        Renv2 = sum(r ** 2 for r in R)
        prof = 0.5 * Renv2 / Hpp.min(axis=0)[:, None]
        pmax = float(prof.max())
        gate = prof >= pmax - self.delta_gate

        log_prior = ((-np.log(2 * np.pi) if self.mphase else 0.0)
                     + (-np.log(np.pi) if self.mpsi else 0.0)
                     + (-np.log(self.Tw) if self.mtime else 0.0))
        log_vol = ((np.log(self.phi_grid[1] - self.phi_grid[0]) if self.mphase and len(self.phi_grid) > 1 else 0.0)
                   + (np.log(self.psi_grid[1] - self.psi_grid[0]) if self.mpsi and len(self.psi_grid) > 1 else 0.0)
                   + (np.log(self.ddt) if self.mtime else 0.0))

        cell_lse = np.full((len(psi_vals), nphi), -np.inf)
        dt_keep_max = np.full(ndt, -np.inf)
        all_dt = np.arange(ndt)
        for ip in range(nphi):
            cols = all_dt[gate[ip]]
            if cols.size == 0:
                continue
            RA = R[0][ip, cols]
            RB = R[1][ip, cols] if len(R) == 2 else 0.0
            D = c[:, None] * RA[None, :] + s[:, None] * RB
            L = self._point_loglr(D, Hpp[:, ip][:, None], s_fix)
            cell_lse[:, ip] = logsumexp(L, axis=1)
            np.maximum.at(dt_keep_max, cols, L.max(axis=0))

        if not np.isfinite(cell_lse).any():
            return -np.inf, None, None, 0.0
        Lmax = float(np.max(cell_lse))

        # refinement (batched: ONE _DH call for all cells)
        refined, refine_delta = np.array([]), 0.0
        if self.refine_fac > 1 and (self.mphase or self.mpsi):
            cells = np.argwhere(cell_lse >= Lmax - self.delta_refine)
            if len(cells) > self.max_refine_cells:
                order = np.argsort(cell_lse[cells[:, 0], cells[:, 1]])[::-1]
                cells = cells[order[:self.max_refine_cells]]
            dt_cols = all_dt[dt_keep_max >= dt_keep_max.max() - self.delta_gate]
            coarse_repl = logsumexp(cell_lse[cells[:, 0], cells[:, 1]])
            refined = self._refine_cells_batched(Z, G, phis, psi_vals, c, s,
                                                 s_fix, cells, dt_cols)
            cell_lse[cells[:, 0], cells[:, 1]] = -np.inf
            refine_delta = float(logsumexp(refined) - coarse_repl)

        parts = [cell_lse[np.isfinite(cell_lse)]]
        if refined.size:
            parts.append(refined)
        log_marg = float(logsumexp(np.concatenate(parts))) + log_vol + log_prior
        return log_marg, None, None, refine_delta




    def _refine_cells_batched(self, Z, G, phis, psi_vals, c, s, s_fix,
                              cells, dt_cols):
        """All refine cells in ONE _DH call. cells[:,0]=iq (psi), cells[:,1]=ip
        (phi). Same values as the looped _refine_cell."""
        f = self.refine_fac
        dphi = phis[1] - phis[0] if len(phis) > 1 else 2 * np.pi
        dpsi = psi_vals[1] - psi_vals[0] if len(psi_vals) > 1 else np.pi
        n_cells = len(cells)

        if self.mphase:
            offs = dphi * (np.arange(f) + 0.5) / f - dphi / 2
            phi_all = (phis[cells[:, 1]][:, None] + offs[None, :]).ravel()
            f_phi = f
        else:
            phi_all = phis[cells[:, 1]]
            f_phi = 1

        Rf, hhf = self._DH(Z, G, phi_all, dt_cols=dt_cols)
        ndt = Rf[0].shape[1]
        RA = Rf[0].reshape(n_cells, f_phi, ndt)
        RB = Rf[1].reshape(n_cells, f_phi, ndt) if len(Rf) == 2 else None
        hh00 = hhf[0, 0].reshape(n_cells, f_phi)
        if hhf.shape[0] == 2:
            hh11 = hhf[1, 1].reshape(n_cells, f_phi)
            hh01 = hhf[0, 1].reshape(n_cells, f_phi)

        if self.mpsi:
            poffs = dpsi * (np.arange(f) + 0.5) / f - dpsi / 2
            psi_all = psi_vals[cells[:, 0]][:, None] + poffs[None, :]
            C, S = np.cos(2 * psi_all), np.sin(2 * psi_all)
            f_psi = f
        else:
            C = c[cells[:, 0]][:, None].copy()
            S = s[cells[:, 0]][:, None].copy()
            f_psi = 1

        n_sub = f_phi * f_psi
        out = np.empty(n_cells)
        chunk = max(1, int(5e7 // max(f_psi * f_phi * ndt, 1)))
        for lo in range(0, n_cells, chunk):
            hi = min(lo + chunk, n_cells)
            Ck, Sk = C[lo:hi], S[lo:hi]
            RAk = RA[lo:hi]
            if RB is not None:
                D = (Ck[:, :, None, None] * RAk[:, None, :, :]
                     + Sk[:, :, None, None] * RB[lo:hi][:, None, :, :])
                H = np.maximum(
                    Ck[:, :, None] ** 2 * hh00[lo:hi][:, None, :]
                    + Sk[:, :, None] ** 2 * hh11[lo:hi][:, None, :]
                    + 2 * Ck[:, :, None] * Sk[:, :, None] * hh01[lo:hi][:, None, :],
                    1e-30)
            else:
                D = Ck[:, :, None, None] * RAk[:, None, :, :]
                H = np.maximum(np.broadcast_to(
                    hh00[lo:hi][:, None, :], (hi - lo, f_psi, f_phi)), 1e-30).copy()
            L = self._point_loglr(D, H[..., None], s_fix)
            out[lo:hi] = logsumexp(L.reshape(hi - lo, -1), axis=1)
        return out - np.log(n_sub)





    def _refine_cell(self, Z, G, phis, psi_vals, c, s, s_fix, ip, iq, dt_cols):
        f = self.refine_fac
        dphi = phis[1] - phis[0] if len(phis) > 1 else 2 * np.pi
        dpsi = psi_vals[1] - psi_vals[0] if len(psi_vals) > 1 else np.pi
        phi_f = float(phis[ip]) - dphi / 2 + dphi * (np.arange(f) + 0.5) / f if self.mphase else phis[ip:ip + 1]
        if self.mpsi:
            psi_f = float(psi_vals[iq]) - dpsi / 2 + dpsi * (np.arange(f) + 0.5) / f
            c_f, s_f = np.cos(2 * psi_f), np.sin(2 * psi_f)
        else:
            psi_f, c_f, s_f = psi_vals[iq:iq + 1], c[iq:iq + 1], s[iq:iq + 1]
        Rf, hhf = self._DH(Z, G, phi_f, dt_cols=dt_cols)
        Hf = self._H_of(hhf, c_f, s_f)
        lse = -np.inf
        for jp in range(len(phi_f)):
            RA = Rf[0][jp]
            RB = Rf[1][jp] if len(Rf) == 2 else 0.0
            D = c_f[:, None] * RA[None, :] + s_f[:, None] * RB
            lse = np.logaddexp(lse, float(logsumexp(self._point_loglr(D, Hf[:, jp][:, None], s_fix))))
        n_sub = (f if self.mphase else 1) * (f if self.mpsi else 1)
        return lse - np.log(n_sub)


# To be in line with very early versions of the scripts that are currently used for reference every now and then
    # if you're looking at the first commit, Hello!, but also yes there was already backwards-compatible code
    # in the first version...
SyntheticPhaseLikelihood = SyntheticExtrinsicLikelihood