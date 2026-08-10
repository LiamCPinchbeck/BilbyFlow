"""
synthetic_phase.py (v6) — Analytic marginalisation of (\varphi_c, \psi, t_c).

Marginal log-likelihood-ratio with:
  \varphi_c : analytic via Bessel I_0 (exact for (2,2)-dominated signals)
  \psi   : dense vectorised grid (π/n_psi spacing)
  t_c : chirp-z transform (CZT) or zero-padded FFT

One waveform call per posterior sample. d_L is fixed (flow-inferred).

References
----------
  Veitch+ 2015 (arXiv:1409.7215)   — analytic phase marginal
  Thrane & Talbot 2019 (1809.02293) — extrinsic structure
  Dax+ 2023 (2210.05686)            — DINGO-IS synthetic likelihood
"""

import numpy as np
from scipy.special import logsumexp, i0e

try:
    from scipy.signal import czt as _czt
    _HAS_CZT = True
except ImportError:
    _HAS_CZT = False


class SyntheticExtrinsicLikelihood:
    """(2,2) analytic marginalisation of extrinsic parameters.

    Parameters
    ----------
    ifos : list[bilby.gw.detector.Interferometer]
    waveform_generator : bilby.gw.WaveformGenerator
    priors : bilby.core.prior.PriorDict
    marg_phase, marg_psi, marg_time : bool
        Which extrinsics to marginalise.
    marg_dist : bool
        Must be False (d_L handled by the flow).
    tc0 : float
        Reference geocent_time for the t_c grid origin.
    dt_res : float
        Time grid resolution in seconds.
    n_psi : int
        Number of polarisation angle grid points (minimum 200).
    window_fn : callable or None
        Applied to polarisations after waveform generation.
    d_ref : float or None
        Reference distance for template scaling (defaults to dL_min).
    """

    def __init__(self, ifos, waveform_generator, priors,
                 marg_phase=True, marg_psi=True, marg_time=True,
                 marg_dist=False, tc0=None, dt_res=2.5e-5, d_ref=None,
                 n_psi=200, window_fn=None, **_ignored):
        self.ifos = list(ifos)
        self.wfg = waveform_generator
        self.mphase = bool(marg_phase)
        self.mpsi = bool(marg_psi)
        self.mtime = bool(marg_time)
        self.window_fn = window_fn

        if bool(marg_dist):
            raise NotImplementedError("d_L handled by the flow (marg_dist=False)")

        # Compatibility attributes (no mode path in v6)
        self.phase_basis = "analytic"
        self.last_refine_delta = 0.0
        self._mode_ok = True
        self._mode_marg_err = 0.0
        self.n_waveform_calls = 0

        # ── Per-detector cached data ────────────────────────────────────
        fa = ifos[0].strain_data.frequency_array
        self.df = float(fa[1] - fa[0])
        self._n_full = len(fa)
        self._det = []
        for ifo in ifos:
            m = np.asarray(ifo.frequency_mask, dtype=bool)
            psd = np.asarray(ifo.power_spectral_density_array, dtype=np.float64)[m]
            w = 4.0 * self.df / psd
            w[~np.isfinite(w)] = 0.0
            d = np.asarray(ifo.strain_data.frequency_domain_strain)[m]
            self._det.append(dict(
                mask=m, w=w, d=d,
                kbin=np.flatnonzero(m),
                wdc=w * np.conj(d),
            ))

        # ── \psi grid ──────────────────────────────────────────────────────
        if self.mpsi:
            self.npsi = max(int(n_psi), 200)
            grid = np.pi * np.arange(self.npsi) / self.npsi
            self.dpsi = np.pi / self.npsi
            self._cp = np.cos(2 * grid)
            self._sp = np.sin(2 * grid)

        # ── t_c grid ────────────────────────────────────────────────────
        self._setup_time_grid(priors, tc0, dt_res)

        # ── Distance reference ──────────────────────────────────────────
        d_pr = priors["luminosity_distance"]
        self.dL_min = float(d_pr.minimum)
        self.dL_max = float(d_pr.maximum)
        self.d_ref = float(d_ref) if d_ref is not None else self.dL_min

    # ── Time grid ───────────────────────────────────────────────────────

    def _setup_time_grid(self, priors, tc0, dt_res):
        if not self.mtime:
            self.tc0 = float(tc0) if tc0 is not None else 0.0
            self.dt_grid = np.zeros(1)
            self.ddt = self.Tw = 1.0
            self._ndt = 1
            return

        if tc0 is None:
            raise ValueError("marg_time requires tc0")
        self.tc0 = float(tc0)

        t_pr = priors["geocent_time"]
        self.t_lo = float(t_pr.minimum) - self.tc0
        self.t_hi = float(t_pr.maximum) - self.tc0
        self.Tw = self.t_hi - self.t_lo

        if _HAS_CZT:
            self.ddt = float(dt_res)
            self._ndt = int(np.floor(self.Tw / self.ddt)) + 1
            self.dt_grid = self.t_lo + self.ddt * np.arange(self._ndt)
            self._czt_a = np.exp(2j * np.pi * self.df * self.t_lo)
            self._czt_w = np.exp(-2j * np.pi * self.df * self.ddt)
        else:
            self.Nfft = int(round(1.0 / (float(dt_res) * self.df)))
            dts = np.arange(self.Nfft) / (self.Nfft * self.df)
            dts[dts > 0.5 / self.df] -= 1.0 / self.df
            sel = (dts >= self.t_lo) & (dts <= self.t_hi)
            order = np.argsort(dts[sel])
            self.dt_idx = np.flatnonzero(sel)[order]
            self.dt_grid = dts[self.dt_idx]
            self.ddt = 1.0 / (self.Nfft * self.df)
            self._ndt = len(self.dt_grid)

    # ── Waveform → detector responses ───────────────────────────────────

    def _templates(self, params):
        """One waveform at \varphi=0, d_ref, tc0 --> h_A(\psi=0), h_B(\psi=\pi/4) per det."""
        p = dict(params, luminosity_distance=self.d_ref, phase=0.0)
        if self.mtime:
            p["geocent_time"] = self.tc0

        pols = self.wfg.frequency_domain_strain(p)
        self.n_waveform_calls += 1
        if self.window_fn is not None:
            pols = {k: self.window_fn(v) for k, v in pols.items()}

        hA, hB = [], []
        for det in self._det:
            ifo = self.ifos[len(hA)]
            pA = dict(p, psi=0.0)
            pB = dict(p, psi=0.25 * np.pi)
            hA.append(ifo.get_detector_response(pols, pA)[det["mask"]])
            hB.append(ifo.get_detector_response(pols, pB)[det["mask"]])
        return hA, hB

    # ── Z(dt) transform ─────────────────────────────────────────────────

    def _Z_dt(self, a, i, czt_a=None):
        """DFT of sparse input --> t_c grid (CZT or zero-padded FFT)."""
        det = self._det[i]
        if _HAS_CZT:
            full = np.zeros(self._n_full, dtype=np.complex64)
            full[det["kbin"]] = a.astype(np.complex64)
            return _czt(full, m=self._ndt, w=self._czt_w,
                        a=(czt_a if czt_a is not None else self._czt_a))
        full = np.zeros(self.Nfft, dtype=np.complex64)
        full[det["kbin"]] = a.astype(np.complex64)
        return np.fft.fft(full)[self.dt_idx]

    # ── Cache: overlaps + Gram scalars ──────────────────────────────────

    def _cache(self, hA, hB, czt_a=None):
        """Z_A(dt), Z_B(dt) and H_AA, H_BB, H_AB summed over detectors."""
        ZA = np.zeros(self._ndt, dtype=complex)
        ZB = np.zeros(self._ndt, dtype=complex)
        HAA = HBB = HAB = 0.0

        for i, det in enumerate(self._det):
            wdc, w = det["wdc"], det["w"]
            if self.mtime:
                ZA += self._Z_dt(wdc * hA[i], i, czt_a=czt_a)
                ZB += self._Z_dt(wdc * hB[i], i, czt_a=czt_a)
            else:
                ZA[0] += np.sum(wdc * hA[i])
                ZB[0] += np.sum(wdc * hB[i])
            HAA += float(np.sum(w * np.abs(hA[i]) ** 2))
            HBB += float(np.sum(w * np.abs(hB[i]) ** 2))
            HAB += float(np.sum(w * np.real(np.conj(hA[i]) * hB[i])))

        return ZA, ZB, HAA, HBB, HAB

    # ── Marginal log-likelihood-ratio ───────────────────────────────────

    def marginal_loglr(self, params, return_extrinsic=False, rng=None,
                       jitter_time=False):
        """Compute the marginal log-likelihood-ratio.

        Parameters
        ----------
        params : dict
            Full BBH parameter dictionary.
        return_extrinsic : bool
            If True, return (log_marg, nan, nan, nan, nan) for interface compat.
        rng : np.random.Generator or None
            For time-grid jittering.
        jitter_time : bool
            Shift the t_c grid by a random sub-bin offset (unbiased estimator).

        Returns
        -------
        float or tuple
        """
        hA, hB = self._templates(params)

        # Optional per-sample grid jitter for coarse dt_res
        czt_a = None
        dt_off = 0.0
        if jitter_time and rng is not None and self.mtime and _HAS_CZT:
            dt_off = float(rng.uniform(0.0, self.ddt))
            czt_a = self._czt_a * np.exp(2j * np.pi * self.df * dt_off)

        ZA, ZB, HAA, HBB, HAB = self._cache(hA, hB, czt_a=czt_a)
        s = self.d_ref / float(params["luminosity_distance"])

        # \psi grid or fixed value
        if self.mpsi:
            cp, sp = self._cp, self._sp
        else:
            psi = float(params["psi"])
            cp = np.array([np.cos(2 * psi)])
            sp = np.array([np.sin(2 * psi)])

        # Z_net(\psi, t_c) and H(\psi)
        Znet = cp[:, None] * ZA[None, :] + sp[:, None] * ZB[None, :]
        Hpsi = np.maximum(cp**2 * HAA + sp**2 * HBB + 2 * cp * sp * HAB, 0.0)

        # \varphi_c: analytic I_0 marginal
        if self.mphase:
            x = s * np.abs(Znet)
            lnL = np.log(i0e(x)) + x - 0.5 * s**2 * Hpsi[:, None]
        else:
            phi = float(params["phase"])
            lnL = (s * np.real(np.exp(2j * phi) * Znet)
                   - 0.5 * s**2 * Hpsi[:, None])

        # Sum over (\psi, t_c) + prior normalisation
        lc = 0.0
        if self.mpsi:
            lc += np.log(self.dpsi) - np.log(np.pi)
        if self.mtime:
            lc += np.log(self.ddt) - np.log(self.Tw)

        log_marg = float(logsumexp(lnL) + lc)

        if not return_extrinsic:
            return log_marg
        return log_marg, np.nan, np.nan, np.nan, np.nan

    # ── Stage-2 helper: narrowed t_c window ─────────────────────────────

    def time_support(self, params, delta=15.0, pad=2e-3):
        """Return (t_lo, t_hi) enclosing regions where lnL > max - delta."""
        if not self.mtime:
            raise RuntimeError("time_support requires marg_time=True")

        hA, hB = self._templates(params)
        ZA, ZB, HAA, HBB, HAB = self._cache(hA, hB)
        s = self.d_ref / float(params["luminosity_distance"])

        if self.mpsi:
            cp, sp = self._cp, self._sp
        else:
            psi = float(params["psi"])
            cp = np.array([np.cos(2 * psi)])
            sp = np.array([np.sin(2 * psi)])

        Znet = cp[:, None] * ZA[None, :] + sp[:, None] * ZB[None, :]
        Hpsi = np.maximum(cp**2 * HAA + sp**2 * HBB + 2 * cp * sp * HAB, 0.0)
        x = s * np.abs(Znet)
        lnL_t = (np.log(i0e(x)) + x - 0.5 * s**2 * Hpsi[:, None]).max(axis=0)

        keep = self.dt_grid[lnL_t >= lnL_t.max() - delta]
        return (float(max(keep.min() - pad, self.t_lo)),
                float(min(keep.max() + pad, self.t_hi)))

    # ── Compatibility stubs ─────────────────────────────────────────────

    def _calibrate_modes(self, params):
        self._mode_ok = True
        self._mode_marg_err = 0.0

    # ── Self-check ──────────────────────────────────────────────────────

    def self_check(self, params, bilby_likelihood, n_test=4, seed=0):
        """Validate against bilby's direct logLR.

        Returns
        -------
        err0 : float
            |point_logLR - bilby_logLR| at \varphi=0, dt=0 (pure convention check,
            should be ~1e-6).
        errT : float
            Max |point - bilby| at random \varphi + dt=t_hi (carries the (2,2) +
            rigid-shift approximation error).
        """
        rng = np.random.default_rng(seed)
        hA, hB = self._templates(params)
        psi0 = float(params["psi"]) if not self.mpsi else 0.3
        dL0 = float(params["luminosity_distance"])

        def _point(phi, psi, dt, dL):
            sc = self.d_ref / dL
            c, sn = np.cos(2 * psi), np.sin(2 * psi)
            out = 0.0
            for i, det in enumerate(self._det):
                h = (c * hA[i] + sn * hB[i]) * np.exp(2j * phi)
                h = h * np.exp(-2j * np.pi * det["kbin"] * self.df * dt)
                D = float(np.sum(det["w"] * np.real(np.conj(det["d"]) * h)))
                H = float(np.sum(det["w"] * np.abs(h) ** 2))
                out += sc * D - 0.5 * sc**2 * H
            return out

        def _bilby_at(phi, psi, dt, dL):
            p = dict(params, phase=phi, psi=psi, luminosity_distance=dL)
            if self.mtime:
                p["geocent_time"] = self.tc0 + dt
            bilby_likelihood.parameters.update(p)
            return float(bilby_likelihood.log_likelihood_ratio())

        err0 = abs(_point(0.0, psi0, 0.0, dL0) - _bilby_at(0.0, psi0, 0.0, dL0))

        errsT = []
        for _ in range(n_test):
            phi = rng.uniform(0, 2 * np.pi)
            psi = rng.uniform(0, np.pi) if self.mpsi else psi0
            dt = self.t_hi if self.mtime else 0.0
            errsT.append(abs(_point(phi, psi, dt, dL0)
                             - _bilby_at(phi, psi, dt, dL0)))

        return float(err0), float(np.max(errsT))


# Backward compatibility alias
SyntheticPhaseLikelihood = SyntheticExtrinsicLikelihood