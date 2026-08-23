"""
bilbyflow.data.canonical — single source of truth for strain -> x conversion.

Every path that builds an x vector for the flow (training __getitem__,
real-event reweighting, injection reweighting, era tests) MUST produce
identical x for identical (signal_fd, noise_fd, PSD). This module holds the
canonical primitives; import them instead of reimplementing whitening /
windowing / PSD estimation / channel-packing.

Training conventions (authoritative, from OnTheFlyGWDataset.__getitem__):
  - Signal:  bilby FD -> window_fd (mask sub-f_min, then irfft->tukey->rfft)
             -> divide by bn -> noiseless whitened signal
  - Noise (gaussian_physical): white FD -> * bn -> window_fd -> / bn
  - Noise (real): rfft(segment * tukey) / sr -> / bn
  - Combine:  whitened_fd = sig_w + noise_w
  - Channels: whitened_fd_to_channels -> (Re_masked, Im_masked, whitened_TD)
  - x = concat([H1_re, H1_im, H1_td, L1_re, L1_im, L1_td])
  - bn = sqrt(psd) * sqrt(4 df), invalid bins -> 1.0
  - td_norm = sqrt(n_masked) / freq_window_factor,
              freq_window_factor = n_masked / len(mask)
  - PSD context: 0.5 log10(psd) z-scored per-bin, clipped at +-psd_clip
  - Tukey alpha = 2 roll_off / duration
  - PSD estimation: Welch, nperseg = duration*sr, 50% overlap, the SAME
    Tukey window, linear detrend, median average, interpolated onto the
    analysis grid with inf below f_min / at non-finite or non-positive bins

Noise realisations and the signal+noise combiner live in data.noise; this
module is deterministic (no RNG). canonical_grid is an alias of
io.config.grid_quantities so the grid has one definition.
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import butter, sosfiltfilt, welch
from scipy.signal.windows import tukey as _tukey

from ..io.config import grid_quantities as canonical_grid

__all__ = [
    "window_fd", "whitened_fd_to_channels", "whiten_fd", "var_q_of",
    "AMP_NAMES", "N_AMP", "compute_amp_context",
    "canonical_grid", "canonical_tukey", "canonical_welch_window",
    "canonical_td_norm", "canonical_bn", "canonical_valid_mask",
    "welch_psd", "welch_psd_floored",
    "signal_to_whitened", "build_x_strain", "build_x_full",
    "canonical_psd_context",
    "real_strain_to_x",
    "real_strain_to_x_training_order", "real_strain_to_x_production_order",
]


# ── core whitening primitives ────────────────────────────────────────────────

def window_fd(strain_phys, freq_mask, tukey_window, n_td):
    """Zero out-of-band bins, apply the TD Tukey window, return the windowed
    physical FD strain (full grid). THE training windowing convention."""
    s = np.asarray(strain_phys).copy()
    s[~freq_mask] = 0.0
    return np.fft.rfft(np.fft.irfft(s, n=n_td) * tukey_window)


def whiten_fd(fd, bn):
    """fd / bn at valid bins, zero elsewhere. THE whitening application;
    every divide-by-bn in the codebase routes through here."""
    valid = canonical_valid_mask(bn)
    out = np.zeros_like(fd)
    out[valid] = fd[valid] / bn[valid]
    return out


def whitened_fd_to_channels(whitened_fd, freq_mask, n_td, td_norm):
    """Whitened FD strain (full grid) ->
    (Re(FD_masked), Im(FD_masked), whitened_TD) as float32, + cleaned FD."""
    wf = np.nan_to_num(whitened_fd * freq_mask, nan=0.0, posinf=0.0,
                       neginf=0.0)
    wtd = np.nan_to_num(np.fft.irfft(wf, n=n_td) * td_norm,
                        nan=0.0, posinf=0.0, neginf=0.0)
    fdm = wf[freq_mask]
    return (np.real(fdm).astype(np.float32), np.imag(fdm).astype(np.float32),
            wtd.astype(np.float32), wf)


def var_q_of(whitened_fd, freq_mask):
    """Per-quadrature in-band variance of a whitened FD strain; ~1 when the
    whitening matches the noise (the var_q diagnostic)."""
    return 0.5 * float(np.mean(np.abs(whitened_fd[freq_mask]) ** 2))


# ── grid helpers (canonical_grid re-exported from io.config) ─────────────────

def _tukey_alpha(cfg):
    return 2.0 * float(cfg.get("tukey_roll_off", 0.2)) / float(cfg["duration"])


def canonical_tukey(cfg, n_td):
    """Tukey window matching training exactly (alpha = 2 roll_off/duration)."""
    return _tukey(n_td, alpha=_tukey_alpha(cfg))


def canonical_welch_window(cfg):
    """scipy.signal.welch `window` argument matching the analysis Tukey."""
    return ("tukey", _tukey_alpha(cfg))


def canonical_td_norm(g):
    """td_norm as computed in training. Does NOT depend on window_factor."""
    return np.sqrt(g["n_masked"]) / (g["n_masked"] / len(g["freq_mask"]))


def canonical_bn(psd, df):
    """Whitening denominator bn = sqrt(psd) sqrt(4 df); bad bins -> 1.0."""
    with np.errstate(invalid="ignore"):
        bn = np.sqrt(np.asarray(psd, dtype=np.float64)) * np.sqrt(4.0 * df)
    bn[~np.isfinite(bn) | (bn <= 0)] = 1.0
    return bn


def canonical_valid_mask(bn):
    """Which FD bins have finite positive whitening factor."""
    return np.isfinite(bn) & (bn > 0)


# ── canonical Welch PSD estimation ───────────────────────────────────────────

def welch_psd(strain, cfg, g, exclude=None):
    """Median-Welch PSD of a strain array on the analysis grid — THE PSD
    estimation convention (segment-matched Tukey, 50% overlap, linear
    detrend, median average; inf below f_min and at bad bins).

    exclude=(s0, s1): excise those samples (off-source estimation) and use
    the longer remaining piece, provided it spans >= 4 Welch segments;
    otherwise fall back to the full strain with a warning.
    """
    sr, f_min, freq_array = g["sr"], g["f_min"], g["freq_array"]
    nperseg = int(g["duration"] * sr)

    est = np.asarray(strain, dtype=np.float64)
    if exclude is not None:
        s0, s1 = exclude
        piece = max(est[:s0], est[s1:], key=len)
        if len(piece) >= 4 * nperseg:
            est = piece
        else:
            print("  ** WARNING: off-source piece too short for Welch "
                  f"({len(piece)} < {4 * nperseg}), PSD estimated on-source.")

    freqs_w, psd_w = welch(est, fs=sr, nperseg=nperseg,
                           noverlap=nperseg // 2,
                           window=canonical_welch_window(cfg),
                           detrend="linear", average="median")

    psd = interp1d(freqs_w, psd_w, bounds_error=False,
                   fill_value=np.inf)(freq_array)
    psd[(freq_array < f_min) | ~np.isfinite(psd) | (psd <= 0)] = np.inf

    finite = psd[np.isfinite(psd)]
    if finite.size and finite.min() < 1e-48:
        print(f"  ** suspect PSD: min in-band {finite.min():.2e} "
              f"(check noise file, will whiten-blow-up)")
    return psd


def welch_psd_floored(strain, cfg, g, floor_factor=5.0):
    """Welch PSD on the analysis grid WITHOUT the f_min inf-mask, with an
    in-band ASD floor at median/floor_factor instead — the convention shared
    by the noise-segment bank and the real-event PSD (their PSDs also feed
    the PSD context, so sub-f_min bins keep finite values; training only
    whitens at masked bins).
    """
    sr, f_min, freq_array = g["sr"], g["f_min"], g["freq_array"]
    nperseg = int(g["duration"] * sr)
    freqs_w, psd_w = welch(np.asarray(strain, dtype=np.float64), fs=sr,
                           nperseg=nperseg, noverlap=nperseg // 2,
                           window=canonical_welch_window(cfg),
                           detrend="linear", average="median")
    psd = interp1d(freqs_w, psd_w, bounds_error=False,
                   fill_value=np.inf)(freq_array).astype(np.float64)
    psd[~np.isfinite(psd) | (psd <= 0)] = np.inf
    asd = np.sqrt(psd)
    finite = np.isfinite(asd) & (asd > 0)
    in_band = freq_array >= f_min
    med = np.nanmedian(np.where(finite & in_band, asd, np.nan))
    floor = med / floor_factor
    asd = np.where(finite & (asd < floor) & in_band, floor, asd)
    return asd ** 2


# ── v4.4 observable amplitude context ────────────────────────────────────────

AMP_NAMES = ["ln_pow_H1", "ln_pow_L1", "ln_pow_ratio",
             "ln_peak_H1", "ln_peak_L1"]
N_AMP = len(AMP_NAMES)


def compute_amp_context(x_strain, n_masked, n_td):
    """5 observable amplitude summaries from the flat strain channel vector
    [Re,Im,TD]x{H1,L1} (RAW, pre-normalise_x). ln total whitened FD power and
    ln peak |whitened TD| per detector, plus the ln power ratio."""
    x = np.asarray(x_strain)
    per_det = 2 * n_masked + n_td
    vals = []
    for o in (0, per_det):
        fd = x[o:o + 2 * n_masked].astype(np.float64)          # [Re || Im]
        td = x[o + 2 * n_masked:o + per_det].astype(np.float64)
        vals.append((np.log(max(float(np.sum(fd ** 2)), 1e-30)),
                     np.log(max(float(np.max(np.abs(td))) if td.size
                                else 0.0, 1e-30))))
    (pH, kH), (pL, kL) = vals
    return np.array([pH, pL, pH - pL, kH, kL], dtype=np.float32)


# ── x-vector construction ────────────────────────────────────────────────────

def signal_to_whitened(signal_fd, bn, freq_mask, tukey_window, n_td):
    """Signal FD -> mask-first windowed -> whitened. Training-matched."""
    windowed = window_fd(signal_fd, freq_mask, tukey_window, n_td)
    return whiten_fd(windowed, bn), windowed


def build_x_strain(sig_w_by_det, noise_w_by_det, freq_mask, n_td, td_norm):
    """Combine whitened signal + noise -> channel-packed x_strain vector,
    ordering [Re, Im, TD] x [H1, L1]."""
    parts = []
    for det in ["H1", "L1"]:
        re_fd, im_fd, w_td, _ = whitened_fd_to_channels(
            sig_w_by_det[det] + noise_w_by_det[det], freq_mask, n_td, td_norm)
        parts.extend([re_fd, im_fd, w_td])
    return np.concatenate(parts).astype(np.float32)


def canonical_psd_context(det_psds, std, cfg, g):
    """PSD context block matching training exactly (per-bin z-scored
    0.5 log10 psd, clipped)."""
    if not (getattr(std, "psd_conditioning", False)
            and getattr(std, "psd_log_mean", None) is not None):
        return None
    freq_mask = g["freq_mask"]
    mean, sd = np.asarray(std.psd_log_mean), np.asarray(std.psd_log_std)
    clip = float(cfg.get("psd_context_clip", 10.0))
    parts = []
    for det in ["H1", "L1"]:
        psd_d = np.asarray(det_psds[det], dtype=np.float64)
        z = np.zeros(g["n_fd_full"], dtype=np.float64)
        m = freq_mask & np.isfinite(psd_d) & (psd_d > 0)
        z[m] = (0.5 * np.log10(psd_d[m]) - mean[m]) / sd[m]
        parts.append(np.clip(z, -clip, clip)[freq_mask].astype(np.float32))
    return np.concatenate(parts)


def build_x_full(x_strain, det_psds, std, cfg, g):
    """x_strain + PSD context + amp context -> full x vector for the flow."""
    parts = [x_strain]
    psd_ctx = canonical_psd_context(det_psds, std, cfg, g)
    if psd_ctx is not None:
        parts.append(psd_ctx)
    if int(getattr(std, "amp_dim", 0) or 0) > 0:
        parts.append(compute_amp_context(x_strain, g["n_masked"], g["n_td"]))
    return np.concatenate(parts).astype(np.float32)


# ── real on-source strain -> x ───────────────────────────────────────────────

def _highpass(strain, sr, fc=15.0, order=8):
    sos = butter(order, fc, btype="highpass", fs=sr, output="sos")
    return sosfiltfilt(sos, strain)


def real_strain_to_x(on_source_td, det_psds, cfg, g, tukey_window, td_norm,
                     sr, highpass_fc=15.0, order="training"):
    """Real on-source strain -> x.

    order='training'  : mask-first, rfft(strain)/sr -> window_fd,
                        eliminating the window-first leakage (D1 fix).
    order='production': window-first, rfft(strain * tukey)/sr (legacy,
                        kept for comparison / backward compatibility).

    Returns (x_strain, whitened_fd_by_det, total_fd_by_det, var_q)."""
    if order not in ("training", "production"):
        raise ValueError(f"order={order!r}")
    freq_mask, n_td, df = g["freq_mask"], g["n_td"], g["df"]
    x_parts, var_q = [], {}
    whitened_fd_by_det, total_fd_by_det = {}, {}
    for det in ["H1", "L1"]:
        strain = _highpass(np.asarray(on_source_td[det], dtype=np.float64),
                           sr, highpass_fc)
        if order == "training":
            windowed_fd = window_fd(np.fft.rfft(strain) / sr,
                                    freq_mask, tukey_window, n_td)
        else:
            windowed_fd = np.fft.rfft(strain * tukey_window) / sr
        whitened = whiten_fd(windowed_fd, canonical_bn(det_psds[det], df))
        re_fd, im_fd, w_td, _ = whitened_fd_to_channels(
            whitened, freq_mask, n_td, td_norm)
        x_parts.extend([re_fd, im_fd, w_td])
        whitened_fd_by_det[det] = whitened
        total_fd_by_det[det] = windowed_fd
        var_q[det] = var_q_of(whitened, freq_mask)
    return (np.concatenate(x_parts).astype(np.float32),
            whitened_fd_by_det, total_fd_by_det, var_q)


def real_strain_to_x_training_order(*args, **kwargs):
    """Back-compat wrapper: order='training', 4-tuple return."""
    return real_strain_to_x(*args, **kwargs, order="training")


def real_strain_to_x_production_order(*args, **kwargs):
    """Back-compat wrapper: order='production', legacy 3-tuple return
    (x_strain, total_fd_by_det, var_q)."""
    x, _wfd, tfd, vq = real_strain_to_x(*args, **kwargs, order="production")
    return x, tfd, vq


