"""
bilbyflow.diagnostics.consistency — data-treatment consistency audits.

Element-wise diffing of two x vectors, the training-vs-production windowing
audit, and the human-readable convention checklist. Every path that builds x
should round-trip to the same vector; these tools locate the first channel and
bin where two paths diverge (the classic culprit is D1, window-first vs
mask-first).

I've had so many issues with this that I just made a whole script for it. :shrug:
"""

import numpy as np

from ..data.canonical import (
    canonical_grid, canonical_tukey, canonical_td_norm,
    real_strain_to_x_training_order, real_strain_to_x_production_order,
)

__all__ = [
    "diff_x", "audit_real_event_windowing", "print_consistency_checklist",
]


def diff_x(x_a, x_b, g, label_a="A", label_b="B"):
    """Element-wise comparison of two x_strain vectors. Prints a per-channel
    summary and returns (max_abs_diff, ok)."""

    x_a, x_b = np.asarray(x_a), np.asarray(x_b)

    assert x_a.shape == x_b.shape, f"Shape mismatch: {x_a.shape} vs {x_b.shape}"

    n_masked, n_td = g["n_masked"], g["n_td"]
    per_det = 2 * n_masked + n_td


    channels = []
    for det in ["H1", "L1"]:
        o = 0 if det == "H1" else per_det
        channels.append((f"{det}_Re", x_a[o:o + n_masked], x_b[o:o + n_masked]))
        o += n_masked
        channels.append((f"{det}_Im", x_a[o:o + n_masked], x_b[o:o + n_masked]))
        o += n_masked
        channels.append((f"{det}_TD", x_a[o:o + n_td], x_b[o:o + n_td]))

    print(f"\n{'channel':<10} {'max|diff|':>12} {'max|rel|':>12} "
          f"{'rms(diff)':>12} {'first!=0 idx':>12}")
    print("-" * 62)


    max_abs_all = 0.0
    for name, a, b in channels:
        d = np.abs(a - b)
        scale = np.maximum(np.abs(a), np.abs(b))
        rel = np.where(scale > 1e-30, d / scale, 0.0)
        ma, mr = float(d.max()), float(rel.max())
        rms = float(np.sqrt(np.mean(d ** 2)))
        first = int(np.argmax(d > 1e-10)) if d.max() > 1e-10 else -1
        print(f"{name:<10} {ma:>12.2e} {mr:>12.2e} {rms:>12.2e} {first:>12d}")
        max_abs_all = max(max_abs_all, ma)


    ok = max_abs_all < 1e-6

    print(f"\nmax|diff| across all channels: {max_abs_all:.2e}  "
          f"{'MATCH' if ok else 'MISMATCH'}  ({label_a} vs {label_b})")
    return max_abs_all, ok


def audit_real_event_windowing(on_source_td, det_psds, cfg):
    """Compare training-order vs production-order x for one real event.
    Returns (x_training, x_production, max_diff)."""

    g = canonical_grid(cfg)
    tukey = canonical_tukey(cfg, g["n_td"])
    td_norm = canonical_td_norm(g)

    sr = g["sr"]
    x_train, _, _, vq_train = real_strain_to_x_training_order(
        on_source_td, det_psds, cfg, g, tukey, td_norm, sr)
    x_prod, _, vq_prod = real_strain_to_x_production_order(
        on_source_td, det_psds, cfg, g, tukey, td_norm, sr)

    print("\n=== Windowing order audit ===")
    print(f"  var_q training-order:   {vq_train}")
    print(f"  var_q production-order: {vq_prod}")

    max_diff, _ = diff_x(x_train, x_prod, g, "training-order", "production-order")
    return x_train, x_prod, max_diff


def print_consistency_checklist(cfg, std=None):
    """Print all data-treatment conventions and flag the known deviations."""


    g = canonical_grid(cfg)
    tukey = canonical_tukey(cfg, g["n_td"])
    wf = float(np.mean(tukey ** 2))


    print("\n" + "=" * 70)
    print("DATA TREATMENT CONSISTENCY CHECKLIST")
    print("=" * 70)


    checks = [
        ("Tukey alpha", f"{2*float(cfg.get('tukey_roll_off',0.2))/float(cfg['duration']):.4f}",
         "2 roll_off / duration"),
        ("window_factor", f"{wf:.6f}", "mean(tukey^2), baseline for var_q"),
        ("f_min", f"{g['f_min']} Hz", "mask cutoff"),
        ("n_masked", f"{g['n_masked']}", "in-band FD bins"),
        ("n_td", f"{g['n_td']}", "TD samples"),
        ("td_norm", f"{canonical_td_norm(g):.6f}", "sqrt(n_masked)/freq_window_factor"),
        ("bn formula", "sqrt(psd) sqrt(4 df)", "bad bins -> 1.0"),
        ("noise_source", cfg.get("noise_source", "gaussian_whitened"),
         "gaussian_physical matches Whittle target"),
        ("dL_param", cfg.get("dL_param", "linear"), "log -> Jacobian in proposal"),
        ("Mc_param", cfg.get("Mc_param", "linear"), "log -> Jacobian in proposal"),
        ("asd_floor_factor", str(cfg.get("psd_bank", {}).get("asd_floor_factor", 5.0)),
         "median(ASD)/factor"),
        ("psd_context_clip", str(cfg.get("psd_context_clip", 10.0)), "+-clip after z-scoring"),
    ]


    if std is not None:
        checks.extend([
            ("strain_dim", str(getattr(std, "strain_dim", "?")), "2 per_det"),
            ("psd_conditioning", str(getattr(std, "psd_conditioning", False)), "PSD block present"),
            ("amp_dim", str(getattr(std, "amp_dim", 0)), "amplitude summaries"),
            ("x_mean", f"{float(getattr(std, 'x_mean', 0)):.4f}", "global strain mean"),
            ("x_std", f"{float(getattr(std, 'x_std', 1)):.4f}", "global strain std"),
        ])


    for name, value, note in checks:
        print(f"  {name:<25} {str(value):<20} ({note})")

    # These at least used to happen pretty often. Should be better after setting up 
        # the canonical data treatment files (which was made to try and solve the
        # exact issues below)
    print("\n  COMMON DEVIATIONS (check manually):")
    print("  D1. Real-data windowing: production window-first vs training")
    print("      mask-first (window_fd). Use real_strain_to_x_training_order.")
    print("  D2. Highpass 15 Hz on real data; training signals already zero")
    print("      below f_min after window_fd.")
    print("  D3. PSD floor: both paths floor -- verify same factor.")
    print("  D4. td_norm_from: ensure no window_factor dependency.")
    print("  D5. Real noise in injection path uses bilby nfft vs rfft/sr;")
    print("      equivalent only if start_time conventions match.")
    print("=" * 70)