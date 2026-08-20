"""bilbyflow.data — data prep, standardisation, banks, whitening + noise."""

# As noted in the main file, yes this is a lot, will refactor later
from .canonical import (
    window_fd, whitened_fd_to_channels, AMP_NAMES, N_AMP, compute_amp_context,
    canonical_grid, canonical_tukey, canonical_welch_window,
    canonical_td_norm, canonical_bn, canonical_valid_mask,
    whiten_fd, var_q_of, welch_psd,
    signal_to_whitened, build_x_strain, build_x_full,
    canonical_psd_context, real_strain_to_x,
    real_strain_to_x_training_order, real_strain_to_x_production_order,
)
from .noise import (
    noise_gaussian_physical, noise_real_segment, noise_gaussian_whitened,
    injection_to_x,
    build_noise_index, load_noise_strain, estimate_psd_from_strain,
    draw_real_noise_pair,
)
from .standardiser import Standardiser, check_aux_stats, check_amp_stats
from .psd import precompute_psd_bank_from_segments
from .banks import (
    D_REF, precompute_waveforms, precompute_sky_bank,
    precompute_noise_segment_bank, load_or_compute,
)
from .dataset import OnTheFlyGWDataset, generate_fixed_dataset


__all__ = [name for name in dir() if not name.startswith("_")]
