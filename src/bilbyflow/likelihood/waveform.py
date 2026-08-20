"""
bilbyflow.likelihood.waveform — waveform generator construction.

NOTE ON reference_frequency: set to f_min here, matching the original
scripts. If cfg["f_min"] != the published LVK f_ref (usually 20 Hz), the
spin angles (tilt_1, tilt_2, phi_jl) are defined at a different reference
frequency than published posteriors, so spin-parameter overlays compare
different quantities. Confirm f_min == f_ref, or set reference_frequency
explicitly, before comparing spins.

I (Liam Pinchbeck) am not currently (20/08) extremely familiar with the whole
Bilby workflow, and hence if there's any optimization that can be done here 
that would be fantastic. Nonetheless, I used the standard tutorial scripts
e.g. https://github.com/bilby-dev/bilby/blob/main/examples/gw_examples/injection_examples/binary_neutron_star_example.py
as a reference. 
"""

import bilby

__all__ = ["make_waveform_generator", "WindowedWaveformGenerator"]


class WindowedWaveformGenerator(bilby.gw.WaveformGenerator):
    """FIX-1 (injection runs): apply the training Tukey window to each
    polarisation, so the likelihood template matches the windowed injected
    signal.

    Windowing the polarisations vs the detector-projected strain differ only
    through the detector time-shift phase ramp across the taper — same order
    as the rigid-shift approximation already used in the synthetic-extrinsic
    likelihood."""

    def __init__(self, *args, freq_mask=None, tukey_window=None, n_td=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._win_args = (freq_mask, tukey_window, n_td)

    def frequency_domain_strain(self, parameters=None):
        from ..data.canonical import window_fd
        pols = super().frequency_domain_strain(parameters)
        if pols is None:
            return None
        freq_mask, tukey_window, n_td = self._win_args
        return {k: window_fd(v, freq_mask, tukey_window, n_td)
                for k, v in pols.items()}


def make_waveform_generator(cfg, start_time=0, approximant=None,
                            windowed=False, g=None, tukey_window=None):
    """Standard generator; windowed=True returns a WindowedWaveformGenerator
    (requires g=grid_quantities(cfg) and the training tukey_window)."""
    cls = WindowedWaveformGenerator if windowed else bilby.gw.WaveformGenerator
    kwargs = dict(
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        sampling_frequency=int(cfg["sampling_frequency"]),
        duration=float(cfg["duration"]),
        start_time=start_time,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
        waveform_arguments=dict(
            waveform_approximant=(approximant or cfg["waveform_approximant"]),
            reference_frequency=float(cfg["f_min"]),
            minimum_frequency=float(cfg["f_min"]),
        ),
    )
    if windowed:
        if g is None or tukey_window is None:
            raise ValueError("windowed=True requires g and tukey_window")
        kwargs.update(freq_mask=g["freq_mask"], tukey_window=tukey_window,
                      n_td=g["n_td"])
    return cls(**kwargs)