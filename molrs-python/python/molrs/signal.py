"""Signal processing — ``molrs::signal``.

FFT autocorrelation and cross-correlation, window functions and frequency
grids. All computation is in Rust.
"""

from ._native import acf_fft, apply_window, frequency_grid, xcorr_fft

__all__ = ["acf_fft", "apply_window", "frequency_grid", "xcorr_fft"]
