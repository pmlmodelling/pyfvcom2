import numpy as np
import scipy.fft


class LanczosFilter:
    """Lanczos low- or high-pass filter.

    Creates a filter object with fixed parameters. Call filter() to apply it
    to a 1-D time series.

    Parameters
    ----------
    dt : float, optional
        Sampling interval in minutes. Defaults to 1.
    cutoff : float, optional
        Cutoff period in minutes. Defaults to half the Nyquist period.
    samples : int, optional
        Number of samples (window length). Defaults to 100.
    passtype : str, optional
        ``'low'`` for low-pass (default) or ``'high'`` for high-pass.

    Notes
    -----
    Python reimplementation of the MATLAB lanczosfilter.m function:
    https://mathworks.com/matlabcentral/fileexchange/14041

    NaN values are replaced by the time-series mean before filtering.

    References
    ----------
    Emery, W. J. and R. E. Thomson. "Data Analysis Methods in Physical
    Oceanography". Elsevier, 2nd ed., 2004, pp. 533-539.
    """

    def __init__(self, dt: float = 1, cutoff: float = None,
                 samples: int = 100, passtype: str = 'low'):
        if passtype == 'low':
            filterindex = 0
        elif passtype == 'high':
            filterindex = 1
        else:
            raise ValueError("passtype must be 'low' or 'high'.")

        self.dt = dt
        self.samples = samples
        self.passtype = passtype
        self.nyquist_frequency = 1 / (2 * dt)

        # Default cutoff to half the Nyquist period if not supplied.
        if cutoff is None:
            cutoff = self.nyquist_frequency / 2

        # Normalise against the Nyquist frequency and store.
        self.cutoff = cutoff / self.nyquist_frequency

        # Compute and select the low- or high-pass coefficients.
        self.coef = _lanczos_filter_coef(self.cutoff, samples)[:, filterindex]

    def filter(self, x: np.ndarray) -> np.ndarray:
        """Apply the filter to a 1-D time series.

        Parameters
        ----------
        x : np.ndarray
            1-D time series values.

        Returns
        -------
        y : np.ndarray
            Filtered time series, same length as x.
        """
        n = len(x)
        window, Ff = _spectral_window(self.coef, n)
        # Scale frequencies from normalised to physical (cycles per minute).
        self.Ff = Ff * self.nyquist_frequency

        # Replace NaNs with the series mean.
        x = x.copy()
        inan = np.isnan(x)
        if np.any(inan):
            x[inan] = np.nanmean(x)

        y, _ = _spectral_filtering(x, window, n)

        if len(y) != n:
            raise ValueError('Filtered output length does not match input.')

        return y


def lanczos(x: np.ndarray, dt: float = 1, cutoff: float = None,
            samples: int = 100, passtype: str = 'low'):
    """Apply a Lanczos low- or high-pass filter to a 1-D time series.

    Parameters
    ----------
    x : np.ndarray
        1-D time series values.
    dt : float, optional
        Sampling interval in minutes. Defaults to 1.
    cutoff : float, optional
        Cutoff period in minutes. Defaults to half the Nyquist period.
    samples : int, optional
        Number of samples (window length). Defaults to 100.
    passtype : str, optional
        ``'low'`` for low-pass (default) or ``'high'`` for high-pass.

    Returns
    -------
    y : np.ndarray
        Filtered time series.
    coef : np.ndarray
        Cosine window coefficients.
    window : np.ndarray
        Frequency-domain window.
    Cx : np.ndarray
        Complex Fourier transform of x (half-spectrum).
    Ff : np.ndarray
        Fourier frequencies from 0 to the Nyquist frequency.

    Notes
    -----
    Python reimplementation of the MATLAB lanczosfilter.m function:
    https://mathworks.com/matlabcentral/fileexchange/14041

    NaN values are replaced by the time-series mean before filtering.

    References
    ----------
    Emery, W. J. and R. E. Thomson. "Data Analysis Methods in Physical
    Oceanography". Elsevier, 2nd ed., 2004, pp. 533-539.
    """
    if passtype == 'low':
        filterindex = 0
    elif passtype == 'high':
        filterindex = 1
    else:
        raise ValueError("passtype must be 'low' or 'high'.")

    nyquist_frequency = 1 / (2 * dt)
    if cutoff is None:
        cutoff = nyquist_frequency / 2
    cutoff = cutoff / nyquist_frequency

    coef = _lanczos_filter_coef(cutoff, samples)[:, filterindex]

    n = len(x)
    window, Ff = _spectral_window(coef, n)
    Ff = Ff * nyquist_frequency

    x = x.copy()
    inan = np.isnan(x)
    if np.any(inan):
        x[inan] = np.nanmean(x)

    y, Cx = _spectral_filtering(x, window, n)

    if len(y) != n:
        raise ValueError('Filtered output length does not match input.')

    return y, coef, window, Cx, Ff


def _lanczos_filter_coef(cutoff: float, samples: int) -> np.ndarray:
    """Compute Lanczos low- and high-pass cosine coefficients.

    Returns an array of shape (samples + 1, 2) where column 0 is the
    low-pass and column 1 is the high-pass coefficients.
    """
    k = np.linspace(1, samples, samples)
    hkcs = cutoff * np.concatenate((
        [1],
        np.sin(np.pi * k * cutoff) / (np.pi * k * cutoff)
    ))
    sigma = np.concatenate((
        [1],
        np.sin(np.pi * k / samples) / (np.pi * k / samples)
    ))
    hkB = hkcs * sigma
    hkA = -hkB.copy()
    hkA[0] += 1
    return np.column_stack([hkB, hkA])


def _spectral_window(coef: np.ndarray, n: int):
    """Compute the cosine filter window in frequency space."""
    eps = np.finfo(np.float32).eps
    Ff = np.arange(0, 1 + eps, 2 / n)
    k = np.arange(1, len(coef))
    window = np.array([
        coef[0] + 2 * np.sum(coef[1:] * np.cos(k * np.pi * f))
        for f in Ff
    ])
    return window, Ff


def _spectral_filtering(x: np.ndarray, window: np.ndarray, n: int):
    """Filter x in frequency space by multiplying with window."""
    Cx = scipy.fft.fft(x.ravel())
    Cx_half = Cx[:(n // 2) + 1]
    CxH = Cx_half * window.ravel()
    # Rebuild the full conjugate-symmetric spectrum.
    CxH = np.concatenate((CxH, np.conj(CxH[1:n - len(CxH) + 1][::-1])))
    y = np.real(scipy.fft.ifft(CxH))
    return y, Cx_half
