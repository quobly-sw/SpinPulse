# --------------------------------------------------------------------------------------
# This code is part of SpinPulse.
#
# (C) Copyright Quobly 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
# --------------------------------------------------------------------------------------
""""""

import matplotlib.pyplot as plt
import numpy as np

from .noise_time_trace import NoiseTimeTrace


def get_pink_noise(segment_duration: int, seed: int | None = None):
    """Generate a single segment of pink noise using an inverse FFT method.

    The generated sequence has length ``segment_duration`` and follows a
    spectral density proportional to ``1/freq``.

    This implementation is adapted from the Matlab implementation of
    the algorithm described in [Little2007].

    References:
        [Little2007] M. A. Little, P. E. McSharry, S. J. Roberts,
        D. A. E. Costello, and I. M. Moroz,
        "Exploiting nonlinear recurrence and fractal scaling properties
        for voice disorder detection",
        BioMedical Engineering OnLine, 6:23 (2007).

    Parameters:
        segment_duration (int): Number of time points in the generated
          pink noise segment.
        seed (int | None): Optional seed for reproducible random
          number generation.

    Returns:
        ndarray: Array of length ``segment_duration`` containing a
          realization of pink noise.

    Raises:
        ValueError: If ``segment_duration`` is odd.

    """
    if segment_duration % 2 != 0:
        raise ValueError("segment_duration must be even")

    rng = np.random.default_rng(seed=seed)
    N = segment_duration
    N2 = N // 2 - 1
    f = np.arange(2, N2 + 2)
    beta = 1.0
    A2 = 1 / (f ** (beta / 2))
    p2 = (rng.uniform(size=N2) - 0.5) * 2 * np.pi
    d2 = A2 * np.exp(1j * p2)
    d = np.concatenate(([0], d2, [1 / ((N2 + 2) ** beta)], np.flipud(np.conj(d2))))
    x = np.real(np.fft.ifft(d))
    return N * x


def get_pink_noise_with_repetitions(
    duration: int, segment_duration: int, seed: int | None = None
):
    """Generate pink noise of a given total duration by repeating segments.

    A base pink noise segment of length ``segment_duration`` is generated
    repeatedly and concatenated until the total length reaches ``duration``.
    A low frequency cutoff of ``1/segment_duration`` is implicitly imposed.

    Parameters:
        duration (int): Total number of time points in the final noise trace.
        segment_duration (int): Length of each repeated pink noise segment.
        seed (int | None): Optional seed for reproducible random
          number generation.

    Returns:
        ndarray: Pink noise trace of length ``duration``.

    """
    segments = []
    total_len = 0

    while total_len < duration:
        segment = get_pink_noise(segment_duration, seed)
        segments.append(segment)
        total_len += len(segment)

    # Concatenate all segments and trim to exact length
    return np.concatenate(segments)[:duration]


class PinkNoiseTimeTrace(NoiseTimeTrace):
    """Generate a pink noise time trace with a 1/f power spectral density.

    This class constructs a noise trace of length ``duration`` where the
    spectral density follows a pink noise distribution proportional to
    ``1/f``. A low-frequency cutoff is enforced at ``1/segment_duration`` by
    building the trace from repeated pink noise segments. The parameter ``T2S``
    determines the noise intensity through the scaling factor ``S0`` defined as::

        S0 = 1 / (4 * pi^2 * log(segment_duration) * T2S^2)

    The internal array ``values`` contains the generated noise samples and
    is used by methods such as ``ramsey_contrast`` to evaluate the effect of
    pink noise on qubit coherence.

    Attributes:
        - segment_duration (int): Length of each pink noise segment.
        - S0 (float): Scaling factor controlling the noise intensity.
        - T2S (float): Coherence time parameter determining the noise intensity.
        - sigma (float): Standard deviation of the generated noise values.
        - values (ndarray): Noise values of length ``duration``.

    """

    def __init__(
        self, T2S: float, duration: int, segment_duration: int, seed: int | None = None
    ):
        """Create a pink noise time trace for spin qubit simulations.

        The generated noise satisfies a spectral density proportional to
        ``1/(f*ts)`` with a low frequency cutoff ``f_min=1/segment_duration``.
        The parameter ``T2S`` sets the noise intensity through the relation
        ``S0 = 1 / (4 * pi^2 * log(segment_duration) * T2S^2)``. The internal
        noise trace has length ``duration`` and is constructed by repeating
        pink noise segments.

        Parameters:
            T2S (float): Coherence time parameter determining the noise
              intensity.
            duration (int): Total number of time steps in the noise trace.
            segment_duration (int): Length of each pink noise segment.
            seed (int | None): Optional seed for reproducible random
              number generation.

        Returns:
            None: The time trace is stored internally in ``self.values``.

        """
        super().__init__(duration)

        self.segment_duration = segment_duration

        S0 = 1 / (4 * np.pi**2 * np.log(segment_duration) * T2S**2)

        self.values = (
            2
            * np.pi
            * np.sqrt(S0)
            * get_pink_noise_with_repetitions(duration, segment_duration, seed)
        )

        self.S0 = S0
        self.T2S = T2S

        self.sigma = np.std(
            np.asarray(self.values), ddof=0
        )  # ddof=0 for population std deviation

    def plot_ramsey_contrast(self, ramsey_duration: int):
        """Plot analytical and numerical Ramsey contrast curves.

        This method overlays:
            - the analytical Gaussian contrast expected for pink noise,
            - a corrected analytical contrast that accounts for the
              frequency cutoff imposed by ``segment_duration``,
            - the numerical contrast obtained from the underlying noise trace.

        Parameters:
            ramsey_duration (int): Number of time steps used in the
              Ramsey experiment evaluation.

        Returns:
            None: The function produces a plot of the Ramsey contrast.

        """
        t = np.arange(ramsey_duration)
        plt.plot(
            np.exp(-(t**2) / self.T2S**2), label=" $e^{-(t/T_2^*)^2}$", color="orange"
        )
        plt.plot(
            t[1:],
            self.get_analytical_contrast(t[1:]),
            label="$e^{-(t/T_2^*(t))^2}$",
            color="green",
        )
        super().plot_ramsey_contrast(ramsey_duration)

    def get_analytical_contrast(self, idle_duration):
        T2_t = self.T2S / np.sqrt(
            np.log(self.segment_duration / idle_duration)
            / np.log(self.segment_duration)
        )
        contrast = np.exp(-(idle_duration**2) / (T2_t**2))
        return contrast
