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


class WhiteNoiseTimeTrace(NoiseTimeTrace):
    r"""Generate a white noise time trace.

    This class constructs a noise trace of length ``duration`` where each
    time step is drawn from an independent Gaussian distribution. The noise
    intensity is determined by the coherence time ``T2S`` that fixes the standard
    deviation :math:`\sqrt{2/T_2^*}`. White noise corresponds to a flat
    power spectral density and requires ``segment_duration = 1``.

    Attributes:
        - segment_duration (int): Always equal to 1 for white noise.
        - sigma (float): Standard deviation controlling the noise intensity.
        - T2S (float): Coherence time parameter determining the noise intensity.
        - values (ndarray): Noise samples of length ``duration``.

    """

    def __init__(
        self, T2S: int, duration: int, segment_duration: int, seed: int | None = None
    ):
        r"""Initialize a white noise time trace for spin qubit simulations.

        Each value of the trace is drawn independently from a Gaussian
        distribution with zero mean and standard deviation :math:`\sqrt{2/T_2^*}`.
        The ``segment_duration`` parameter must be equal to 1, since white
        noise has no temporal correlations.

        Parameters:
            T2S (float): Coherence time parameter determining the noise intensity.
            duration (int): Total number of time steps in the noise trace.
            segment_duration (int): Must be equal to 1 for white noise.
            seed (int | None): Optional seed for reproducible random
              number generation.

        Returns:
            None: The generated noise values are stored in ``self.values``.

        """
        super().__init__(duration)
        if segment_duration != 1:
            raise ValueError("White noise must have segment_duration=1")

        rng = np.random.default_rng(seed=seed)

        self.segment_duration = 1
        self.sigma = np.sqrt(2 / T2S)
        self.values = np.zeros(duration)
        x = rng.normal(loc=0.0, scale=1.0, size=duration)
        self.values = self.sigma * x
        self.T2S = T2S

    def plot_ramsey_contrast(self, ramsey_duration: int):
        """Plot analytical and numerical Ramsey contrast for white noise.

        The plot overlays:

        - the analytical exponential decay expected for white noise,
        - the numerical contrast computed from the generated time trace.

        Parameters:
            ramsey_duration (int): Number of time steps used in the
              Ramsey experiment evaluation.

        Returns:
            None: The function produces a plot of the Ramsey contrast.

        """
        t = np.arange(ramsey_duration)
        plt.plot(
            self.get_analytical_contrast(t), label="$e^{-t/T_2^*}$", color="orange"
        )
        super().plot_ramsey_contrast(ramsey_duration)

    def get_analytical_contrast(self, idle_duration):
        return np.exp(-idle_duration / self.T2S)
