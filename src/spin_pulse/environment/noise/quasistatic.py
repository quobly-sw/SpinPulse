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


class QuasistaticNoiseTimeTrace(NoiseTimeTrace):
    r"""Generate a quasi-static noise time trace for spin-qubit simulations.

    This noise type produces a sequence of piecewise-constant Gaussian
    noise segments. Each segment has duration ``segment_duration``, and all
    samples within a segment share the same random value. The resulting
    noise corresponds to a quasi-static fluctuation with standard deviation
    determined by the coherence time ``T2S``.

    Attributes:
        - segment_duration (int): Duration of each constant-noise segment.
        - sigma (float): Standard deviation of the quasistatic noise,
            computed as :math:`\sqrt{2}/T_2^*`.
        - T2S (float): Coherence time parameter used to scale the noise.
        - values (ndarray): Array of noise values of length ``duration``.

    """

    def __init__(
        self, T2S: float, duration: int, segment_duration: int, seed: int | None = None
    ):
        r"""Initialize a quasi-static Gaussian noise trace.

        The noise trace is constructed from repeated Gaussian segments of
        length ``segment_duration``. Each segment takes a single Gaussian
        random value with zero mean and standard deviation :math:`\sqrt{2}/T_2^*`.
        The total duration must be an integer multiple of the segment length.

        Parameters:
            T2S (float): Coherence time parameter defining the noise intensity.
            duration (int): Total number of time steps in the noise trace.
            segment_duration (int): Length of each piecewise-constant segment.
            seed (int | None): Optional seed for reproducible random generation.

        Raises:
            ValueError: If ``duration`` is not divisible by ``segment_duration``.

        """

        super().__init__(duration)

        self.segment_duration = segment_duration

        if duration % segment_duration != 0:
            raise ValueError("segment_duration must be commensurate with duration")

        repeat = duration // segment_duration

        self.sigma = np.sqrt(2) / (T2S)
        self.T2S = T2S
        self.values = np.zeros(duration)
        rng = np.random.default_rng(seed=seed)
        for i in range(repeat):
            self.values[i * segment_duration : (i + 1) * segment_duration] = (
                self.sigma * rng.normal(loc=0.0, scale=1.0) * np.ones(segment_duration)
            )

    def plot_ramsey_contrast(self, ramsey_duration: int):
        r"""Plot the analytical and simulated Ramsey contrast.

        The analytical contrast :math:`\exp(-t^2 / T_2^{*2})` is plotted together
        with the contrast returned by the parent ``NoiseTimeTrace`` class.

        Parameters:
            ramsey_duration (int): Duration over which the Ramsey signal
                is evaluated.

        """
        t = np.arange(ramsey_duration)
        plt.plot(
            self.get_analytical_contrast(t), label="$e^{-(t/T_2^*)^2}$", color="orange"
        )
        super().plot_ramsey_contrast(ramsey_duration)

    def get_analytical_contrast(self, idle_duration):
        return np.exp(-((idle_duration / self.T2S) ** 2))
