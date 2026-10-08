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
# Modified: reduce pink-noise memory usage and add optional disk storage.
""""""

import math
import os
import shutil
import weakref
from tempfile import NamedTemporaryFile, TemporaryFile, mkdtemp

import matplotlib.pyplot as plt
import numpy as np

from .noise_time_trace import NoiseTimeTrace

_BLOCK_SAMPLES = 65536


def _remove_noise_directory(directory):
    try:
        shutil.rmtree(directory)
    except FileNotFoundError:
        # A caller may already have removed its private job scratch directory.
        pass


def _empty_noise_array(length, dtype, noise_directory, *, shared=False):
    if noise_directory is None:
        return np.empty(length, dtype=dtype)
    # Process workers reopen histories by filename; keep them in a private
    # directory. FFT workspaces stay anonymous.
    private_directory = (
        mkdtemp(dir=noise_directory, prefix="spin-pulse-") if shared else None
    )
    try:
        file = (
            NamedTemporaryFile(dir=private_directory, delete=False)
            if shared
            else TemporaryFile(dir=noise_directory)
        )
        with file:
            size = length * np.dtype(dtype).itemsize
            if hasattr(os, "posix_fallocate"):
                # Fail with ENOSPC before a sparse mapping could cause SIGBUS.
                os.posix_fallocate(file.fileno(), 0, size)
            else:
                file.truncate(size)
            values = np.memmap(file, dtype=dtype, mode="r+", shape=(length,))
    except BaseException:
        if private_directory is not None:
            _remove_noise_directory(private_directory)
        raise
    if shared:
        # Tie deletion to the underlying mapping, not a particular array view.
        # Joblib workers borrow the filename while the parent owns its arrays.
        cleanup = weakref.finalize(
            values._mmap, _remove_noise_directory, private_directory
        )
        # Windows cannot unlink a still-open mapping at interpreter shutdown.
        # Normal last-view cleanup runs after mmap has closed its OS handles.
        cleanup.atexit = os.name == "posix"
    return values


def _flush(values):
    if isinstance(values, np.memmap):
        # Clean file pages can be reclaimed under a process/cgroup memory limit.
        values.flush()


def get_pink_noise(
    segment_duration: int,
    seed: int | None = None,
    *,
    noise_directory: str | os.PathLike | None = None,
):
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
        noise_directory: Existing directory on a disk filesystem for temporary
          arrays. If supplied, return a float64 memmap; otherwise return an
          in-memory array. Files live as long as the arrays and their views.
          Avoid tmpfs when the goal is to reduce RAM usage.

    Returns:
        ndarray: Array of length ``segment_duration`` containing a
          realization of pink noise.

    Raises:
        ValueError: If ``segment_duration`` is odd.

    """
    if segment_duration < 2 or segment_duration % 2 != 0:
        raise ValueError("segment_duration must be even and at least 2")

    rng = np.random.default_rng(seed=seed)
    n = segment_duration
    spectrum = _empty_noise_array(n, np.complex128, noise_directory)
    spectrum[0] = 0
    spectrum[n // 2] = 1 / (n // 2 + 1)
    for start in range(1, n // 2, _BLOCK_SAMPLES):
        stop = min(n // 2, start + _BLOCK_SAMPLES)
        frequencies = np.arange(start + 1, stop + 1)
        phases = (rng.uniform(size=stop - start) - 0.5) * 2 * np.pi
        spectrum[start:stop] = (1 / frequencies**0.5) * np.exp(1j * phases)
        spectrum[n - stop + 1 : n - start + 1] = np.conj(spectrum[start:stop][::-1])
    _flush(spectrum)

    # Exact-length Cooley-Tukey factorization. Never pad to a convenient FFT
    # length: doing so changes the frequencies and the low-frequency cutoff.
    # Bound the first dimension so column transforms and twiddles stay small.
    rows = max(
        (d for d in range(2, min(512, math.isqrt(n)) + 1) if n % d == 0),
        default=2,
    )
    columns = n // rows
    matrix = spectrum.reshape(rows, columns)
    tile = max(1, _BLOCK_SAMPLES // rows)
    row_indices = np.arange(rows)[:, None]
    for start in range(0, columns, tile):
        stop = min(columns, start + tile)
        block = np.fft.ifft(matrix[:, start:stop], axis=0)
        block *= np.exp(2j * np.pi * row_indices * np.arange(start, stop)[None, :] / n)
        matrix[:, start:stop] = block
    _flush(spectrum)
    for row in range(rows):
        matrix[row, :] = np.fft.ifft(matrix[row, :])
    _flush(spectrum)

    output = _empty_noise_array(n, np.float64, noise_directory, shared=True)
    for start in range(0, columns, tile):
        stop = min(columns, start + tile)
        output[start * rows : stop * rows] = (
            matrix[:, start:stop].real.T.reshape(-1) * n
        )
    _flush(output)
    return output


def get_pink_noise_with_repetitions(
    duration: int,
    segment_duration: int,
    seed: int | None = None,
    *,
    noise_directory: str | os.PathLike | None = None,
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
        noise_directory: Optional disk directory, as in ``get_pink_noise``.

    Returns:
        ndarray: Pink noise trace of length ``duration``.

    """
    if duration < 1:
        raise ValueError("duration must be positive")
    if duration == segment_duration:
        return get_pink_noise(segment_duration, seed, noise_directory=noise_directory)
    if segment_duration < 2 or segment_duration % 2:
        raise ValueError("segment_duration must be even and at least 2")
    output = _empty_noise_array(duration, np.float64, noise_directory, shared=True)
    for start in range(0, duration, segment_duration):
        # Preserve seed semantics: integer/SeedSequence seeds repeat, Generator
        # seeds advance, and None draws a new independent segment each time.
        segment = get_pink_noise(
            segment_duration, seed, noise_directory=noise_directory
        )
        count = min(segment_duration, duration - start)
        for offset in range(0, count, _BLOCK_SAMPLES):
            stop = min(count, offset + _BLOCK_SAMPLES)
            output[start + offset : start + stop] = segment[offset:stop]
        del segment
    _flush(output)
    return output


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
        self,
        T2S: float,
        duration: int,
        segment_duration: int,
        seed: int | None = None,
        *,
        noise_directory: str | os.PathLike | None = None,
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
            noise_directory: Optional disk directory for temporary trace and
              FFT arrays. ``values`` remains a NumPy array (a memmap).

        Returns:
            None: The time trace is stored internally in ``self.values``.

        """
        # The base initializer allocates a zero-filled array which would be
        # discarded immediately. Allocate the final noise values directly.
        self.duration = duration

        self.segment_duration = segment_duration

        S0 = 1 / (4 * np.pi**2 * np.log(segment_duration) * T2S**2)

        self.values = get_pink_noise_with_repetitions(
            duration, segment_duration, seed, noise_directory=noise_directory
        )
        scale = 2 * np.pi * np.sqrt(S0)
        for start in range(0, duration, _BLOCK_SAMPLES):
            self.values[start : start + _BLOCK_SAMPLES] *= scale
        _flush(self.values)

        self.S0 = S0
        self.T2S = T2S

        # Two-pass population standard deviation with bounded temporaries.
        # np.std on a complete memmap allocates a trace-sized deviation array.
        mean = (
            math.fsum(
                float(np.sum(self.values[start : start + _BLOCK_SAMPLES]))
                for start in range(0, duration, _BLOCK_SAMPLES)
            )
            / duration
        )
        variance = (
            math.fsum(
                float(np.sum((self.values[start : start + _BLOCK_SAMPLES] - mean) ** 2))
                for start in range(0, duration, _BLOCK_SAMPLES)
            )
            / duration
        )
        self.sigma = math.sqrt(variance)

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
