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
"""Description of the noisy environment associated to a hardware."""

import os

import numpy as np
from joblib import Parallel, delayed

from ..transpilation.hardware_specs import HardwareSpecs
from .noise import (
    NoiseType,
    PinkNoiseTimeTrace,
    QuasistaticNoiseTimeTrace,
    WhiteNoiseTimeTrace,
)


class ExperimentalEnvironment:
    """
    Contain a quantum experimental environment with configurable noise models.

    Attributes:
        - hardware_specs (class): HardwareSpecs class, that defines the hardware settings.
        - noise_type (NoiseType): Type of noise to simulate. Must be "pink", "white", or "quasistatic".
        - T2S (float): Characteristic time of individual qubits.
        - TJS (float or None): Characteristic time of coupled two-qubit system at maximal J coupling without noise on the qubit's frequency.
        - duration (int): Total duration of the simulation.
        - segment_duration (int): Duration of each noise segment; used to partition the time trace.
        - only_idle (bool): Flag to apply noise only to idle qubits.
        - time_traces (list[float]): List of time traces for each qubit.
        - time_traces_coupling (list[float]): List of time traces for coupling noise for each pair of qubits (if TJS is set).
        - seed (int or None): seed integer for random number generation. If not specified, no seed used.

    """

    # noise_generator (class): Noise generator class based on noise_type. :no-index:

    noise_generator: (
        type[WhiteNoiseTimeTrace]
        | type[QuasistaticNoiseTimeTrace]
        | type[PinkNoiseTimeTrace]
    )

    def __init__(
        self,
        hardware_specs: HardwareSpecs,
        noise_type: NoiseType = NoiseType.PINK,
        T2S: float = 100.0,
        TJS: float | None = None,
        duration: int = 2**10,
        only_idle: bool = False,
        segment_duration: int = 2**10,
        seed: int | None = None,
    ):
        """
        Initialize the ExperimentalEnvironment with specified noise characteristics and simulation parameters.

        Parameters:
            hardware_specs (class): HardwareSpecs class, that defines the hardware settings.
            noise_type (NoiseType): Type of noise to simulate. Must be "pink", "white", or "quasistatic".
            T2S (float): Characteristic time of individual qubits.
            TJS (float or None): Characteristic time of coupled two-qubit system at maximal J coupling with no noise on the qubit's frequency.
            duration (int): Total duration of the simulation.
            only_idle (bool): Flag to apply noise only to idle qubits.
            segment_duration (int): Duration of each noise segment; used to partition the time trace.
            seed (int or None): seed integer for random number generation. If not specified, no seed used.
        Raises:
            ValueError: If an invalid noise_type is provided.

        """

        self.hardware_specs: HardwareSpecs = hardware_specs
        self.noise_type: NoiseType = noise_type
        self.T2S: float = T2S
        self.TJS: float | None = TJS
        self.duration: int = duration
        self.segment_duration: int = segment_duration

        self.seed: int | None = seed
        self._seed_sequence = (
            np.random.SeedSequence(self.seed)
            if self.seed is not None
            else np.random.SeedSequence()
        )

        match noise_type:
            case NoiseType.PINK:
                self.noise_generator = PinkNoiseTimeTrace
            case NoiseType.WHITE:
                self.noise_generator = WhiteNoiseTimeTrace
            case NoiseType.QUASISTATIC:
                self.noise_generator = QuasistaticNoiseTimeTrace
            case _:
                raise ValueError("unknown noise type")

        self.only_idle = only_idle
        self.generate_time_traces()

    def _generate_time_trace_on_qubit(self, seed: int):
        """Generate the time trace for one qubit."""
        return self.noise_generator(
            self.T2S,
            self.duration,
            self.segment_duration,
            seed=seed,  # type: ignore
        )

    def _generate_coupling_time_trace_on_qubit(self, seed: int):
        """Generate the coupling time trace for one qubit."""
        assert self.TJS is not None, (
            "self.TJS cannot be None when generating coupling traces."
        )

        return self.noise_generator(
            self.TJS,
            self.duration,
            self.segment_duration,
            seed=seed,  # type: ignore
        )

    def generate_time_traces(self, n_jobs: int = 1):
        """
        Generate noise time traces for each qubit's frequency and J coupling to each pair of qubits if TJS is defined.

        Parameters:
            n_jobs (int): Number of core used to generate the time trace.

        Behavior:
            For each qubit, instantiate a noise generator using the selected noise_type.
            The generator uses T2S, duration, and segment_duration to produce a time trace.
            If TJS is provided, generate additional time traces for J coupling noise for each pair of qubits.

        Effects:
            Populate self.time_traces with one noise trace per qubit.
            If TJS is set, populate self.time_traces_coupling with one trace per pair of qubits (n-1 traces for n qubits).

        """

        n_jobs = min(n_jobs, os.cpu_count() or 1)

        # Spawn two child SeedSequences (one for qubit traces, one for coupling traces).
        # Each call to .spawn() yield independent streams, even though the root's
        # entropy (derived from self.seed) is unchanged.
        qubit_root, coupling_root = self._seed_sequence.spawn(2)
        qubit_seeds: list[np.random.SeedSequence] = list(
            qubit_root.spawn(self.hardware_specs.num_qubits)
        )

        self.time_traces = Parallel(n_jobs=n_jobs)(
            delayed(self._generate_time_trace_on_qubit)(qubit_seeds[index])
            for index in range(self.hardware_specs.num_qubits)
        )

        if self.TJS is not None:
            coupling_seeds: list[np.random.SeedSequence] = list(
                coupling_root.spawn(self.hardware_specs.num_qubits - 1)
            )

            self.time_traces_coupling = Parallel(n_jobs=n_jobs)(
                delayed(self._generate_coupling_time_trace_on_qubit)(
                    coupling_seeds[index]
                )
                for index in range(self.hardware_specs.num_qubits - 1)
            )

    def __str__(self):
        """
        Return a string representation of the ExperimentalEnvironment instance.

        Includes:
            Number of qubits
            Noise type
            T2S and TJS values
            Duration and segment duration
            Whether noise is only appplied to idle qubits
            The J coupling value set in HardwareSpecs
            The total number of generated time traces
            The total number of generated coupling time traces if there exists some
        """
        summary = [
            "ExperimentalEnvironment:",
            f"  Qubits: {self.hardware_specs.num_qubits}",
            f"  Noise Type: {self.noise_type}",
            f"  T2S (qubit dephasing): {self.T2S}",
            f"  TJS (coupling dephasing): {self.TJS if self.TJS is not None else 'None'}",
            f"  Duration: {self.duration}",
            f"  Segment Duration: {self.segment_duration}",
            f"  Only Idle: {self.only_idle}",
            f"  J Coupling: {self.hardware_specs.J_coupling if self.hardware_specs.J_coupling is not None else 'None'}",
            f"  Time Traces Generated: {len(self.time_traces)}",
            f"  Seed: {self.seed}",
        ]
        if hasattr(self, "time_traces_coupling"):
            summary.append(
                f"  Coupling Time Traces Generated: {len(self.time_traces_coupling)}"
            )
        return "\n".join(summary)

    def get_analytical_contrast(self, idle_duration: int | np.ndarray):
        return self.time_traces[0].get_analytical_contrast(idle_duration)
