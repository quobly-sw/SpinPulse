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
"""Utility functions to interact with circuits."""

from __future__ import annotations

import warnings
from functools import reduce
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from quimb.tensor import CircuitMPS
from quimb.tensor.circuit import register_constant_gate
from scipy.linalg import expm

from .hardware_specs import HardwareSpecs
from .instructions import (
    IdleInstruction,
    SquareRotationInstruction,
)
from .pulse_sequence import PulseSequence

if TYPE_CHECKING:
    from qiskit import QuantumCircuit
    from quimb.tensor.circuit import CircuitMPS

    from .pulse_circuit import PulseCircuit


def gate_to_pulse_sequences(
    gate, hardware_specs: HardwareSpecs
) -> tuple[list[PulseSequence], list[PulseSequence]]:
    """
    Translate a Qiskit gate (`RX`, `RY`, `RZ`, or `RZZ`) into hardware-compatible
    pulse sequences.

    This function maps high-level Qiskit rotation gates into low-level pulse
    instructions that follow the hardware specifications. Single-qubit rotations
    generate a single PulseSequence, while the ``rzz`` interaction generate two
    qubit sequence followed by a single qubit sequence on each qubit.

    Parameters:
        gate (CircuitInstruction): The Qiskit instruction to translate.
            Supported operations are ``rx``, ``ry``, ``rz``, ``rzz`` and ``delay``.
        hardware_specs (HardwareSpecs): Hardware configuration including available
            fields, ramp durations, and rotation generation routines.

    Returns:
        tuple[list[PulseSequence], list[PulseSequence]]:
            A pair of lists:
            * one-qubit pulse sequences generated from the gate,
            * two-qubit pulse sequences (only for ``rzz``).

    Raises:
        ValueError: If the gate type is not supported.

    Notes:
        For ``rzz`` gates:
            * A central Heisenberg pulse is generated.
            * Pre/post idle ramps are added to ensure smooth pulse shaping.
            * Each qubit receives a compensating detuned Z rotation of equal duration.

        For ``delay`` gates:
            A single IdleInstruction is wrapped in a PulseSequence.

        For single-qubit rotations:
            The corresponding rotation generator in ``hardware_specs`` is invoked.

    """
    # Gate being a Qiskit.CircuitInstruction
    generator = hardware_specs.rotation_generator
    ramp_duration = hardware_specs.ramp_duration
    oneq_pulse_sequences: list[PulseSequence] = []
    twoq_pulse_sequences: list[PulseSequence] = []
    name_: str = gate.operation.name
    if name_ in ["rx", "ry", "rz", "rzz"]:
        qubits = gate.qubits
        angle = gate.operation.params[0]
        if not np.isclose(angle, 0.0, atol=1e-12):
            if name_ == "rzz":
                heis_instruction = generator.from_angle(
                    "Heisenberg", qubits, angle, hardware_specs
                )
                pre_instruction = IdleInstruction(qubits, hardware_specs.ramp_duration)
                post_instruction = IdleInstruction(qubits, hardware_specs.ramp_duration)
                heis_sequence = PulseSequence(
                    [pre_instruction, heis_instruction, post_instruction]
                )
                twoq_pulse_sequences.append(heis_sequence)
                duration = heis_sequence.duration
                amplitude = hardware_specs.fields["z"]
                detuned_instruction_0 = SquareRotationInstruction(
                    "z", [qubits[0]], amplitude, -1.0, ramp_duration, duration
                )
                detuned_instruction_1 = SquareRotationInstruction(
                    "z", [qubits[1]], amplitude, 1.0, ramp_duration, duration
                )
                oneq_pulse_sequences.append(PulseSequence([detuned_instruction_0]))
                oneq_pulse_sequences.append(PulseSequence([detuned_instruction_1]))
            else:
                rotation_instruction = generator.from_angle(
                    name_[1:], qubits, angle, hardware_specs
                )
                oneq_pulse_sequences.append(PulseSequence([rotation_instruction]))
        else:
            warnings.warn(
                "Replacing angle=0 rotations with an idle instruction of duration 1."
            )
            for qubit in qubits:
                oneq_pulse_sequences.append(
                    PulseSequence([IdleInstruction([qubit], duration=1)])
                )

    elif gate.operation.name in ["delay"]:
        duration = gate.operation.duration
        if duration == 0:
            duration = 1
            warnings.warn(
                "Found a delay instruction with duration 0. Set the duration to 1."
            )
        oneq_pulse_sequences.append(
            PulseSequence([IdleInstruction(gate.qubits, duration=duration)])
        )
    else:
        raise ValueError(
            f"The gate {name_} pulse is not implemented. Possible gates are rx,ry,rz,rzz"
        )
    return oneq_pulse_sequences, twoq_pulse_sequences


def propagate(H: np.ndarray, coeff: np.ndarray) -> np.ndarray:
    """
    Compute the total unitary evolution operator for a quantum system governed by
    a time-dependent Hamiltonian, expressed as a linear combination of basis Hamiltonians.

    Parameters:
        H (np.ndarray): array containing the Hamiltonian matrices [H1, H2, ..., Hn], each of shape (d, d).
        coeff (np.ndarray): array of time-dependent coefficients for each Hamiltonian. coeff[j, i] is the coefficient for Hamiltonian H[j, :, :] at time step i.

    Returns:
        np.ndarray: The final unitary matrix U of shape (d, d) representing the total time evolution.

    Raises:
        ValueError: If the number of Hamiltonians does not coincide with the number of time-dependent coefficient lists given or if
          the number of coefficients per Hamiltonian is not always the same.

    """
    H_tots = np.einsum("jkl,ji->ikl", H, coeff)
    us = expm(-1j * H_tots)
    u = reduce(np.matmul, us[::-1, :, :])
    return u


def deshuffle_qiskit(mat: NDArray[np.complexfloating]) -> NDArray[np.complexfloating]:
    """Reverse Qiskit's bit-ordering convention in a matrix representation.

    This function permutes the rows and columns of a square matrix by reversing
    the binary representation of their indices. It is typically used when
    converting multi-qubit operators between Qiskit and libraries that follow a
    different qubit-index (endianness) convention, such as Quimb.

    Parameters:
        mat: A square matrix of shape ``(2**n, 2**n)`` representing an ``n``-qubit
            operator.

    Returns:
        A matrix of the same shape as ``mat`` with reversed bit-ordering applied
          to both row and column indices.

    """
    d = mat.shape[0]
    mymat = np.zeros_like(mat)
    n_bits = int(np.log2(d))
    for i in range(d):
        i_bin = format(i, f"0{n_bits}b")
        i_bin_p = i_bin[::-1]
        i_p = int(i_bin_p, 2)
        for j in range(d):
            j_bin = format(j, f"0{n_bits}b")
            j_bin_p = j_bin[::-1]
            j_p = int(j_bin_p, 2)
            mymat[i_p, j_p] = mat[i, j]
    return mymat


def qiskit_to_quimb(circuit: QuantumCircuit) -> CircuitMPS:
    """Convert a Qiskit quantum circuit into a Quimb MPS circuit.

    Each instruction of the input ``QuantumCircuit`` is translated into a
    constant gate applied to a ``CircuitMPS``. For multi-qubit gates, the gate
    matrix is first reordered using ``deshuffle_qiskit`` to match Quimb's qubit
    ordering convention.

    Parameters:
        circuit: The input ``qiskit.QuantumCircuit`` to convert.

    Returns:
        A ``quimb.tensor.circuit.CircuitMPS`` representing the same sequence of
          operations as the input circuit.

    Notes:
        This function assumes that each instruction in ``circuit.data`` provides
          a matrix representation via ``ins.matrix`` and qubit operands via
          ``ins.qubits``.

    """
    tot_qbs = circuit.num_qubits
    quimb_circ = CircuitMPS(tot_qbs)
    for ins in circuit.data:
        n_qb = len(ins.qubits)
        if n_qb == 1:
            register_constant_gate("NOISY", num_qubits=len(ins.qubits), G=ins.matrix)
            quimb_circ.apply_gate(
                "noisy", qubits=[circuit.qubits.index(q) for q in ins.qubits]
            )
        else:
            register_constant_gate(
                "NOISY", num_qubits=len(ins.qubits), G=deshuffle_qiskit(ins.matrix)
            )
            quimb_circ.apply_gate(
                "noisy", qubits=[circuit.qubits.index(q) for q in ins.qubits]
            )
    return quimb_circ


def my_quimb_fidelity(
    pulse_circuit: PulseCircuit, quimb_circ_ideal: CircuitMPS
) -> float:
    r"""Compute the state fidelity against an ideal Quimb reference circuit.

    The input `pulse_circuit` is converted to a `qiskit.QuantumCircuit` via
    `pulse_circuit.to_circuit()`, then translated to a `CircuitMPS` using
    `qiskit_to_quimb`. The fidelity is computed as

    .. math::

        F = \left| \langle \psi_{\mathrm{ideal}} \mid \psi \rangle \right|^2



    Parameters:
        pulse_circuit: Pulse-level circuit to evaluate.
        quimb_circ_ideal: Ideal reference `CircuitMPS` circuit.

    Returns:
        The state fidelity as a float in $[0, 1]$.

    """
    quimb_circ = qiskit_to_quimb(pulse_circuit.to_circuit())
    psi_ideal = quimb_circ_ideal.psi
    psi = quimb_circ.psi
    return np.abs(psi_ideal.H @ psi) ** 2
