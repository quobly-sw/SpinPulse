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
"""
Utilities to analyze and visualize quantum super-Operators.
"""

import itertools

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import qiskit as qi
from qiskit.quantum_info import (
    Chi,
    Operator,
    Pauli,
    SuperOp,
)


def compare_circuits(circ1: qi.QuantumCircuit, circ2: qi.QuantumCircuit):
    """
    Compare two quantum circuits by plotting the matrix elements of their
    corresponding unitary operators.

    This function converts both circuits into unitary matrices using
    qiskit.quantum_info.Operator.
    The global phase is removed before comparison. Three scatter plots are
    generated: real parts, imaginary parts, and absolute values of the matrix
    elements. A diagonal reference line is shown, and the squared distance
    between the two matrices is displayed.

    Parameters:
        circ1 (qiskit.QuantumCircuit): First circuit to compare.
        circ2 (qiskit.QuantumCircuit): Second circuit to compare.

    Notes:
        The global phase is aligned using the matrix element with maximum magnitude.
        This function is useful to visually validate the equivalence of two circuits
        after transformations such as transpilation or pulse compilation.

    """
    data1 = (Operator.from_circuit(circ1).data).flatten()
    data2 = (Operator.from_circuit(circ2).data).flatten()
    i_phase = np.argmax(abs(data1))
    phase1 = np.angle(data1[i_phase])
    data1 *= np.exp(-1j * phase1)
    phase2 = np.angle(data2[i_phase])
    data2 *= np.exp(-1j * phase2)
    plt.plot(np.real(data1), np.real(data2), "o", label="real")
    plt.plot(np.imag(data1), np.imag(data2), "x", label="imag")
    plt.plot(np.abs(data1), np.abs(data2), "*", label="abs")

    a = np.max(abs(data1))
    plt.plot([-a, a], [-a, a], "--k")
    plt.text(-a, a, f"distance  {np.sum(np.abs(data1 - data2) ** 2)}")
    plt.xlabel("circ 1 matrix elements")
    plt.ylabel("circ 2 matrix elements")
    plt.legend(loc=0)


def plot_chi_matrix(superop: dict[str, SuperOp], threshold=None) -> plt.Figure:
    """Plot the chi-matrix elements for one or multiple quantum superop.

    The chi-matrix is computed for each channel and plotted as bar plots
    (real and imaginary parts). If a threshold is provided, only elements
    with absolute value greater than the threshold (from the first channel)
    are shown.

    Channels whose key contains the substring ``"analytical"`` are plotted
    with transparent bars and line styles, while the others are plotted as
    semi-transparent filled bars.

    Parameters:
        superop (dict[str, qiskit.quantum_info.SuperOp or qiskit.quantum_info.Channel]):
            Dictionary mapping labels to quantum super-Operator. Each value must
            be compatible with ``qiskit.quantum_info.Choi``/``Chi`` so that
            ``Chi(superop[key]).data`` returns the chi-matrix.
        threshold (float | None): If not ``None``, only chi-matrix elements with
            absolute value greater than ``threshold`` (as determined from the
            first channel in ``superop``) are included in the plot.

    Returns:
        matplotlib.figure.Figure: The figure object containing the chi-matrix plot.

    """
    n_qb = int(np.log2(next(iter(superop.values())).data.shape[0]) / 2)
    mpl.rcParams["font.size"] = 22
    fig = plt.figure(figsize=(10, 5))
    # Generate Pauli basis labels
    paulis = ["I", "X", "Y", "Z"]
    pauli_labels = ["".join(p) for p in itertools.product(paulis, repeat=n_qb)]
    full_labels = [f"${p1}.{p2}$" for p1 in pauli_labels for p2 in pauli_labels]

    counter = 0
    lines_style = ["-", "--", ":", "-."]

    x = np.arange(len(full_labels))

    for index, key in enumerate(superop.keys()):
        chi_matrix = Chi(superop[key]).data

        # Flatten matrix into list of values and labels
        values = chi_matrix.flatten()

        if threshold is not None and index == 0:
            indices = [i for i in range(len(values)) if np.abs(values[i]) > threshold]
            x = np.array(range(len(indices)))
            full_labels = [full_labels[i] for i in indices]
        if threshold is not None:
            values = values[indices]

        if "analytical" in key:
            plt.bar(
                x,
                np.real(values),
                tick_label=full_labels,
                label=f"{key}" + " (real)",
                facecolor="none",
                edgecolor="tab:blue",
                linestyle=lines_style[counter % len(lines_style)],
                linewidth=1.0,
            )
            plt.bar(
                x,
                np.imag(values),
                tick_label=full_labels,
                label=f"{key}" + " (imag)",
                facecolor="none",
                edgecolor="tab:orange",
                linestyle=lines_style[counter % len(lines_style)],
                linewidth=1.0,
            )
            counter += 1

        else:
            plt.bar(
                x,
                np.real(values),
                alpha=0.3,
                tick_label=full_labels,
                label=f"{key}" + " (real)",
            )
            plt.bar(
                x,
                np.imag(values),
                alpha=0.3,
                tick_label=full_labels,
                label=f"{key}" + " (imag)",
            )

    plt.xticks(rotation=90)
    plt.ylabel(r"$\chi$")
    plt.legend()
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    return fig


def get_superop_from_paulidict(pauli_dict: dict[str, complex]) -> SuperOp:
    r"""Return the SuperOp corresponding to a Pauli decomposition given as
    a dictionary.

    The input is a mapping from tensor-product Pauli labels (e.g., "IXZ",
    "ZZ", "I") to complex coefficients. For an n-qubit system, each label
    must be a string of length n, with characters drawn from
    ``{"I", "X", "Y", "Z"}``.

    The function builds the operator

    .. math::

        O = \sum_{P} c_P P,

    where :math:`P` runs over Pauli strings and :math:`c_P` are the provided
    coefficients, and then wraps it as a ``qiskit.quantum_info.SuperOp``.

    Parameters:
        pauli_dict (dict[str, complex]): Dictionary mapping Pauli labels
            (e.g., "IX", "ZZI") to complex coefficients.

    Returns:
        qiskit.quantum_info.SuperOp: The resulting quantum channel represented
        as a ``SuperOp`` acting on the corresponding Hilbert space dimension.

    """
    keys = list(pauli_dict.keys())

    # Validate that all keys are even in Pauli matrices
    for i in range(len(keys)):
        if len(keys[i]) % 2 != 0:
            raise ValueError(
                "All Pauli keys must have an even number of single-qubit Pauli matrices "
                f"(got {len(keys[i])})."
            )
    # Validate that all keys have the same length
    for i in range(1, len(keys)):
        if len(keys[i - 1]) != len(keys[i]):
            raise ValueError(
                "All keys must have the same length "
                f"(got {len(keys[i - 1])} and {len(keys[i])})."
            )

    super_ops = []
    for label, coeff in pauli_dict.items():
        P = Pauli(label[::-1])
        super_ops.append(coeff * SuperOp(P.to_matrix()))
    return sum(super_ops)
