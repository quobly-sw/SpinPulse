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
"""Layer of pulses applied to the target qubits simultaneously."""

import matplotlib.pyplot as plt
import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate

from .hardware_specs import HardwareSpecs
from .instructions import IdleInstruction
from .pulse_sequence import PulseSequence
from .utils import gate_to_pulse_sequences, propagate


class PulseLayer:
    """Layer of one- and two-qubit gate pulse sequences.

    A PulseLayer groups single-qubit and two-qubit PulseSequence objects that
    act over a common time interval. Pulse sequences in the layer are eventually
    padded with idle instructions such that all of them share the same duration. The
    layer keeps track of which qubits are driven by one-qubit or two-qubit
    pulses and which qubits remain idle.

    Attributes:
        - duration (int): Duration of the layer, which is the maximal duration
            of the pulse sequences given.
        - qubits (list[qiskit.circuit.Qubit]): Ordered list of qubits included in the layer.
        - num_qubits (int): Number of qubits in the layer.
        - oneq_pulse_sequences (list[PulseSequence]): One-qubit pulse sequence
            assigned to each qubit. For qubits without an explicit drive,
            an idle-only PulseSequence of length duration is created.
        - twoq_pulse_sequences (list[PulseSequence]): Two-qubit pulse sequences
            acting on pair of qubits in the layer.
        - pulse_sequences (list[PulseSequence]): Concatenation of all one- and
            two-qubit pulse sequences in the layer.
        - n_pulses (int): Total number of pulse sequences in the layer.
        - qubits_oneq_active (list[qiskit.circuit.Qubit]): Qubits that are manipulated during the layer
            through single-qubit gate.
        - qubits_twoq_active (list[qiskit.circuit.Qubit]): Qubits that are manipulated during the layer
            through two-qubit gate.
        - qubits_idle (list[qiskit.circuit.Qubit]): Qubits that are idle during this layer.

    """

    def __init__(
        self,
        qubits: list,
        oneq_pulse_sequences: list,
        twoq_pulse_sequences: list,
    ):
        """
        Initialize the PulseLayer from lists of pulse sequences and qubits.

        Parameters:
            qubits (list[qiskit.circuit.Qubit]): Ordered list of qubits included in the layer.
            oneq_pulse_sequences (list[PulseSequence]): One-qubit pulse sequences
                to be applied in the layer.
            twoq_pulse_sequences (list[PulseSequence]): Two-qubit pulse sequences
                to be applied in the layer.

        """

        durations = [_.duration for _ in oneq_pulse_sequences + twoq_pulse_sequences]
        duration = max(durations)
        ##Adjust durations accros the layer by adding idle instructions
        for sequence in oneq_pulse_sequences:
            sequence.adjust_duration(duration)
        for sequence in twoq_pulse_sequences:
            sequence.adjust_duration(duration)

        self.duration = duration

        self.oneq_pulse_sequences = oneq_pulse_sequences
        self.twoq_pulse_sequences = twoq_pulse_sequences
        self.pulse_sequences = oneq_pulse_sequences + twoq_pulse_sequences
        self.qubits = qubits
        self.n_pulses: int = len(self.pulse_sequences)
        self.num_qubits: int = len(qubits)

        self.qubits_oneq_active = []
        self.qubits_twoq_active = []
        self.qubits_idle = []

        self.oneq_pulse_sequences = [0] * self.num_qubits
        for _ in oneq_pulse_sequences:
            if _.name != "delay":
                self.qubits_oneq_active += _.qubits
            else:
                self.qubits_idle += _.qubits
            self.oneq_pulse_sequences[_.qubits[0]._index] = _

        for _ in twoq_pulse_sequences:
            self.qubits_twoq_active += _.qubits

        qubits_forgotten = [
            q
            for q in self.qubits
            if q not in self.qubits_oneq_active + self.qubits_idle
        ]
        self.qubits_idle += qubits_forgotten
        # We add IdleInstructions so that all qubits are exactly subject to a PulseSequence
        for qubit in qubits_forgotten:
            self.oneq_pulse_sequences[qubit._index] = PulseSequence(
                [IdleInstruction([qubit], duration=self.duration)]
            )

    @classmethod
    def from_circuit_layer(
        cls,
        qubits: list,
        circuit_layer,
        hardware_specs: HardwareSpecs,
    ):
        """Construct a PulseLayer from a circuit layer.

        This method converts each gate in a circuit layer into a
        PulseSequence object according to the hardware specifications. Single-
        and two-qubit gates are translated separately, and the resulting pulse
        sequences are grouped into a PulseLayer.

        Parameters:
            qubits (list[qiskit.circuit.Qubit]): Ordered list of qubits included in the layer.
            circuit_layer (QuantumCircuit): QuantumCircuit instance representing one layer of a bigger circuit. Its instructions are passed to
                ``gate_to_pulse_sequences`` to be converted into pulses.
            hardware_specs (HardwareSpecs): HardwareSpecs class instance that defines
                the hardware specifications.

        Returns:
            PulseLayer: A PulseLayer containing the pulse sequences that corresponds to the circuit_layer.

        """

        oneq_pulse_sequences = []
        twoq_pulse_sequences = []
        for data in circuit_layer.data:
            oneq_, twoq_ = gate_to_pulse_sequences(data, hardware_specs)
            oneq_pulse_sequences += oneq_
            twoq_pulse_sequences += twoq_
        return cls(qubits, oneq_pulse_sequences, twoq_pulse_sequences)

    def __str__(self):
        """Return a string representation of the pulse layer.

        Includes:
            Duration of the layer

        """
        return f"Pulse Layer of duration={self.duration}"

    def plot(self, axs=None, label_gates: bool | str = True):
        """Plot all single-qubit and two-qubit pulse sequences in the layer.

        For each qubit, the corresponding one-qubit PulseSequence is displayed on
        its own axis. Two-qubit PulseSequences are plotted on intermediate axes
        between the qubits they act on. If no axis array is provided, a new
        matplotlib figure is created.

        Parameters:
            axs (list[Axes] | None): Array of axes on which to draw the sequences.
                If None, a new figure and axis array is created.
            label_gates (bool | str): Controls gate labelling. If True, each pulse
                is annotated with a default gate label. If a string, the value is
                forwarded to the underlying ``PulseSequence.plot`` methods to
                customize labelling.

        """
        if axs is None:
            _, axs = plt.subplots(ncols=1, nrows=2 * self.num_qubits - 1)
            for ax in axs.flat:
                ax.tick_params(
                    left=False, bottom=False, labelleft=False, labelbottom=False
                )
        for i in range(self.num_qubits):
            self.oneq_pulse_sequences[i].plot(ax=axs[2 * i], label_gates=label_gates)

        for seq in self.twoq_pulse_sequences:
            i = seq.qubits[0]._index
            seq.plot(ax=axs[2 * i + 1], label_gates=label_gates)

    def to_circuit(self):
        """Convert the pulse layer into an equivalent qiskit.QuantumCircuit.

        Each PulseSequence is translated into a unitary matrix by propagating the
        corresponding Hamiltonian terms over the layer duration. Two-qubit
        sequences generate 4x4 unitaries, while single-qubit sequences generate
        2x2 unitaries. The resulting unitaries are appended to a new qiskit.QuantumCircuit
        in an order consistent with the layer structure.

        Returns:
            qiskit.QuantumCircuit: A circuit representation of the pulse layer, where UnitaryGate
                corresponds to a pulse sequence.

        """
        circ = QuantumCircuit(self.qubits)
        indices_treated = []
        for pulse_sequence in self.twoq_pulse_sequences:
            qubits = pulse_sequence.qubits
            H, coeff = pulse_sequence.to_hamiltonian()
            ## Adding the corresponding one qubit hamiltonians
            H0, coeff0 = self.oneq_pulse_sequences[qubits[0]._index].to_hamiltonian()
            #  Little-endian convention: qubits[0] is the least significant (I \otimes H0)
            H0 = np.einsum("dij,ab->daibj", H0, np.eye(2)).reshape(
                H0.shape[0], H0.shape[1] * 2, H0.shape[2] * 2
            )
            H1, coeff1 = self.oneq_pulse_sequences[qubits[1]._index].to_hamiltonian()
            H1 = np.einsum("dij,ab->diajb", H1, np.eye(2)).reshape(
                H1.shape[0], H1.shape[1] * 2, H1.shape[2] * 2
            )
            H = np.concatenate((H, H0, H1), axis=0)
            coeff = np.concatenate((coeff, coeff0, coeff1), axis=0)

            indices_treated += [qubits[0]._index, qubits[1]._index]
            matrix = propagate(H, coeff)
            instruction = UnitaryGate(matrix, "2q_" + f"{self.duration}")
            circ.append(instruction, qubits)

        for i in range(self.num_qubits):
            if i not in indices_treated:
                H, coeff = self.oneq_pulse_sequences[i].to_hamiltonian()
                matrix = propagate(H, coeff)
                instruction = UnitaryGate(matrix, "1q_" + f"{self.duration}")
                circ.append(instruction, [self.qubits[i]])

        return circ

    def attach_dynamical_decoupling(self, hardware_specs: HardwareSpecs) -> None:
        """Apply a dynamical decoupling sequence to all one-qubit pulse sequences
        if possible.

        If a dynamical decoupling mode is specified in the hardware
        specifications, this method transforms each one-qubit PulseSequence
        in the layer with a new sequence obtained by applying a dynamical decoupling
        transformation according to the specified hardware parameters. Two-qubit
        sequences are left unchanged.

        Parameters:
            hardware_specs (HardwareSpecs): Hardware configuration specifying
                the dynamical decoupling mode and available pulse shapes.

        """

        for i in range(len(self.oneq_pulse_sequences)):
            self.oneq_pulse_sequences[i] = self.oneq_pulse_sequences[
                i
            ].to_dynamical_decoupling(hardware_specs)
