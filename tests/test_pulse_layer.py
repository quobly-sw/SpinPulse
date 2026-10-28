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
import pytest
import qiskit as qi
from qiskit.quantum_info import Operator

from spin_pulse import HardwareSpecs, Shape
from spin_pulse.transpilation.instructions import (
    IdleInstruction,
    SquareRotationInstruction,
)
from spin_pulse.transpilation.pulse_layer import PulseLayer
from spin_pulse.transpilation.pulse_sequence import PulseSequence


def make_hardware(num_qubits):
    # Create a minimal HardwareSpecs instance.
    B_field, delta, J_coupling = 0.5, 0.2, 0.01
    ramp_duration = 5
    return HardwareSpecs(
        num_qubits, B_field, delta, J_coupling, Shape.GAUSSIAN, ramp_duration, optim=3
    )


def make_single_qubit_layer(qubits):
    # Return a simple one-qubit PulseSequence of fixed duration.
    instr = IdleInstruction([qubits[0]], duration=10)
    return [PulseSequence([instr])]


def make_two_qubit_layer(qubits):
    # Return a simple two-qubit PulseSequence.
    instr = IdleInstruction(qubits, duration=12)
    return [PulseSequence([instr])]


def test_pulse_layer_init():
    qreg = qi.QuantumRegister(2)
    qubits = list(qreg)

    oneq = make_single_qubit_layer(qubits)
    twoq = make_two_qubit_layer(qubits)

    pl = PulseLayer(qubits, oneq, twoq)

    assert pl.duration == 12  # max of (10,12)
    assert pl.num_qubits == 2
    assert len(pl.oneq_pulse_sequences) == 2
    assert len(pl.twoq_pulse_sequences) == 1


def test_pulse_layer_str():
    qreg = qi.QuantumRegister(1)
    qubits = list(qreg)

    oneq = make_single_qubit_layer(qubits)
    pl = PulseLayer(qubits, oneq, [])

    s = str(pl)
    assert "Pulse Layer" in s
    assert str(pl.duration) in s


@pytest.mark.parametrize(
    "num_qubits, gate, angle, depth",
    [
        (3, "rx", np.pi / 2, 2),
        (5, "ry", np.pi / 2, 3),
        (5, "rz", np.pi / 2, 3),
        (5, "rzz", np.pi / 2, 1),
    ],
)
def test_from_circuit_layer_runs(num_qubits, gate, angle, depth):
    hardware = make_hardware(num_qubits)

    qreg = qi.QuantumRegister(num_qubits)
    circ = qi.QuantumCircuit(qreg)
    for _ in range(depth):
        if gate == "rzz":
            getattr(circ, gate)(angle, 0, 1)
        else:
            getattr(circ, gate)(angle, 0)

    layer = PulseLayer.from_circuit_layer(list(qreg), circ, hardware)

    assert isinstance(layer, PulseLayer)
    assert layer.duration > 0
    assert len(layer.oneq_pulse_sequences) == num_qubits
    if gate == "rzz":
        assert len(layer.twoq_pulse_sequences) == 1
    else:
        assert len(layer.twoq_pulse_sequences) == 0


def test_plot_runs():
    qreg = qi.QuantumRegister(2)
    qubits = list(qreg)

    oneq = make_single_qubit_layer(qubits)
    twoq = make_two_qubit_layer(qubits)

    pl = PulseLayer(qubits, oneq, twoq)

    pl.plot()  # must not raise
    plt.close("all")


def test_to_circuit_builds_valid_qiskit_circuit():
    qreg = qi.QuantumRegister(2)
    qubits = list(qreg)

    oneq = make_single_qubit_layer(qubits)
    twoq = make_two_qubit_layer(qubits)

    pl = PulseLayer(qubits, oneq, twoq)
    qc = pl.to_circuit()

    assert isinstance(qc, qi.QuantumCircuit)
    assert qc.num_qubits == 2
    print(qc.data)
    assert len(qc.data) == 1


def test_attach_dynamical_decoupling_runs():
    num_qubits = 1
    hardware = make_hardware(num_qubits)

    qreg = qi.QuantumRegister(num_qubits)
    qubits = list(qreg)

    oneq = make_single_qubit_layer(qubits)
    pl = PulseLayer(qubits, oneq, [])

    pl.attach_dynamical_decoupling(hardware)  # should not crash


def make_z_sequence(qubit, hardware, duration=21):
    # Return a one-qubit Z PulseSequence and its rotation angle.
    instr = SquareRotationInstruction(
        "z", [qubit], hardware.fields["z"], 1.0, hardware.ramp_duration, duration
    )
    return PulseSequence([instr]), instr.to_angle()


def make_idle_sequence(qubits, duration=21):
    # Return an idle PulseSequence on the given qubits.
    return PulseSequence([IdleInstruction(qubits, duration=duration)])


@pytest.mark.parametrize("reverse_operands", [False, True])
def test_to_circuit_two_qubit_layer_local_term_on_right_qubit(reverse_operands):
    """With a zero-coupling two-qubit sequence and a Z pulse on qubit 0 only, to_circuit
    must equal rz(angle) on qubit 0.
    """

    hardware = make_hardware(2)
    qreg = qi.QuantumRegister(2)
    qubits = list(qreg)
    pair = qubits[::-1] if reverse_operands else qubits

    z_seq, angle = make_z_sequence(qubits[0], hardware)
    oneq = [z_seq, make_idle_sequence([qubits[1]])]
    twoq = [make_idle_sequence(pair)]
    pl = PulseLayer(qubits, oneq, twoq)

    expected = qi.QuantumCircuit(qreg)
    expected.rz(angle, 0)

    assert Operator(pl.to_circuit()).equiv(Operator(expected))


@pytest.mark.parametrize("reverse_operands", [False, True])
def test_to_circuit_two_qubit_layer_different_pulses_on_both_qubits(reverse_operands):
    """With a zero-coupling two-qubit sequence and Z pulses of different angles on
    qubits 0 and 1, to_circuit must apply each rz to its own qubit.
    """

    hardware = make_hardware(2)
    qreg = qi.QuantumRegister(2)
    qubits = list(qreg)
    pair = qubits[::-1] if reverse_operands else qubits

    z_seq_0, angle_0 = make_z_sequence(qubits[0], hardware, duration=21)
    z_seq_1, angle_1 = make_z_sequence(qubits[1], hardware, duration=31)
    oneq = [z_seq_0, z_seq_1]
    twoq = [make_idle_sequence(pair, duration=31)]
    pl = PulseLayer(qubits, oneq, twoq)

    expected = qi.QuantumCircuit(qreg)
    expected.rz(angle_0, 0)
    expected.rz(angle_1, 1)

    unitary = Operator(pl.to_circuit())
    assert unitary.equiv(Operator(expected))
    assert not unitary.equiv(Operator(expected.reverse_bits()))


@pytest.mark.parametrize("reverse_operands", [False, True])
def test_to_circuit_two_qubit_layer_inside_larger_register(reverse_operands):
    """With a zero-coupling two-qubit sequence on qubits 1 and 2 of a 3-qubit register,
    Z pulses on qubits 0 and 2 must give rz on qubits 0 and 2.
    """

    hardware = make_hardware(3)
    qreg = qi.QuantumRegister(3)
    qubits = list(qreg)
    pair = [qubits[2], qubits[1]] if reverse_operands else [qubits[1], qubits[2]]

    z_seq_0, angle_0 = make_z_sequence(qubits[0], hardware, duration=31)
    z_seq_2, angle_2 = make_z_sequence(qubits[2], hardware, duration=21)
    oneq = [z_seq_0, make_idle_sequence([qubits[1]]), z_seq_2]
    twoq = [make_idle_sequence(pair)]
    pl = PulseLayer(qubits, oneq, twoq)

    expected = qi.QuantumCircuit(qreg)
    expected.rz(angle_0, 0)
    expected.rz(angle_2, 2)

    assert Operator(pl.to_circuit()).equiv(Operator(expected))
