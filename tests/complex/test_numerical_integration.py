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

import math as m

import pytest
import qiskit as qi
from qiskit.circuit.random import random_circuit
from qiskit.quantum_info import Operator, process_fidelity

from spin_pulse import HardwareSpecs, PulseCircuit, Shape

one_qubit_gates = ["id", "x", "y", "z", "h", "s", "sdg", "t", "tdg", "sx", "sxdg"]
two_qubit_gates = ["cx", "cy", "cz", "ch", "cs", "csdg", "swap", "ecr", "dcx"]


params = [
    (B0, delta, J_coupling, rotation_shape, ramp_duration, coeff_duration)
    for (B0, delta, J_coupling, rotation_shape, ramp_duration, coeff_duration) in [
        (0.2, 0.5, 0.2, Shape.GAUSSIAN, 2, 7),
        (110, 120, 10, Shape.SQUARE, 5, 0),
        (0.01, 0.01, 0.2, Shape.GAUSSIAN, 0, 7),
        (1000, 10000, 10, Shape.SQUARE, 5, 7),
    ]
]

all_params = [
    (gate, B0, delta, J_coupling, rotation_shape, ramp_duration, coeff_duration)
    for gate in one_qubit_gates
    for (B0, delta, J_coupling, rotation_shape, ramp_duration, coeff_duration) in params
]


@pytest.mark.parametrize(
    "gate, B0, delta, J_coupling, rotation_shape, ramp_duration, coeff_duration",
    all_params,
)
def test_singlequbit_gate(
    gate, B0, delta, J_coupling, rotation_shape, ramp_duration, coeff_duration
):
    n_qb = 1
    qreg = qi.QuantumRegister(n_qb)
    circ = qi.QuantumCircuit(qreg)

    hardware_specs = HardwareSpecs(
        num_qubits=n_qb,
        B_field=B0,
        delta=delta,
        J_coupling=J_coupling,
        rotation_shape=rotation_shape,
        ramp_duration=ramp_duration,
        coeff_duration=coeff_duration,
    )

    getattr(circ, gate)(0)

    isa_circ = hardware_specs.gate_transpile(circ)
    pulse_circuit = PulseCircuit.from_circuit(isa_circ, hardware_specs)
    implemented_circ = pulse_circuit.to_circuit()

    assert m.isclose(
        process_fidelity(
            Operator.from_circuit(circ), Operator.from_circuit(implemented_circ)
        ),
        1,
    )


params = [
    (B0, delta, J_coupling, rotation_shape, ramp_duration, coeff_duration)
    for (B0, delta, J_coupling, rotation_shape, ramp_duration, coeff_duration) in [
        (0.2, 0.5, 0.2, Shape.GAUSSIAN, 2, 7),
        (0.2, 0.5, 10, Shape.SQUARE, 5, 0),
        (0.2, 0.5, 1000, Shape.SQUARE, 5, 0),
        (0.2, 0.5, 0.01, Shape.SQUARE, 5, 0),
    ]
]

all_params = [
    (gate, B0, delta, J_coupling, rotation_shape, ramp_duration, coeff_duration)
    for gate in one_qubit_gates
    for (B0, delta, J_coupling, rotation_shape, ramp_duration, coeff_duration) in params
]


@pytest.mark.parametrize(
    "gate, B0, delta, J_coupling, rotation_shape, ramp_duration, coeff_duration",
    all_params,
)
def test_twoqubit_gate(
    gate, B0, delta, J_coupling, rotation_shape, ramp_duration, coeff_duration
):
    n_qb = 1
    qreg = qi.QuantumRegister(n_qb)
    circ = qi.QuantumCircuit(qreg)

    hardware_specs = HardwareSpecs(
        num_qubits=n_qb,
        B_field=B0,
        delta=delta,
        J_coupling=J_coupling,
        rotation_shape=rotation_shape,
        ramp_duration=ramp_duration,
        coeff_duration=coeff_duration,
    )

    getattr(circ, gate)(0)

    isa_circ = hardware_specs.gate_transpile(circ)
    pulse_circuit = PulseCircuit.from_circuit(isa_circ, hardware_specs)
    implemented_circ = pulse_circuit.to_circuit()

    assert m.isclose(
        process_fidelity(
            Operator.from_circuit(circ), Operator.from_circuit(implemented_circ)
        ),
        1,
    )


@pytest.mark.parametrize(
    "n_qb, depth",
    [
        (1, 2),
        (1, 10),
        (5, 2),
    ],
)
def test_random_circuit(n_qb, depth):
    circ = random_circuit(n_qb, depth, measure=False)

    B0, delta, J_coupling, rotation_shape, ramp_duration, coeff_duration = (
        5,
        1,
        10,
        Shape.GAUSSIAN,
        5,
        7,
    )
    hardware_specs = HardwareSpecs(
        num_qubits=n_qb,
        B_field=B0,
        delta=delta,
        J_coupling=J_coupling,
        rotation_shape=rotation_shape,
        ramp_duration=ramp_duration,
        coeff_duration=coeff_duration,
    )

    isa_circ = hardware_specs.gate_transpile(circ)

    assert m.isclose(
        process_fidelity(Operator.from_circuit(circ), Operator.from_circuit(isa_circ)),
        1,
    )


@pytest.mark.parametrize(
    "B0, delta, J_coupling, rotation_shape, ramp_duration, coeff_duration",
    params,
)
@pytest.mark.parametrize("angle", [m.pi, m.pi / 2, m.pi / 8])
def test_rzz_from_circuit_matches_gate_transpiled_flow(
    B0, delta, J_coupling, rotation_shape, ramp_duration, coeff_duration, angle
):
    """``PulseCircuit.from_circuit`` on a native-basis circuit must give the same
    implemented unitary as the documented ``gate_transpile`` -> ``from_circuit`` flow.

    Regression test: ``from_circuit`` now applies ``RZZEchoPass`` itself, so an
    ``rzz`` gate fed in directly (without ``gate_transpile``) is echoed -- and the X
    echo cancels the large spurious single-qubit Z (Stark) phase the detuning pulses
    introduce. Without the echo the direct path and the documented path disagree
    badly (the direct one has fidelity well below 1 against the ideal gate).
    """
    direct_circ = qi.QuantumCircuit(2)
    direct_circ.rzz(angle, 0, 1)

    hardware_specs = HardwareSpecs(
        num_qubits=2,
        B_field=B0,
        delta=delta,
        J_coupling=J_coupling,
        rotation_shape=rotation_shape,
        ramp_duration=ramp_duration,
        coeff_duration=coeff_duration,
    )

    # direct: no gate_transpile() -- the circuit is already in the native basis
    direct_impl = PulseCircuit.from_circuit(direct_circ, hardware_specs).to_circuit()
    # documented flow: gate_transpile() (which echoes), then from_circuit()
    isa_circ = hardware_specs.gate_transpile(direct_circ)
    flow_impl = PulseCircuit.from_circuit(isa_circ, hardware_specs).to_circuit()

    assert m.isclose(
        process_fidelity(
            Operator.from_circuit(direct_impl), Operator.from_circuit(flow_impl)
        ),
        1,
        abs_tol=1e-6,
    )


def test_rzz_echo_pass_is_idempotent():
    """Running ``RZZEchoPass`` twice must not double the echo (``from_circuit`` may run
    it after ``gate_transpile`` already did)."""
    from qiskit.converters import circuit_to_dag, dag_to_circuit

    from spin_pulse.transpilation.passes.rzz_echo import RZZEchoPass

    circ = qi.QuantumCircuit(2)
    circ.rzz(m.pi / 3, 0, 1)

    once = dag_to_circuit(RZZEchoPass().run(circuit_to_dag(circ)))
    twice = dag_to_circuit(RZZEchoPass().run(circuit_to_dag(once)))

    assert once.count_ops() == twice.count_ops()
    assert m.isclose(
        process_fidelity(Operator.from_circuit(once), Operator.from_circuit(twice)), 1
    )
    # and the marker did not leak back onto the original circuit
    assert "spin_pulse_rzz_echoed" not in (circ.metadata or {})
