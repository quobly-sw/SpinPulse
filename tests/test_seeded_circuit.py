import pytest
from numpy import array_equal
from qiskit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit.primitives import StatevectorSampler
from qiskit.transpiler import CouplingMap

from spin_pulse import ExperimentalEnvironment, HardwareSpecs, PulseCircuit, Shape
from spin_pulse.environment.noise import NoiseType


@pytest.mark.parametrize(
    "duration, segment_duration,noise",
    [
        (5000, 1000, NoiseType.PINK),
        (4000, 1, NoiseType.WHITE),
        (5000, 1000, NoiseType.QUASISTATIC),
    ],
)
def test_seeded_circuit(duration, segment_duration, noise):
    ### Generating ENV
    B0, delta, J_coupling = 0.3, 0.3, 0.03
    duration = duration
    ramp_dur = 5
    J_coupling = 0.005
    T2S = 1_000_000
    TJS = 500

    hardware_specs = HardwareSpecs(
        num_qubits=5,
        B_field=B0,
        delta=delta,
        J_coupling=J_coupling,
        rotation_shape=Shape.SQUARE,
        ramp_duration=ramp_dur,
    )

    exp_env1 = ExperimentalEnvironment(
        hardware_specs=hardware_specs,
        noise_type=noise,
        T2S=T2S,
        duration=duration,
        segment_duration=segment_duration,
        TJS=TJS,
        seed=100,
    )

    exp_env2 = ExperimentalEnvironment(
        hardware_specs=hardware_specs,
        noise_type=noise,
        T2S=T2S,
        duration=duration,
        segment_duration=segment_duration,
        TJS=TJS,
        seed=100,
    )

    ### Testing time trace (across fresh instances with same seed)
    assert len(exp_env1.time_traces_coupling) == len(exp_env2.time_traces_coupling)
    assert len(exp_env1.time_traces) == len(exp_env2.time_traces)

    for tt_1, tt_2 in zip(exp_env1.time_traces, exp_env2.time_traces):
        assert array_equal(tt_1.values, tt_2.values)

    for tt_1, tt_2 in zip(exp_env1.time_traces_coupling, exp_env2.time_traces_coupling):
        assert array_equal(tt_1.values, tt_2.values)

    ### Testing no seed collision across qubits within a single instance
    qubit_values = [tt.values for tt in exp_env1.time_traces]
    for i in range(len(qubit_values)):
        for j in range(i + 1, len(qubit_values)):
            assert not array_equal(qubit_values[i], qubit_values[j]), (
                f"Qubit {i} and qubit {j} time traces are identical: seed collision across qubits."
            )

    coupling_values = [tt.values for tt in exp_env1.time_traces_coupling]
    for i in range(len(coupling_values)):
        for j in range(i + 1, len(coupling_values)):
            assert not array_equal(coupling_values[i], coupling_values[j]), (
                f"Coupling pair {i} and {j} time traces are identical: seed collision across coupling pairs."
            )

    ### Testing no collision between qubit traces and coupling traces
    for i, qv in enumerate(qubit_values):
        for j, cv in enumerate(coupling_values):
            if qv.shape == cv.shape:
                assert not array_equal(qv, cv), (
                    f"Qubit {i} trace collides with coupling trace {j}."
                )

    ### Testing from_circuit
    circuit = QuantumCircuit(4)
    circuit.rx(3.14, 0)
    circuit.cx(0, 1)
    circuit.cx(1, 2)
    circuit.cx(2, 3)
    circuit = transpile(
        circuits=circuit,
        seed_transpiler=100,
        optimization_level=0,
        basis_gates=["rx", "rz", "ry", "rzz"],
    )

    c_1 = PulseCircuit.from_circuit(
        circuit, hardware_specs=hardware_specs, exp_env=exp_env1
    ).to_circuit()
    c_2 = PulseCircuit.from_circuit(
        circuit, hardware_specs=hardware_specs, exp_env=exp_env2
    ).to_circuit()

    for g_1, g_2 in zip(c_1.data, c_2.data):  # We check gate matrix are equal
        assert (g_1.matrix == g_2.matrix).all()

    ### Testing simulation
    simu: StatevectorSampler = StatevectorSampler(seed=1000)
    c_1.measure_all()
    c_2.measure_all()
    r1 = list(
        next(iter(simu.run(pubs=[c_1], shots=1000).result()[0].data.values()))
        .get_counts()
        .values()
    )
    r2 = list(
        next(iter(simu.run(pubs=[c_2], shots=1000).result()[0].data.values()))
        .get_counts()
        .values()
    )

    assert array_equal(r1, r2)

    ### Calling generate_time_traces() twice on the same seeded instance must NOT silently reproduce the same traces
    prev_traces = [tt.values.copy() for tt in exp_env1.time_traces]
    prev_coupling = [tt.values.copy() for tt in exp_env1.time_traces_coupling]

    exp_env1.generate_time_traces(n_jobs=1)

    for prev, new in zip(prev_traces, exp_env1.time_traces):
        assert not array_equal(prev, new.values), (
            "Qubit time traces are identical across successive generate_time_traces() calls on the same instance: seed sequence is not advancing."
        )

    for prev, new in zip(prev_coupling, exp_env1.time_traces_coupling):
        assert not array_equal(prev, new.values), (
            "Coupling time traces are identical across successive generate_time_traces() calls on the same instance: seed sequence is not advancing."
        )


def test_bistring_conversion_smaller_qubit_than_qpu_max():
    ### Generating ENV
    B0, delta, J_coupling = 0.3, 0.3, 0.03
    ramp_dur = 5
    J_coupling = 0.005

    coupling: list[list[int]] = []
    for qubit in range(10 - 1):
        coupling.append([qubit, qubit + 1])

    c_m: CouplingMap = CouplingMap(coupling)

    hardware_specs = HardwareSpecs(
        num_qubits=10,
        B_field=B0,
        delta=delta,
        J_coupling=J_coupling,
        rotation_shape=Shape.SQUARE,
        ramp_duration=ramp_dur,
    )

    # Define a quantum circuit with two parameters.
    circuit = QuantumCircuit(3)
    circuit.h(0)
    circuit.cx(0, 2)
    circuit.measure_all()
    circuit = transpile(
        circuits=circuit,
        seed_transpiler=100,
        optimization_level=0,
        basis_gates=["rx", "rz", "ry", "rzz"],
        coupling_map=c_m,
    )
    p_circ = PulseCircuit.from_circuit(circuit, hardware_specs=hardware_specs)
    assert p_circ.get_logical_bitstring("00010") == "00100"


def test_bistring_conversion_user_warning():
    ### Generating ENV
    B0, delta, J_coupling = 0.3, 0.3, 0.03
    ramp_dur = 5
    J_coupling = 0.005

    hardware_specs = HardwareSpecs(
        num_qubits=10,
        B_field=B0,
        delta=delta,
        J_coupling=J_coupling,
        rotation_shape=Shape.SQUARE,
        ramp_duration=ramp_dur,
    )

    # Define a quantum circuit with two parameters.
    circuit = QuantumCircuit(3)
    circuit.h(0)
    circuit.cx(0, 2)
    circuit.measure_all()
    circuit = transpile(
        circuits=circuit,
        seed_transpiler=100,
        optimization_level=0,
        basis_gates=["rx", "rz", "ry", "rzz"],
    )
    p_circ = PulseCircuit.from_circuit(circuit, hardware_specs=hardware_specs)
    with pytest.warns(UserWarning):
        assert p_circ.get_logical_bitstring("00010") == "00010"
