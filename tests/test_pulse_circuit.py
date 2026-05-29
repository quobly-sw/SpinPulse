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

import warnings
from math import isclose
from unittest.mock import ANY, MagicMock, patch

import numpy as np
import pytest
from qiskit.circuit import Barrier, Measure, QuantumCircuit

import tests.fixtures.dummy_objects as dm
from spin_pulse import DynamicalDecoupling, PulseCircuit
from spin_pulse.transpilation.pulse_circuit import IdleInstruction

#
# -----------------------
# Tests for __init__, assign_starting_times, attach_dynamical_decoupling in constructor
# -----------------------
#


def test_init_and_basic_properties(two_qubits, pulse_layers):
    # Covers:
    # - __init__
    # - assign_starting_times()
    # - attach_dynamical_decoupling() (no DD mode)
    # - attach_time_traces(exp_env=None)

    hw = dm.DummyHardwareSpecs(dynamical_decoupling=None)
    qc = QuantumCircuit(len(two_qubits))
    pc = PulseCircuit(qc, two_qubits, pulse_layers, hardware_specs=hw, exp_env=None)

    # Check core properties
    assert pc.original_circ is qc
    assert pc.num_qubits == 2
    assert pc.qubits is two_qubits
    assert pc.pulse_layers == pulse_layers
    assert pc.n_layers == 2
    assert pc.duration == 12  # duration = 5 + 7

    # assign_starting_times() should set .t_start on each layer
    assert pulse_layers[0].t_start == 0
    assert pulse_layers[1].t_start == 5

    # t_lab should be initialized to 0
    assert pc.t_lab == 0

    # since exp_env=None, attach_time_traces() should just set no attributes like time_traces
    assert not hasattr(pc, "time_traces")


def test_init_with_dynamical_decoupling_applies_to_layers(two_qubits):
    # make our own layers so we can inspect .applied_dd
    layerA = dm.DummyPulseLayer(two_qubits, duration=4)
    layerB = dm.DummyPulseLayer(two_qubits, duration=6)
    hw = dm.DummyHardwareSpecs(dynamical_decoupling=DynamicalDecoupling.FULL_DRIVE)
    qc = QuantumCircuit(len(two_qubits))

    pc = PulseCircuit(qc, two_qubits, [layerA, layerB], hw, exp_env=None)

    assert getattr(layerA, "applied_dd", None) == DynamicalDecoupling.FULL_DRIVE
    assert getattr(layerB, "applied_dd", None) == DynamicalDecoupling.FULL_DRIVE
    assert pc.duration == 10
    assert pc.n_layers == 2


# -----------------------
# Test from_circuit() classmethod
# -----------------------


def test_from_circuit_builds_layers_and_calls_constructor():
    hw = dm.DummyHardwareSpecs(dynamical_decoupling=None, J_coupling=1)
    exp_env = dm.DummyExpEnv(duration=100, hardware_specs=hw)

    fake_circ = QuantumCircuit(2)
    two_qubits = fake_circ.qubits

    # Fake DAG and its .layers() iterator
    fake_layer_graph = MagicMock()
    fake_layer = {"graph": fake_layer_graph}

    fake_dag = MagicMock()
    N = 4
    fake_dag.layers.return_value = [fake_layer] * N

    dummy_layer_obj = dm.DummyPulseLayer(two_qubits, duration=3)

    class Operation:
        name = "gate"

    class Gate:
        operation: Operation = Operation()

    class LayerAsCirc(str):
        data: list = [Gate()]

    with (
        patch(
            "spin_pulse.transpilation.pulse_circuit.circuit_to_dag",
            return_value=fake_dag,
        ),
        patch(
            "spin_pulse.transpilation.pulse_circuit.dag_to_circuit",
            return_value=LayerAsCirc("layer_as_circ"),
        ),
        patch(
            "spin_pulse.transpilation.pulse_circuit.PulseLayer.from_circuit_layer",
            return_value=dummy_layer_obj,
        ) as mock_from_layer,
        # patch("spin_pulse.transpilation.pulse_circuit.PulseCircuit") as mock_pc,
    ):
        # PulseCircuit( fake_circ, two_qubits, fake_dag, hw, exp_env=None)

        PulseCircuit.from_circuit(fake_circ, hw, exp_env=exp_env)

    # Ensure we called from_circuit_layer for each layer in the DAG
    assert mock_from_layer.call_count == N


def test_from_circuit_strip_measurement():
    """We mock circuit_to_dag, dag.layers(), dag_to_circuit, and PulseLayer.from_circuit_layer
    to ensure from_circuit() iterates layers and returns PulseCircuit.
    """

    hw = dm.DummyHardwareSpecs(dynamical_decoupling=None, J_coupling=1)
    exp_env = dm.DummyExpEnv(duration=100, hardware_specs=hw)

    fake_circ = QuantumCircuit(2)
    two_qubits = fake_circ.qubits

    class Operation:
        name = "gate"

    class Gate:
        operation: Operation = Operation()

    class LayerAsCirc(str):
        data: list = [Gate()]

    # Fake DAG and its .layers() iterator
    fake_layer_graph = MagicMock()
    fake_layer = {"graph": fake_layer_graph}

    fake_dag = MagicMock()
    N = 4
    fake_dag.layers.return_value = [fake_layer] * N

    dummy_layer_obj = dm.DummyPulseLayer(two_qubits, duration=3)

    fake_circ = QuantumCircuit(2)
    fake_circ.rx(0.0, 0)
    fake_circ.rx(0.0, 1)
    fake_circ.measure_all()

    with (
        patch(
            "spin_pulse.transpilation.pulse_circuit.circuit_to_dag",
            return_value=fake_dag,
        ) as dag_func,
        patch(
            "spin_pulse.transpilation.pulse_circuit.dag_to_circuit",
            return_value=LayerAsCirc("layer_as_circ"),
        ),
        patch(
            "spin_pulse.transpilation.pulse_circuit.PulseLayer.from_circuit_layer",
            return_value=dummy_layer_obj,
        ),
    ):
        PulseCircuit.from_circuit(fake_circ, hw, exp_env=exp_env)

    assert all(
        not isinstance(x, Barrier | Measure)
        for x in dag_func.call_args_list[0].data.operation
    )


def test_empty_layer_graph():
    # Test if from_circuit pass empty graph.

    hw = dm.DummyHardwareSpecs(dynamical_decoupling=None, J_coupling=1)
    exp_env = dm.DummyExpEnv(duration=100, hardware_specs=hw)

    fake_circ = QuantumCircuit(2)
    two_qubits = fake_circ.qubits

    # Fake DAG and its .layers() iterator
    fake_layer_graph = MagicMock()
    fake_layer = {"graph": fake_layer_graph}

    fake_dag = MagicMock()

    fake_dag.layers.return_value = [fake_layer] * 4

    dummy_layer_obj = dm.DummyPulseLayer(two_qubits, duration=3)

    class Operation:
        name = "barrier"

    class Gate:
        operation: Operation = Operation()

    class LayerAsCirc(str):
        data: list = [Gate()]

    with (
        patch(
            "spin_pulse.transpilation.pulse_circuit.circuit_to_dag",
            return_value=fake_dag,
        ),
        patch(
            "spin_pulse.transpilation.pulse_circuit.dag_to_circuit",
            return_value=LayerAsCirc("layer_as_circ"),
        ),
        patch(
            "spin_pulse.transpilation.pulse_circuit.PulseLayer.from_circuit_layer",
            return_value=dummy_layer_obj,
        ) as mock_from_layer,
    ):
        PulseCircuit.from_circuit(fake_circ, hw, exp_env=exp_env)

    assert mock_from_layer.call_count == 0


#
# -----------------------
# Test __str__
# -----------------------
#


def test_str(two_qubits, pulse_layers):
    hw = dm.DummyHardwareSpecs()
    qc = QuantumCircuit(len(two_qubits))
    pc = PulseCircuit(qc, two_qubits, pulse_layers, hw, exp_env=None)
    s = str(pc)
    assert "PulseCircuit of duration=" in s
    assert f"{pc.duration}" in s
    assert f"{pc.n_layers}" in s


#
# -----------------------
# Test plot() branches:
#   - with hardware_specs=None
#   - with hardware_specs != None
# and we make sure nothing crashes
# -----------------------
#


def test_plot_without_hardware_specs(two_qubits, pulse_layers):
    hw = dm.DummyHardwareSpecs()
    qc = QuantumCircuit(len(two_qubits))

    pc = PulseCircuit(qc, two_qubits, pulse_layers, hw, exp_env=None)

    # Case 1: hardware_specs=None branch (ymax_qubit=0.5 etc.)
    pc.plot(hardware_specs=None, label_gates=False)


def test_plot_with_hardware_specs(two_qubits, pulse_layers):
    hw = dm.DummyHardwareSpecs()
    qc = QuantumCircuit(len(two_qubits))
    pc = PulseCircuit(qc, two_qubits, pulse_layers, hw, exp_env=None)

    # Case 2: hardware_specs provided
    pc.plot(hardware_specs=hw, label_gates=True)


#
# -----------------------
# Test to_circuit()
#   - measure_all=False
#   - measure_all=True
# -----------------------
#


def test_to_circuit_measure_and_no_measure(two_qubits):
    # build custom pulse layers so we control to_circuit() content
    layerA = dm.DummyPulseLayer(two_qubits, duration=4)
    layerB = dm.DummyPulseLayer(two_qubits, duration=6)

    hw = dm.DummyHardwareSpecs()
    qc = QuantumCircuit(len(two_qubits))
    pc = PulseCircuit(qc, two_qubits, [layerA, layerB], hw, exp_env=None)

    circ_nom = pc.to_circuit(measure_all=False)
    assert isinstance(circ_nom, QuantumCircuit)
    # total qubits should match
    assert circ_nom.num_qubits == len(two_qubits)

    circ_meas = pc.to_circuit(measure_all=True)
    # measure_all() should add classical bits equal to num_qubits
    assert circ_meas.num_clbits == len(two_qubits)


#
# -----------------------
# Test circuit_samples()
#   - exp_env is None  -> returns 1
#   - exp_env provided -> returns duration_ratio
# -----------------------
#


def test_circuit_samples(two_qubits, pulse_layers):
    hw = dm.DummyHardwareSpecs()
    qc = QuantumCircuit(len(two_qubits))
    pc = PulseCircuit(qc, two_qubits, pulse_layers, hw, exp_env=None)

    # exp_env None => 1
    assert pc.circuit_samples(None) == 1

    # exp_env with matching duration
    exp_env = dm.DummyExpEnv(duration=120, hardware_specs=hw)
    # pc.duration = 12 (5+7 from fixture pulse_layers)
    assert pc.circuit_samples(exp_env) == 120 // 12


# -------------------------------------------------------------------
# get_logical_bitstring()
# -------------------------------------------------------------------


def test_get_logical_bitstring_transpiled_layout(two_qubits, pulse_layers, hw_no_dd):
    qc = QuantumCircuit(len(two_qubits))
    # Fake a "transpiled layout" with final_index_layout()
    layout = MagicMock()
    layout.final_index_layout.return_value = [1, 0]  # swap 0<->1
    qc._layout = layout

    pc = PulseCircuit(qc, two_qubits, pulse_layers, hw_no_dd, exp_env=None)

    out = pc.get_logical_bitstring("10")  # physical "10"
    assert out == "01"


@pytest.mark.filterwarnings("default")
def test_get_logical_bitstring_no_layout(two_qubits, pulse_layers, hw_no_dd):
    qc = QuantumCircuit(len(two_qubits))
    qc._layout = None  # trigger fallback branch

    pc = PulseCircuit(qc, two_qubits, pulse_layers, hw_no_dd, exp_env=None)

    with warnings.catch_warnings(record=True) as w:
        out = pc.get_logical_bitstring("10")

    assert out == "10"

    assert any("not have a TranspileLayout nor a Layout" in str(wi.message) for wi in w)


def test_averaging_over_samples(two_qubits, pulse_layers, hw_no_dd):
    hw = dm.DummyHardwareSpecs()
    exp_env = dm.DummyExpEnv(duration=36, hardware_specs=hw, only_idle=True)
    qc = QuantumCircuit(len(two_qubits))
    pc = PulseCircuit(qc, two_qubits, pulse_layers, hw_no_dd, exp_env=None)

    with (
        patch("spin_pulse.transpilation.pulse_circuit.tqdm", side_effect=lambda x: x),
        patch.object(pc, "attach_time_traces") as mock_attach,
    ):

        def fake_eval(pulse_circ):
            # always return scalar 10.0
            return 10.0

        avg = pc.averaging_over_samples(fake_eval, exp_env)

        # avg should be dict with exactly 1 key, value ~10
        assert isinstance(avg, float)
        assert isclose(avg, 10.0)

        # attach_time_traces called once per sample
        # num_samples = 36 // 12 = 3
        assert mock_attach.call_count == 3

        # t_lab reset to 0 before loop, and mock doesn't change it
        assert pc.t_lab == 0


# -------------------------------------------------------------------
# fidelity(), mean_fidelity(), mean_channel()
# -------------------------------------------------------------------


def test_mean_fidelity_and_mean_channel_wrappers(two_qubits, pulse_layers, hw_no_dd):
    qc = QuantumCircuit(len(two_qubits))
    pc = PulseCircuit(qc, two_qubits, pulse_layers, hw_no_dd, exp_env=None)

    dummy_ref_circ = QuantumCircuit(2)

    with (
        patch(
            "spin_pulse.transpilation.pulse_circuit.average_gate_fidelity",
            return_value=0.9,
        ) as mock_fid,
        patch(
            "spin_pulse.transpilation.pulse_circuit.Operator.from_circuit"
        ) as mock_from_op,
        patch.object(pc, "averaging_over_samples", return_value=0.5) as mock_avg,
    ):
        # prepare Operator.from_circuit(...) return so fidelity() works
        mock_from_op.return_value = MagicMock()

        # fidelity()
        val_fid = pc.fidelity(dummy_ref_circ)
        assert val_fid == 0.9
        mock_fid.assert_called_once()

        # mean_fidelity()
        out_mean_fid = pc.mean_fidelity(exp_env="ENV", circ_ref=dummy_ref_circ)
        assert pytest.approx(out_mean_fid, rel=1e-12) == 0.5
        mock_avg.assert_any_call(ANY, "ENV")

        # mean_channel() -> lambda x: SuperOp(Operator.from_circuit(x.to_circuit()))
        out_channel = pc.mean_channel(exp_env="ENV2")
        assert pytest.approx(out_channel, rel=1e-12) == 0.5
        mock_avg.assert_any_call(ANY, "ENV2")

        assert mock_avg.call_count == 2


# -------------------------------------------------------------------
# attach_time_traces()
#   - too short (prints warning)
#   - normal: fills .time_traces, per-layer time_traces, and distort_factor
# -------------------------------------------------------------------


def test_attach_time_traces_too_short(two_qubits, pulse_layers, hw_no_dd):
    qc = QuantumCircuit(len(two_qubits))
    pc = PulseCircuit(qc, two_qubits, pulse_layers, hw_no_dd, exp_env=None)

    hw = dm.DummyHardwareSpecs()
    # Force a too-short env: exp_env.duration < pc.duration
    exp_env = dm.DummyExpEnv(duration=5, hardware_specs=hw, with_coupling=True)
    pc.t_lab = 0
    with pytest.warns(UserWarning):
        pc.attach_time_traces(exp_env)


def test_attach_time_traces_normal(hw_no_dd):
    # Build predictable layers so we can reason about indexing/slicing.
    # We'll do duration=4 then 6 => total=10
    qc = QuantumCircuit(2)
    two_qubits = qc.qubits
    layerA = dm.DummyPulseLayer(two_qubits, duration=4)
    layerB = dm.DummyPulseLayer(two_qubits, duration=6)
    pc = PulseCircuit(qc, two_qubits, [layerA, layerB], hw_no_dd, exp_env=None)

    hw = dm.DummyHardwareSpecs(J_coupling=1)
    # Now exp_env is long enough: duration >= pc.duration
    # We'll keep with_coupling=True so distort_factor branch runs.
    exp_env = dm.DummyExpEnv(duration=50, hardware_specs=hw, with_coupling=True)
    pc.t_lab = 0

    pc.attach_time_traces(exp_env)

    # After attach_time_traces with enough duration:
    # pc.time_traces should exist
    assert hasattr(pc, "time_traces")
    assert len(pc.time_traces) == len(exp_env.time_traces)

    # layerA / layerB should have .time_traces sliced correctly
    assert hasattr(layerA, "time_traces")
    assert hasattr(layerB, "time_traces")
    assert len(layerA.time_traces) == len(exp_env.time_traces)

    # For twoq_pulse_sequences, we expect distort_factor to be set
    # except when instruction is IdleInstruction.
    for layer in [layerA, layerB]:
        for seq in layer.twoq_pulse_sequences:
            for k in range(seq.n_pulses):
                instr = seq.pulse_instructions[k]
                if type(instr) is not IdleInstruction:
                    assert hasattr(instr, "distort_factor")
                    # distort_factor should be array-like
                    assert isinstance(instr.distort_factor, np.ndarray)

    # t_lab should have advanced by pc.duration
    assert pc.t_lab == pc.duration


@pytest.mark.parametrize("operands", [(0, 1), (1, 0), (1, 2), (2, 1)])
def test_attach_time_traces_two_qubit_operand_order(operands):
    """Coupling time traces must be indexed by the coupling edge, not by the gate's
    first operand. A two-qubit gate whose operands are in reverse order (e.g.
    ``rzz q[i+1], q[i]``) used to pick the wrong coupling trace -- and raise
    ``IndexError`` when that operand was the last qubit."""

    from spin_pulse import ExperimentalEnvironment, HardwareSpecs, Shape
    from spin_pulse.environment.noise import NoiseType

    specs = HardwareSpecs(
        num_qubits=3,
        B_field=0.06,
        delta=0.06,
        J_coupling=0.003,
        rotation_shape=Shape.GAUSSIAN,
        ramp_duration=5,
        coeff_duration=5,
    )
    env = ExperimentalEnvironment(
        specs,
        noise_type=NoiseType.PINK,
        T2S=10_000,
        TJS=5_000,
        duration=2**16,
        segment_duration=2**16,
        seed=1,
    )

    circ = QuantumCircuit(3)
    circ.rzz(0.5, *operands)

    # Must not raise; and re-attaching across "shots" must keep working.
    pc = PulseCircuit.from_circuit(circ, specs, exp_env=env)
    pc.attach_time_traces(env)
    pc.attach_time_traces(env)
