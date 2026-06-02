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

import numpy as np
import pytest

import tests.fixtures.dummy_objects as dm
from spin_pulse import ExperimentalEnvironment
from spin_pulse.environment.noise import (
    NoiseType,
    PinkNoiseTimeTrace,
    QuasistaticNoiseTimeTrace,
    WhiteNoiseTimeTrace,
)


@pytest.mark.parametrize(
    "num_qubits, noise_type, expected_class, segment_duration",
    [
        (2, NoiseType.PINK, PinkNoiseTimeTrace, 2**10),
        (3, NoiseType.WHITE, WhiteNoiseTimeTrace, 1),
        (1, NoiseType.QUASISTATIC, QuasistaticNoiseTimeTrace, 2**4),
    ],
)
def test_noise_generator_selection(
    num_qubits, noise_type, expected_class, segment_duration
):
    """Check that the correct noise generator is assigned based on noise_type."""
    hw = dm.DummyHardwareSpecs(num_qubits=num_qubits)
    env = ExperimentalEnvironment(
        hardware_specs=hw, noise_type=noise_type, segment_duration=segment_duration
    )
    assert env.noise_generator is expected_class

    assert type(env.T2S) is float
    assert env.segment_duration == segment_duration
    assert len(env.time_traces) == num_qubits
    assert env.hardware_specs.num_qubits == num_qubits
    assert all(isinstance(trace, expected_class) for trace in env.time_traces)


def test_invalid_noise_type_raises():
    """Ensure invalid noise type raises an assertion."""
    hw = dm.DummyHardwareSpecs()
    with pytest.raises(ValueError, match="unknown noise type"):
        ExperimentalEnvironment(hardware_specs=hw, noise_type="invalid")


@pytest.mark.parametrize(
    "num_qubits, TJS",
    [
        (2, 50),
        (11, 30),
        (6, 2),
    ],
)
def test_coupling_traces_created_when_TJ_is_set(num_qubits, TJS):
    """When TJS is provided, coupling noise traces must be generated."""
    hw = dm.DummyHardwareSpecs(num_qubits=num_qubits)
    env = ExperimentalEnvironment(hardware_specs=hw, TJS=TJS)
    assert hasattr(env, "time_traces_coupling")
    assert len(env.time_traces_coupling) == num_qubits - 1  # n-1 coupling traces


@pytest.mark.parametrize(
    "num_qubits, noise_type, T2S, TJS, duration, segment_duration, only_idle, J_coupling",
    [
        (2, NoiseType.QUASISTATIC, 150, None, 1024, 512, False, None),
        (3, NoiseType.WHITE, 200, None, 2048, 1, True, 0.5),
        (3, NoiseType.PINK, 200, 20, 2048, 1024, True, 0.5),
    ],
)
def test_str_representation_contains_all_info(
    num_qubits, noise_type, T2S, TJS, duration, segment_duration, only_idle, J_coupling
):
    """Ensure __str__ output includes all relevant attributes of ExperimentalEnvironment."""
    hw = dm.DummyHardwareSpecs(num_qubits=num_qubits, J_coupling=J_coupling)
    env = ExperimentalEnvironment(
        hardware_specs=hw,
        noise_type=noise_type,
        T2S=T2S,
        TJS=TJS,
        duration=duration,
        segment_duration=segment_duration,
        only_idle=only_idle,
    )
    summary = str(env)

    # Check presence of all key information
    expected_strings = [
        "ExperimentalEnvironment:",
        f"  Qubits: {num_qubits}",
        f"  Noise Type: {noise_type}",
        f"  T2S (qubit dephasing): {T2S}",
        f"  TJS (coupling dephasing): {TJS}",
        f"  Duration: {duration}",
        f"  Segment Duration: {segment_duration}",
        f"  Only Idle: {only_idle}",
        f"  J Coupling: {J_coupling if J_coupling is not None else 'None'}",
        f"  Time Traces Generated: {len(env.time_traces)}",
    ]

    for expected in expected_strings:
        assert expected in summary, (
            f"Missing '{expected}' in __str__ output:\n{summary}"
        )
        if TJS is None:
            assert "Coupling Time Traces Generated:" not in summary
        else:
            assert f"  Coupling Time Traces Generated: {len(env.time_traces_coupling)}"


def test_regenerate_time_traces_updates_length():
    """Verify generate_time_traces regenerates correct number of traces."""
    hw = dm.DummyHardwareSpecs(num_qubits=4)
    env = ExperimentalEnvironment(hardware_specs=hw)
    old_traces = env.time_traces
    env.hardware_specs.num_qubits = 2
    env.generate_time_traces()
    assert len(env.time_traces) == 2
    assert env.time_traces is not old_traces


def test_generate_time_traces_seeded():
    """Verify seeded generate_time_traces generate correct unique trace for qubits."""
    hw = dm.DummyHardwareSpecs(num_qubits=4)
    env = ExperimentalEnvironment(hardware_specs=hw, seed=101010)
    times_traces = env.time_traces
    while times_traces:
        trace_to_check = times_traces[0]
        times_traces.remove(trace_to_check)
        for trace in times_traces:
            assert not np.array_equal(trace_to_check.values, trace.values)


def test_generate_time_traces_coupling_seeded():
    """Verify seeded generate_time_traces generate correct unique trace for qubits."""
    hw = dm.DummyHardwareSpecs(num_qubits=4)
    env = ExperimentalEnvironment(hardware_specs=hw, seed=101010, TJS=10)
    times_traces = env.time_traces_coupling
    while times_traces:
        trace_to_check = times_traces[0]
        times_traces.remove(trace_to_check)
        for trace in times_traces:
            assert not np.array_equal(trace_to_check.values, trace.values)
