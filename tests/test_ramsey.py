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
import qiskit as qi

from spin_pulse import (
    DynamicalDecoupling,
    ExperimentalEnvironment,
    HardwareSpecs,
    PulseCircuit,
    Shape,
)
from spin_pulse.characterization.ramsey import (
    get_average_ramsey_contrast,
    get_contrast,
    get_ramsey_circuit,
)


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------
@pytest.mark.parametrize(
    "num_qubits, B_field, delta, J_coupling, rotation_shape, ramp_duration, coeff_duration, dynamical_decoupling",
    [
        (3, 1.0, 0.2, 0.5, Shape.GAUSSIAN, 2, 7, DynamicalDecoupling.FULL_DRIVE),
        (5, 0.1, 10, 0.2, Shape.SQUARE, 5, 7, None),
    ],
)
def test_get_ramsey_circuit_runs(
    num_qubits,
    B_field,
    delta,
    J_coupling,
    rotation_shape,
    ramp_duration,
    coeff_duration,
    dynamical_decoupling,
):
    # Test initialization with multiple qubits and Gaussian & square rotation shape.

    hw = HardwareSpecs(
        num_qubits=num_qubits,
        B_field=B_field,
        delta=delta,
        J_coupling=J_coupling,
        rotation_shape=rotation_shape,
        ramp_duration=ramp_duration,
        coeff_duration=coeff_duration,
        dynamical_decoupling=dynamical_decoupling,
        seed=10,
    )
    T2S = 50
    duration = 200
    segment_duration = 10
    env = ExperimentalEnvironment(
        T2S=T2S,
        duration=duration,
        segment_duration=segment_duration,
        hardware_specs=hw,
    )

    pulse_circ = get_ramsey_circuit(20, hw, exp_env=env)

    qreg = qi.QuantumRegister(1)
    circ = qi.QuantumCircuit(qreg)
    circ.h(0)
    circ.delay(int(duration))
    circ.h(0)
    isa_circ = hw.gate_transpile(circ)

    assert isinstance(pulse_circ, PulseCircuit)
    assert 0 < pulse_circ.duration <= 200
    assert pulse_circ.n_layers == hw.gate_transpile(isa_circ).depth()
    # assert pulse_circ.n_layers == 5


@pytest.mark.parametrize(
    "num_qubits, B_field, delta, J_coupling, rotation_shape, ramp_duration, coeff_duration, dynamical_decoupling",
    [
        (3, 1.0, 0.2, 0.5, Shape.GAUSSIAN, 2, 7, DynamicalDecoupling.FULL_DRIVE),
        (5, 0.1, 10, 0.2, Shape.SQUARE, 5, 7, None),
    ],
)
def test_contrast_runs_and_returns_float(
    num_qubits,
    B_field,
    delta,
    J_coupling,
    rotation_shape,
    ramp_duration,
    coeff_duration,
    dynamical_decoupling,
):
    # Test initialization with multiple qubits and Gaussian & square rotation shape.

    hw = HardwareSpecs(
        num_qubits=num_qubits,
        B_field=B_field,
        delta=delta,
        J_coupling=J_coupling,
        rotation_shape=rotation_shape,
        ramp_duration=ramp_duration,
        coeff_duration=coeff_duration,
        dynamical_decoupling=dynamical_decoupling,
    )
    T2S = 50
    duration = 200
    segment_duration = 10
    env = ExperimentalEnvironment(
        T2S=T2S,
        duration=duration,
        segment_duration=segment_duration,
        hardware_specs=hw,
    )

    pulse_circ = get_ramsey_circuit(10, hw, exp_env=env)
    c = get_contrast(pulse_circ)

    assert isinstance(c, float)
    assert -1.0 <= c <= 1.0  # contrast must be between -1 and 1


@pytest.mark.parametrize(
    "num_qubits, B_field, delta, J_coupling, rotation_shape, ramp_duration, coeff_duration, dynamical_decoupling",
    [
        (3, 1.0, 0.2, 0.5, Shape.GAUSSIAN, 2, 7, DynamicalDecoupling.FULL_DRIVE),
        (5, 0.1, 10, 0.2, Shape.SQUARE, 5, 7, None),
    ],
)
def test_average_ramsey_contrast_runs(
    num_qubits,
    B_field,
    delta,
    J_coupling,
    rotation_shape,
    ramp_duration,
    coeff_duration,
    dynamical_decoupling,
):
    # Test initialization with multiple qubits and Gaussian & square rotation shape.

    hw = HardwareSpecs(
        num_qubits=num_qubits,
        B_field=B_field,
        delta=delta,
        J_coupling=J_coupling,
        rotation_shape=rotation_shape,
        ramp_duration=ramp_duration,
        coeff_duration=coeff_duration,
        dynamical_decoupling=dynamical_decoupling,
    )
    T2S = 50
    duration = 200
    segment_duration = 10
    env = ExperimentalEnvironment(
        T2S=T2S,
        duration=duration,
        segment_duration=segment_duration,
        hardware_specs=hw,
    )

    durations = [5, 10, 15]

    avg_contrast = get_average_ramsey_contrast(hw, env, durations)

    assert isinstance(avg_contrast, np.ndarray)
    assert len(avg_contrast) == len(durations)

    # contrast is always a real scalar
    assert np.all(np.isfinite(avg_contrast))


@pytest.mark.parametrize(
    "num_qubits, B_field, delta, J_coupling, rotation_shape, ramp_duration, coeff_duration, dynamical_decoupling",
    [
        (3, 1.0, 0.2, 0.5, Shape.GAUSSIAN, 2, 7, DynamicalDecoupling.FULL_DRIVE),
        (5, 0.1, 10, 0.2, Shape.SQUARE, 5, 7, None),
    ],
)
def test_contrast_zero_delay_close_to_one(
    num_qubits,
    B_field,
    delta,
    J_coupling,
    rotation_shape,
    ramp_duration,
    coeff_duration,
    dynamical_decoupling,
):
    # Test initialization with multiple qubits and Gaussian & square rotation shape.

    hw = HardwareSpecs(
        num_qubits=num_qubits,
        B_field=B_field,
        delta=delta,
        J_coupling=J_coupling,
        rotation_shape=rotation_shape,
        ramp_duration=ramp_duration,
        coeff_duration=coeff_duration,
        dynamical_decoupling=dynamical_decoupling,
    )
    T2S = 50000
    duration = 200
    segment_duration = 10
    env = ExperimentalEnvironment(
        T2S=T2S,
        duration=duration,
        segment_duration=segment_duration,
        hardware_specs=hw,
    )

    pulse_circ = get_ramsey_circuit(1, hw, exp_env=env)
    c = get_contrast(pulse_circ)

    assert np.isfinite(c)
    assert c > 0.5  # Should be near +1 even with noise
