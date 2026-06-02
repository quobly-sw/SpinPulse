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

import pytest
from qiskit import QuantumCircuit
from qiskit.transpiler import PassManager
from qiskit.transpiler.passmanager import StagedPassManager

from spin_pulse import DynamicalDecoupling, HardwareSpecs, Shape
from spin_pulse.transpilation.instructions import (
    GaussianRotationInstruction,
    SquareRotationInstruction,
)

# To do list:
# I need to do a more complete test of gate-transpile with some more complex circuit.
# Will be done after the test of the circ file


@pytest.mark.parametrize(
    "num_qubits, B_field, delta, J_coupling, rotation_shape, ramp_duration, coeff_duration, dynamical_decoupling",
    [
        (3, 1.0, 0.2, 0.5, Shape.GAUSSIAN, 2, 7, DynamicalDecoupling.FULL_DRIVE),
        (5, 0.1, 10, 0.2, Shape.SQUARE, 5, 7, None),
    ],
)
def test_init_gaussian_shape_multiple_qubits(
    num_qubits,
    B_field,
    delta,
    J_coupling,
    rotation_shape,
    ramp_duration,
    coeff_duration,
    dynamical_decoupling,
):
    """Test initialization with multiple qubits and Gaussian & square rotation shape."""

    specs = HardwareSpecs(
        num_qubits=num_qubits,
        B_field=B_field,
        delta=delta,
        J_coupling=J_coupling,
        rotation_shape=rotation_shape,
        ramp_duration=ramp_duration,
        coeff_duration=coeff_duration,
        dynamical_decoupling=dynamical_decoupling,
    )

    # Check attributes
    assert specs.num_qubits == num_qubits
    assert specs.fields == {
        "x": B_field,
        "y": B_field,
        "Heisenberg": J_coupling,
        "z": delta,
    }
    assert specs.rotation_shape == rotation_shape

    if rotation_shape == Shape.GAUSSIAN:
        assert specs.rotation_generator is GaussianRotationInstruction
        assert specs.coeff_duration == coeff_duration
    elif rotation_shape == Shape.SQUARE:
        assert specs.rotation_generator is SquareRotationInstruction
    assert specs.ramp_duration == ramp_duration
    assert specs.dynamical_decoupling == dynamical_decoupling

    # Check that coupling_map and passes are properly initialized
    circ = QuantumCircuit(num_qubits)
    circ.rx(0.5, 0)
    transpiled = specs.gate_transpile(circ)
    assert isinstance(transpiled, QuantumCircuit)
    assert transpiled.num_qubits == num_qubits

    assert type(specs.first_pass) == StagedPassManager  # noqa: E721
    assert type(specs.second_pass) == PassManager  # noqa: E721


def test_invalid_rotation_shape_raises():
    """Test that invalid rotation shape raises an AssertionError."""
    with pytest.raises(ValueError, match="not currently available"):
        HardwareSpecs(
            num_qubits=2,
            B_field=1.0,
            delta=0.2,
            J_coupling=0.2,
            rotation_shape="triangular",
        )


def test_str_representation_for_gaussian():
    """Test string representation includes all key attributes."""
    specs = HardwareSpecs(
        num_qubits=2,
        B_field=1.2,
        delta=0.3,
        J_coupling=0.7,
        rotation_shape=Shape.GAUSSIAN,
        coeff_duration=8,
        dynamical_decoupling=None,
    )
    text = str(specs)

    expected_strings = [
        "HardwareSpec:",
        f"num_qubits: {specs.num_qubits}",
        f"B_field: {specs.fields['x']}",
        f"delta: {specs.fields['z']}",
        f"J_coupling: {specs.fields['Heisenberg']}",
        f"rotation_shape: {specs.rotation_shape}",
        f"ramp_duration: {specs.ramp_duration}",
        f"coeff_duration: {specs.coeff_duration}",
        f"dynamical_decoupling: {specs.dynamical_decoupling}",
    ]
    for expected in expected_strings:
        assert expected in text, f"Missing '{expected}' in output:\n{text}"


def test_str_representation_for_square():
    """Ensure coeff_duration is marked as N/A for square pulses."""
    specs = HardwareSpecs(
        num_qubits=1,
        B_field=0.9,
        delta=0.1,
        J_coupling=0.2,
        rotation_shape=Shape.SQUARE,
    )
    text = str(specs)
    assert "coeff_duration: N/A" in text


@pytest.mark.parametrize(
    "B_field, delta, J_coupling",
    [
        (0.0001, 0.2, 0.2),
        (0.1, 0.0002, 0.2),
    ],
)
def test_invalid_hardware_specs_raises(B_field, delta, J_coupling):
    """Test that invalid hardware specs raise appropriate errors."""
    with pytest.raises(ValueError):
        HardwareSpecs(
            num_qubits=1,
            B_field=B_field,
            delta=delta,
            J_coupling=J_coupling,
            rotation_shape=Shape.GAUSSIAN,
        )


@pytest.mark.parametrize(
    "B_field, delta, J_coupling",
    [
        (0.1, 0.1, 0.0002),
    ],
)
def test_invalid_hardware_specs_warns(B_field, delta, J_coupling):
    """Test that invalid hardware specs raise appropriate errors."""
    with pytest.warns():
        HardwareSpecs(
            num_qubits=1,
            B_field=B_field,
            delta=delta,
            J_coupling=J_coupling,
            rotation_shape=Shape.GAUSSIAN,
        )


def test_invalid_hardware_dynamic():
    """Test that invalid hardware specs raise appropriate errors."""
    with pytest.raises(ValueError):
        HardwareSpecs(
            num_qubits=2,
            B_field=1.2,
            delta=0.3,
            J_coupling=0.7,
            rotation_shape=Shape.GAUSSIAN,
            coeff_duration=8,
            dynamical_decoupling=False,
        )
