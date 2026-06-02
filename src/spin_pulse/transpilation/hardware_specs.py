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
"""Description of the hardware to simulate circuit execution."""

import warnings
from enum import Enum
from typing import Union

from qiskit import QuantumCircuit
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.transpiler import (
    PassManager,
    generate_preset_pass_manager,
)
from qiskit.transpiler.passes import Optimize1qGatesDecomposition

from .dynamical_decoupling import DynamicalDecoupling
from .instructions import (
    GaussianRotationInstruction,
    SquareRotationInstruction,
)
from .passes.rzz_echo import RZZEchoPass


class Shape(Enum):
    """Enumeration of pulse envelope shapes.

    This enum defines the functional form of the pulse envelope used for
    control operations in spin-qubit pulse sequences.
    """

    GAUSSIAN = "gaussian"
    """Gaussian-shaped pulse envelope."""

    SQUARE = "square"
    """Square (rectangular) pulse envelope."""


class HardwareSpecs:
    """Defines the hardware specifications for a spin qubit device model.

    This class stores all physical and control parameters required to build
    pulse instructions, generate rotation gates, and configure backend-level
    compilation. It also defines the available couplings, supported gate set,
    and optional dynamical decoupling strategy. The specifications in this
    object determine how qubit operations and timing constraints are modeled
    throughout the simulation workflow.

    Attributes:
        - rotation_generator (RotationInstruction): Class used to generate rotation
          instructions based on the selected pulse shape.
        - num_qubits (int): Number of qubits in the device model.
        - J_coupling (float): Maximum coupling strength between adjacent qubits.
        - fields (dict): Dictionary mapping interaction types ("x", "y",
          "z", "Heisenberg") to their corresponding field strengths.
        - rotation_shape (Shape): Shape of the pulses used for qubit control.
        - coeff_duration (int): Duration coefficient for Gaussian pulses
          when applicable.
        - ramp_duration (int): Duration of the ramp used in square pulses.
        - first_pass (PassManager): Preset first-stage Qiskit pass manager.
        - second_pass (PassManager): Additional optimization pass manager.
        - dynamical_decoupling (DynamicalDecoupling | None): Optional
          dynamical decoupling sequence applied to idle qubits.
        - optim (int): Qiskit optimization level

    """

    rotation_generator: Union[
        type[SquareRotationInstruction], type[GaussianRotationInstruction]
    ]  # should add any new rotation type. should be reviewed to be more versatile

    def __init__(
        self,
        num_qubits: int,
        B_field: float,
        delta: float,
        J_coupling: float,
        rotation_shape: Shape,
        ramp_duration: int = 1,
        coeff_duration: int = 5,
        dynamical_decoupling: DynamicalDecoupling | None = None,
        optim: int = 0,
    ):
        """
        Initialize the HardwareSpecs object that defines the hardware specifications.

        Parameters:
            num_qubits (int): Number of qubits considered.
            B_field (float): Maximal magnetic field strength.
            delta (float): Maximal energy difference between two qubits.
            J_coupling (float): Maximal coupling strength between two qubit.
            rotation_shape (Shape): Pulse shape used for qubit manipulation.
            ramp_duration (int, optional): Duration of the pulse ramp for square pulse. Default is 1.
            coeff_duration (int, optional): Duration coefficient for Gaussian pulses. Default is 5.
            dynamical_decoupling (DynamicalDecoupling, optional): If not None, defines the dynamical decoupling sequence to be applied to Idle qubits.

        """

        self.num_qubits: int = num_qubits
        self.J_coupling: float = J_coupling

        if num_qubits > 1:
            coupling_map = [(i, i + 1) for i in range(num_qubits - 1)]
        else:
            coupling_map = None
        if num_qubits > 1:
            basis_gates = ["rx", "ry", "rz", "rzz"]
        else:
            basis_gates = ["rx", "ry", "rz"]
        backend: GenericBackendV2 = GenericBackendV2(
            num_qubits=num_qubits, coupling_map=coupling_map, basis_gates=basis_gates
        )

        if B_field < 1e-3:
            raise ValueError(f"B_field must be greater than 1e-3, got {B_field}")

        if delta < 1e-3:
            raise ValueError(f"delta must be greater than 1e-3, got {delta}")

        if J_coupling < 1e-3:
            warnings.warn(
                f"J_coupling value smaller than 1e-3 ({J_coupling}), simulation might be long."
            )

        if dynamical_decoupling is not None and not isinstance(
            dynamical_decoupling, DynamicalDecoupling
        ):
            raise ValueError(
                "dynamical_decoupling parameter must be None or a DynamicalDecoupling enum."
            )

        fields = {"x": B_field, "y": B_field, "Heisenberg": J_coupling, "z": delta}
        self.fields: dict[str, float] = fields

        self.rotation_shape: Shape = rotation_shape
        match self.rotation_shape:
            case Shape.GAUSSIAN:
                self.rotation_generator = GaussianRotationInstruction
                self.coeff_duration = coeff_duration
            case Shape.SQUARE:
                self.rotation_generator = SquareRotationInstruction
            case _:
                raise ValueError(f"{rotation_shape} not currently available")

        self.ramp_duration: int = ramp_duration

        self.first_pass = generate_preset_pass_manager(
            target=backend.target, optimization_level=optim
        )
        self.second_pass = PassManager(
            [RZZEchoPass(), Optimize1qGatesDecomposition(target=backend.target)]
        )

        self.dynamical_decoupling: DynamicalDecoupling | None = dynamical_decoupling

    def gate_transpile(self, circ: QuantumCircuit) -> QuantumCircuit:
        """Transpile a quantum circuit into an ISA circuit using hardware specifications.

        Parameters:
            circ (qiskit.QuantumCircuit): The quantum circuit to be converted.

        Returns:
            qiskit.QuantumCircuit: The ISA quantum circuit composed of spin qubit native gates.

        """
        both_passes = PassManager(
            [
                self.first_pass.to_flow_controller(),
                self.second_pass.to_flow_controller(),
            ]
        )
        return both_passes.run(circ)

    def __str__(self):
        """
        Return a string representation of the HardwareSpecs instance.

        Includes:
            Number of qubits.
            B_field, delta and J_coupling values.
            Shape of manipulation pulses.
            Pulse ramp duration for square pulse.
            Duration coefficient for Gaussian pulses if the pulses are Gaussian, otherwise N/A.
            Dynamical decoupling sequence chosen. None if no dynamical decoupling.
        """
        summary = [
            "HardwareSpec:",
            f"  num_qubits: {self.num_qubits}",
            f"  B_field: {self.fields['x']}",
            f"  delta: {self.fields['z']}",
            f"  J_coupling: {self.fields['Heisenberg']}",
            f"  rotation_shape: {self.rotation_shape}",
            f"  ramp_duration: {self.ramp_duration}",
            f"  coeff_duration: {getattr(self, 'coeff_duration', 'N/A')}",
            f"  dynamical_decoupling: {self.dynamical_decoupling}",
        ]

        return "\n".join(summary)
