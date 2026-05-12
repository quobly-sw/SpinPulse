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
from qiskit.circuit import QuantumRegister
from qiskit.circuit.library import RXGate, RZZGate
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.basepasses import TransformationPass


# Key written into ``DAGCircuit.metadata`` once the echo has been applied, so the
# pass can be a no-op if it runs again (e.g. once during ``HardwareSpecs.gate_transpile``
# and once again inside ``PulseCircuit.from_dag_circuit``).
_ECHOED_METADATA_KEY = "spin_pulse_rzz_echoed"


class RZZEchoPass(TransformationPass):
    r"""Echo :math:`R_{ZZ}` gates with :math:`X` pulses to mitigate Stark shifts.

    This transpiler pass replaces each two-qubit :math:`R_{ZZ}` gate by an echoed
    sequence consisting of interleaved :math:`\pi` rotations around the :math:`x` axis and
    :math:`R_{ZZ}` gates with half the original angle. The echo sequence is applied
    locally on each occurrence of an RZZ gate in the input DAG circuit.

    The pass is idempotent: it records a marker in ``dag.metadata`` and returns the
    DAG untouched if it is run a second time.
    """

    def run(
        self,
        dag: DAGCircuit,
    ) -> DAGCircuit:
        r"""Apply the :math:`R_{ZZ}` echo transformation to all :math:`R_{ZZ}` gates in the DAG.

        Each :math:`R_{ZZ}(\theta)` operation is replaced by the following pattern
        on the two involved qubits:

        - :math:`R_X(\pi)` on both qubits
        - :math:`R_{ZZ}(\theta / 2)`
        - :math:`R_X(\pi)` on both qubits
        - :math:`R_{ZZ}(\theta / 2)`

        All other operations are left unchanged.

        Parameters:
            dag (DAGCircuit): Input circuit represented as a DAG on which
              the transformation is applied.

        Returns:
            DAGCircuit: New DAG circuit where each :math:`R_{ZZ}` gate has been replaced by the echoed :math:`R_{ZZ}` sequence.

        """
        metadata = dag.metadata
        if metadata is None:
            metadata = {}
        elif not isinstance(metadata, dict):
            # Not a real metadata mapping (e.g. a test double); leave the DAG untouched.
            return dag
        elif metadata.get(_ECHOED_METADATA_KEY):
            # Already echoed (e.g. by HardwareSpecs.gate_transpile) -- do not echo twice.
            return dag
        # Work on a fresh dict: ``DAGCircuit.metadata`` aliases the source circuit's
        # ``metadata`` (``circuit_to_dag`` shares the reference), so mutating it in
        # place would leak the marker back onto the caller's circuit.
        metadata = dict(metadata)

        for node in dag.op_nodes():
            if not node.op.name == "rzz":
                continue
            theta = node.op.params[0]
            # instantiate mini_dag and attach quantum register
            mini_dag = DAGCircuit()
            register = QuantumRegister(2)
            mini_dag.add_qreg(register)

            for _ in range(2):
                mini_dag.apply_operation_back(RXGate(np.pi), [register[0]])
                mini_dag.apply_operation_back(RXGate(np.pi), [register[1]])
                mini_dag.apply_operation_back(
                    RZZGate(theta / 2), [register[0], register[1]]
                )

            dag.substitute_node_with_dag(node, mini_dag)

        metadata[_ECHOED_METADATA_KEY] = True
        dag.metadata = metadata
        return dag
