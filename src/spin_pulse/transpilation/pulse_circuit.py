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
"""Pulse-level representation of a quantum circuit."""

import warnings
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Qubit
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.dagcircuit import DAGCircuit
from qiskit.quantum_info import Operator, SuperOp, average_gate_fidelity
from qiskit_aer import AerSimulator
from tqdm import tqdm

from ..environment.experimental_environment import ExperimentalEnvironment
from .hardware_specs import HardwareSpecs
from .instructions import IdleInstruction
from .passes.rzz_echo import RZZEchoPass
from .pulse_layer import PulseLayer


class PulseCircuit:
    """Pulse-level representation of a quantum circuit.

    A PulseCircuit stores a layered decomposition of a qiskit.QuantumCircuit into
    PulseLayer objects and, optionally, a stochastic noise environment. It
    provides utilities to visualize the pulse schedule, convert it back to a
    gate-level circuit, and compute fidelities or average channels under
    sampled noise realizations.

    Attributes:
        - original_circ (qiskit.QuantumCircuit): Original quantum circuit from which
            the PulseCircuit was constructed.
        - num_qubits (int): Number of qubits in the circuit.
        - qubits (list[qiskit.circuit.Qubit]): Ordered list of qubits acted on by the circuit.
        - pulse_layers (list[PulseLayer]): Ordered list of pulse layers that
            implement the circuit at the pulse level.
        - n_layers (int): Number of pulse layers.
        - duration (int): Total duration of the pulse circuit, given by the sum
            of all layer durations (in the discrete time unit of the model).
        - t_lab (int): Current position in the laboratory time axis, used to
            attach successive noise time traces when averaging over samples.

    """

    def __init__(
        self,
        circ: QuantumCircuit,
        qubits: list[Qubit],
        pulse_layers,
        hardware_specs: HardwareSpecs,
        exp_env: ExperimentalEnvironment | None = None,
    ):
        """Initialize a PulseCircuit from pulse layers.

        The constructor records the original circuit, the list of qubits and
        the sequence of PulseLayer objects. It computes the total duration,
        assigns starting times to all layers and computes a dynamical decoupling
        sequence according to the hardware specifications. If an ExperimentalEnvironment
        is provided, noise time traces are attached to the circuit and its layers.
        Optionally, a container for storing per-sample time series can be created.

        Parameters:
            circ (QuantumCircuit): Original quantum circuit represented at
                the pulse level.
            qubits (list[qiskit.circuit.Qubit]): Ordered list of qubits involved in the
                circuit.
            pulse_layers (list[PulseLayer]): Pulse layers that realize the
                circuit in time order.
            hardware_specs (HardwareSpecs): Hardware configuration specifying
                fields, couplings and dynamical decoupling options.
            exp_env (ExperimentalEnvironment | None): Noise environment from
                which time traces are drawn and attached to this circuit. If
                None, no noise trace is attached.

        """
        self.original_circ: QuantumCircuit = circ
        self.qubits: list[Qubit] = qubits
        self.num_qubits = len(self.qubits)
        self.pulse_layers = pulse_layers
        self.n_layers = len(self.pulse_layers)
        self.duration = sum([_.duration for _ in pulse_layers])

        self.assign_starting_times()
        self.attach_dynamical_decoupling(hardware_specs)

        self.t_lab = 0
        self.attach_time_traces(exp_env)

    @classmethod
    def from_circuit(
        cls,
        circ: QuantumCircuit,
        hardware_specs: HardwareSpecs,
        exp_env: ExperimentalEnvironment | None = None,
    ):
        """Construct a PulseCircuit from a qiskit.QuantumCircuit.

        The input is first converted to a qiskit.dagcircuit.DAGCircuit then all barrier and
        measurement gate are removed, and each DAG layer is translated
        into a PulseLayer using the hardware specifications. Empty layers
        (containing only barriers) are skipped. The resulting list of
        pulse layers is used to build a PulseCircuit.

        Parameters:
            circ (qiskit.QuantumCircuit): Circuit to translate into pulse layers.
            hardware_specs (HardwareSpecs): Hardware specification used to
                map gate operations to pulse sequences.
            exp_env (ExperimentalEnvironment | None): Experimental
                environment providing noise specification and noise time traces.
                If None, no noise is attached to the pulses.

        Returns:
            PulseCircuit: A pulse-level representation of the input
            circuit.

        """

        dag = circuit_to_dag(circ)
        return cls.from_dag_circuit(
            dag,
            hardware_specs=hardware_specs,
            exp_env=exp_env,
            original_circuit=circ,
        )

    @classmethod
    def from_dag_circuit(
        cls,
        dag: DAGCircuit,
        hardware_specs: HardwareSpecs,
        exp_env: ExperimentalEnvironment | None = None,
        original_circuit: QuantumCircuit | None = None,
    ):
        """Construct a PulseCircuit from a qiskit.dagcircuit.DAGCircuit.

        The input circuit is first stripped of measurements and
        barrier, if present. Then each DAG layer is translated
        into a PulseLayer using the hardware specifications.
        Empty layers (containing only barriers) are skipped.
        The resulting list of pulse layers is used to build a PulseCircuit.

        Parameters:
            dag (qiskit.dagcircuit.DAGCircuit): Circuit to translate into pulse layers.
            hardware_specs (HardwareSpecs): Hardware specification used to
                map gate operations to pulse sequences.
            exp_env (ExperimentalEnvironment | None): Experimental
                environment providing noise specification and noise time traces.
                If None, no noise is attached to the pulses.

        Returns:
            PulseCircuit: A pulse-level representation of the input
            circuit.

        """

        ## strip measurement

        dag.remove_all_ops_named("measure")
        dag.remove_all_ops_named("barrier")

        # Echo every RZZ gate. spin-pulse realises an ``rzz`` gate by detuning the two
        # qubits (a strong, opposite-sign Z field on each) for the whole exchange pulse;
        # that detuning suppresses the unwanted XX + YY part of the Heisenberg coupling
        # but also imprints a large spurious single-qubit Z (Stark) phase that only
        # cancels when each ``rzz`` is wrapped in an X echo. Applying it here makes
        # ``from_circuit``/``from_dag_circuit`` produce a faithful pulse circuit even
        # when the caller has not run ``HardwareSpecs.gate_transpile`` first. The pass
        # is idempotent (it marks ``dag.metadata``), so it is a no-op when the circuit
        # already went through ``gate_transpile``.
        dag = RZZEchoPass().run(dag)

        """dag.data = [
            inst for inst in circ.data if not isinstance(inst.operation, Measure)
        ]
        if circ.data and isinstance(circ.data[-1].operation, Barrier):
            circ.data.remove(circ.data[-1])
        """
        qubits: list[Qubit] = dag.qubits
        pulse_layers = []
        for layer in dag.layers():
            layer_: QuantumCircuit = dag_to_circuit(layer["graph"])

            flag_empty_layer: bool = (
                True  # Layer containing no gate that can be translated to pulse
            )
            for gate in layer_.data:
                if gate.operation.name not in ["barrier"]:
                    flag_empty_layer = False
            if flag_empty_layer:
                continue

            pulse_layer = PulseLayer.from_circuit_layer(qubits, layer_, hardware_specs)
            pulse_layers.append(pulse_layer)

        circ: QuantumCircuit
        if original_circuit:
            circ = original_circuit
        else:
            circ = dag_to_circuit(dag)

        return cls(
            circ,
            qubits,
            pulse_layers,
            hardware_specs,
            exp_env=exp_env,
        )

    def assign_starting_times(self):
        """Assign a starting time to each pulse layer to synchronize the
        different layers and pulse instructions.

        The starting time ``t_start`` of each PulseLayer is computed as the
        cumulative duration of all previous layers, with the first layer
        starting at time zero. This defines a global time axis for the
        PulseCircuit.

        """
        for i in range(self.n_layers):
            if i == 0:
                t_start = 0
            else:
                t_start = t_start + self.pulse_layers[i - 1].duration
            self.pulse_layers[i].t_start = t_start

    def __str__(self) -> str:
        """Return a readable description of the pulse circuit.

        Returns:
            str: A formatted string containing the total duration and the
            number of layers in the circuit.

        """
        return f"PulseCircuit of duration={self.duration} with {self.n_layers} layers "

    def plot(
        self, hardware_specs: HardwareSpecs | None = None, label_gates: bool = True
    ):
        """Plot the pulse schedule of the circuit.

        This method creates a multi-panel matplotlib figure showing, for each
        layer, the one-qubit and two-qubit pulse envelopes applied to each
        qubit and coupling. The vertical scale is chosen
        consistently with the hardware field and coupling amplitudes if given.

        Parameters:
            hardware_specs (HardwareSpecs | None): Hardware specification used
                to set the limits of manipulation parameter amplitudes.
                If None, default symmetric limits are used.
            label_gates (bool): If True, gate labels are shown.

        """
        width_ratios = [self.pulse_layers[i].duration for i in range(self.n_layers)] + [
            0.1 * self.duration
        ]
        nrows = self.num_qubits + (self.num_qubits - 1) + 1
        height_ratios = [1, 0.67] * (self.num_qubits - 1) + [1] + [0.1]
        fig, axs = plt.subplots(
            figsize=(13, 2.5 * self.num_qubits),
            ncols=self.n_layers + 1,
            nrows=nrows,
            height_ratios=height_ratios,
            width_ratios=width_ratios,
            layout=None,
        )

        if hardware_specs is None:
            ymax_qubit = 0.5
            ymax_coupling = 0.5
        else:
            ymax_qubit = 1.05 * max(
                hardware_specs.fields["x"], hardware_specs.fields["z"]
            )
            ymax_coupling = 1.5 * hardware_specs.fields["Heisenberg"]

        for ax in axs.flat:
            ax.set_rasterization_zorder(0)
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

        for j in range(self.n_layers):
            self.pulse_layers[j].plot(axs=axs[:-1, j], label_gates=label_gates)

            for i in range(self.num_qubits):
                ax_q = axs[2 * i, j]
                ax_q.set_ylim(-ymax_qubit, ymax_qubit)
                for spine in ax_q.spines.values():
                    spine.set_alpha(0.1)

                if i < self.num_qubits - 1:
                    ax_c = axs[2 * i + 1, j]
                    ax_c.set_ylim(-ymax_coupling, ymax_coupling)
                    for spine in ax_c.spines.values():
                        spine.set_alpha(0.1)

            ax_bottom = axs[-1, j]
            ax_bottom.set_axis_off()

            if self.n_layers < 7:
                is_first = j == 0
                layer_text = f"Layer {j}" if is_first else f"L{j}"
                t_text = (
                    f"$t=${self.pulse_layers[j].t_start}"
                    if is_first
                    else f"{self.pulse_layers[j].t_start}"
                )
                font_weight = "bold" if is_first else "normal"

                ax_bottom.text(
                    0,
                    1,
                    layer_text,
                    color="#24185E",
                    fontsize=12,
                    fontweight=font_weight,
                )
                ax_bottom.text(0, -1, t_text, color="#24185E", fontsize=12)

        for i in range(self.num_qubits):
            ax_q = axs[2 * i, -1]
            ax_q.set_axis_off()
            ax_q.set_xlim(-0.5, 0.5)
            ax_q.set_ylim(-0.5, 0.5)
            ax_q.text(
                0.0, 0.2, f"qubit {i}", color="#E61873", fontsize=10, fontweight="bold"
            )
            ax_q.add_patch(plt.Circle((0, 0), 0.1, fc="#E61873"))
            qubit_coords = np.array([[0, 0], [-0.5, 0.5], [-0.5, -0.5]])
            ax_q.add_patch(plt.Polygon(qubit_coords, fc="#E61873", alpha=0.5))

            if i < self.num_qubits - 1:
                ax_c = axs[2 * i + 1, -1]
                ax_c.set_axis_off()
                ax_c.set_xlim(-0.5, 0.5)
                ax_c.set_ylim(-0.5, 0.5)
                coupling_coords = np.array(
                    [[0.0, 0.5], [-0.1, 0.5], [-0.1, -0.5], [0.0, -0.5]]
                )
                ax_c.add_patch(
                    plt.Polygon(
                        coupling_coords,
                        fc="#24185E",
                    )
                )

        axs[-1, -1].set_axis_off()

        fig.subplots_adjust(right=0.95, top=0.97)
        fig.text(
            0.92,
            0.975,
            "Spin",
            color="#24185E",
            fontsize=14,
            fontweight="bold",
            va="top",
        )
        fig.text(
            0.965,
            0.975,
            "Pulse",
            color="#E61873",
            fontsize=14,
            fontweight="bold",
            va="top",
        )

    def to_circuit(self, measure_all: bool = False) -> QuantumCircuit:
        """Convert the pulse circuit back to a qiskit.QuantumCircuit.

        Each PulseLayer is converted into equivalent circuit that are
        composed into a copy of the original circuit structure.
        Optionally, a final measurement on all qubits is added.

        Parameters:
            measure_all (bool): If True, all qubits are measured
                at the end of the reconstructed circuit.

        Returns:
            qiskit.QuantumCircuit: A circuit implementing the same unitary evolution
            as the PulseCircuit (up to numerical approximations).

        """
        circ = self.original_circ.copy()
        circ.data = []
        for pulse_layer in self.pulse_layers:
            circuit_ = pulse_layer.to_circuit()
            circ.compose(circuit_, inplace=True)
        if measure_all:
            circ.measure_all()
        return circ

    def circuit_samples(self, exp_env: ExperimentalEnvironment | None = None) -> int:
        """Compute the number of noisy circuit samples that can be obtained with the given environment.

        The number of circuit samples is given by the integer ratio between
        the environment duration and the pulse-circuit duration. If no
        environment is provided, a single sample is assumed.

        Parameters:
            exp_env (ExperimentalEnvironment | None): Experimental environment
                providing the total duration.

        Returns:
            int: Number of non-overlapping samples that can be drawn from the
            duration of the experiment.

        """
        if exp_env is not None:
            return exp_env.duration // self.duration
        else:
            return 1

    def get_logical_bitstring(self, physical_bistring: str) -> str:
        """
        Restore the original logical qubit layout before transpilation.
        During Qiskit transpilation process, the qubit register may be
        reordered to produce an optimized physical layout that minimizes gate
        costs. To recover the logical bitstring and interpret measurement
        outcomes correctly, it is necessary to invert this remapping and reconstruct
        the initial logical ordering of the qubits.

        The mapping is derived from the layout associated with the original
        circuit used to build the PulseCircuit. If a TranspileLayout is
        present, the final index layout is used. If only a Layout is
        available, its permutation is used and a warning is emitted, as this
        typically indicates that the circuit was not transpiled. If no layout
        information is present, the physical and logical orders are assumed
        to coincide and a warning is emitted.

        Parameters:
            physical_bistring (str): Bitstring obtained from a measurement in
                the physical qubit ordering.

        Returns:
            str: Bitstring reordered into the logical qubit basis.

        """
        physical_bits = list(physical_bistring[::-1])
        if self.original_circ.layout:
            try:
                permut = self.original_circ.layout.final_index_layout()
            except AttributeError:
                permut = self.original_circ.layout.to_permutation()
                warnings.warn(
                    "The original circuit used to create the PulseCircuit does"
                    " not have a TranspileLayout but a Layout. This means the circuit was"
                    " likely not transpiled. Continuing using the ordering given by to_permutation()."
                )
            while len(permut) < len(
                physical_bits
            ):  # Full the permutation to the full bitstring
                permut.append(len(permut))
            logical_bits = [physical_bits[permut[i]] for i in range(len(physical_bits))]
        else:
            logical_bits = physical_bits
            warnings.warn(
                "The original circuit used to create the PulseCircuit does"
                " not have a TranspileLayout nor a Layout. Continuing using same ordering"
                " for logical and physical bits."
            )
        return "".join(logical_bits[::-1])

    def averaging_over_samples(
        self,
        f,
        exp_env: ExperimentalEnvironment | None,
        *args,
        num_samples: int | None = None,
        progress_bar: bool = True,
    ):
        r"""Estimate the average value over as many noisy samples as the experimental environment
        allows it, of a user-provided function using the pulse circuit.

        This method repeatedly attaches new pieces of the noise time traces from the
        experimental environment to the PulseCircuit and evaluates a
        user-provided function ``f(self, *args)`` for each realization. The
        results are averaged over all samples, effectively performing a
        Monte Carlo average over noise trajectories.

        Parameters:
            f (Callable): Function that takes the PulseCircuit as first
                argument and returns a numerical quantity to be averaged.
            exp_env (ExperimentalEnvironment | None): Noise environment from
                which time traces are drawn. If None, a single deterministic
                evaluation is performed.
            \*Parameters: Additional positional arguments forwarded to ``f``.
            num_samples (int | None): Number of samples to be used. Default if None,
                in this case result of the circuit_samples function if taken.
            progress_bar (bool): if True, shows the progress. Default is True.

        Returns:
            Any: The sample-averaged value of ``f(self, *args)``.

        """
        if num_samples is None:
            num_samples = self.circuit_samples(exp_env)
        elif num_samples > self.circuit_samples(exp_env):
            warnings.warn(
                f"You requested a number of samples ({num_samples}) that is too high for a single instance of the experimental \
                environment specified. I will use several generations of the time traces to compute the average"
            )

        if num_samples == 0:
            raise ValueError(
                "Number of sample is 0. This is caused because the"
                " duration of the circuit is greater than environment duration."
            )
        self.t_lab = 0
        for i in tqdm(range(num_samples), disable=(not progress_bar)):
            # If we reach the end of the time trace, we generate the another one and set the internal cursor to 0
            if exp_env is not None:
                if self.t_lab + self.duration > exp_env.duration:
                    self.t_lab = 0
                    exp_env.generate_time_traces()

                self.attach_time_traces(exp_env)
            if i == 0:
                f_sum = f(self, *args)
            else:
                f_sum += f(self, *args)
        return f_sum / num_samples

    def run_experiment(
        self,
        exp_env: ExperimentalEnvironment,
        simulator=AerSimulator(),
        num_samples: int | None = None,
        progress_bar: bool = True,
    ):
        """
        Simulate single-shot measurement on noisy instances of the circuit.

        This method repeatedly attaches new pieces of the noise time traces from the
        experimental environment to the PulseCircuit and simulates with MPS method a single-shot measurement thanks to Qiskit Aer. The
        results are stored in a dictionary that gather the counts obtained for all possible bitstrings.

        Parameters:
            exp_env (ExperimentalEnvironment | None): Noise environment from
                which time traces are drawn. If None, a single deterministic
                evaluation is performed.
            simulator: an instance of qiksit's AerSimulator.
            num_samples (int | None): Number of samples to be used. Default if None,
                in this case result of the circuit_samples function if taken.
            progress_bar (bool): if True, shows the progress. Default is True.

        Returns:
            dict: A dictionary which keys are the obtained bitstrings and their respective number of occurences.

        """
        if num_samples is None:
            num_samples = self.circuit_samples(exp_env)
        elif num_samples > self.circuit_samples(exp_env):
            warnings.warn(
                f"You requested a number of samples ({num_samples}) that is too high for a single instance of the experimental \
                environment specified. I will use several generations of the time traces to compute the average"
            )

        if num_samples == 0:
            raise ValueError(
                "Number of sample is 0. This is caused because the"
                " duration of the circuit is greater than environment duration."
            )
        self.t_lab = 0
        result: defaultdict[str, int] = defaultdict(int)

        simulator = AerSimulator()
        for _ in tqdm(range(num_samples), disable=(not progress_bar)):
            # If we reach the end of the time trace, we generate the another one and set the internal cursor to 0
            if exp_env is not None:
                if self.t_lab + self.duration > exp_env.duration:
                    self.t_lab = 0
                    exp_env.generate_time_traces()

                self.attach_time_traces(exp_env)
            circuit = self.to_circuit()
            circuit.measure_all()

            counts = simulator.run(circuit, shots=1).result().get_counts()
            obtained_str = self.get_logical_bitstring(next(iter(counts.keys())))
            result[obtained_str] += 1
        return result

    def fidelity(self, circ_ref: QuantumCircuit | None = None) -> float:
        """Compute the average gate fidelity with respect to a reference circuit.

        The PulseCircuit is converted to a qiskit.QuantumCircuit, and the average
        gate fidelity between its unitary and that of the reference circuit
        is computed using the qiskit.quantum_info.Operator representation.

        Parameters:
            circ_ref (qiskit.QuantumCircuit): Reference circuit used to define the
                target unitary. Default is self.original_circ.

        Returns:
            float: Average gate fidelity between the PulseCircuit unitary and
            the reference circuit unitary.

        """

        if circ_ref is None:
            circ_ref = self.original_circ
        return average_gate_fidelity(
            Operator.from_circuit(self.to_circuit()), Operator.from_circuit(circ_ref)
        )

    def mean_fidelity(
        self,
        exp_env: ExperimentalEnvironment | None,
        circ_ref: QuantumCircuit | None = None,
        num_samples: int | None = None,
        progress_bar: bool = True,
    ) -> float:
        """Estimate the mean fidelity under a stochastic noise environment.

        The average gate fidelity with respect to a reference circuit is
        computed for multiple noise realizations drawn from the experimental
        environment, and the results are averaged using
        ``averaging_over_samples``.

        Parameters:
            circ_ref (qiskit.QuantumCircuit): Reference circuit used to define the
                target unitary.
            exp_env (ExperimentalEnvironment | None): Noise environment from
                which time traces are drawn.
            num_samples (int | None): Number of samples to be used. Default if None,
                in this case result of the circuit_samples function if taken.
            progress_bar (bool): if True, shows the progress. Default is True.


        Returns:
            float: Sample-averaged gate fidelity under the specified noise
            model.

        """
        return self.averaging_over_samples(
            lambda pulse_circ: pulse_circ.fidelity(circ_ref),
            exp_env,
            num_samples=num_samples,
            progress_bar=progress_bar,
        )

    def mean_channel(
        self,
        exp_env: ExperimentalEnvironment | None = None,
        num_samples: int | None = None,
        progress_bar: bool = True,
    ):
        """Estimate the mean quantum channel generated by the pulse circuit.

        For each noise realization, the PulseCircuit is converted to a
        qiskit.QuantumCircuit and then to a SuperOp representing the corresponding
        quantum channel. The channels are averaged over samples using
        ``averaging_over_samples``.

        Parameters:
            exp_env (ExperimentalEnvironment | None): Noise environment from
                which time traces are drawn.
            num_samples (int | None): Number of samples to be used. Default if None,
                in this case result of the circuit_samples function if taken.
            progress_bar (bool): if True, shows the progress. Default is True.

        Returns:
            qiskit.quantum_info.SuperOp: Sample-averaged quantum channel acting on the qubit
            register.

        """
        return self.averaging_over_samples(
            lambda x: SuperOp(Operator.from_circuit(x.to_circuit())),
            exp_env,
            num_samples=num_samples,
            progress_bar=progress_bar,
        )

    def attach_time_traces(self, exp_env: ExperimentalEnvironment | None = None):
        """Attach noise time traces from the experimental environment
        to the pulse circuit.

        If an experimental environment is provided, this method extracts
        segments of the environment time traces that match the total circuit
        duration and assigns them to the PulseCircuit and each PulseLayer.
        For one-qubit pulse sequences, the corresponding single-qubit time
        traces are attached (qubit's frequency deviation).
        For two-qubit sequences, coupling time traces are attached if
        available (i.e. J coupling noise) and used to set the J noise deviation
        during two-qubit gates.

        The internal laboratory time pointer ``t_lab`` is incremented by the
        circuit duration after attaching the time traces, so that subsequent
        calls use the next segment of the noise time trace.

        In case of the "t_lab" plus circuit duration is greater than the exp_env duration,
        a ValueError is raised.

        Parameters:
            exp_env (ExperimentalEnvironment | None): Experimental
                environment providing qubit's frequency deviation time trace
                and, optionally, J coupling deviation time traces.
                If None, no trace is attached.

        """
        if exp_env is not None:
            t_lab = self.t_lab
            if t_lab + self.duration > exp_env.duration:
                raise ValueError(
                    f"Environment duration {exp_env.duration} trace too short"
                    f"for circuit duration {self.duration} at t_lab time {t_lab}"
                )
            else:
                self.time_traces = [
                    tt.values[t_lab : t_lab + self.duration]
                    for tt in exp_env.time_traces
                ]
                if hasattr(exp_env, "time_traces_coupling"):
                    self.time_traces_coupling = [
                        tt.values[t_lab : t_lab + self.duration]
                        for tt in exp_env.time_traces_coupling
                    ]
                for layer_i in range(self.n_layers):
                    layer = self.pulse_layers[layer_i]
                    t_start = layer.t_start
                    duration = layer.duration
                    layer.time_traces = [
                        tt[t_start : t_start + duration] for tt in self.time_traces
                    ]
                    for sequence in layer.oneq_pulse_sequences:
                        j = sequence.qubits[0]._index
                        sequence.attach_time_trace(
                            layer.time_traces[j], only_idle=exp_env.only_idle
                        )

                    if hasattr(exp_env, "time_traces_coupling"):
                        layer.time_traces_coupling = [
                            tt[t_start : t_start + duration]
                            for tt in self.time_traces_coupling
                        ]
                        for sequence in layer.twoq_pulse_sequences:
                            # Coupling time traces are indexed by the lower of the two
                            # (adjacent) qubit indices, i.e. by the coupling edge
                            # (i, i + 1). Two-qubit gates may be given with their
                            # operands in either order (e.g. ``rzz q[i+1], q[i]``), so
                            # use ``min`` here -- using ``qubits[0]`` directly picks the
                            # wrong trace, and is out of range when that operand is the
                            # last qubit.
                            j = min(q._index for q in sequence.qubits)
                            for k in range(sequence.n_pulses):
                                instruction = sequence.pulse_instructions[k]
                                ta, tb = (
                                    sequence.t_start_relative[k],
                                    sequence.t_start_relative[k] + instruction.duration,
                                )
                                if type(instruction) is not IdleInstruction:
                                    instruction.distort_factor = (
                                        layer.time_traces_coupling[j][ta:tb]
                                        / exp_env.hardware_specs.J_coupling
                                    )

                self.t_lab += self.duration

    def attach_dynamical_decoupling(self, hardware_specs: HardwareSpecs):
        """Insert dynamical decoupling sequences into all pulse layers,
        when possible (i.e. Idle time sufficiently long).

        If a dynamical decoupling mode is specified in the hardware
        specifications, each PulseLayer in the circuit is updated by calling
        its ``attach_dynamical_decoupling`` method with the chosen mode. If
        no dynamical decoupling is configured, the circuit is left unchanged.

        Parameters:
            hardware_specs (HardwareSpecs): Hardware configuration specifying
                the dynamical decoupling mode and available pulse shapes.

        """
        if hardware_specs.dynamical_decoupling is not None:
            for layer in self.pulse_layers:
                layer.attach_dynamical_decoupling(hardware_specs)
