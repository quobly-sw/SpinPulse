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
"""Description of rotations at the pulse level."""

import warnings

import matplotlib.pyplot as plt
import numpy as np
from qiskit.quantum_info import Pauli

from .pulse_instruction import PulseInstruction

MAX_DURATION = 1e5
MAX_ITER = 1e5


class RotationInstruction(PulseInstruction):
    """Base class for single- and two-qubit rotation pulse instructions.

    A rotation instruction represents a shaped control pulse that generates
    a rotation around a given axis or interaction, such as x, y, z, or
    Heisenberg. Subclasses define the time-dependent envelope via the
    ``eval`` method and provide constructors from a target rotation angle.

    Attributes:
        - name (str): Label of the generating operator, for example
          "x", "y", "z", or "Heisenberg".
        - qubits (list[qiskit.circuit.Qubit]): List of qubits on which the rotation acts.
        - num_qubits (int): Number of qubits targeted by the instruction.
        - duration (int): Duration of the pulse in time steps.
        - amplitude (float): Optional amplitude parameter used to rescale
          the pulse envelope when adjusting the duration.

    """

    def __init__(self, name, qubits, duration):
        """Initialize a rotation instruction with a given name and duration.

        Parameters:
            name (str): Name of the generating operator, for example
              "x", "y", "z", or "Heisenberg".
            qubits (list[qiskit.circuit.Qubit]): List of qubits on which the rotation is applied.
            duration (int): Duration of the pulse in time steps.

        Returns:
            None: The rotation instruction is stored in the created object.

        """
        super().__init__(qubits, duration)
        self.name = name

    @classmethod
    def from_angle(cls, name, qubits, angle, hardware_specs, low_duration, shape_param):
        """Construct a rotation instruction from a target angle.

        This class method must be called by subclasses to map a target
        rotation angle to a concrete pulse shape and duration, using the
        hardware specifications.

        Parameters:
            name (str): Name of the generating operator, for example
              "x", "y", "z", or "Heisenberg".
            qubits (list[qiskit.circuit.Qubit]): List of qubits on which the rotation is applied.
            angle (float): Target rotation angle in radians.
            hardware_specs (HardwareSpecs): Hardware configuration used
              to determine pulse shape and duration.

        Returns:
            RotationInstruction: Instance of the subclass implementing the requested rotation.

        """
        sign = np.sign(angle)

        prev_duration = -1
        prev_low = -1
        prev_high = -1
        high_duration = MAX_DURATION
        cpt = 0

        while cpt < MAX_ITER and low_duration <= high_duration:
            duration = low_duration + (high_duration - low_duration) // 2

            if (
                duration == prev_duration
                and low_duration == prev_low
                and high_duration == prev_high
            ):
                break

            prev_duration = duration
            prev_low = low_duration
            prev_high = high_duration

            amplitude_1 = 1
            instruction = cls(name, qubits, amplitude_1, sign, shape_param, duration)
            angle_1 = instruction.to_angle()
            if np.abs(angle_1) > 1e-15:
                amplitude = np.abs(angle) / np.abs(angle_1)
            else:
                amplitude = np.inf
            if amplitude < hardware_specs.fields[name]:
                high_duration = duration - 1
            elif np.isclose(amplitude, hardware_specs.fields[name], atol=1e-16):
                pass
            else:
                low_duration = duration + 1
            cpt += 1

        if cpt >= MAX_ITER:
            warnings.warn(
                f"Maximum of iterations reached for angle={angle}, duration={duration}"
            )

        # if amplitude >= hardware_specs.fields[name] + 1e-10:
        #     duration += 1
        #     amplitude_1 = 1
        #     instruction = cls(name, qubits, amplitude_1, sign, shape_param, duration)
        #     angle_1 = instruction.to_angle()
        #     if np.abs(angle_1) > 1e-15:
        #         amplitude = np.abs(angle) / np.abs(angle_1)
        #     if amplitude >= hardware_specs.fields[name] + 1e-10:
        #         warnings.warn(
        #             f"Amplitude is higher ({amplitude}) than what the hardware allows ({hardware_specs.fields[name]})"
        #         )

        instruction = cls(name, qubits, amplitude, sign, shape_param, duration)

        return instruction

    def eval(self, t):
        """Evaluate the pulse envelope at the given time indices.

        Subclasses must implement this method to return the pulse amplitude
        at each time step contained in the input array.

        Parameters:
            t (ndarray or array-like): Discrete time indices at which
              the pulse envelope is evaluated.

        Returns:
            ndarray: Pulse amplitudes at the specified time indices.

        Raises:
            NotImplementedError: If the method is not implemented in
              the subclass.

        """
        raise NotImplementedError

    def to_pulse(self):
        """Return the full pulse envelope over the instruction duration.

        The envelope is evaluated on the interval from 0 to duration-1.
        If the instruction has an attribute ``distort_factor``, the pulse
        is corrected by an additional multiplicative factor.

        Returns:
            ndarray: Pulse amplitudes over the full duration.

        """
        t_rel = np.arange(self.duration)
        pulse = self.eval(t_rel)
        if hasattr(self, "distort_factor"):
            pulse += pulse * self.distort_factor
        return pulse

    def to_angle(self):
        """Compute the effective rotation angle generated by the pulse.

        The effective angle is taken as the sum of all pulse amplitudes
        over the duration of the instruction.

        Returns:
            float: Total rotation angle in radians.

        """
        return sum(self.to_pulse())

    def to_hamiltonian(self):
        """Build the generator Hamiltonian associated with this rotation.

        The method returns the static Hamiltonian matrix corresponding to
        the chosen generating operator, together with the time-dependent
        scalar coefficients given by the pulse envelope.

        For example:

        - "x" uses the Pauli :math:`X` operator.
        - "y" uses the Pauli :math:`Y` operator.
        - "z" uses the Pauli :math:`Z` operator.
        - "Heisenberg" uses the sum of :math:`XX`, :math:`YY`, and :math:`ZZ` interactions.

        Returns:
            - tuple(ndarray,ndarray): Tuple containing the Hamiltonian matrix of size :math:`2^n` by :math:`2^n`, where :math:`n` is the number of qubits, and
            the time-dependent coefficients defined by the pulse envelope.

        """
        coeff = self.to_pulse()
        name = self.name
        if name == "x":
            H = 0.5 * Pauli("X").to_matrix()
        elif name == "y":
            H = 0.5 * Pauli("Y").to_matrix()
        elif name == "z":
            H = 0.5 * Pauli("Z").to_matrix()
        elif name == "Heisenberg":
            H = 0.5 * (
                Pauli("XX").to_matrix()
                + Pauli("YY").to_matrix()
                + Pauli("ZZ").to_matrix()
            )

        return H, coeff

    def adjust_duration(self, duration):
        """Rescale the pulse amplitude to match a new duration.

        The total rotation angle is preserved by adjusting the pulse
        amplitude after changing the duration. The method computes the
        original angle, sets a new duration, evaluates the new angle, and
        rescales the amplitude so that the final effective angle remains
        unchanged.

        Parameters:
            duration (int): New duration of the pulse in time steps.

        Returns:
            None: The internal duration and amplitude are updated in place.

        """
        angle = self.to_angle()
        self.duration = duration
        self.amplitude = 1
        angle_1 = self.to_angle()

        self.amplitude = np.abs(angle) / np.abs(angle_1)

    def plot(self, ax=None, t_start=0, label_gates=True):
        """Plot the pulse envelope as a filled region.

        The pulse is drawn over the interval from ``t_start`` to
        ``t_start + duration`` using a color that depends on the
        generating operator name. Optionally, a label containing the gate
        name and rotation angle is appended to the axis title.

        Parameters:
            ax (matplotlib axis, optional): Axis on which to draw the pulse.
              If None, the current axis is used.
            t_start (int): Starting time index of the pulse on the plot.
              Default is 0.
            label_gates (bool): If True, appends a textual label with the
              gate name and rotation angle to the axis title. Default is True.

        Returns:
            None: The pulse envelope is drawn on the provided axis.

        """
        if ax is None:
            ax = plt.gca()

        name = self.name
        color_dict = {
            "x": "brown",
            "y": "brown",
            "z": "deepskyblue",
            "Heisenberg": "deepskyblue",
        }

        a = self.to_angle() / np.pi
        if np.abs(a - 1) < 1e-10:
            latex_str = r" $\pi$"
        elif np.abs(a + 1) < 1e-10:
            latex_str = r" $-\pi$"
        elif np.abs(1 / a - round(1 / a)) < 1e-10:
            if a > 0:
                latex_str = rf" $\frac{{\pi}}{{{abs(round(1 / a))}}}$"
            else:
                latex_str = rf" $-\frac{{\pi}}{{{abs(round(1 / a))}}}$"
        else:
            latex_str = " {:.1f}".format(a) + r"$\pi$"

        ax.fill_between(
            range(t_start, t_start + self.duration),
            self.to_pulse(),
            0,
            color=color_dict[name],
            alpha=0.5,
            lw=2,
        )
        if label_gates:
            ax.set_title(ax.get_title() + " " + name + latex_str, color="#24185E")


class SquareRotationInstruction(RotationInstruction):
    """Square-shaped rotation pulse with optional linear ramps.

    This instruction implements a piecewise-linear pulse envelope composed of
    a rising ramp, a constant plateau, and a falling ramp. The total duration
    and amplitude determine the effective rotation angle. The sign parameter
    controls the direction of the rotation.

    Attributes:
        - name (str): Name of the generating operator, for example
          "x", "y", "z", or "Heisenberg".
        - qubits (list[qiskit.circuit.Qubit]): List of qubits on which the rotation is applied.
        - num_qubits (int): Number of qubits targeted by the instruction.
        - duration (int): Total duration of the pulse in time steps.
        - amplitude (float): Pulse amplitude during the plateau.
        - sign (float): Sign of the rotation, typically +1 or -1.
        - ramp_duration (int): Duration of each ramp region at the beginning
          and end of the pulse.

    """

    def __init__(self, name, qubits, amplitude, sign, ramp_duration, duration):
        """Initialize a square rotation pulse with ramps and plateau.

        The pulse is defined by a linear ramp up, a constant-amplitude plateau,
        and a linear ramp down. The plateau duration is given by
        ``duration - 2 * ramp_duration`` and must be non-negative.

        Parameters:
            name (str): Name of the generating operator, for example
              "x", "y", "z", or "Heisenberg".
            qubits (list[qiskit.circuit.Qubit]): List of qubits on which the rotation is applied.
            amplitude (float): Amplitude of the square plateau.
            sign (float): Sign of the rotation, typically +1 or -1.
            ramp_duration (int): Duration of each linear ramp region.
            duration (int): Total duration of the pulse in time steps.

        Returns:
            None: The square rotation instruction is stored in the created object.

        Raises:
            ValueError: If the plateau duration is negative, i.e.,
                ``duration < 2 * ramp_duration``.

        """
        super().__init__(name, qubits, duration)
        self.amplitude = amplitude
        self.sign = sign
        self.ramp_duration = ramp_duration
        self.duration = int(duration)
        plateau_duration = duration - 2 * ramp_duration
        if plateau_duration < 0:
            raise ValueError(
                f"Negative plateau duration: duration={duration}, "
                f"ramp_duration={ramp_duration}, plateau={plateau_duration}"
            )

    @classmethod
    def from_angle(cls, name, qubits, angle, hardware_specs):
        """Build a square pulse that performs a rotation of the target angle.

        A binary search is used over the pulse duration to find a combination
        of duration and amplitude that matches the requested angle while
        respecting the hardware field limits specified in ``hardware_specs``.
        The procedure starts from the minimal duration compatible with the
        ramp time and iteratively refines the duration.

        Parameters:
            name (str): Name of the generating operator, for example
              "x", "y", "z", or "Heisenberg".
            qubits (list[qiskit.circuit.Qubit]): List of qubits on which the rotation is applied.
            angle (float): Target rotation angle in radians.
            hardware_specs (HardwareSpecs): Hardware configuration providing
              ramp duration and maximum allowed field strengths.

        Returns:
            SquareRotationInstruction: A square pulse instruction that performs the requested rotation with the target angle within the hardware limits.

        """
        return super().from_angle(
            name,
            qubits,
            angle,
            hardware_specs,
            2 * hardware_specs.ramp_duration + 1,
            hardware_specs.ramp_duration,
        )

    def eval(self, t):
        """Evaluate the square pulse envelope at the given time indices.

        The envelope is composed of:

        - a linear rise from 0 to the target amplitude during the first ramp,
        - a constant plateau with full amplitude,
        - a symmetric linear fall back to 0 during the final ramp.

        If ``ramp_duration`` is zero, a pure square pulse with constant height
        is returned over the whole duration.

        Parameters:
            t (ndarray or array-like): Discrete time indices at which
              the pulse envelope is evaluated.

        Returns:
            ndarray: Pulse amplitudes at the specified time indices.

        """
        height = self.sign * self.amplitude
        plateau_duration = self.duration - 2 * self.ramp_duration
        if not self.ramp_duration:  # "pure" square?
            return height * (t < self.duration)
        else:
            t1 = 0
            t2 = t1 + self.ramp_duration
            t3 = t2 + plateau_duration
            t4 = t3 + self.ramp_duration
            y_rise = height * (t / t2) * (t < t2)
            y_constant = height * (t >= t2) * (t < t3)
            y_fall = height * (1 + ((t3 - t - 1) / t2)) * (t >= t3) * (t < t4)
            return y_rise + y_constant + y_fall

    def __str__(self):
        """Return a readable string representation of the square pulse.

        Returns:
            str: Description including the operator name, amplitude,
              and duration.

        """
        return f"SquarePulse for {self.name}, amplitude={self.amplitude}, duration={self.duration} "


class GaussianRotationInstruction(RotationInstruction):
    """Gaussian-shaped rotation pulse instruction.

    This instruction implements a Gaussian pulse envelope centered in time,
    with width controlled by ``coeff_duration`` and amplitude chosen to
    generate a target rotation angle. The sign parameter controls the
    direction of the rotation.

    Attributes:
        - name (str): Name of the generating operator, for example
          "x", "y", "z", or "Heisenberg".
        - qubits (list[qiskit.circuit.Qubit]): List of qubits on which the rotation is applied.
        - num_qubits (int): Number of qubits targeted by the instruction.
        - duration (int): Total duration of the pulse in time steps.
        - amplitude (float): Peak amplitude of the Gaussian pulse.
        - coeff_duration (float): Factor relating pulse duration to the
          Gaussian standard deviation.
        - sign (float): Sign of the rotation, typically +1 or -1.

    """

    def __init__(self, name, qubits, amplitude, sign, coeff_duration, duration):
        """Initialize a Gaussian rotation pulse with given parameters.

        The pulse is modeled as a Gaussian envelope with a peak amplitude
        given by ``amplitude``, centered in the middle of the time window,
        and with a standard deviation derived from ``duration`` and
        ``coeff_duration``.

        Parameters:
            name (str): Name of the generating operator, for example
              "x", "y", "z", or "Heisenberg".
            qubits (list[qiskit.circuit.Qubit]): List of qubits on which the rotation is applied.
            amplitude (float): Peak amplitude of the Gaussian pulse.
            sign (float): Sign of the rotation, typically +1 or -1.
            coeff_duration (float): Factor used to set the Gaussian width
              ``sigma = duration / coeff_duration``.
            duration (int): Total duration of the pulse in time steps.

        Returns:
            None: The Gaussian rotation instruction is stored in the created object.

        """
        super().__init__(name, qubits, duration=int(duration))
        self.amplitude = amplitude
        self.coeff_duration = coeff_duration
        self.sign = sign

    @classmethod
    def from_angle(cls, name, qubits, angle, hardware_specs):
        """Build a Gaussian pulse that performs a target rotation angle.

        A search over pulse duration is performed to find a configuration
        where the peak amplitude remains below the hardware field limit and
        the total integrated area matches the requested angle. The Gaussian
        width is controlled by ``hardware_specs.coeff_duration``.

        Parameters:
            name (str): Name of the generating operator, for example
              "x", "y", "z", or "Heisenberg".
            qubits (list[qiskit.circuit.Qubit]): List of qubits on which the rotation is applied.
            angle (float): Target rotation angle in radians.
            hardware_specs (HardwareSpecs): Hardware configuration providing
              maximum allowed field strengths and the duration coefficient.

        Returns:
            GaussianRotationInstruction: A Gaussian pulse instruction that performs the requested angle within the hardware limits.

        """

        return super().from_angle(
            name,
            qubits,
            angle,
            hardware_specs,
            1,
            hardware_specs.coeff_duration,
        )

    def eval(self, t):
        """Evaluate the Gaussian pulse envelope at the given time indices.

        The envelope is defined as ``amplitude * sign * exp( - (t - t0)^2 / (2 * sigma^2) )``.

        where ``t0`` is the center of the pulse window and ``sigma`` is given by
        ``duration/coeff_duration``.

        Parameters:
            t (ndarray or array-like): Discrete time indices at which
              the pulse envelope is evaluated.

        Returns:
            ndarray: Pulse amplitudes at the specified time indices.

        """

        sigma = self.duration / self.coeff_duration
        t0 = self.duration / 2
        return self.sign * self.amplitude * np.exp(-((t - t0) ** 2) / (2 * sigma**2))

    def __str__(self):
        """Return a readable string representation of the Gaussian pulse.

        Returns:
            str: Description including the operator name, amplitude,
              and duration.

        """
        return f"GaussianPulse for {self.name}, amplitude={self.amplitude}, duration={self.duration} "
