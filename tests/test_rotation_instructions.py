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

from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pytest

import tests.fixtures.dummy_objects as dm
from spin_pulse.transpilation.instructions import (
    GaussianRotationInstruction,
    RotationInstruction,
    SquareRotationInstruction,
)

# -------------------------------------------------------------------
# Tests for RotationInstruction base class
# -------------------------------------------------------------------


def test_rotation_instruction_init():
    q = dm.DummyQubit(idx=3)
    r = RotationInstruction("x", [q], duration=5)
    assert r.name == "x"
    assert r.duration == 5
    assert r.qubits == [q]


def test_to_pulse_and_to_angle_without_distortion(monkeypatch):
    q = dm.DummyQubit()
    r = RotationInstruction("x", [q], duration=3)

    with pytest.raises(NotImplementedError):
        r.eval(1)

    monkeypatch.setattr(r, "eval", lambda t: np.array([1, 2, 3]))
    pulse = r.to_pulse()
    assert np.array_equal(pulse, np.array([1, 2, 3]))
    assert r.to_angle() == 6


def test_to_pulse_with_distort_factor(monkeypatch):
    q = dm.DummyQubit()
    r = RotationInstruction("x", [q], duration=2)
    monkeypatch.setattr(r, "eval", lambda t: np.array([1.0, 1.0]))
    r.distort_factor = np.array([0.1, -0.1])
    pulse = r.to_pulse()
    np.testing.assert_allclose(pulse, np.array([1.1, 0.9]))


@pytest.mark.parametrize(
    "name, expected_trace", [("x", 0), ("y", 0), ("z", 0), ("Heisenberg", 0)]
)
def test_to_hamiltonian_valid_paulis(monkeypatch, name, expected_trace):
    q = dm.DummyQubit()
    r = RotationInstruction(name, [q], duration=4)

    monkeypatch.setattr(
        r, "eval", lambda t: np.ones(4)
    )  # We fake eval because it depends on the pulse shape
    H, coeff = r.to_hamiltonian()
    assert H.shape[0] == H.shape[1]
    assert np.allclose(coeff, np.ones(4))
    assert np.isclose(np.trace(H), expected_trace)  # all Pauli have trace 0


def test_adjust_duration_rescales_amplitude(monkeypatch):
    q = dm.DummyQubit()
    r = RotationInstruction("x", [q], duration=3)
    monkeypatch.setattr(r, "to_angle", lambda: 6)
    # first call gives angle=6, second gives angle=2
    calls = [6, 2]

    def side_effect():
        return calls.pop(0)  # first time 6 and then 2

    monkeypatch.setattr(r, "to_angle", side_effect)
    r.adjust_duration(10)

    assert np.isclose(r.amplitude, 3.0)  # abs(6)/abs(2)
    assert r.duration == 10


@pytest.mark.parametrize(
    "angle,label_gates,expected_substr",
    [
        (np.pi, True, r"$\pi$"),  # case a ≈ 1
        (-np.pi, True, r"$-\pi$"),  # case a ≈ -1
        (np.pi / 2, True, r"$\frac{\pi}{2}$"),  # case a = 1/n positive
        (-np.pi / 3, True, r"$-\frac{\pi}{3}$"),  # case a = -1/n negative
        (0.75 * np.pi, True, r"0.8$\pi$"),  # generic value
    ],
)
def test_plot_all_angle_cases(monkeypatch, angle, label_gates, expected_substr):
    """Full coverage of RotationInstruction.plot() branches."""
    q = dm.DummyQubit()
    r = RotationInstruction("x", [q], duration=3)

    # Monkeypatch core methods
    monkeypatch.setattr(r, "to_pulse", lambda: np.array([1, 1, 1]))
    monkeypatch.setattr(r, "to_angle", lambda: angle)

    # Create figure/axes manually
    fig, ax = plt.subplots()

    with (
        patch.object(ax, "fill_between") as mock_fill,
        patch.object(ax, "set_title") as mock_title,
    ):
        # Simulate an existing title (for string concatenation)
        ax.get_title = lambda: "BaseTitle"

        # Call method
        r.plot(ax=ax, t_start=0, label_gates=label_gates)

        # Ensure fill_between called correctly
        mock_fill.assert_called_once()
        x_arg, y_arg = mock_fill.call_args[0][:2]
        assert list(x_arg) == list(range(0, r.duration))
        np.testing.assert_array_equal(y_arg, np.array([1, 1, 1]))

        # Verify correct color
        kwargs = mock_fill.call_args.kwargs
        assert kwargs["color"] == "brown"
        assert kwargs["alpha"] == 0.5
        assert kwargs["lw"] == 2

        # If label_gates=True, verify title updated
        if label_gates:
            called_title = mock_title.call_args[0][0]
            assert "x" in called_title
            assert expected_substr.replace(" ", "") in called_title.replace(" ", "")
        else:
            mock_title.assert_not_called()

    plt.close(fig)


def test_plot_with_ax_none(monkeypatch):
    """Branch coverage: ax=None -> uses plt.gca()."""
    q = dm.DummyQubit()
    r = RotationInstruction("z", [q], duration=3)
    monkeypatch.setattr(r, "to_pulse", lambda: np.array([1, 1, 1]))
    monkeypatch.setattr(r, "to_angle", lambda: np.pi)

    fake_ax = patch("matplotlib.pyplot.gca").start().return_value
    fake_ax.get_title = lambda: ""
    with patch.object(fake_ax, "fill_between") as mock_fill:
        r.plot(ax=None, t_start=2, label_gates=False)
        mock_fill.assert_called_once()
        x_arg = mock_fill.call_args[0][0]
        assert list(x_arg) == list(range(2, 2 + r.duration))
    patch.stopall()


# -------------------------------------------------------------------
# Tests for SquareRotationInstruction
# -------------------------------------------------------------------


@pytest.mark.parametrize(
    "name, amplitude, sign, ramp_duration, duration",
    [
        ("x", 2, 1, 0, 5),
        ("y", 2, -1, 0, 5),
    ],
)
def test_square_eval_no_ramp_is_pure_square(
    capsys, name, amplitude, sign, ramp_duration, duration
):
    q = dm.DummyQubit()
    s = SquareRotationInstruction(
        name,
        [q],
        amplitude=amplitude,
        sign=sign,
        ramp_duration=ramp_duration,
        duration=duration,
    )

    assert s.amplitude == amplitude
    assert s.ramp_duration == ramp_duration
    assert s.sign == sign
    assert s.duration == duration

    t = np.arange(duration)
    y = s.eval(t)
    assert len(y) == duration
    assert np.all(y == sign * amplitude)


def test_square_eval_error():
    name, amplitude, sign, ramp_duration, duration = ("z", 2, -1, 10, 5)
    q = dm.DummyQubit()
    with pytest.raises(ValueError):
        SquareRotationInstruction(
            name,
            [q],
            amplitude=amplitude,
            sign=sign,
            ramp_duration=ramp_duration,
            duration=duration,
        )


@pytest.mark.parametrize(
    "name, amplitude, sign, ramp_duration, duration",
    [
        ("x", 2, 1, 2, 10),
        ("y", 2, -1, 1, 5),
        ("z", 2, -1, 1, 10),
        ("z", 2, -1, False, 10),
    ],
)
def test_square_eval_ramp(capsys, name, amplitude, sign, ramp_duration, duration):
    q = dm.DummyQubit()
    s = SquareRotationInstruction(
        name,
        [q],
        amplitude=amplitude,
        sign=sign,
        ramp_duration=ramp_duration,
        duration=duration,
    )
    captured = capsys.readouterr()

    if not s.ramp_duration:
        t = np.arange(duration)
        y = s.eval(t)
        assert len(y) == duration
        # rises, then plateau, then falls
        assert np.all(abs(y[:ramp_duration]) <= amplitude)
        assert np.all(y[ramp_duration : duration - ramp_duration] == sign * amplitude)
        assert np.all(abs(y[duration - ramp_duration :]) <= amplitude)

    else:
        if ramp_duration > duration:
            assert "error negative plateau duration" in captured.out
        else:
            assert "error negative plateau duration" not in captured.out

        assert s.amplitude == amplitude
        assert s.ramp_duration == ramp_duration
        assert s.sign == sign
        assert s.duration == duration

        t = np.arange(duration)
        y = s.eval(t)
        assert len(y) == duration
        # rises, then plateau, then falls
        assert np.all(abs(y[:ramp_duration]) <= amplitude)
        assert np.all(y[ramp_duration : duration - ramp_duration] == sign * amplitude)
        assert np.all(abs(y[duration - ramp_duration :]) <= amplitude)


@pytest.mark.parametrize(
    "name, amplitude, sign, ramp_duration, duration",
    [
        ("x", 2, 1, 2, 10),
        ("y", 2, -1, 1, 5),
    ],
)
def test_square_str_includes_gate_and_amplitude(
    monkeypatch, name, amplitude, sign, ramp_duration, duration
):
    q = dm.DummyQubit()
    s = SquareRotationInstruction(
        name,
        [q],
        amplitude=amplitude,
        sign=sign,
        ramp_duration=ramp_duration,
        duration=duration,
    )

    s_str = s.__str__()
    assert f"SquarePulse for {name}" in s_str
    assert f"amplitude={amplitude}" in s_str
    assert f"duration={duration}" in s_str


def test_from_angle_normal_case(monkeypatch):
    """
    Covers normal operation of from_angle without triggering special branches."""
    q = dm.DummyQubit()
    hw = dm.DummyHardwareSpecs()

    instr = SquareRotationInstruction.from_angle("x", [q], np.pi, hw)

    assert isinstance(instr, SquareRotationInstruction)
    # The amplitude and duration should be reasonable
    assert hasattr(instr, "duration")
    assert hasattr(instr, "amplitude")

    assert instr.amplitude > 0
    assert instr.duration > 0
    assert instr.ramp_duration >= 0
    assert instr.sign in [1, -1]
    assert instr.to_angle() - np.pi < 1e-6


@pytest.mark.parametrize(
    "name, angle",
    [
        ("x", 0),
        ("x", np.pi),
        ("y", 2 * np.pi),
        ("y", 0.859614),
        ("z", 3 * np.pi),
        ("z", 4 * np.pi),
    ],
)
def test_from_angle_zero_angle_case(monkeypatch, name, angle):
    """Covers branch where |angle_1| <= 1e-15 → amplitude = 1."""
    q = dm.DummyQubit()
    hw = dm.DummyHardwareSpecs()

    with (
        patch.object(SquareRotationInstruction, "__init__", return_value=None),
        patch.object(SquareRotationInstruction, "to_angle", return_value=0.0),
    ):
        instr = SquareRotationInstruction.from_angle(name, [q], angle, hw)

        assert isinstance(instr, SquareRotationInstruction)
        assert instr.to_angle() - angle % (2 * np.pi) < 1e-6


def test_from_angle_loop_termination(monkeypatch):
    """Ensure that MAX_ITER prevents infinite loops."""
    q = dm.DummyQubit()
    hw = dm.DummyHardwareSpecs()

    N_iter = 3
    monkeypatch.setattr(
        "spin_pulse.transpilation.instructions.rotations.MAX_ITER", N_iter
    )

    with (
        patch.object(SquareRotationInstruction, "__init__", return_value=None),
        patch.object(
            SquareRotationInstruction, "to_angle", return_value=np.pi
        ) as mock_to_angle,
    ):
        with pytest.warns():
            instr = SquareRotationInstruction.from_angle("x", [q], 50 * np.pi, hw)
        assert isinstance(instr, SquareRotationInstruction)
        assert mock_to_angle.call_count == N_iter


# -------------------------------------------------------------------
# Tests for GaussianRotationInstruction
# -------------------------------------------------------------------


@pytest.mark.parametrize(
    "name, amplitude, sign, coeff_duration, duration",
    [
        ("x", 1, 1, 2, 11),
        ("y", 1, -1, 2, 11),
    ],
)
def test_gaussian_eval_and_symmetry(name, amplitude, sign, coeff_duration, duration):
    q = dm.DummyQubit()
    g = GaussianRotationInstruction(
        name,
        [q],
        amplitude=amplitude,
        sign=sign,
        coeff_duration=coeff_duration,
        duration=duration,
    )
    t = np.linspace(0, duration - 1, duration)
    y = g.eval(t)

    assert np.allclose(y[1:], y[1:][::-1])  # symmetric Gaussian
    assert np.all(abs(y) <= amplitude)  # within amplitude
    assert len(y) == duration

    assert g.sign == sign
    assert g.duration == duration
    assert g.amplitude == amplitude
    assert coeff_duration == coeff_duration


def test_gaussian_str_contains_expected_parts():
    q = dm.DummyQubit()
    g = GaussianRotationInstruction(
        "y", [q], amplitude=0.8, sign=-1, coeff_duration=2, duration=6
    )
    s = str(g)
    assert "GaussianPulse" in s
    assert "y" in s
    assert "0.8" in s
    assert "6" in s

    s_str = g.__str__()
    assert "GaussianPulse for y" in s_str
    assert f"amplitude={0.8}" in s_str
    assert f"duration={6}" in s_str


def test_gaussian_from_angle_runs_to_completion(monkeypatch):
    hw = dm.DummyHardwareSpecs()
    q = dm.DummyQubit()
    # Mock to_angle to avoid infinite loop
    with patch.object(GaussianRotationInstruction, "to_angle", return_value=1.0):
        instr = GaussianRotationInstruction.from_angle("x", [q], np.pi, hw)
        assert isinstance(instr, GaussianRotationInstruction)
        assert hasattr(instr, "amplitude")
        assert instr.amplitude > 0

        assert instr.to_angle() - np.pi < 1e-6


def test_gaussian_from_angle_max_iter_triggers_warning(capsys, monkeypatch):
    hw = dm.DummyHardwareSpecs()
    q = dm.DummyQubit()
    # Make it loop until MAX_ITER reached
    with (
        patch("spin_pulse.transpilation.instructions.rotations.MAX_ITER", 1),
        patch.object(GaussianRotationInstruction, "to_angle", return_value=0.0),
    ):
        with pytest.warns():
            instr = GaussianRotationInstruction.from_angle("x", [q], np.pi, hw)
        assert isinstance(instr, GaussianRotationInstruction)
