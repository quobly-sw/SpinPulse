# Copyright (C) Quobly 2025. Licensed under the Apache License, Version 2.0.
# Added by qBraid in 2026 to verify memory optimizations against SpinPulse 1.1.3.
"""Differential tests against the released Fourier construction."""

import numpy as np
import pytest

from spin_pulse import ExperimentalEnvironment, HardwareSpecs, PulseCircuit, Shape
from spin_pulse.environment.noise.pink import (
    PinkNoiseTimeTrace,
    get_pink_noise,
    get_pink_noise_with_repetitions,
)


def released_pink_noise(n, seed):
    """Unmodified numerical construction from SpinPulse v1.1.3, pink.py."""
    rng = np.random.default_rng(seed=seed)
    n2 = n // 2 - 1
    f = np.arange(2, n2 + 2)
    amplitude = 1 / (f ** (1.0 / 2))
    phase = (rng.uniform(size=n2) - 0.5) * 2 * np.pi
    positive = amplitude * np.exp(1j * phase)
    spectrum = np.concatenate(
        ([0], positive, [1 / ((n2 + 2) ** 1.0)], np.flipud(np.conj(positive)))
    )
    return n * np.real(np.fft.ifft(spectrum))


@pytest.mark.parametrize("n", [2, 10, 190, 65536, 2000006, 2301610])
@pytest.mark.parametrize("seed", [0, 42])
def test_noise_matches_released_samples(n, seed):
    expected = released_pink_noise(n, seed)
    actual = get_pink_noise(n, seed)
    np.testing.assert_allclose(
        actual, expected, rtol=0, atol=5e-14 * max(1, np.max(np.abs(expected)))
    )


def test_disk_trace_preserves_repetitions_scaling_and_ramsey_contrast(tmp_path):
    duration, segment_duration, t2s = 381, 190, 100.0
    expected = np.tile(released_pink_noise(segment_duration, 0), 3)[:duration]
    expected *= (
        2 * np.pi * np.sqrt(1 / (4 * np.pi**2 * np.log(segment_duration) * t2s**2))
    )
    trace = PinkNoiseTimeTrace(
        t2s, duration, segment_duration, seed=0, noise_directory=tmp_path
    )
    assert isinstance(trace.values, np.memmap)
    assert trace.values.dtype == np.float64
    np.testing.assert_allclose(trace.values, expected, rtol=0, atol=1e-15)
    assert trace.sigma == pytest.approx(np.std(expected), rel=1e-14)
    contrast = np.mean(
        [np.real(np.exp(-1j * np.cumsum(x))) for x in expected[:380].reshape(20, 19)],
        axis=0,
    )
    np.testing.assert_allclose(trace.ramsey_contrast(19), contrast, atol=1e-14)


def test_repetition_generator_continues_rng_instead_of_restarting():
    reference_rng = np.random.default_rng(7)
    expected = np.concatenate(
        [released_pink_noise(190, reference_rng) for _ in range(3)]
    )[:401]
    actual = get_pink_noise_with_repetitions(401, 190, np.random.default_rng(7))
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-13)


@pytest.mark.parametrize("n", [2, 65536, 2000006, 2301610])
def test_disk_fft_matches_released_samples(tmp_path, n):
    expected = released_pink_noise(n, np.random.SeedSequence(7, spawn_key=(0, 1)))
    actual = get_pink_noise(
        n, np.random.SeedSequence(7, spawn_key=(0, 1)), noise_directory=tmp_path
    )
    assert isinstance(actual, np.memmap)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)


def hardware_specs():
    return HardwareSpecs(
        num_qubits=2,
        B_field=0.05,
        delta=0.1,
        J_coupling=0.03,
        rotation_shape=Shape.GAUSSIAN,
    )


def test_disk_environment_keeps_streams_across_parallel_redraws(tmp_path):
    from joblib import parallel_config

    options = {
        "hardware_specs": hardware_specs(),
        "duration": 1026,
        "segment_duration": 1026,
        "T2S": 100,
        "TJS": 200,
        "seed": 0,
    }
    reference = ExperimentalEnvironment(**options)
    actual = ExperimentalEnvironment(**options, noise_directory=tmp_path)
    for generation in range(3):
        if generation:
            reference.generate_time_traces()
            # An ambient process preference must not move the backing-file
            # owner into a worker that exits before the parent can reopen it.
            with parallel_config(backend="loky"):
                actual.generate_time_traces(n_jobs=2)
        for old, new in zip(
            reference.time_traces + reference.time_traces_coupling,
            actual.time_traces + actual.time_traces_coupling,
            strict=True,
        ):
            assert isinstance(new.values, np.memmap)
            np.testing.assert_array_equal(old.values, new.values)


def test_disk_environment_uses_custom_pink_noise_generator(tmp_path):
    class CustomPinkNoise(PinkNoiseTimeTrace):
        def get_analytical_contrast(self, idle_duration):
            return 0.25

    env = ExperimentalEnvironment(hardware_specs(), noise_directory=tmp_path)
    env.noise_generator = CustomPinkNoise
    env.generate_time_traces()
    assert env.get_analytical_contrast(10) == 0.25


def test_older_pickled_environment_can_regenerate_noise():
    import pickle

    from tests.fixtures.dummy_objects import DummyHardwareSpecs

    env = ExperimentalEnvironment(DummyHardwareSpecs(num_qubits=2), seed=7)
    # Serialized pre-option environments contain precisely these older fields.
    del env.noise_directory
    restored = pickle.loads(pickle.dumps(env))
    old_values = restored.time_traces[0].values.copy()
    restored.generate_time_traces()
    assert restored.noise_directory is None
    assert not np.array_equal(old_values, restored.time_traces[0].values)


def test_history_files_have_a_private_directory_even_in_shared_scratch(tmp_path):
    import os
    import stat
    from pathlib import Path

    values = get_pink_noise(1026, 7, noise_directory=tmp_path)
    directory = Path(values.filename).parent
    assert directory.parent == tmp_path
    assert directory != tmp_path
    if os.name == "posix":
        assert stat.S_IMODE(directory.stat().st_mode) == 0o700
        assert stat.S_IMODE(Path(values.filename).stat().st_mode) == 0o600


def test_failed_disk_reservation_cleans_up_history_directory(tmp_path):
    import os
    import subprocess
    import sys

    if os.name != "posix":
        pytest.skip("Requires POSIX file-size resource limits")
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import errno
import resource
import signal
import sys
from spin_pulse.environment.noise.pink import get_pink_noise_with_repetitions

signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
resource.setrlimit(resource.RLIMIT_FSIZE, (8192, 8192))
try:
    get_pink_noise_with_repetitions(1000000, 2, seed=0, noise_directory=sys.argv[1])
except OSError as exc:
    assert exc.errno == errno.EFBIG, exc
else:
    raise AssertionError("Allocation unexpectedly bypassed the file-size limit")
""",
            str(tmp_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert list(tmp_path.iterdir()) == []


def test_noisy_unitaries_match_released_noise_across_shot_windows(tmp_path):
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator, Statevector

    circuit = QuantumCircuit(2)
    circuit.rx(1.0, 1)
    circuit.rz(0.7, 0)
    circuit.rx(0.3, 0)
    circuit.rzz(1.2, 0, 1)
    specs = hardware_specs()
    pulse = PulseCircuit.from_circuit(specs.gate_transpile(circuit), specs)
    duration = pulse.duration * 13
    duration += duration % 2
    env = ExperimentalEnvironment(
        specs,
        T2S=100,
        TJS=200,
        duration=duration,
        segment_duration=duration,
        seed=42,
        noise_directory=tmp_path,
    )
    root_qubits, root_couplings = np.random.SeedSequence(42).spawn(2)
    seeds = root_qubits.spawn(2) + root_couplings.spawn(1)
    traces = env.time_traces + env.time_traces_coupling
    released_values = [
        released_pink_noise(duration, seed) * (2 * np.pi * np.sqrt(trace.S0))
        for trace, seed in zip(traces, seeds, strict=True)
    ]
    mapped_values = [trace.values for trace in traces]
    for shot in [0, 1, 6, 11, 12]:
        operators, probabilities = [], []
        for values in [released_values, mapped_values]:
            for trace, value in zip(traces, values, strict=True):
                trace.values = value
            pulse.t_lab = shot * pulse.duration
            pulse.attach_time_traces(env)
            noisy = pulse.to_circuit()
            operators.append(Operator(noisy).data)
            probabilities.append(Statevector.from_instruction(noisy).probabilities())
        np.testing.assert_allclose(operators[0], operators[1], rtol=0, atol=1e-12)
        np.testing.assert_allclose(
            probabilities[0], probabilities[1], rtol=0, atol=1e-12
        )


def test_disk_arrays_release_files_when_last_view_is_released(tmp_path):
    import gc
    import os

    if not os.path.isdir("/proc/self/fd"):
        pytest.skip("Linux file descriptor lifecycle check")

    def backing_files():
        found = []
        for name in os.listdir("/proc/self/fd"):
            try:
                target = os.readlink("/proc/self/fd/" + name)
            except FileNotFoundError:
                continue
            if str(tmp_path) in target:
                found.append(target)
        return found

    values = get_pink_noise(1026, 7, noise_directory=tmp_path)
    view = values[10:20]
    del values
    gc.collect()
    assert len(backing_files()) == 1
    assert np.all(np.isfinite(view))
    del view
    gc.collect()
    assert backing_files() == []
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("length", [0, -2, 1, 3])
def test_invalid_segment_fails_before_creating_disk_files(tmp_path, length):
    with pytest.raises(ValueError, match="even and at least 2"):
        get_pink_noise(length, noise_directory=tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_seeded_counts_match_with_disk_histories_and_process_workers(tmp_path):
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator

    circuit = QuantumCircuit(2)
    circuit.rx(1.0, 1)
    circuit.rz(0.7, 0)
    circuit.rx(0.3, 0)
    circuit.rzz(1.2, 0, 1)
    circuit.measure_all()
    specs = hardware_specs()
    pulse = PulseCircuit.from_circuit(specs.gate_transpile(circuit), specs)
    duration = pulse.duration * 102
    duration += duration % 2
    results = []
    for directory, workers in [(None, 1), (tmp_path, 1), (tmp_path, 2)]:
        env = ExperimentalEnvironment(
            specs,
            T2S=100,
            TJS=200,
            duration=duration,
            segment_duration=duration,
            seed=7,
            noise_directory=directory,
        )
        simulator = AerSimulator(seed_simulator=7, max_parallel_threads=1)
        results.append(
            pulse.run_experiment(
                env,
                simulator,
                num_samples=100,
                progress_bar=False,
                seed_progression_function=lambda seed: seed + 70,
                n_jobs=workers,
            )
        )
    assert sum(results[0].values()) == 100
    assert results[0] == results[1] == results[2]
