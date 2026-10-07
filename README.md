<img src="https://github.com/quobly-sw/SpinPulse/raw/main/assets/SpinPulse.png" width=200>

[![Doc](https://img.shields.io/badge/Doc-dev-green.svg)](https://quobly-sw.github.io/SpinPulse)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![arXiv](https://img.shields.io/badge/arXiv-2601.10435-b31b1b.svg)](https://arxiv.org/abs/2601.10435)


**`SpinPulse`** is an open-source **`python`** package for simulating spin qubit-based quantum computers at the pulse-level.
**`SpinPulse`** models the specific physics of spin qubits, particularly through the inclusion of classical non-Markovian noise.
This enables realistic simulations of native gates and noise-accurate quantum circuits, in order to support hardware development.

This code is licensed under the Apache License, Version 2.0.

<img src="https://github.com/quobly-sw/SpinPulse/raw/main/docs/source/customapi/figures/global_fig.png">


## Installation

You can install **SpinPulse** by running the following command from the root of the repository:
```
    pip install spin-pulse
```

For more information consult the [installation documentation](https://quobly-sw.github.io/SpinPulse/customapi/installation/index.html) page.

## API & Documentation

The API documentation can be found at [APIdoc](https://quobly-sw.github.io/SpinPulse/) and provides a detailed description of the **`SpinPulse`** package architecture, including its core modules, classes and functions.

Detailed information on our model and the code structure is presented in our [publication](https://arxiv.org/abs/2601.10435)

## Large pink-noise environments

*Documentation added by qBraid in 2026 for the pink-noise memory optimization.*

Long experiments can keep noise histories on disk by passing an existing disk
directory to `ExperimentalEnvironment`:

```python
env = ExperimentalEnvironment(
    hardware_specs=specs,
    T2S=10_000,
    TJS=5_000,
    duration=experiment_duration,
    segment_duration=experiment_duration,
    seed=42,
    noise_directory="/path/to/job-scratch",
)
```

The trace values are NumPy `memmap` arrays, so pulse circuits continue to read
the same windows from the full history. This preserves correlations across
shots, the spectral amplitudes, random phase stream, trace duration,
low-frequency cutoff, and noise normalization. The exact-length FFT is split
into smaller transforms without changing the time grid or precision
(`complex128` transforms and `float64` samples). Different FFT evaluation order
can change the last few floating-point bits; bitwise identity with earlier
versions, including a seeded measurement exactly at a sampling boundary, is
not guaranteed.

The default remains in-memory storage. Both modes avoid full-size phase and
variance temporaries and concatenation copies. Disk mode also stores the FFT
spectrum on disk. Memory still depends on the largest constituent FFT and the
number of concurrent workers; it is not constant for every trace length.
Clean mapped pages are reclaimable by the operating system under memory
pressure. A RAM-backed filesystem such as tmpfs does not provide these savings.

Allow disk space for eight bytes per sample per retained trace, plus a
sixteen-byte-per-sample FFT spectrum and any additional segment needed when
`duration != segment_duration`. Pink noise has one frequency trace per qubit
and, when `TJS` is set, one coupling trace per neighboring pair. Temporary
history files are created in private subdirectories, and backing files and
subdirectories are reclaimed when their arrays and all views are released,
including after exceptions. Use a private job directory and remove it after
the job exits: forced termination (or Windows interpreter shutdown) can leave
named history files behind. On platforms with
`posix_fallocate`, space is reserved before mapping so disk exhaustion raises
an exception before writes to that array.

`noise_directory` currently applies to pink noise. Trace generation uses
threads in disk mode to preserve shared mappings. Shot execution through
`PulseCircuit.run_experiment` can use process workers, which reopen the same
named history files. Prefer `n_jobs=1` when minimizing concurrent workspaces.
Keep the original environment alive while workers use it. Other serialization
methods or explicitly copying a whole mapped array can materialize its complete
contents in RAM.

## Citing

If you use **`SpinPulse`** in your research work, please cite our publication

```latex
@misc{vermersch2026spinpulse,
      title={The SpinPulse library for transpilation and noise-accurate simulation of spin qubit quantum computers},
      author={Beno\^it Vermersch, Oscar Gravier, Nathan Miscopein, Julia Guignon, Carlos Ramos Marim\'on, Jonathan Durandau, Matthieu Dartiailh, Tristan Meunier and Valentin Savin},
      year={2026},
      eprint={2601.10435},
      archivePrefix={arXiv},
      primaryClass={quant-ph},
      url={https://arxiv.org/abs/2601.10435},
}
```
