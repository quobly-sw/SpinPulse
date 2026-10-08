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
