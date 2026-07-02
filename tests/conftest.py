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

import matplotlib

from tests.fixtures.pulse_circuit_fixtures import *  # noqa: F403
from tests.fixtures.utils_fixtures import *  # noqa: F403

matplotlib.use(
    "Agg"
)  # To avoid RuntimeError: main thread is not in main loop from tkinter
