"""pytest plugin: a ``ulp`` fixture exposing :func:`specaudit.testing.assert_max_ulp`.

Enabled automatically through the ``pytest11`` entry point once specaudit is installed::

    def test_my_erfinv(ulp):
        ulp.assert_max_ulp(my_erfinv, "erfinv", np.linspace(-0.99, 0.99, 999), max_ulp=2)
"""

from __future__ import annotations

import pytest

from . import testing


@pytest.fixture
def ulp():
    return testing
