"""specaudit: how accurate are the special functions you call?

Evaluates SciPy, JAX and PyTorch implementations against mpmath references at 30 significant
digits and reports the error in units in the last place (ULP), per function, per library, per
dtype, over wide parameter domains.
"""

from .functions import FUNCTIONS, FunctionSpec
from .metrics import ulp_error
from .run import run_audit
from .testing import assert_max_ulp

__version__ = "0.1.0"
__all__ = ["FUNCTIONS", "FunctionSpec", "ulp_error", "run_audit", "assert_max_ulp", "__version__"]
