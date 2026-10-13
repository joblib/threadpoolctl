"""Test script for ``_get_cached_controller``."""

from threadpoolctl import ThreadpoolController, get_cached_controller

controller = get_cached_controller()
assert isinstance(controller, ThreadpoolController)
# Same controller is returned on another call:
assert controller is get_cached_controller()
# No libraries loaded:
assert len(controller.lib_controllers) == 0

import numpy as np  # noqa: F401 (also gets us BLAS)

# Now a new controller is returned:
controller2 = get_cached_controller()
assert controller is not controller2
assert isinstance(controller2, ThreadpoolController)
# And it noticed the newly loaded controllable library:
assert len(controller2.lib_controllers) >= 1

# Special success exit code:
raise SystemExit(17)
