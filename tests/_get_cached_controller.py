"""Test script for ``_get_cached_controller``."""

from threadpoolctl import get_cached_controller, ThreadpoolController


def assert_or_exit(condition):
    if not condition:
        raise RuntimeError("condition not met")


controller = get_cached_controller()
assert isinstance(controller, ThreadpoolController)
# Same controller is returned on another call:
assert_or_exit(controller is get_cached_controller())
# No libraries loaded:
assert_or_exit(len(controller.lib_controllers) == 0)

import numpy as np

# Now a new controller is returned:
controller2 = get_cached_controller()
assert_or_exit(controller is not controller2)
assert isinstance(controller2, ThreadpoolController)
# And it noticed the newly loaded controllable library:
assert_or_exit(len(controller2.lib_controllers) >= 1)

# Special success exit code:
raise SystemExit(17)
