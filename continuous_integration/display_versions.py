from threadpoolctl import threadpool_info
from pprint import pprint

try:
    import numpy as np
    print("numpy", np.__version__)
except ImportError:
    pass

try:
    import scipy
    import scipy.linalg
    print("scipy", scipy.__version__)
except ImportError:
    pass

try:
    from tests._openmp_test_helper import *  # noqa
except ImportError:
    pass


pprint(threadpool_info())
