# Used by test_setting_limit_on_thread_local_blas_api_is_actually_thread_local()

from concurrent.futures import ThreadPoolExecutor
from time import sleep
import sys

import numpy as np
import threadpoolctl

ARR = np.ones((1500, 1500))


def in_thread(_):
    with threadpoolctl.threadpool_limits(limits=int(sys.argv[1]), user_api="blas"):
        ARR.dot(ARR)
    # Make sure jobs are evenly distributed and don't end up in one thread.
    sleep(0.01)


if __name__ == "__main__":
    with ThreadPoolExecutor(2) as pool:
        list(pool.map(in_thread, range(2)))
