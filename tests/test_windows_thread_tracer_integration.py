import ctypes
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from threadpoolctl import threadpool_info

from _thread_tracer import ThreadTracerError, WindowsThreadSpawnTracer
from tests.utils import cython_extensions_compiled

pytestmark = [
    pytest.mark.skipif(
        sys.platform != "win32",
        reason="Windows ETW integration tests require Windows",
    ),
    pytest.mark.skipif(
        sys.platform == "win32" and not ctypes.windll.shell32.IsUserAnAdmin(),
        reason="Windows ETW kernel tracing requires an elevated process",
    ),
]

REPO_ROOT = Path(__file__).resolve().parents[1]
TRACER_ATTACH_DELAY_SECONDS = 0.5
TRACER_FLUSH_DELAY_SECONDS = 0.3
SUBPROCESS_TIMEOUT_SECONDS = 60
BLAS_THREAD_ENV_VARS = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "4",
    "MKL_NUM_THREADS": "4",
}


def _expected_pool_thread_count(user_api):
    thread_counts = [
        module["num_threads"]
        for module in threadpool_info()
        if module.get("user_api") == user_api and module.get("num_threads")
    ]
    if not thread_counts:
        pytest.skip("No {0} thread pool detected".format(user_api))
    # Native pools report team size including the calling thread, while ETW
    # Start events only observe newly created OS threads.
    return max(1, max(thread_counts) - 1)


def _configure_blas_thread_env():
    for name, value in BLAS_THREAD_ENV_VARS.items():
        os.environ[name] = value


def _child_script(body):
    return textwrap.dedent(
        """
        import time

        time.sleep({attach_delay})
        {body}
        time.sleep({flush_delay})
        """
    ).format(
        attach_delay=TRACER_ATTACH_DELAY_SECONDS,
        body=textwrap.dedent(body).strip(),
        flush_delay=TRACER_FLUSH_DELAY_SECONDS,
    )


def _run_traced_child(body, minimum_spawn_count):
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        str(REPO_ROOT) if not pythonpath else str(REPO_ROOT) + os.pathsep + pythonpath
    )

    proc = subprocess.Popen(
        [sys.executable, "-c", _child_script(body)],
        cwd=str(REPO_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    tracer = WindowsThreadSpawnTracer(proc.pid)
    try:
        tracer.start()
    except ThreadTracerError as exc:
        proc.kill()
        proc.wait(timeout=SUBPROCESS_TIMEOUT_SECONDS)
        pytest.fail("failed to start Windows ETW tracer: {0}".format(exc))

    try:
        stdout, stderr = proc.communicate(timeout=SUBPROCESS_TIMEOUT_SECONDS)
        return_code = proc.returncode
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, stderr = proc.communicate(timeout=SUBPROCESS_TIMEOUT_SECONDS)
        tracer.stop()
        pytest.fail("timed out waiting for traced child process")

    stats = tracer.stop()
    if return_code != 0:
        pytest.fail(
            "traced child exited with code {0}\nstdout:\n{1}\nstderr:\n{2}".format(
                return_code,
                stdout,
                stderr,
            )
        )
    assert (
        stats.spawn_count >= minimum_spawn_count
    ), "expected at least {expected} thread spawn events, got {actual} ({stats})".format(
        expected=minimum_spawn_count,
        actual=stats.spawn_count,
        stats=stats,
    )
    return stats


def test_tracer_counts_python_thread_spawns():
    body = """
    import threading

    def work():
        pass

    threads = [threading.Thread(target=work) for _ in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    """
    _run_traced_child(body, minimum_spawn_count=3)


@pytest.mark.skipif(
    not cython_extensions_compiled,
    reason="OpenMP test helper is not built",
)
def test_tracer_counts_openmp_thread_spawns():
    from tests._openmp_test_helper.openmp_helpers_inner import check_openmp_num_threads

    os.environ["OMP_NUM_THREADS"] = "4"
    check_openmp_num_threads(10)
    expected_spawn_count = _expected_pool_thread_count("openmp")

    body = """
    import os

    os.environ["OMP_NUM_THREADS"] = "4"
    from tests._openmp_test_helper.openmp_helpers_inner import check_openmp_num_threads

    used = check_openmp_num_threads(100)
    assert used >= 1
    """
    _run_traced_child(body, minimum_spawn_count=expected_spawn_count)


def test_tracer_counts_blas_thread_spawns():
    pytest.importorskip("numpy")
    _configure_blas_thread_env()

    import numpy as np

    rng = np.random.RandomState(0)
    warmup = rng.rand(100, 100)
    np.dot(warmup, warmup)
    expected_spawn_count = _expected_pool_thread_count("blas")

    body = """
    import os

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"

    import numpy as np

    rng = np.random.RandomState(0)
    a = rng.rand(2000, 2000)
    np.dot(a, a)
    """
    _run_traced_child(body, minimum_spawn_count=expected_spawn_count)
