import ctypes
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from threadpoolctl._thread_tracer import ThreadTracerError, WindowsThreadSpawnTracer

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

REPO_ROOT = Path(__file__).resolve().parents[2]
TRACER_ATTACH_DELAY_SECONDS = 0.5
TRACER_FLUSH_DELAY_SECONDS = 0.3
SUBPROCESS_TIMEOUT_SECONDS = 60
BLAS_THREAD_ENV_VARS = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "4",
    "MKL_NUM_THREADS": "4",
}


def _threadpool_info():
    try:
        from threadpoolctl import threadpool_info

        return threadpool_info()
    except ImportError:
        from threadpoolctl import get_threadpool_limits

        return get_threadpool_limits()


def _module_num_threads(module):
    if "num_threads" in module:
        return module["num_threads"]
    return module.get("n_thread")


def _expected_pool_thread_count(user_api):
    thread_counts = [
        _module_num_threads(module)
        for module in _threadpool_info()
        if module.get("user_api") == user_api and _module_num_threads(module)
    ]
    if not thread_counts:
        pytest.skip("No {0} thread pool detected".format(user_api))
    return max(thread_counts)


def _configure_blas_thread_env():
    for name, value in BLAS_THREAD_ENV_VARS.items():
        os.environ[name] = value


def _child_script(body):
    return textwrap.dedent("""
        import time

        time.sleep({attach_delay})
        {body}
        time.sleep({flush_delay})
        """).format(
        attach_delay=TRACER_ATTACH_DELAY_SECONDS,
        body=textwrap.indent(textwrap.dedent(body).strip(), "    "),
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
    )
    tracer = WindowsThreadSpawnTracer(proc.pid)
    try:
        tracer.start()
    except ThreadTracerError as exc:
        proc.kill()
        proc.wait(timeout=SUBPROCESS_TIMEOUT_SECONDS)
        pytest.fail("failed to start Windows ETW tracer: {0}".format(exc))

    try:
        return_code = proc.wait(timeout=SUBPROCESS_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=SUBPROCESS_TIMEOUT_SECONDS)
        tracer.stop()
        pytest.fail("timed out waiting for traced child process")

    stats = tracer.stop()
    assert return_code == 0, "traced child exited with code {0}".format(return_code)
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

    num_threads = 3
    started = threading.Barrier(num_threads)

    def work():
        started.wait(timeout=5)

    threads = [threading.Thread(target=work) for _ in range(num_threads)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)
    """
    _run_traced_child(body, minimum_spawn_count=3)


def test_tracer_counts_openmp_thread_spawns():
    try:
        from threadpoolctl.tests._openmp_test_helper import check_openmp_n_threads
    except ImportError:
        pytest.skip("OpenMP test helper is not built")

    os.environ["OMP_NUM_THREADS"] = "4"
    check_openmp_n_threads(10)
    expected_spawn_count = _expected_pool_thread_count("openmp")

    body = """
    import os

    os.environ["OMP_NUM_THREADS"] = "4"
    from threadpoolctl.tests._openmp_test_helper import check_openmp_n_threads

    used = check_openmp_n_threads(1000)
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
