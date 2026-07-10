import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from threadpoolctl._thread_tracer import (
    ThreadTracerError,
    WindowsThreadSpawnTracer,
    windows_etw_admin_available,
    windows_tracer_available,
)

pytestmark = [
    pytest.mark.skipif(
        not windows_tracer_available(),
        reason="Windows ETW integration tests require Windows",
    ),
    pytest.mark.skipif(
        not windows_etw_admin_available(),
        reason="Windows ETW kernel tracing requires an elevated process",
    ),
]

REPO_ROOT = Path(__file__).resolve().parents[2]
TRACER_ATTACH_DELAY_SECONDS = 0.5
TRACER_FLUSH_DELAY_SECONDS = 0.3
SUBPROCESS_TIMEOUT_SECONDS = 60


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
        body=textwrap.indent(textwrap.dedent(body).strip(), "    "),
        flush_delay=TRACER_FLUSH_DELAY_SECONDS,
    )


def _run_traced_child(body, minimum_spawn_count):
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        str(REPO_ROOT)
        if not pythonpath
        else str(REPO_ROOT) + os.pathsep + pythonpath
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
    assert stats.spawn_count >= minimum_spawn_count, (
        "expected at least {expected} thread spawn events, got {actual} ({stats})"
        .format(
            expected=minimum_spawn_count,
            actual=stats.spawn_count,
            stats=stats,
        )
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
        from threadpoolctl.tests._openmp_test_helper import check_openmp_n_threads  # noqa: F401
    except ImportError:
        pytest.skip("OpenMP test helper is not built")

    body = """
    import os

    os.environ["OMP_NUM_THREADS"] = "4"
    from threadpoolctl.tests._openmp_test_helper import check_openmp_n_threads

    used = check_openmp_n_threads(1000)
    assert used >= 1
    """
    _run_traced_child(body, minimum_spawn_count=4)


def test_tracer_counts_blas_thread_spawns():
    pytest.importorskip("numpy")
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
    _run_traced_child(body, minimum_spawn_count=2)
