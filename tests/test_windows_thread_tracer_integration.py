import ctypes
import os
import subprocess
import sys
import textwrap
import time
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
# Attach after the child body has created and blocked its worker threads so ETW
# rundown can observe them as pre-existing (DCStart) rather than new spawns.
EXISTING_THREAD_TRACER_DELAY_SECONDS = TRACER_ATTACH_DELAY_SECONDS + 0.5
SUBPROCESS_TIMEOUT_SECONDS = 60
BLAS_THREAD_ENV_VARS = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "4",
    "MKL_NUM_THREADS": "4",
}

NOOP_CHILD_BODY = "pass"
# Keep the child alive long enough for a delayed tracer attach when measuring
# DCStart rundown events (see EXISTING_THREAD_TRACER_DELAY_SECONDS).
NOOP_EXISTING_CHILD_BODY = """
import time
time.sleep(2)
"""


@pytest.fixture(scope="module")
def noop_child_spawn_count():
    """Spawn count for a traced child that only sleeps (no extra work)."""
    stats = _run_traced_child(NOOP_CHILD_BODY)
    return stats.spawn_count


@pytest.fixture(scope="module")
def noop_child_existing_thread_count():
    """Existing-thread rundown count for a noop child traced after a delay."""
    stats = _run_traced_child(
        NOOP_EXISTING_CHILD_BODY,
        include_existing_threads=True,
        tracer_start_delay=EXISTING_THREAD_TRACER_DELAY_SECONDS,
    )
    return stats.existing_thread_count


def _expected_pool_spawn_count(user_api):
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


def _assert_spawn_count_relative_to_baseline(
    stats, baseline_spawn_count, extra_spawn_count
):
    expected_total = baseline_spawn_count + extra_spawn_count
    assert stats.spawn_count == expected_total, (
        "expected {expected_total} spawns "
        "({baseline} noop baseline + {extra} extra), "
        "got {actual} ({stats})".format(
            expected_total=expected_total,
            baseline=baseline_spawn_count,
            extra=extra_spawn_count,
            actual=stats.spawn_count,
            stats=stats,
        )
    )


def _assert_existing_count_relative_to_baseline(
    stats, baseline_existing_count, extra_existing_count
):
    expected_total = baseline_existing_count + extra_existing_count
    assert stats.existing_thread_count == expected_total, (
        "expected {expected_total} existing threads "
        "({baseline} noop baseline + {extra} extra), "
        "got {actual} ({stats})".format(
            expected_total=expected_total,
            baseline=baseline_existing_count,
            extra=extra_existing_count,
            actual=stats.existing_thread_count,
            stats=stats,
        )
    )


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


def _run_traced_child(
    body,
    include_existing_threads=False,
    tracer_start_delay=0,
):
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
    if tracer_start_delay:
        time.sleep(tracer_start_delay)
    tracer = WindowsThreadSpawnTracer(
        proc.pid,
        include_existing_threads=include_existing_threads,
    )
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
    return stats


def test_tracer_counts_python_thread_spawns(noop_child_spawn_count):
    num_threads = 3
    body = """
    import threading

    def work():
        pass

    threads = [threading.Thread(target=work) for _ in range({num_threads})]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    """.format(
        num_threads=num_threads
    )
    stats = _run_traced_child(body)
    _assert_spawn_count_relative_to_baseline(
        stats,
        noop_child_spawn_count,
        num_threads,
    )


def test_tracer_counts_existing_python_threads(
    noop_child_existing_thread_count,
):
    num_threads = 3
    body = """
    import threading
    import time

    hold = threading.Event()

    def work():
        hold.wait(timeout=10)

    threads = [threading.Thread(target=work) for _ in range({num_threads})]
    for thread in threads:
        thread.start()
    time.sleep(2)
    hold.set()
    for thread in threads:
        thread.join()
    """.format(
        num_threads=num_threads
    )
    stats = _run_traced_child(
        body,
        include_existing_threads=True,
        tracer_start_delay=EXISTING_THREAD_TRACER_DELAY_SECONDS,
    )
    _assert_existing_count_relative_to_baseline(
        stats,
        noop_child_existing_thread_count,
        num_threads,
    )


@pytest.mark.skipif(
    not cython_extensions_compiled,
    reason="OpenMP test helper is not built",
)
def test_tracer_counts_openmp_thread_spawns(noop_child_spawn_count):
    from tests._openmp_test_helper.openmp_helpers_inner import check_openmp_num_threads

    os.environ["OMP_NUM_THREADS"] = "4"
    check_openmp_num_threads(10)
    expected_extra_spawn_count = _expected_pool_spawn_count("openmp")

    body = """
    import os

    os.environ["OMP_NUM_THREADS"] = "4"
    from tests._openmp_test_helper.openmp_helpers_inner import check_openmp_num_threads

    used = check_openmp_num_threads(100)
    assert used >= 1
    """
    stats = _run_traced_child(body)
    _assert_spawn_count_relative_to_baseline(
        stats,
        noop_child_spawn_count,
        expected_extra_spawn_count,
    )


def test_tracer_counts_blas_thread_spawns(noop_child_spawn_count):
    pytest.importorskip("numpy")
    _configure_blas_thread_env()

    import numpy as np

    rng = np.random.RandomState(0)
    warmup = rng.rand(100, 100)
    np.dot(warmup, warmup)
    expected_extra_spawn_count = _expected_pool_spawn_count("blas")

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
    stats = _run_traced_child(body)
    _assert_spawn_count_relative_to_baseline(
        stats,
        noop_child_spawn_count,
        expected_extra_spawn_count,
    )
