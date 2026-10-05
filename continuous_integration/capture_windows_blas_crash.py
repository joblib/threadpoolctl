"""Run the empirical BLAS diagnostic under ProcDump on Windows CI."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

from threadpoolctl import threadpool_info


def main():
    output_dir = Path("crash_diagnostics").resolve()
    output_dir.mkdir(exist_ok=True)
    completed = output_dir / "completed.txt"
    completed.unlink(missing_ok=True)

    # Record exact builds and runtime library paths for local dump analysis.
    libraries = threadpool_info()
    packages = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((Path(sys.prefix) / "conda-meta").glob("*.json"))
    ]
    (output_dir / "environment.json").write_text(
        json.dumps(
            {
                "python": sys.version,
                "executable": sys.executable,
                "numpy": np.__version__,
                "libraries": libraries,
                "conda_packages": packages,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_dir / "conda-explicit.txt").write_text(
        "@EXPLICIT\n" + "".join(f"{package['url']}\n" for package in packages),
        encoding="utf-8",
    )

    # ProcDump's exit status is not the Python child's status. Write a marker
    # only after the diagnostic returns normally, so a crash still fails CI.
    child_code = (
        "import runpy; "
        "runpy.run_path('tests/empirical_scope_observation.py', "
        "run_name='__main__'); "
        f"open({str(completed)!r}, 'w').write('completed')"
    )
    command = [
        os.environ["PROCDUMP_PATH"],
        "-accepteula",
        "-ma",
        "-e",
        "1",
        "-f",
        "C0000005",
        "-n",
        "1",
        "-x",
        str(output_dir),
        sys.executable,
        "-u",
        "-X",
        "faulthandler",
        "-c",
        child_code,
        "blas",
    ]
    # Capture first-chance access violations before faulthandler can terminate
    # the process. Stream debugger and Python output to CI and the artifact.
    with (output_dir / "procdump.log").open("w", encoding="utf-8") as log:
        with subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
        ) as process:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            returncode = process.wait()

    dumps = list(output_dir.glob("*.dmp"))
    if dumps:
        # Keep the exact BLAS DLL, rather than relying on a later conda solve.
        for library in libraries:
            if library["internal_api"] == "openblas":
                shutil.copy2(library["filepath"], output_dir)
    if returncode != 0 or not completed.exists() or dumps:
        raise SystemExit("BLAS diagnostic failed; see crash_diagnostics artifact")


if __name__ == "__main__":
    main()
