"""Commandline interface to display thread-pool information and exit."""

__all__ = ()


def _main() -> None:
    import argparse
    import importlib
    import json
    import sys

    from threadpoolctl import threadpool_info

    parser = argparse.ArgumentParser(
        usage="python -m threadpoolctl -i numpy scipy.linalg xgboost",
        description="Display thread-pool information and exit.",
    )
    parser.add_argument(
        "-i",
        "--import",
        dest="modules",
        nargs="*",
        default=(),
        help="Python modules to import before introspecting thread-pools.",
    )
    parser.add_argument(
        "-c",
        "--command",
        help="a Python statement to execute before introspecting thread-pools.",
    )
    options = parser.parse_args(sys.argv[1:])

    for module in options.modules:
        try:
            importlib.import_module(module, package=None)
        except ImportError:
            print("WARNING: could not import", module, file=sys.stderr)

    if options.command:
        exec(options.command)

    print(json.dumps(threadpool_info(), indent=2))


if __name__ == "__main__":
    _main()
