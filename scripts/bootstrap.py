"""Create a local virtual environment and install this checkout.

Run from the repository root with ``python scripts/bootstrap.py``.  The
script deliberately uses only the Python standard library so it works before
the project's dependencies have been installed.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import venv


MINIMUM_PYTHON = (3, 10)
MAXIMUM_PYTHON = (3, 14)
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
VENV_DIRECTORY = REPOSITORY_ROOT / ".venv"


def venv_python(venv_directory: Path) -> Path:
    """Return the Python executable for a virtual environment on this OS."""

    if sys.platform == "win32":
        return venv_directory / "Scripts" / "python.exe"
    return venv_directory / "bin" / "python"


def run(command: list[str]) -> None:
    print("+", " ".join(command))
    subprocess.run(command, check=True, cwd=REPOSITORY_ROOT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Recreate the virtual environment before installing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not MINIMUM_PYTHON <= sys.version_info[:2] < MAXIMUM_PYTHON:
        minimum = ".".join(map(str, MINIMUM_PYTHON))
        maximum = f"{MAXIMUM_PYTHON[0]}.{MAXIMUM_PYTHON[1] - 1}"
        raise SystemExit(f"Python {minimum} through {maximum} is required.")

    if args.recreate and VENV_DIRECTORY.exists():
        print(f"Removing {VENV_DIRECTORY}")
        shutil.rmtree(VENV_DIRECTORY)
    if not VENV_DIRECTORY.exists():
        print(f"Creating {VENV_DIRECTORY}")
        venv.EnvBuilder(with_pip=True).create(VENV_DIRECTORY)

    python = venv_python(VENV_DIRECTORY)
    run([str(python), "-m", "pip", "install", "--upgrade", "pip"])
    run([str(python), "-m", "pip", "install", "-e", "."])

    print("\nInstallation complete. Activate the environment, then run a command:")
    if sys.platform == "win32":
        print(r"  .venv\Scripts\Activate.ps1")
    else:
        print("  source .venv/bin/activate")
    print("  train-cross-fitted-pcs --analysis 12 --epochs 100")


if __name__ == "__main__":
    main()
