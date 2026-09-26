"""Create or reuse Gagent's local Python environment, then launch the CLI."""

import hashlib
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
REQUIREMENTS = ROOT / "requirements.txt"
STAMP = VENV / ".gagent-requirements.sha256"


def environment_python() -> Path:
    if sys.version_info < (3, 11):
        raise RuntimeError("Python 3.11 or newer is required")

    python = VENV / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    if not python.is_file():
        print(f"Gagent: creating environment at {VENV}", flush=True)
        subprocess.run([sys.executable, "-m", "venv", str(VENV)], check=True)
    if not python.is_file():
        raise RuntimeError(f"environment Python was not created: {python}")

    check = subprocess.run(
        [str(python), "-c", "import sys; print(sys.prefix)"],
        capture_output=True, text=True,
    )
    if check.returncode or Path(check.stdout.strip()).resolve() != VENV.resolve():
        raise RuntimeError(f"environment is invalid: {VENV}; repair or remove it and retry")
    return python


def install_requirements(python: Path) -> None:
    digest = hashlib.sha256(REQUIREMENTS.read_bytes()).hexdigest()
    if STAMP.is_file() and STAMP.read_text(encoding="ascii").strip() == digest:
        return

    print("Gagent: installing project dependencies...", flush=True)
    subprocess.run(
        [str(python), "-m", "pip", "install", "-r", str(REQUIREMENTS)],
        cwd=ROOT, check=True,
    )
    STAMP.write_text(digest + "\n", encoding="ascii")


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    try:
        python = environment_python()
        install_requirements(python)
        if argv == ["--setup-only"]:
            return 0
        return subprocess.run(
            [str(python), "-X", "utf8", str(ROOT / "cli.py"), *argv],
            cwd=ROOT,
        ).returncode
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"Gagent: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
