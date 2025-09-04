import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

CONSTRAINTS_MAP = {
    ("Darwin", "arm64"): "constraints.macos-arm64.txt",
    ("Windows", "AMD64"): "constraints.windows-x86_64.txt",
}

def run(cmd, **kwargs):
    print(f"\n>> {cmd}")
    subprocess.check_call(cmd, shell=True, **kwargs)

def main():
    system = platform.system()
    machine = platform.machine()
    constraints_file = CONSTRAINTS_MAP.get((system, machine))

    if constraints_file is None:
        print(f"No constraints defined for {system} {machine}. Falling back to plain requirements.")
        constraints_path = None
    else:
        constraints_path = ROOT / constraints_file
        print(f"Using constraints: {constraints_path}")

    venv_dir = ROOT / "venv"
    python = sys.executable

    if not venv_dir.exists():
        run(f"{python} -m venv {venv_dir}")

    pip = venv_dir / ("Scripts/pip.exe" if system == "Windows" else "bin/pip")

    run(f"{pip} install --upgrade pip setuptools wheel")

    run(
        f"{pip} install --no-deps "
        "git+https://github.com/DennisSoemers/ml-agents.git@fix-numpy-release-21-branch#egg=mlagents&subdirectory=ml-agents "
        "git+https://github.com/DennisSoemers/ml-agents.git@fix-numpy-release-21-branch#egg=mlagents-envs&subdirectory=ml-agents-envs"
    )

    req_file = ROOT / "requirements.txt"
    if constraints_path and constraints_path.exists():
        run(f"{pip} install -r {req_file} -c {constraints_path}")
    else:
        run(f"{pip} install -r {req_file}")

    print("\n✅ Environment setup complete!")
    print(f"To activate the venv, run:")
    if system == "Windows":
        print(rf"   {venv_dir}\Scripts\Activate.ps1")
    else:
        print(f"   source {venv_dir}/bin/activate")

if __name__ == "__main__":
    main()