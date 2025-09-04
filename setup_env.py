import platform
import subprocess
import sys
from pathlib import Path

def main():
    base_dir = Path(__file__).parent
    venv_dir = base_dir / "training" / "venv"

    # create venv if it doesn't exist
    if not venv_dir.exists():
        print("Creating virtual environment...")
        subprocess.check_call([sys.executable, "-m", "venv", str(venv_dir)])

    if platform.system() == "Windows":
        python_exe = venv_dir / "Scripts" / "python.exe"
    else:
        python_exe = venv_dir / "bin" / "python"

    # requirements file per OS
    system = platform.system()
    if system == "Darwin":
        req_file = base_dir / "training" / "requirements.mac.txt"
    elif system == "Windows":
        req_file = base_dir / "training" / "requirements.windows.txt"
    else:
        req_file = base_dir / "training" / "requirements.mac.txt"  # fallback in case of Linux

    print(f"Installing dependencies from {req_file} ...")
    subprocess.check_call([str(python_exe), "-m", "pip", "install", "-r", str(req_file)])

    print("\nSetup complete!")
    if system == "Windows":
        print(f"To activate: {venv_dir}\\Scripts\\activate")
    else:
        print(f"To activate: source {venv_dir}/bin/activate")

if __name__ == "__main__":
    main()