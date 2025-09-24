### Installation
**1. Clone Repository**
```
git clone https://github.com/omarelfiki/unity-ml-drl-data.git
cd unity-ml-drl-data
```

**2. Install Unity**
* Download and Install Unity Hub
* Install recommended Unity Editor version: 2023.2.12f1
* Open the ```/unity``` folder as a Unity project.

**3. Set up Python enviroment (Python 3.10.12)**   
From the root of the repository:
```
cd training
python setup_env.py
```

This will:
* Create a Python virtual environment in training/venv.
* Install all dependencies from the appropriate requirements file (based on your system).

> Ensure you have Python 3.10.12 available (via pyenv, conda, or system Python).

### Windows Python Version Note
Some Windows users reported issues installing **Python 3.10.12** (the version officially required by ML-Agents).  
If you encounter errors during installation, try instead:

* **Python 3.10.11** → this version is confirmed to work on Windows and is easier to install with the official installer.  
* After installation, re-run:

### Windows Setup Guide (Detailed)

If you’re on Windows and run into issues, follow these platform-specific steps.

**A) Install Python 3.10.x on Windows**
1. Download **3.10.12** (preferred) or **3.10.11** from the official Python releases page.
2. In the installer:
 - Check **“Add Python 3.10 to PATH.”**
 - Click **Customize installation** → ensure **pip** and **venv** are selected.
3. Verify versions:
 ```
 py -3.10 --version
 python --version
 ```
**B) Create & Activate the Virtual Environment**
From the repo root:
```
cd training
py -3.10 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python setup_env.py
```
To reactivate later:
```
cd training
.\venv\Scripts\Activate.ps1
```
**C) PowerShell Execution Policy (if activation is blocked)**
If you see a script execution error:
```
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
.\venv\Scripts\Activate.ps1
```
**D) Common Windows Fixes**

 1. ```mlagents-learn``` **not found** → Activate venv:
```
cd training
.\venv\Scripts\Activate.ps1
```
 2.  **Multiple Python versions conflict** → Force 3.10.x:
```
py -3.10 -m venv venv
.\venv\Scripts\Activate.ps1
py -3.10 -m pip install --upgrade pip
py -3.10 training\setup_env.py
```
 * **Venv active** but ```python --version``` ≠ 3.10.x → Use ```py -3.10``` in commands above.
 * If problems persist, please **open a GitHub issue** with the exact error message and your Windows version so we can help and improve the docs.
