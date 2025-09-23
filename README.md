# unity-ml-drl-data
### Group 6 - P2-1: Artificial Intelligence & Machine Learning

unity-ml-drl-data is a GitHub repository for experienting with Deep Reinforcement Learning (DRL) using Unity and ML-Agents. This project uses simulated 3D enviroments to study and train agents with DRL algorithms, while logging performance and behavioral data for analysis using Machine Learning Techniques.


### Project Structure
```
unity-ml-drl-data/
│
├── unity/                 # Unity project files (scenes, agents, environment scripts)
├── training/              # Python training scripts, configs, and utilities
├── data/                  # Collected data and schema definitions
├── docs/                  # Documentation, research notes, and reports
└── README.md              # This file
```

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
```mlagents-learn``` **not found** → Activate venv:
```
cd training
.\venv\Scripts\Activate.ps1
```
**Multiple Python versions conflict** → Force 3.10.x:
```
py -3.10 -m venv venv
.\venv\Scripts\Activate.ps1
py -3.10 -m pip install --upgrade pip
py -3.10 training\setup_env.py
```
 * **Venv active** but ```python --version``` ≠ 3.10.x → Use ```py -3.10``` in commands above.
 * If problems persist, please **open a GitHub issue** with the exact error message and your Windows version so we can help and improve the docs.

### Starting a Training Run
Once Unity and Python are set up, you can train agents using the provides example enviroments or your own custom ones

1. Open an Enviroment in Unity
   * Example: ```Assets/ML-Agents/Examples/3DBall/Scenes/3DBall.unity```
2. Launch Training from Python
   * Inside your virtual enviroment created earlier, run:
     ```
     mlagents-learn configs_official/3DBall.yaml --run-id=3dball_experiment --force
     ```
   * You should see output like this:
     ```
     [INFO] Listening on port 5004. Start training by pressing the Play button in the Unity Editor.
     [INFO] Connected to Unity environment with package version 2.0.1 and communication version 1.5.0
     ```
     > If you have exited your virtual environment you can access it again by running:
     > 
     > ```cd training```
     > 
     > ```source venv/bin/activate``` for Unix systems
     >
     > ```.\venv\Scripts\Activate.ps1``` for Windows systems
     
3. Start the Unity Simulation
   * Back in Unity, press the play button in the top-center of the screen
   * The python process will detect the environment and begin training
   * Training statistics such as reward, step, and loss will be printed in the Python console

4. Check Results
   * Trained models and logs are saved under:
     ```training/results/3dball_experiment/```
   * The trained .onnx model can be loaded back into Unity:
       * Drag the .onnx file into the Agent's **Behavior Parameters > Model** field
       * Change **Behavior Type** to ```Inference Only```
       * Press Play to watch the agent use its learned model!
     
### Dependencies
- **Unity side**: ML-Agents 2.0.1 (installed automatically via Unity Package Manager)
- **Python side**: Dennis Soemers’ ML-Agents fork (see `training/requirements.*.txt`)

## Attributions
This project includes the official **Unity ML-Agents Examples and corrosponding training configuration files**, sourced from the [Unity ML-Agents GitHub Repository](https://github.com/Unity-Technologies/ml-agents).

All rights to these examples belong to Unity Technologies. We claim no ownership over them.

## License Notice
The Unity ML-Agents Examples included here remain under their original [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0), as provided by Unity Technologies.  
All other code and assets created for this repository are licensed under the terms specified in this project’s LICENSE file.

