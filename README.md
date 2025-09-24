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

###Installation Steps
See SETUP.md for in-depth installation steps.

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

