# unity-ml-drl-data
### Group 6 - P2-1: Artificial Intelligence & Machine Learning
unity-ml-drl-data is a GitHub repository for experimenting with Deep Reinforcement Learning (DRL) using Unity and ML-Agents. 
This project uses simulated 3D environments to study and train agents with DRL algorithms, while logging performance and behavioral data using TensorBoard for analysis using Machine Learning Techniques to make predictions on training results. 
More information can be found in the [`docs/`](https://www.github.com/omarelfiki/unity-ml-drl-data/tree/main/docs) directory.

### Repository Structure
```
unity-ml-drl-data/
│
├── unity/                 # Unity project files (scenes, agents, environment scripts)
├── training/              # Python training scripts, configs, and utilities
├── data/                  # Collected data and schema definitions
├── docs/                  # Documentation, research notes, and reports
├── README.md              # This file
├── CONTRIBUTING.md        # Guidelines for making contributions
└── SETUP.md               # In-depth installation steps
```

### Installation
Clone Repository and setup Python Environment
```
git clone https://github.com/omarelfiki/unity-ml-drl-data.git
cd unity-ml-drl-data/training
python setup_env.py
```
> Unity 2023.2.12f1 is a requirement for this project. Unity versions can be downloaded via Unity Hub.

See [```SETUP.md```](https://www.github.com/omarelfiki/unity-ml-drl-data/tree/main/SETUP.md)for in-depth installation steps.

### Training
```
python -m scripts.train_model [-h] --config <config_file> --run-id <run_id> [--num-steps <int>] [--headless <env_path>]
```
> Example Usage for training script. Results will be appended to shared dataset. Local results are directed to gitignore.

See [```training/README.md```](https://www.github.com/omarelfiki/unity-ml-drl-data/tree/main/training/README.md) for more information on the `training` package.

### Results
Results are obtained in a shared dataset updated each training run with new metrics. Upon being pushed to the repository, data validation checks are executed automatically by GitHub workflows. A summary report is also available providing average dataset values and plots from TensorBoard data for visualization

See [```data/```](https://www.github.com/omarelfiki/unity-ml-drl-data/tree/main/data) for results and documentation.


## Attributions
This project includes the official **Unity ML-Agents Examples and corresponding training configuration files**, sourced from the [Unity ML-Agents GitHub Repository](https://github.com/Unity-Technologies/ml-agents).
All rights to these examples belong to Unity Technologies. We claim no ownership over them.

#### Dependencies
1. **Unity side**: ML-Agents 2.0.1 (installed automatically via Unity Package Manager)
2. **Python side**: Dennis Soemers’ [ML-Agents fork](https://www.github.com/dennissoemers/ml-agents)

### License Notice
The Unity ML-Agents Examples included here remain under their original [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0), as provided by Unity Technologies. All other code and assets created for this repository are licensed under the terms specified in this project’s LICENSE file.

