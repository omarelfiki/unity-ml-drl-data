# unity-ml-drl-data
### Group 6 - P2-1: Artificial Intelligence & Machine Learning

unity-ml-drl-data is a GitHub repository for experienting with Deep Reinforcement Learning (DRL) using Unity and ML-Agents. This project uses simulated 3D enviroments to study and train agents with DRL algorithms, while logging performance and behavioral data for analysis using Machine Learning Techniques.


This project uses the ML-Agent package by Unity : https://github.com/DennisSoemers/ml-agents/tree/fix-numpy-release-21-branch


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
1. Clone Repository
```
git clone https://github.com/omarelfiki/unity-ml-drl-data.git
cd unity-ml-drl-data
```

2. Install Unity
* Download and Install Unity Hub and recommended Unity Editor version: 6.2 (6000.2.2f1)
* Open the ```/unity``` folder as a Unity project.

3. Set up Python enviroment
```
cd training
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Contributors
* @omarelfiki
* @AlexPayn
* @AshourKaria
* @Lorenzo-D-Coder2
* @ntlonggx
* Alexandru Mihăilă
