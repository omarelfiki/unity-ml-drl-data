# Headless training

A way to train the model without opening Unity Editor, just run the code in the correct format and you will train it automatically

### Unity Editor

- Open an Enviroment in Unity

Example: ```Assets/ML-Agents/Examples/3DBall/Scenes/3DBall.unity```

- Edit (Top left) -> Project Settings -> Player -> Resolution and Presentation -> Run in Background
- File (Top left) -> Build Settings -> Scenes In Build Section -> Add Open Scenes (the current scene you opening)
- If any scenes are shown in the Scenes in Build list, make sure that the 3DBall Scene (or any different scene) is the only one checked. (If the list is empty, then only the current scene is included in the build).
- Choose your platform -> Build -> Select where you want it to be placed

### Run the training without graphics

```python -m scripts.train --config configs_official/3DBall.yaml --headless "directory to your build" --run-id ball_test_01```
