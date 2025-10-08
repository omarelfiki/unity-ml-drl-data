
## Starting a Training Run
Once Unity and Python are set up, you can train agents using the provides example enviroments or your own custom ones

**Open an Enviroment in Unity**

Example: ```Assets/ML-Agents/Examples/3DBall/Scenes/3DBall.unity```


**Launch Training from Python**

Inside your virtual enviroment created earlier, run from ```/training```:

    python scripts/train_model.py --config configs_official/3DBall.yaml --run-id ball_test_001

> Replace ```configs_official/3DBall.yaml``` with the path to your training configuration file and  ```ball_test_001``` with a unique run ID for your training run. Check the contributing guidelines for naming conventions.
     
You should see output like this:

     Starting ML-Agents training
     Config file: configs_edited/basic.yaml
     Run ID:      OE_081025
     Launching ML-Agents... please wait.
    

   > If you have exited your virtual environment you can access it again by running:
   >
   > 
   > ```source venv/bin/activate``` for Unix systems
   >
   > ```.\venv\Scripts\Activate.ps1``` for Windows systems


**Start the Unity Simulation**

Back in Unity, press the play button in the top-center of the screen.
The python process will detect the environment and begin training.

Training statistics such as reward, step, and loss will be printed in the Python console

**Check Results**

After the training run is complete, you can check data metrics in the terminal. Your output should look like this

    ===============================================
    | Metric                          | Value      |
    ===============================================
    | Run ID                          | OE_081025  |
    | Environment                     | Basic      |
    | Seed                            | 123        |
    | Number of Agents                | 12         |
    | Algorithm                       | ppo        |
    | Steps                           | 5000       |
    | Batch Size                      | 32         |
    | Buffer Size                     | 256        |
    | Learning Rate                   | 0.0003     |
    | Epochs                          | 3          |
    | Total Time (s)                  | 184        |
    | Average CPU (%)                 | 71.3       |
    | Average RAM (%)                 | 63.2       |
    | Mean Policy Reward              | 0.0562     |
    | Mean Policy Reward (start step) | 24192      |
    | Mean Policy Loss                | 0.3214     |
    | Mean Value Loss                 | 0.6935     |
    | Mean Entropy                    | 1.7821     |
    ===============================================

> Trained models and logs are saved under ```training/results/{run_id}/```
> 
> Note: These files are part of the gitignore. Results are available under the ```data``` directory.

       
### Using TensorBoard
You can visualize training progress using TensorBoard during or after training by running the following command in your virtual environment:

```
cd training
tensorboard --logdir results
```
Open a web browser and navigate to `http://localhost:6006` to view the TensorBoard dashboard.

You can monitor various metrics such as reward, loss, and other statistics logged during training.

To stop TensorBoard, simply return to your terminal and press `Ctrl+C`.

For the training run naming convention, please see CONTRIBUTING.md.