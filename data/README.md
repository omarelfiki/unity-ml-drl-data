# Understanding Results

### Research Question
Can we predict the time-to-threshold, denoting number of training steps and wall-clock time required for
an agent to reach a performance threshold using initial configuration and early learning-curve signals?

#### Targets (to predict)
* Wall clock time (s) (observed seconds).
* Time-to-threshold (derived) from per-step reward logs: steps/time to reach a target reward 

### Column Dictionary

| **Column**                 | **Type** | **Description**                                                 |
|----------------------------|----------|-----------------------------------------------------------------|
| `run_id`                   | string   | Unique identifier for a run                                     |
| `behavior`                 | string   | Unity environment (e.g., `3DBall`, `Basic`, `Sorter`)           |
| `seed`                     | int      | Random seed for reproducibility (`-1` indicates default/random) |
| `num_agents`               | int      | Number of simultaneous agents/instances                         |
| `algorithm`                | string   | DRL algorithm (e.g., `PPO`)                                     |
| `steps`                    | int      | Completed total training steps                                  |
| `batch_size`               | int      | Minibatch size per update                                       |
| `buffer_size`              | int      | Experience buffer size                                          |
| `learning_rate`            | float    | Optimizer learning rate                                         |
| `epochs`                   | int      | Epochs per update (if applicable)                               |
| `cpu_usage_mean`           | float    | Average CPU utilization (0–1)                                   |
| `ram_usage_mean`           | float    | Average RAM utilization (0–1)                                   |
| `wall_clock_time (s)`      | float    | Wall-clock time in seconds                                      |
| `step_average`             | int      | Step interval used for computing running means                  |
| `mean_policy_reward`       | float    | Average policy reward                                           |
| `policy_reward_start_step` | int      | Step index where reward aggregation starts                      |
| `mean_policy_loss`         | float    | Average policy loss                                             |
| `policy_loss_start_step`   | int      | Step index where policy-loss aggregation starts                 |
| `mean_value_loss`          | float    | Average value loss                                              |
| `value_loss_start_step`    | int      | Step index where value-loss aggregation starts                  |
| `mean_entropy`             | float    | Average policy entropy                                          |
| `entropy_start_step`       | int      | Step index where entropy aggregation starts                     |



# Updating Shared Results

All training runs automatically append a new entry to:

data/combined_results.csv
data/combined_results.json

These files are shared by the whole team — so follow this workflow to keep everyone’s results organized and avoid data loss.

---

## 1. Before running a new training
Always pull the latest data first:

```bash
git pull --rebase
```
This ensures your local results file is up to date.

## 2. After training completes

The script automatically adds your new results to the files above.

Commit and push your updates:

```bash
git add data/combined_results.csv data/combined_results.json
git commit -m "Add results for Run [YourRunID] ([Environment/ConfigName])"
git push
```

## 3. If you see a merge conflict

If Git reports something like:
```abpublidot
CONFLICT (content): Merge conflict in data/combined_results.csv
```

Open the file, keep both lines (each is a valid run), then:
```abpublidot
git add data/combined_results.csv
git commit -m "Resolve CSV merge conflict"
git push
```

> **Never delete or reorder rows** in the CSV
> 
> **Always pull before** running new trainings
