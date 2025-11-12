"""Configuration constants for ML-Agents training runner."""

VERSION = "2.5"
AUTO_COMMIT_BRANCH = "refactor"
N_STEPS = 12000

TAGS = {
    "Environment/Cumulative Reward": "Mean Policy Reward",
    "Losses/Policy Loss": "Mean Policy Loss",
    "Losses/Value Loss": "Mean Value Loss",
    "Policy/Entropy": "Mean Entropy",
    "Environment/NumAgents": "Number of Agents"
}

CSV_HEADERS = [
    "run_id", "environment", "seed", "num_agents",
    "algorithm", "steps", "batch_size", "buffer_size",
    "learning_rate", "epochs", "total_time", "average_cpu", "average_ram",
    "step_interval", "reward_mean", "reward_mean_step",
    "p_loss_mean", "p_loss_mean_step", "v_loss_mean", "v_loss_mean_step",
    "entropy_mean", "entropy_mean_step",
    "threshold_value", "steps_to_threshold", "time_to_threshold",
    "threshold_version", "run_reached_threshold",
    "best_reward_before_timeout", "step_of_best_reward"
]

KEY_MAPPING = {
        "Run ID": "run_id",
        "Environment": "environment",
        "Seed": "seed",
        "Number of Agents": "num_agents",
        "Algorithm": "algorithm",
        "Steps": "steps",
        "Batch Size": "batch_size",
        "Buffer Size": "buffer_size",
        "Learning Rate": "learning_rate",
        "Epochs": "epochs",
        "Total Time": "total_time",
        "Average CPU": "average_cpu",
        "Average RAM": "average_ram",
        "Step Interval (Running Mean)": "step_interval",
        "Mean Policy Reward": "reward_mean",
        "Mean Policy Reward (start step)": "reward_mean_step",
        "Mean Policy Loss": "p_loss_mean",
        "Mean Policy Loss (start step)": "p_loss_mean_step",
        "Mean Value Loss": "v_loss_mean",
        "Mean Value Loss (start step)": "v_loss_mean_step",
        "Mean Entropy": "entropy_mean",
        "Mean Entropy (start step)": "entropy_mean_step",
        "Threshold Value": "threshold_value",
        "Steps to Threshold": "steps_to_threshold",
        "Time to Threshold (s)": "time_to_threshold",
        "Threshold Version": "threshold_version",
        "Run Reached Threshold": "run_reached_threshold",
        "Best Reward Before Timeout": "best_reward_before_timeout",
        "Step Of Best Reward": "step_of_best_reward",
}