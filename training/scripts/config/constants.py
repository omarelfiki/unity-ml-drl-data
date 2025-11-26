"""Configuration constants for ML-Agents training runner."""

VERSION = "2.5"
AUTO_COMMIT_BRANCH = "Week2PuchBlock"

# Default number of tail steps to compute running means / best metrics
DEFAULT_N_STEPS = 50000  # fallback for unknown environments

# number of steps for tail-window analysis
ENV_N_STEPS = {
    "3DBall": 50000,
    "Basic": 20000,
    "Sorter": 20000,
    "Crawler": 100000,
    "Hallway": 100000,
    "WallJump": 100000,
}

def get_n_steps_for_env(env_name: str) -> int:
    return ENV_N_STEPS.get(env_name, DEFAULT_N_STEPS)

def get_early_window_for_env(env):
    mapping = {
        "3DBall": 12000,
        "Basic": 1000,
        "Sorter": 10000,
        "Crawler": 50000,
        "Hallway": 50000,
        "WallJump": 10000,
    }
    return mapping.get(env, 10000)

def get_final_window_for_env(env):
    mapping = {
        "3DBall": 50000,
        "Basic": 5000,
        "Sorter": 40000,
        "Crawler": 200000,
        "Hallway": 200000,
        "WallJump": 40000,
    }
    return mapping.get(env, 50000)

TAGS = {
    "Losses/Policy Loss": "Mean Policy Loss",
    "Losses/Value Loss": "Mean Value Loss",
    "Policy/Entropy": "Mean Entropy",
    "Environment/NumAgents": "Number of Agents"
}

CSV_HEADERS = [
    # Identifiers
    "run_id", "environment", "seed", "num_agents",
    "algorithm", "steps", "batch_size", "buffer_size",
    "learning_rate", "epochs",
    # System Performance Metrics
    "total_time", "average_cpu", "average_ram",
    # Training Metrics
    "reward_mean", "reward_mean_step", "early_reward_mean", "early_reward_mean_step",
    "final_reward_mean", "final_reward_mean_step", "best_reward", "best_reward_step",
    "step_interval", "p_loss_mean", "p_loss_mean_step", "v_loss_mean",
    "v_loss_mean_step", "entropy_mean", "entropy_mean_step",
    # Threshold Analysis
    "threshold_value", "steps_to_threshold", "time_to_threshold",
    "threshold_version", "run_reached_threshold",
    # misc
    "notes"
]

KEY_MAPPING = {
    # Identifiers
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
    # System Performance Metrics
        "Total Time": "total_time",
        "Average CPU": "average_cpu",
        "Average RAM": "average_ram",
    # Training Metrics
        "Mean Policy Reward": "reward_mean",
        "Mean Policy Reward (start step)": "reward_mean_step",
        "Early Reward Mean": "early_reward_mean",
        "Early Reward Mean (start step)": "early_reward_mean_step",
        "Final Reward Mean": "final_reward_mean",
        "Final Reward Mean (start step)": "final_reward_mean_step",
        "Best Reward": "best_reward",
        "Best Reward (step)": "best_reward_step",
        "Step Interval (Running Mean)": "step_interval",
        "Mean Policy Loss": "p_loss_mean",
        "Mean Policy Loss (start step)": "p_loss_mean_step",
        "Mean Value Loss": "v_loss_mean",
        "Mean Value Loss (start step)": "v_loss_mean_step",
        "Mean Entropy": "entropy_mean",
        "Mean Entropy (start step)": "entropy_mean_step",
    # Threshold Analysis (for ML prediction)
        "Threshold Value": "threshold_value",
        "Steps to Threshold": "steps_to_threshold",
        "Time to Threshold (s)": "time_to_threshold",
        "Threshold Version": "threshold_version",
        "Run Reached Threshold": "run_reached_threshold",
}
