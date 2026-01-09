"""Data models for training results."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainingArgs:
    config: str
    run_id: str
    num_steps: int
    ac: bool
    env_path: Optional[str]
    seed: Optional[int]
    no_thresholds: bool
    verbose: bool
    batch_range: Optional[tuple[int, int]]
    randomize: bool

@dataclass
class SystemMetrics:
    total_time: float
    mean_cpu: float
    mean_ram: float

@dataclass
class TrainingResult:
    combined_data: dict
    behavior_name: str
    environment: str
    total_time: float
    log_dir: str