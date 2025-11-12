"""System resource monitoring during training."""
from typing import Optional

import psutil
import threading
import time

from scripts.models.data_models import SystemMetrics

class SystemMonitor:
    """Monitors CPU and RAM usage during training."""

    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._cpu_readings = []
        self._ram_readings = []
        self._start_time = None

    def start(self):
        """Start monitoring in background thread."""
        self._start_time = time.time()
        self._thread = threading.Thread(target=self._monitor_loop)
        self._thread.start()

    def stop(self) -> SystemMetrics:
        """Stop monitoring and return metrics."""
        self._stop_event.set()
        if self._thread:
            self._thread.join()

        total_time = time.time() - self._start_time
        mean_cpu = sum(self._cpu_readings) / len(self._cpu_readings)
        mean_ram = sum(self._ram_readings) / len(self._ram_readings)

        return SystemMetrics(total_time, mean_cpu, mean_ram)

    def _monitor_loop(self):
        """Continuous monitoring loop."""
        while not self._stop_event.is_set():
            self._cpu_readings.append(psutil.cpu_percent(interval=self.interval))
            self._ram_readings.append(psutil.virtual_memory().percent)