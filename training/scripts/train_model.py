import psutil
import subprocess
import threading
import time
import argparse
from datetime import datetime

def monitor_system(stop_event, interval=1):
    """Monitor CPU and RAM usage while training runs."""
    print("\n🖥️ Monitoring system usage... (will stop when training ends)\n")
    print(f"{'Timestamp':<25} {'CPU %':<8} {'RAM Used (GB)':<15} {'RAM %':<8} "
          f"{'Mean CPU %':<10} {'Mean RAM %':<10}")
    print("-" * 85)

    cpu_readings = []
    ram_readings = []
    start_time = time.time()

    while not stop_event.is_set():
        cpu_percent = psutil.cpu_percent(interval=interval)
        ram = psutil.virtual_memory()
        ram_used_gb = ram.used / (1024 ** 3)
        ram_percent = ram.percent

        cpu_readings.append(cpu_percent)
        ram_readings.append(ram_percent)

        mean_cpu = sum(cpu_readings) / len(cpu_readings)
        mean_ram = sum(ram_readings) / len(ram_readings)

        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S'): <25} "
              f"{cpu_percent: <8.1f} {ram_used_gb: <15.2f} {ram_percent: <8.1f} "
              f"{mean_cpu: <10.1f} {mean_ram: <10.1f}")

    total_time = time.time() - start_time
    mean_cpu = sum(cpu_readings) / len(cpu_readings)
    mean_ram = sum(ram_readings) / len(ram_readings)
    return mean_cpu, mean_ram, total_time


def main():
    # --- Parse CLI arguments ---
    parser = argparse.ArgumentParser(description="Run ML-Agents training with system monitoring.")
    parser.add_argument("--config", required=True, help="Path to the ML-Agents YAML config file")
    parser.add_argument("--run-id", required=True, help="Run ID for the training session")
    args = parser.parse_args()

    config_file = args.config
    run_id = args.run_id

    print(f"\n🚀 Starting ML-Agents training")
    print(f"   Config file: {config_file}")
    print(f"   Run ID:      {run_id}")

    stop_event = threading.Event()
    results = {}

    def monitor_wrapper():
        results["metrics"] = monitor_system(stop_event)

    monitor_thread = threading.Thread(target=monitor_wrapper)
    monitor_thread.start()

    start_time = time.time()

    # --- Launch ML-Agents training ---
    process = subprocess.Popen(
        ["mlagents-learn", config_file, "--run-id", run_id, "--force"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )

    # Stream ML-Agents output
    for line in iter(process.stdout.readline, ""):
        print(line, end="")
    process.wait()

    stop_event.set()
    monitor_thread.join()

    total_time = time.time() - start_time
    mean_cpu, mean_ram, _ = results["metrics"]

    print("\n✅ Training complete.")
    print(f"⏱️  Total wall-clock time: {total_time/60:.2f} minutes ({total_time:.0f} seconds)")
    print(f"📊 Average CPU usage: {mean_cpu:.1f}%")
    print(f"📊 Average RAM usage: {mean_ram:.1f}%")


if __name__ == "__main__":
    main()