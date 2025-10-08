import psutil
from datetime import datetime

print("Monitoring system usage... (press Ctrl+C to stop)\n")
print(f"{'Timestamp':<25} {'CPU %':<8} {'RAM Used (GB)':<15} {'RAM %':<8}")
print("-" * 60)

try:
    while True:
        cpu_percent = psutil.cpu_percent(interval=1)
        ram = psutil.virtual_memory()
        ram_used_gb = ram.used / (1024 ** 3)
        ram_percent = ram.percent

        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S'): <25} "
              f"{cpu_percent: <8.1f} {ram_used_gb: <15.2f} {ram_percent: <8.1f}")

except KeyboardInterrupt:
    print("\n Monitoring stopped.")