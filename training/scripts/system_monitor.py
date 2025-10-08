import psutil
from datetime import datetime

print("Monitoring system usage... (press Ctrl+C to stop)\n")
print(f"{'Timestamp':<25} {'CPU %':<8} {'RAM Used (GB)':<15} {'RAM %':<8} {'Mean CPU %':<10} {'Mean RAM %':<10}")
print("-" * 85)

cpu_readings = []
ram_readings = []

try:
    while True:
        cpu_percent = psutil.cpu_percent(interval=1)
        ram = psutil.virtual_memory()
        ram_used_gb = ram.used / (1024 ** 3)
        ram_percent = ram.percent

        # Update running means
        cpu_readings.append(cpu_percent)
        ram_readings.append(ram_percent)
        mean_cpu = sum(cpu_readings) / len(cpu_readings)
        mean_ram = sum(ram_readings) / len(ram_readings)

        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S'): <25} "
              f"{cpu_percent: <8.1f} {ram_used_gb: <15.2f} {ram_percent: <8.1f} "
              f"{mean_cpu: <10.1f} {mean_ram: <10.1f}")

except KeyboardInterrupt:
    print("\nMonitoring stopped.")
    print(f"\nFinal mean CPU usage: {sum(cpu_readings)/len(cpu_readings):.1f}%")
    print(f"Final mean RAM usage: {sum(ram_readings)/len(ram_readings):.1f}%")