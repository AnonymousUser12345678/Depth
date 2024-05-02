import subprocess
import time
import re
import threading


def run_script():
    # subprocess.run(["python", "time_check.py", "--student_arch", "resnet18", "--student_input_size", "64", "--data", "kitti"])
    # subprocess.run(["python", "time_check.py", "--student_arch", "resnet18", "--student_input_size", "32", "--data", "kitti"])
    # subprocess.run(["python", "time_check.py", "--student_arch", "resnet18", "--student_input_size", "64", "--data", "kitti", "--global_info"])
    # subprocess.run(["python", "time_check.py", "--student_arch", "resnet18", "--student_input_size", "32", "--data", "kitti", "--global_info"])
    # subprocess.run(["python", "time_check.py", "--student_arch", "resnet34", "--student_input_size", "64", "--data", "kitti"])
    subprocess.run(["python", "time_check.py", "--student_arch", "resnet34", "--student_input_size", "32", "--data", "kitti"])
    # subprocess.run(["python", "time_check.py", "--student_arch", "resnet34", "--student_input_size", "64", "--data", "kitti", "--global_info"])
    # subprocess.run(["python", "time_check.py", "--student_arch", "resnet34", "--student_input_size", "32", "--data", "kitti", "--global_info"])
    

def monitor_power(usage_list):
    while script_running[0]:
        try:
            result = subprocess.run(["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader"], text=True, capture_output=True)
            power_usage = float(re.search(r"(\d+.\d+) W", result.stdout).group(1))
            usage_list.append(power_usage)
        except Exception as e:
            print("Failed to read power usage:", e)
        time.sleep(1)


if __name__ == "__main__":
    power_usage_list = []
    script_running = [True]

    script_thread = threading.Thread(target=run_script)
    script_thread.start()

    monitor_thread = threading.Thread(target=monitor_power, args=(power_usage_list,))
    monitor_thread.start()

    script_thread.join()
    script_running[0] = False
    monitor_thread.join()

    if power_usage_list:
        average_power = sum(power_usage_list) / len(power_usage_list)
        print(f"Average power usage: {average_power:.2f} W")
    else:
        print("No power usage data collected.")
