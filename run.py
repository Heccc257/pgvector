import subprocess
import sys

def run_command(command):
    try:
        result = subprocess.run(command, check=True, shell=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: Command '{command}' failed with exit code {e.returncode}.")
        sys.exit(e.returncode)

if __name__ == "__main__":
    run_command("python3 opq.py --config_file config.json")
    run_command("python3 construct.py --config_file config.json")
    
    print("Both scripts ran successfully.")
