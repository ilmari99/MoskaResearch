"""
Runs simulate_train_loop.py with a wrapper that fixes a bug in the code.
This runs the code for as long as there are new files coming in to the dataset, i.e. the process is not hanging.
"""
import os

import os
import sys
import time
import subprocess
from simulate_train_loop import SIMULATING_GAMES_FLAG

def get_time_from_latest_change(folder):
    """ Return the time in seconds, since the last change to the folder."""
    last_changed = os.path.getmtime(folder)
    return time.time() - last_changed

def kill_process(p):
    while p.poll() is None:
        p.kill()
        time.sleep(1)
    print("Process killed.")

if __name__ == "__main__":
    stdout_file = "simulate_train_loop_stdout.txt"
    for i in range(10):
        KILLED_PROCESS = False
        print(f"Starting iteration {i}")
        py_exe = sys.executable
        # Start the process
        # Write the stdout to a file
        with open(stdout_file,"a") as f:
            p = subprocess.Popen([py_exe,os.path.abspath("./ModelTraining/simulate_train_loop.py")], stdout=f, stderr=f)
        # Start monitoring file changes WHEN the process is in simulation stage
        time.sleep(10)
        while SIMULATING_GAMES_FLAG:
            # If there are no changes for 60 seconds, the proces is hanging. Kill it.
            if get_time_from_latest_change("./NotRandomDataset_1") > 60:
                print("No changes for 60 seconds. Killing process.")
                kill_process(p)
                KILLED_PROCESS = True
                break
            time.sleep(10)
            
            
            
        