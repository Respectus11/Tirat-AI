import shutil
import subprocess
from pathlib import Path

output_file = Path("C:/Users/hp/Desktop/Tirat AI/model_training/redchili/env_search.txt")

out = ""
# Check for conda
conda_path = shutil.which("conda")
out += f"conda path: {conda_path}\n"
if conda_path:
    # Run conda env list
    try:
        res = subprocess.run(["conda", "env", "list"], capture_output=True, text=True)
        out += f"conda envs:\n{res.stdout}\n"
    except Exception as e:
        out += f"conda env error: {e}\n"

# Check for wsl
wsl_path = shutil.which("wsl")
out += f"wsl path: {wsl_path}\n"
if wsl_path:
    try:
        res = subprocess.run(["wsl", "which", "python3"], capture_output=True, text=True)
        out += f"wsl python3: {res.stdout.strip()}\n"
        res2 = subprocess.run(["wsl", "python3", "--version"], capture_output=True, text=True)
        out += f"wsl python3 version: {res2.stdout.strip()}\n"
    except Exception as e:
        out += f"wsl error: {e}\n"

with open(output_file, "w", encoding="utf-8") as f:
    f.write(out)
