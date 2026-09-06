import subprocess
from pathlib import Path

output_file = Path("C:/Users/hp/Desktop/Tirat AI/model_training/redchili/wsl_check.txt")

out = ""
try:
    res = subprocess.run(["wsl", "-l", "-v"], capture_output=True, text=True)
    out += f"wsl -l -v stdout:\n{res.stdout}\n"
    out += f"wsl -l -v stderr:\n{res.stderr}\n"
except Exception as e:
    out += f"wsl list error: {e}\n"

try:
    res = subprocess.run(["wsl", "docker", "--version"], capture_output=True, text=True)
    out += f"wsl docker: {res.stdout.strip()} {res.stderr.strip()}\n"
except Exception as e:
    out += f"docker check error: {e}\n"

with open(output_file, "w", encoding="utf-8") as f:
    f.write(out)
