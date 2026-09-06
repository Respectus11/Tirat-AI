import sys
import subprocess
from pathlib import Path

output_file = Path("C:/Users/hp/Desktop/Tirat AI/model_training/redchili/installed_packages.txt")
try:
    res = subprocess.run([sys.executable, "-m", "pip", "list"], capture_output=True, text=True, check=True)
    out = res.stdout
except Exception as e:
    out = f"Error running pip: {e}"

with open(output_file, "w", encoding="utf-8") as f:
    f.write(out)
