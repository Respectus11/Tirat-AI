import sys
import subprocess
from pathlib import Path

output_file = Path("C:/Users/hp/Desktop/Tirat AI/model_training/redchili/pip_install_log.txt")

out = ""
for pkg in ["tflite-runtime", "ai-edge-litert", "tensorflow"]:
    out += f"\n--- Trying to install {pkg} ---\n"
    res = subprocess.run([sys.executable, "-m", "pip", "install", pkg], capture_output=True, text=True)
    out += f"stdout:\n{res.stdout}\n"
    out += f"stderr:\n{res.stderr}\n"
    if res.returncode == 0:
        out += f"Successfully installed {pkg}!\n"
        break
    else:
        out += f"Failed to install {pkg}\n"

with open(output_file, "w", encoding="utf-8") as f:
    f.write(out)
