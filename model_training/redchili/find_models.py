import os
from pathlib import Path

output_file = Path("C:/Users/hp/Desktop/Tirat AI/model_training/redchili/models_found.txt")

search_paths = [
    Path("C:/Users/hp/Downloads"),
    Path("C:/Users/hp/Desktop"),
]

found = []
for sp in search_paths:
    if sp.exists():
        for root, dirs, files in os.walk(sp):
            # speed up by skipping node_modules
            if "node_modules" in dirs:
                dirs.remove("node_modules")
            if ".venv" in dirs:
                dirs.remove(".venv")
            for file in files:
                if file.endswith(".tflite") or file.endswith(".keras") or file.endswith(".h5"):
                    p = Path(root) / file
                    try:
                        found.append(f"{p} ({p.stat().st_size} bytes)")
                    except Exception as e:
                        found.append(f"{p} (error: {e})")

with open(output_file, "w", encoding="utf-8") as f:
    f.write("Found models:\n")
    for item in found:
        f.write(item + "\n")
