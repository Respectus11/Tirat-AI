import os
from pathlib import Path

models_dir = Path("C:/Users/hp/Desktop/Tirat AI/app/assets/models")
output_file = Path("C:/Users/hp/Desktop/Tirat AI/model_training/redchili/model_sizes.txt")

with open(output_file, "w", encoding="utf-8") as f:
    for p in models_dir.iterdir():
        f.write(f"File: {p.name}, Size: {p.stat().st_size} bytes\n")
