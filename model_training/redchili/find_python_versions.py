import os
from pathlib import Path

output_file = Path("C:/Users/hp/Desktop/Tirat AI/model_training/redchili/python_search.txt")

locations = [
    Path("C:/"),
    Path("C:/Program Files"),
    Path("C:/Program Files (x86)"),
    Path("C:/Users/hp/AppData/Local/Programs"),
    Path("C:/Users/hp/AppData/Local/Python"),
]

found = []
for loc in locations:
    if loc.exists():
        for p in loc.iterdir():
            if p.is_dir() and "python" in p.name.lower():
                found.append(str(p))
                # Check for python.exe inside
                py_exe = p / "python.exe"
                if py_exe.exists():
                    found.append(f"  -> Found: {py_exe}")

with open(output_file, "w") as f:
    f.write("Searching Python dirs:\n")
    for item in found:
        f.write(item + "\n")
