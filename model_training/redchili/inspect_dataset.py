import os
from pathlib import Path

base_path = Path("C:/Users/hp/Downloads/redchili_dataset")
output_file = Path("C:/Users/hp/Desktop/Tirat AI/model_training/redchili/dataset_structure.txt")

with open(output_file, "w", encoding="utf-8") as f:
    f.write(f"Base path exists: {base_path.exists()}\n")
    
    def write_tree(path, depth=0):
        if depth > 5:
            return
        for item in sorted(path.iterdir()):
            if item.is_dir():
                subdirs = [x for x in item.iterdir() if x.is_dir()]
                files = list(item.glob("*.jpg")) + list(item.glob("*.jpeg")) + list(item.glob("*.png"))
                f.write("  " * depth + f"📁 {item.name}/ ({len(files)} files in this dir directly)\n")
                write_tree(item, depth + 1)
                
    write_tree(base_path)

print("Done. Structure written to dataset_structure.txt")
