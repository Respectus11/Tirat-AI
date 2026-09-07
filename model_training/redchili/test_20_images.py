import os
import glob
from pathlib import Path
import numpy as np
from PIL import Image

# Exact reproduction of app/src/ml/preprocess.ts pipeline:
# 1. Resize shortest side to 224 (aspect ratio preserved)
# 2. Center-crop to exactly 224x224
# 3. Decode RGB pixels in float32 format (0..255)
# 4. Interleave RGB floats in shape [1, 224, 224, 3]

def preprocess_app_pipeline(image_path: str) -> np.ndarray:
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        w, h = img.size
        
        # Step 1: resize shortest side to 224
        if w < h:
            new_w = 224
            new_h = int(h * (224 / w))
        else:
            new_h = 224
            new_w = int(w * (224 / h))
            
        img_resized = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
        
        # Step 2: center-crop to 224x224
        left = max(0, (new_w - 224) // 2)
        top = max(0, (new_h - 224) // 2)
        right = left + 224
        bottom = top + 224
        img_cropped = img_resized.crop((left, top, right, bottom))
        
        # Step 3 & 4: RGB floats 0..255, shape [1, 224, 224, 3]
        arr = np.array(img_cropped, dtype=np.float32)
        tensor_input = np.expand_dims(arr, axis=0) # [1, 224, 224, 3]
        return tensor_input

# Select 20 sample images across classes from dataset
dataset_dir = Path("C:/Users/hp/Downloads/redchili_dataset")
image_paths = []

# Search for pure and adulterated subfolders in DS I and DS II
all_jpgs = list(dataset_dir.rglob("*.jpg")) + list(dataset_dir.rglob("*.png")) + list(dataset_dir.rglob("*.jpeg"))

pure_imgs = [p for p in all_jpgs if "c1_pwh" in str(p).lower() or "wh00" in str(p).lower()]
adulterated_imgs = [p for p in all_jpgs if p not in pure_imgs]

print(f"Total dataset images found: {len(all_jpgs)} (Pure: {len(pure_imgs)}, Adulterated: {len(adulterated_imgs)})")

# Take 10 pure and 10 adulterated images for 20-image test set
selected_pure = pure_imgs[:10]
selected_adulterated = adulterated_imgs[:10]
sample_20 = selected_pure + selected_adulterated

output_log = []
output_log.append(f"Selected 20 Test Images from Dataset:")
output_log.append("=" * 80)

for idx, p in enumerate(sample_20, 1):
    expected_cls = "pure" if p in selected_pure else "adulterated"
    rel_path = p.relative_to(dataset_dir)
    input_tensor = preprocess_app_pipeline(str(p))
    output_log.append(f"[{idx:02d}] {rel_path}")
    output_log.append(f"     Expected Ground Truth: {expected_cls}")
    output_log.append(f"     Preprocessed Shape: {input_tensor.shape}, Range: [{input_tensor.min():.1f}, {input_tensor.max():.1f}]")

# Attempt LiteRT / TFLite Model Loading
try:
    from ai_edge_litert.interpreter import Interpreter
    output_log.append("\nLiteRT Interpreter loaded successfully!")
    
    redchili_model_path = "C:/Users/hp/Desktop/Tirat AI/app/assets/models/redchili_model.tflite"
    tirat_model_path = "C:/Users/hp/Desktop/Tirat AI/app/assets/models/tirat_model.tflite"
    
    output_log.append(f"\nModel File Status:")
    output_log.append(f"  redchili_model.tflite: {os.path.getsize(redchili_model_path)} bytes")
    output_log.append(f"  tirat_model.tflite: {os.path.getsize(tirat_model_path)} bytes")
    
    # Try loading redchili model
    try:
        interpreter = Interpreter(model_path=redchili_model_path)
        interpreter.allocate_tensors()
        output_log.append("  [!] redchili_model.tflite loaded into Interpreter.")
    except Exception as e:
        output_log.append(f"  [X] redchili_model.tflite loading failed: {e}")
        
    # Try loading tirat_model
    try:
        interpreter2 = Interpreter(model_path=tirat_model_path)
        interpreter2.allocate_tensors()
        output_log.append("  [!] tirat_model.tflite loaded into Interpreter.")
    except Exception as e:
        output_log.append(f"  [X] tirat_model.tflite loading failed: {e}")

except Exception as e:
    output_log.append(f"\nLiteRT import failed: {e}")

report_txt = "\n".join(output_log)
print(report_txt)
with open("C:/Users/hp/Desktop/Tirat AI/model_training/redchili/test_20_results.txt", "w", encoding="utf-8") as f:
    f.write(report_txt)
