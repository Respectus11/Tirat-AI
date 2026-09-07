import os
from pathlib import Path
import numpy as np
from PIL import Image
from ai_edge_litert.interpreter import Interpreter

# Exact app/src/ml/preprocess.ts reproduction
def preprocess_app_pipeline(image_path: str) -> np.ndarray:
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        w, h = img.size
        
        if w < h:
            new_w = 224
            new_h = int(h * (224 / w))
        else:
            new_h = 224
            new_w = int(w * (224 / h))
            
        img_resized = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
        
        left = max(0, (new_w - 224) // 2)
        top = max(0, (new_h - 224) // 2)
        right = left + 224
        bottom = top + 224
        img_cropped = img_resized.crop((left, top, right, bottom))
        
        arr = np.array(img_cropped, dtype=np.float32)
        tensor_input = np.expand_dims(arr, axis=0) # [1, 224, 224, 3]
        return tensor_input

dataset_dir = Path("C:/Users/hp/Downloads/redchili_dataset")
all_jpgs = list(dataset_dir.rglob("*.jpg")) + list(dataset_dir.rglob("*.png")) + list(dataset_dir.rglob("*.jpeg"))

pure_imgs = [p for p in all_jpgs if "c1_pwh" in str(p).lower() or "wh00" in str(p).lower()]
adulterated_imgs = [p for p in all_jpgs if p not in pure_imgs]

# Sample 10 pure, 10 adulterated
sample_20 = pure_imgs[:10] + adulterated_imgs[:10]

# Load redchili model
redchili_model_path = "C:/Users/hp/Desktop/Tirat AI/app/assets/models/redchili_model.tflite"
interpreter = Interpreter(model_path=redchili_model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f"Red Chili Model Input Details: {input_details}")
print(f"Red Chili Model Output Details: {output_details}")

redchili_classes = ["pure", "adulterated"]

results = []
correct_count = 0

report_lines = []
report_lines.append(f"{'#':<3} | {'Image File':<45} | {'Ground Truth':<12} | {'Predicted':<12} | {'Confidence':<10} | {'Match?':<6}")
report_lines.append("-" * 105)

for idx, p in enumerate(sample_20, 1):
    expected_cls = "pure" if p in pure_imgs[:10] else "adulterated"
    rel_name = p.name
    short_rel_path = f"{p.parent.name}/{p.name}"
    
    input_tensor = preprocess_app_pipeline(str(p))
    
    # Set tensor input
    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()
    
    # Get tensor output
    output_data = interpreter.get_tensor(output_details[0]['index'])
    probs = output_data[0] # softmax array
    
    pred_idx = int(np.argmax(probs))
    pred_cls = redchili_classes[pred_idx]
    confidence = float(probs[pred_idx])
    
    is_match = (pred_cls == expected_cls)
    if is_match:
        correct_count += 1
        
    match_str = "YES" if is_match else "NO"
    
    report_lines.append(f"{idx:02d}  | {short_rel_path:<45} | {expected_cls:<12} | {pred_cls:<12} | {confidence*100:>8.2f}% | {match_str:<6}")
    results.append({
        "idx": idx,
        "path": str(p),
        "expected": expected_cls,
        "predicted": pred_cls,
        "confidence": confidence,
        "probs": probs.tolist(),
        "match": is_match
    })

accuracy = (correct_count / len(sample_20)) * 100
report_lines.append("-" * 105)
report_lines.append(f"Overall Test Accuracy: {correct_count}/{len(sample_20)} ({accuracy:.1f}%)")

final_report = "\n".join(report_lines)
print(final_report)

with open("C:/Users/hp/Desktop/Tirat AI/model_training/redchili/phase2_evaluation_table.txt", "w", encoding="utf-8") as f:
    f.write(final_report)
