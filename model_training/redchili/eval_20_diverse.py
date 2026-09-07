import os
from pathlib import Path
import numpy as np
from PIL import Image
from ai_edge_litert.interpreter import Interpreter

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
        tensor_input = np.expand_dims(arr, axis=0)
        return tensor_input

dataset_dir = Path("C:/Users/hp/Downloads/redchili_dataset")

# Sample across various folders:
# Pure: WH00 (5 images), C1_PWH (5 images)
# Adulterated: WHBM_5 (2), WHGM_10 (2), WHRB_15 (2), WHWB_5 (2), WHWS_10 (2)

pure_wh00 = list((dataset_dir / "Red Chilli Adulteration Digital image Dataset (DS-WH-1)/DS-WH-1/DS II (WH)/Test/WH00").glob("*.jpg"))[:5]
pure_c1 = list((dataset_dir / "Red Chilli Adulteration Digital image Dataset (DS-WH-1)/DS-WH-1/DS I (WH)/Test/C1_PWH").glob("*.jpg"))[:5]
adulterated_bm = list((dataset_dir / "Red Chilli Adulteration Digital image Dataset (DS-WH-1)/DS-WH-1/DS II (WH)/Test/WHBM_5").glob("*.jpg"))[:2]
adulterated_gm = list((dataset_dir / "Red Chilli Adulteration Digital image Dataset (DS-WH-1)/DS-WH-1/DS II (WH)/Test/WHGM_10").glob("*.jpg"))[:2]
adulterated_rb = list((dataset_dir / "Red Chilli Adulteration Digital image Dataset (DS-WH-1)/DS-WH-1/DS II (WH)/Test/WHRB_15").glob("*.jpg"))[:2]
adulterated_wb = list((dataset_dir / "Red Chilli Adulteration Digital image Dataset (DS-WH-1)/DS-WH-1/DS II (WH)/Test/WHWB_5").glob("*.jpg"))[:2]
adulterated_ws = list((dataset_dir / "Red Chilli Adulteration Digital image Dataset (DS-WH-1)/DS-WH-1/DS II (WH)/Test/WHWS_10").glob("*.jpg"))[:2]

test_20_diverse = pure_wh00 + pure_c1 + adulterated_bm + adulterated_gm + adulterated_rb + adulterated_wb + adulterated_ws

redchili_model_path = "C:/Users/hp/Desktop/Tirat AI/app/assets/models/redchili_model.tflite"
interpreter = Interpreter(model_path=redchili_model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

redchili_classes = ["pure", "adulterated"]

report_lines = []
report_lines.append(f"{'#':<3} | {'Folder / File':<35} | {'Ground Truth':<12} | {'Predicted':<12} | {'Confidence':<10} | {'Match?':<6}")
report_lines.append("-" * 95)

correct_count = 0
for idx, p in enumerate(test_20_diverse, 1):
    expected_cls = "pure" if (p in pure_wh00 or p in pure_c1) else "adulterated"
    folder_file = f"{p.parent.name}/{p.name}"
    
    input_tensor = preprocess_app_pipeline(str(p))
    
    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()
    
    output_data = interpreter.get_tensor(output_details[0]['index'])
    probs = output_data[0]
    
    pred_idx = int(np.argmax(probs))
    pred_cls = redchili_classes[pred_idx]
    confidence = float(probs[pred_idx])
    
    is_match = (pred_cls == expected_cls)
    if is_match:
        correct_count += 1
        
    match_str = "YES" if is_match else "NO"
    report_lines.append(f"{idx:02d}  | {folder_file:<35} | {expected_cls:<12} | {pred_cls:<12} | {confidence*100:>8.2f}% | {match_str:<6}")

accuracy = (correct_count / len(test_20_diverse)) * 100
report_lines.append("-" * 95)
report_lines.append(f"Diverse 20-Image Test Set Accuracy: {correct_count}/{len(test_20_diverse)} ({accuracy:.1f}%)")

final_report = "\n".join(report_lines)
print(final_report)

with open("C:/Users/hp/Desktop/Tirat AI/model_training/redchili/diverse_20_evaluation.txt", "w", encoding="utf-8") as f:
    f.write(final_report)
