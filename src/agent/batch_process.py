import os
import glob
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import torch
import datetime

# --- CONFIGURATION ---
CROPS_FOLDER = "data/crops"
ALERTS_FOLDER = "data/alerts"
os.makedirs(ALERTS_FOLDER, exist_ok=True)

print("[OK] Loading Moondream VLM...")
model_id = "vikhyatk/moondream2"
revision = "main"
model = AutoModelForCausalLM.from_pretrained(
    model_id, trust_remote_code=True, revision=revision, torch_dtype=torch.float32
)
tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
print("[OK] Model loaded!")

# 1. Fetch all .jpg images in the folder automatically
image_files = glob.glob(f"{CROPS_FOLDER}/*.jpg")
print(f"[OK] Found {len(image_files)} images to process.")

# 2. Loop through them one by one
for img_path in image_files:
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[{current_time}] Processing: {img_path}")
    
    try:
        image = Image.open(img_path).convert("RGB")
        
        # 3. Ask the VLM
        prompt = "Describe exactly what you see in this image. What is the main object?"
        enc_image = model.encode_image(image)
        answer = model.answer_question(enc_image, prompt, tokenizer)
        
        # 4. Print and save the result
        print(f"--- VLM RESPONSE ---")
        print(answer)
        print("--------------------")
        
        # Save the text output automatically
        base_name = os.path.basename(img_path).replace('.jpg', '')
        txt_path = f"{ALERTS_FOLDER}/{base_name}_analysis.txt"
        with open(txt_path, "w") as f:
            f.write(f"Image: {img_path}\nTime: {current_time}\n\n{answer}")
            
        print(f"[OK] Saved text to {txt_path}")

    except Exception as e:
        print(f"[ERROR] Could not process {img_path}: {e}")

print("\n[OK] Finished processing all images!")