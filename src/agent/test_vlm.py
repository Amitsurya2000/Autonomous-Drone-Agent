from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import torch
import time

# 1. Load the VLM (Moondream - a fast, lightweight vision model)
print("[OK] Loading Moondream VLM... (This downloads ~3GB the first time, please wait)")
model_id = "vikhyatk/moondream2"
revision = "main"
model = AutoModelForCausalLM.from_pretrained(
    model_id, trust_remote_code=True, revision=revision, torch_dtype=torch.float32
)
tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
print("[OK] Model loaded!")

# 2. Load your crop image
# CHANGE THIS to a real filename from your data/crops folder!
image_path = "data/crops/track2_1792821710_cell phone.jpg" 


try:
    image = Image.open(image_path).convert("RGB")
    print(f"[OK] Loaded image: {image_path}")
except FileNotFoundError:
    print(f"[ERROR] Could not find {image_path}. Check the filename!")
    exit()

# 3. Ask the VLM a question
prompt = "Describe exactly what you see in this image. What is the main object?"
print(f"\nPrompt: {prompt}")
print("Thinking...")

# 4. Generate the answer
start_time = time.time()
enc_image = model.encode_image(image)
answer = model.answer_question(enc_image, prompt, tokenizer)
end_time = time.time()

print("\n--- VLM RESPONSE ---")
print(answer)
print("--------------------")
print(f"Time taken: {end_time - start_time:.2f} seconds")