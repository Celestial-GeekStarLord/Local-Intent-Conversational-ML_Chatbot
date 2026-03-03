# train_pretrained.py
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
import os

MODEL_NAME = "facebook/blenderbot-400M-distill"
SAVE_DIR = "Model_Pretrained/blenderbot"

os.makedirs(SAVE_DIR, exist_ok=True)

print("Downloading pretrained BlenderBot model (first run only)...")

tokenizer = BlenderbotTokenizer.from_pretrained(MODEL_NAME)
model = BlenderbotForConditionalGeneration.from_pretrained(MODEL_NAME)

tokenizer.save_pretrained(SAVE_DIR)
model.save_pretrained(SAVE_DIR)

print(f"✅ Model saved to {SAVE_DIR}")