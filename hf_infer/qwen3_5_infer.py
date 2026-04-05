"""
Qwen3.5-9B local inference script using transformers.
Runs multimodal (image + text) inference locally.

Usage:
    cd /home/wangyh/OrinMLLM/hf_infer/
    python qwen3_5_infer.py
"""
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image

MODEL_PATH = "/mnt/ssd/QwenModels/Qwen3.5-9B/"
IMAGE_PATH = "/mnt/ssd/workspace/OrinMLLM/hf_infer/mathv-1327.jpg"

# Load processor and model
print("Loading model from", MODEL_PATH)
processor = AutoProcessor.from_pretrained(MODEL_PATH)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto",
    attn_implementation="eager",
)
print("Model loaded successfully")

# Load image
image = Image.open(IMAGE_PATH).convert("RGB")
print(f"Image loaded: {image.size}")

# Build messages
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": IMAGE_PATH},
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

# Apply chat template
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(f"Prompt: {text[:200]}...")

# Process inputs
inputs = processor(
    text=[text],
    images=[image],
    return_tensors="pt",
    padding=True,
)
inputs = inputs.to(model.device)
print(f"Input ids shape: {inputs['input_ids'].shape}")

# Generate
print("Generating...")
with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=1024,
        do_sample=False,  # greedy for reproducibility
    )

# Decode output (skip input tokens)
generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print("\n=== Response ===")
print(response)
print("================")
