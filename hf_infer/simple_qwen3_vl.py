#!/usr/bin/env python3
"""Simple Qwen3-VL inference script using transformers directly."""
import torch
from transformers import AutoProcessor
from transformers import Qwen3VLForConditionalGeneration
from PIL import Image

print("=" * 60)
print("Qwen3-VL Inference Test")
print("=" * 60)

model_path = "/mnt/ssd/QwenModels/Qwen3-VL-8B-Instruct"
image_path = "/mnt/ssd/QwenModels/demo.jpg"

print(f"Model: {model_path}")
print(f"Image: {image_path}")

# Load model
print("\nLoading model...")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    attn_implementation="eager"
)
print(f"Model loaded on device: {model.device}")

# Load processor
print("Loading processor...")
processor = AutoProcessor.from_pretrained(model_path)

# Load image
print("Loading image...")
image = Image.open(image_path)
print(f"Image size: {image.size}")

# Prepare messages
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

# Process
print("\nProcessing inputs...")
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
)

# Check input details
if "input_ids" in inputs:
    print(f"Input tokens: {inputs['input_ids'].shape}")
if "pixel_values" in inputs:
    print(f"Pixel values shape: {inputs['pixel_values'].shape}")

# Move to GPU
inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

# Generate
print("\nGenerating response...")
with torch.no_grad():
    generated_ids = model.generate(**inputs, max_new_tokens=256)

# Decode
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)

print("\n" + "=" * 60)
print("OUTPUT:")
print("=" * 60)
print(output_text[0])
print("=" * 60)
