#!/usr/bin/env python3
"""Test Qwen3-VL inference with transformers."""

import torch
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor

print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 可用: {torch.cuda.is_available()}")

model_name = "/mnt/ssd/QwenModels/Qwen3-VL-8B-Instruct"
image_path = "/mnt/ssd/QwenModels/demo.jpg"

print(f"Loading model from {model_name}...")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="eager"  # Use eager attention for compatibility
)

processor = Qwen3VLProcessor.from_pretrained(model_name)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": image_path,
            },
            {"type": "text", "text": "What is in this image?"},
        ],
    }
]

print("Processing input...")
# Preparation for inference
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
)

# Move inputs to GPU
inputs = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

print(f"Input shape: input_ids={inputs['input_ids'].shape}")
print(f"Grid thw: {inputs.get('image_grid_thw', 'N/A')}")

# Inference: Generation of the output
print("Generating...")
generated_ids = model.generate(**inputs, max_new_tokens=256)

print(f"Generated {generated_ids.shape[1] - inputs['input_ids'].shape[1]} new tokens")

generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)

print("=" * 50)
print("Response:")
print(output_text[0])
print("=" * 50)
