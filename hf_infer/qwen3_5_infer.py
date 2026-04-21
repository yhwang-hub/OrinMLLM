"""
Qwen3.5-9B local inference script using transformers.
Runs multimodal (image + text) inference locally.

Usage:
    cd /home/wangyh/OrinMLLM/hf_infer/
    python qwen3_5_infer.py                # no-think mode (default)
    python qwen3_5_infer.py --think        # thinking mode
"""
import argparse
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image

MODEL_PATH = "/mnt/ssd/Qwen3.5-9B/"
IMAGE_PATH = "/mnt/ssd/workspace/OrinMLLM/hf_infer/mathv-1327.jpg"

parser = argparse.ArgumentParser(description="Qwen3.5-9B inference")
parser.add_argument("--think", action="store_true", help="Enable thinking mode")
parser.add_argument("--max-new-tokens", type=int, default=4096, help="Max new tokens")
args = parser.parse_args()

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

# Build messages (match C++ template with system message)
messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant.",
    },
    {
        "role": "user",
        "content": [
            {"type": "image", "image": IMAGE_PATH},
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

# Apply chat template (enable_thinking=False matches C++ --no-think)
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True,
    enable_thinking=args.think,
)
print(f"Thinking mode: {'ON' if args.think else 'OFF'}")
print(f"Prompt: {text[:300]}...")

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
print(f"Generating (max_new_tokens={args.max_new_tokens})...")
with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,  # greedy for reproducibility
    )

# Decode output (skip input tokens)
generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
raw_response = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

# Remove EOS markers
for eos in ["<|endoftext|>", "<|im_end|>"]:
    raw_response = raw_response.replace(eos, "")

# Separate thinking from response
thinking_text = ""
response_text = raw_response
if "<think>" in raw_response:
    think_end = raw_response.find("</think>")
    think_start = raw_response.find("<think>") + len("<think>")
    if think_end != -1:
        thinking_text = raw_response[think_start:think_end].strip()
        response_text = raw_response[think_end + len("</think>"):].strip()
    else:
        thinking_text = raw_response[think_start:].strip()
        response_text = "(thinking truncated, increase --max-new-tokens)"
else:
    response_text = raw_response.strip()

if thinking_text:
    print(f"\n=== Thinking ({len(thinking_text.split())} words) ===")
    print(thinking_text)
    print("================")
print("\n=== Response ===")
print(response_text)
print("================")
