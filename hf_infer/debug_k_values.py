import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from PIL import Image
import sys

device = "cuda"
model_path = "/mnt/ssd/QwenModels/Qwen3-VL-8B-Instruct"
image_path = "/mnt/ssd/QwenModels/demo.jpg"

model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_path, torch_dtype=torch.float16, device_map=device
)
processor = AutoProcessor.from_pretrained(model_path)
model.eval()

image = Image.open(image_path).convert("RGB")
messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": "描述这张图片"}]}]
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt").to(device)

with torch.no_grad():
    # Get vision encoder hidden states
    pixel_values = inputs["pixel_values"]
    image_grid_thw = inputs["image_grid_thw"]
    
    visual = model.visual
    # Get patch embeddings
    hidden_states = visual.patch_embed(pixel_values)
    rotary_pos_emb = visual.rot_pos_emb(image_grid_thw)
    
    # cu_seqlens setup
    cu_seqlens = torch.repeat_interleave(image_grid_thw[:, 1] * image_grid_thw[:, 2], image_grid_thw[:, 0]).cumsum(dim=0, dtype=torch.int32)
    cu_seqlens = torch.nn.functional.pad(cu_seqlens, (1, 0), value=0)
    
    print(f"cu_seqlens: {cu_seqlens}")
    
    # Process first block only
    block0 = visual.blocks[0]
    
    # Layer norm
    hidden_states = block0.norm1(hidden_states)
    
    # QKV projection  
    qkv = block0.attn.qkv(hidden_states)  # [1, num_tokens, 3*hidden_dim]
    print(f"qkv shape: {qkv.shape}")
    
    # Split into Q, K, V
    hidden_size = visual.config.hidden_size  # 1152
    head_dim = visual.config.hidden_size // visual.config.num_heads  # 72
    num_heads = visual.config.num_heads  # 16
    
    q, k, v = qkv.reshape(qkv.shape[0], qkv.shape[1], 3, num_heads, -1).permute(2, 0, 3, 1, 4).unbind(0)
    print(f"q shape: {q.shape}")  # [batch, heads, tokens, head_dim]
    print(f"k shape: {k.shape}")
    
    # Apply RoPE
    from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb_vision
    q, k = apply_rotary_pos_emb_vision(q, k, rotary_pos_emb)
    
    # Print K for head 0, tokens 0-4
    print("\n=== K values after RoPE, head 0 ===")
    for t in range(5):
        print(f"K[head=0, token={t}, :5]: {k[0, 0, t, :5].cpu().numpy()}")
    
    # Also print K for tokens 0-10 to see the pattern
    print("\n=== K[head=0] dot products with Q[head=0, token=0] ===")
    q0 = q[0, 0, 0, :]  # [head_dim]
    for t in range(5):
        kt = k[0, 0, t, :]  # [head_dim]
        score = (q0 * kt).sum().item()
        scaled_score = score / (head_dim ** 0.5)
        print(f"Q[0] dot K[{t}] = {score:.4f}, scaled = {scaled_score:.4f}")
    
    print("\n=== Manual attention scores computation ===")
    # Let's compute the full attention scores for head 0
    q0_all = q[0, 0, :, :]  # [tokens, head_dim]
    k0_all = k[0, 0, :, :]  # [tokens, head_dim]
    
    # scores = Q @ K.T / sqrt(d)
    scores = torch.matmul(q0_all, k0_all.transpose(0, 1)) / (head_dim ** 0.5)
    print(f"scores shape: {scores.shape}")
    print(f"scores[0, :5]: {scores[0, :5].cpu().numpy()}")
    print(f"scores[0, :10]: {scores[0, :10].cpu().numpy()}")

