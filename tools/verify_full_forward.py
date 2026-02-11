#!/usr/bin/env python3
"""
Complete verification of EAGLE-3 forward pass using the same hidden states as C++.
This script computes the full forward pass and compares with C++ output.
"""

import torch
import numpy as np
import struct
import os
import sys

# Add EAGLE path
sys.path.insert(0, '/mnt/ssd/workspace/EAGLE')

def load_binary_fc_input(path="/tmp/cpp_fc_input.bin"):
    """Load FC input saved by C++ code."""
    with open(path, 'rb') as f:
        data = np.fromfile(f, dtype=np.float32)
    return torch.from_numpy(data)


def load_binary_input_embed(path="/tmp/cpp_input_embed.bin"):
    """Load input embedding saved by C++ code."""
    with open(path, 'rb') as f:
        data = np.fromfile(f, dtype=np.float32)
    return torch.from_numpy(data)


def rms_norm(x, weight, eps=1e-6):
    """RMSNorm implementation - compute in FP32 for numerical stability."""
    x_fp32 = x.float()
    weight_fp32 = weight.float()
    variance = x_fp32.pow(2).mean(-1, keepdim=True)
    x_normed = x_fp32 * torch.rsqrt(variance + eps)
    result = (x_normed * weight_fp32).to(x.dtype)
    return result


def main():
    print("Loading EAGLE-3 model weights...")
    model_path = "/mnt/ssd/QwenModels/Qwen3-8B_eagle3/pytorch_model.bin"
    state_dict = torch.load(model_path, map_location='cpu')
    
    # Load FC input from C++
    fc_input = load_binary_fc_input().to(torch.float16)
    print(f"FC input: {fc_input.shape}, first 4: {fc_input[:4].tolist()}")
    
    # Load weights
    fc_weight = state_dict['fc.weight'].to(torch.float16)
    hidden_norm_weight = state_dict['midlayer.hidden_norm.weight'].to(torch.float16)
    input_norm_weight = state_dict['midlayer.input_layernorm.weight'].to(torch.float16)
    q_proj_weight = state_dict['midlayer.self_attn.q_proj.weight'].to(torch.float16)
    k_proj_weight = state_dict['midlayer.self_attn.k_proj.weight'].to(torch.float16)
    v_proj_weight = state_dict['midlayer.self_attn.v_proj.weight'].to(torch.float16)
    o_proj_weight = state_dict['midlayer.self_attn.o_proj.weight'].to(torch.float16)
    post_attn_norm_weight = state_dict['midlayer.post_attention_layernorm.weight'].to(torch.float16)
    gate_proj_weight = state_dict['midlayer.mlp.gate_proj.weight'].to(torch.float16)
    up_proj_weight = state_dict['midlayer.mlp.up_proj.weight'].to(torch.float16)
    down_proj_weight = state_dict['midlayer.mlp.down_proj.weight'].to(torch.float16)
    final_norm_weight = state_dict['norm.weight'].to(torch.float16)
    lm_head_weight = state_dict['lm_head.weight'].to(torch.float16)
    d2t = state_dict['d2t']
    
    print(f"\nWeight shapes:")
    print(f"  FC: {fc_weight.shape}")
    print(f"  Q_proj: {q_proj_weight.shape}")
    print(f"  K_proj: {k_proj_weight.shape}")
    print(f"  V_proj: {v_proj_weight.shape}")
    print(f"  O_proj: {o_proj_weight.shape}")
    print(f"  Gate: {gate_proj_weight.shape}")
    print(f"  Up: {up_proj_weight.shape}")
    print(f"  Down: {down_proj_weight.shape}")
    print(f"  LM Head: {lm_head_weight.shape}")
    print(f"  D2T: {d2t.shape}")
    
    # Step 1: FC forward
    fc_output = torch.nn.functional.linear(fc_input.unsqueeze(0), fc_weight).squeeze(0)
    print(f"\n=== Step 1: FC Output ===")
    print(f"FC output first 4: {fc_output[:4].tolist()}")
    
    # Step 2: Apply hidden_norm (RMSNorm to FC output)
    hidden_normed = rms_norm(fc_output, hidden_norm_weight)
    print(f"\n=== Step 2: Hidden Norm Output ===")
    print(f"hidden_normed first 4: {hidden_normed[:4].tolist()}")
    
    # We need an input embedding to complete the forward pass
    # Load the input embedding saved by C++
    try:
        input_embed = load_binary_input_embed().to(torch.float16)
        print(f"\nLoaded input_embed from C++: {input_embed.shape}, first 4: {input_embed[:4].tolist()}")
    except FileNotFoundError:
        print("\nInput embedding file not found! Using dummy embedding.")
        # Fallback: use a dummy embedding
        input_embed = torch.zeros(4096, dtype=torch.float16)
    
    # Step 3: Apply input_layernorm
    input_normed = rms_norm(input_embed, input_norm_weight)
    print(f"\n=== Step 3: Input Norm Output ===")
    print(f"input_normed first 4: {input_normed[:4].tolist()}")
    
    # Step 4: Concatenate [input_normed, hidden_normed]
    concat = torch.cat([input_normed, hidden_normed], dim=-1)
    print(f"\n=== Step 4: Concatenated Hidden States ===")
    print(f"concat shape: {concat.shape}")
    print(f"concat first 4 (input_normed part): {concat[:4].tolist()}")
    print(f"concat mid 4 (hidden_normed part): {concat[4096:4100].tolist()}")
    
    # Step 5: Q, K, V projections
    Q = torch.nn.functional.linear(concat.unsqueeze(0), q_proj_weight).squeeze(0)
    K = torch.nn.functional.linear(concat.unsqueeze(0), k_proj_weight).squeeze(0)
    V = torch.nn.functional.linear(concat.unsqueeze(0), v_proj_weight).squeeze(0)
    print(f"\n=== Step 5: Q, K, V Projections ===")
    print(f"Q first 4: {Q[:4].tolist()}")
    print(f"K first 4: {K[:4].tolist()}")
    print(f"V first 4: {V[:4].tolist()}")
    
    # Note: We're skipping RoPE for now since it requires position info
    # In actual usage, Q and K would be rotated
    
    # Step 6: Attention (simplified for single position with no past KV)
    # For position 0 with no past, attention is just self-attention with one token
    # score = softmax(Q @ K^T / sqrt(d)) @ V = V (since it's just one token)
    # But actually for position 0, we only have self-attention
    
    # In the prefill case, we have multiple positions
    # For now, let's just do a simplified single-position attention
    
    num_heads = 32
    num_kv_heads = 8
    head_dim = 128
    
    # Reshape for attention
    Q_reshaped = Q.view(num_heads, head_dim)  # [32, 128]
    K_reshaped = K.view(num_kv_heads, head_dim)  # [8, 128]
    V_reshaped = V.view(num_kv_heads, head_dim)  # [8, 128]
    
    # GQA: repeat KV heads
    kv_repeat = num_heads // num_kv_heads  # 4
    K_repeated = K_reshaped.repeat_interleave(kv_repeat, dim=0)  # [32, 128]
    V_repeated = V_reshaped.repeat_interleave(kv_repeat, dim=0)  # [32, 128]
    
    # Single position attention: each head attends only to itself
    # attn_weight = softmax(Q @ K^T / sqrt(d)) = softmax(score)
    # For single position: score = Q @ K^T / sqrt(d) is a scalar per head
    scores = (Q_reshaped * K_repeated).sum(dim=-1) / (head_dim ** 0.5)  # [32]
    attn_weights = torch.softmax(scores, dim=-1)  # [32]
    
    # Output: weighted sum of values (but for single position, it's just V)
    attn_output = V_repeated  # [32, 128]
    attn_output_flat = attn_output.reshape(-1)  # [4096]
    
    print(f"\n=== Step 6: Attention Output ===")
    print(f"Attention output first 4: {attn_output_flat[:4].tolist()}")
    
    # Step 7: O projection
    o_output = torch.nn.functional.linear(attn_output_flat.unsqueeze(0), o_proj_weight).squeeze(0)
    print(f"\n=== Step 7: O Projection Output ===")
    print(f"O output first 4: {o_output[:4].tolist()}")
    
    # Step 8: Residual add (add to fc_output, which is the residual)
    hidden_out = fc_output + o_output
    print(f"\n=== Step 8: After Residual Add ===")
    print(f"hidden_out first 4: {hidden_out[:4].tolist()}")
    
    # Step 9: Post attention norm
    post_normed = rms_norm(hidden_out, post_attn_norm_weight)
    print(f"\n=== Step 9: Post Attention Norm ===")
    print(f"post_normed first 4: {post_normed[:4].tolist()}")
    
    # Step 10: FFN (SwiGLU)
    gate = torch.nn.functional.linear(post_normed.unsqueeze(0), gate_proj_weight).squeeze(0)
    up = torch.nn.functional.linear(post_normed.unsqueeze(0), up_proj_weight).squeeze(0)
    
    # SiLU activation on gate
    gate_activated = gate * torch.sigmoid(gate.float()).to(gate.dtype)
    
    # Element-wise multiply
    ffn_hidden = gate_activated * up
    
    # Down projection
    down = torch.nn.functional.linear(ffn_hidden.unsqueeze(0), down_proj_weight).squeeze(0)
    print(f"\n=== Step 10: FFN Output ===")
    print(f"down first 4: {down[:4].tolist()}")
    
    # Step 11: Residual add
    hidden_out2 = hidden_out + down
    print(f"\n=== Step 11: After FFN Residual Add ===")
    print(f"hidden_out2 first 4: {hidden_out2[:4].tolist()}")
    
    # Step 12: Final norm
    final_normed = rms_norm(hidden_out2, final_norm_weight)
    print(f"\n=== Step 12: Final Norm ===")
    print(f"final_normed first 4: {final_normed[:4].tolist()}")
    
    # Step 13: LM Head
    logits = torch.nn.functional.linear(final_normed.unsqueeze(0), lm_head_weight).squeeze(0)
    print(f"\n=== Step 13: LM Head Output (Logits) ===")
    print(f"logits shape: {logits.shape}")
    print(f"logits first 4: {logits[:4].tolist()}")
    print(f"logits stats: min={logits.min():.4f}, max={logits.max():.4f}")
    
    # Step 14: Argmax and D2T mapping
    draft_idx = logits.argmax().item()
    target_idx = draft_idx + d2t[draft_idx].item()
    print(f"\n=== Step 14: Prediction ===")
    print(f"Draft index (argmax): {draft_idx}")
    print(f"D2T offset: {d2t[draft_idx].item()}")
    print(f"Target token ID: {target_idx}")
    
    # Top-5 predictions
    top5 = torch.topk(logits, 5)
    print(f"\nTop-5 predictions:")
    for i, (score, idx) in enumerate(zip(top5.values.tolist(), top5.indices.tolist())):
        target = idx + d2t[idx].item()
        print(f"  {i}: draft={idx}, score={score:.4f}, target={target}")
    
    print("\n=== C++ Reported Values ===")
    print("FC output first 4: [343.7500, 710.0000, -734.5000, -103.6250]")
    print("hidden_norm first 4: [0.8540, 1.7783, -1.8613, -0.2428]")
    print("input_norm first 4: [0.8208, -1.1885, 0.4331, -0.0593]")
    print("concat first 4 (input_norm): [0.8208, -1.1885, 0.4331, -0.0593]")
    print("Q before RoPE first 4: [-1.2588, -0.4592, 2.7578, 1.3418]")
    print("K before RoPE first 4: [-0.0811, -0.1012, -0.2773, 0.0967]")


if __name__ == '__main__':
    main()
