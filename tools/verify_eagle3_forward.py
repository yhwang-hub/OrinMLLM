#!/usr/bin/env python3
"""
Complete verification of EAGLE-3 forward pass between C++ and Python.
"""

import torch
import numpy as np
import struct
import os


def load_eagle3_model(model_path="/mnt/ssd/QwenModels/Qwen3-8B_eagle3/pytorch_model.bin"):
    """Load EAGLE-3 model weights."""
    state_dict = torch.load(model_path, map_location='cpu')
    return state_dict


def load_binary_fc_input(path="/tmp/cpp_fc_input.bin"):
    """Load FC input saved by C++ code."""
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
    state_dict = load_eagle3_model()
    
    # Print all keys
    print("\nAvailable weights:")
    for k, v in sorted(state_dict.items()):
        print(f"  {k}: {v.shape}, {v.dtype}")
    
    # Load FC input from C++
    fc_input = load_binary_fc_input().to(torch.float16)
    print(f"\nFC input: {fc_input.shape}, first 4: {fc_input[:4].tolist()}")
    
    # Load weights
    fc_weight = state_dict['fc.weight'].to(torch.float16)
    hidden_norm_weight = state_dict['midlayer.hidden_norm.weight'].to(torch.float16)
    input_norm_weight = state_dict['midlayer.input_layernorm.weight'].to(torch.float16)
    
    print(f"\nFC weight: {fc_weight.shape}")
    print(f"hidden_norm weight: {hidden_norm_weight.shape}, first 4: {hidden_norm_weight[:4].tolist()}")
    print(f"input_norm weight: {input_norm_weight.shape}, first 4: {input_norm_weight[:4].tolist()}")
    
    # Step 1: FC forward
    fc_output = torch.nn.functional.linear(fc_input.unsqueeze(0), fc_weight).squeeze(0)
    print(f"\n=== Step 1: FC Output ===")
    print(f"FC output: {fc_output.shape}, first 4: {fc_output[:4].tolist()}")
    print(f"FC output stats: min={fc_output.min():.4f}, max={fc_output.max():.4f}")
    
    # Step 2: Apply hidden_norm (RMSNorm to FC output)
    hidden_normed = rms_norm(fc_output, hidden_norm_weight)
    print(f"\n=== Step 2: Hidden Norm Output ===")
    print(f"hidden_normed: {hidden_normed.shape}, first 4: {hidden_normed[:4].tolist()}")
    print(f"hidden_normed stats: min={hidden_normed.min():.4f}, max={hidden_normed.max():.4f}")
    
    # We also need the input embedding - let's just use zeros for now to check the flow
    # In real usage, this would be embed_tokens(input_id)
    
    print("\n=== C++ Reported Values ===")
    print("FC output first 4: [343.7500, 710.0000, -734.5000, -103.6250]")
    print("hidden_norm first 4: [0.8540, 1.7783, -1.8613, -0.2428]")
    print("input_norm first 4: [0.8208, -1.1885, 0.4331, -0.0593]")
    
    print("\n=== Comparison ===")
    print(f"FC match: {torch.allclose(fc_output[:4], torch.tensor([343.75, 710.0, -734.5, -103.625], dtype=torch.float16), rtol=0.01)}")
    print(f"hidden_norm match: {torch.allclose(hidden_normed[:4], torch.tensor([0.8540, 1.7783, -1.8613, -0.2428], dtype=torch.float16), rtol=0.05)}")
    
    # Check the actual computed values
    print(f"\nActual hidden_norm first 4: {hidden_normed[:4].tolist()}")
    expected_hidden_norm = torch.tensor([0.8540, 1.7783, -1.8613, -0.2428], dtype=torch.float16)
    print(f"Expected hidden_norm first 4: {expected_hidden_norm.tolist()}")
    print(f"Difference: {(hidden_normed[:4] - expected_hidden_norm).tolist()}")


if __name__ == '__main__':
    main()
