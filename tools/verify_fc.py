#!/usr/bin/env python3
"""
Verify FC layer computation between C++ and Python EAGLE-3.
This script loads the same hidden states and FC weights, then computes the output.
"""

import torch
import numpy as np
import struct


def load_binary_fc_input(path="/tmp/cpp_fc_input.bin"):
    """Load FC input saved by C++ code."""
    with open(path, 'rb') as f:
        data = np.fromfile(f, dtype=np.float32)
    return torch.from_numpy(data)


def load_eagle3_fc_weights(model_path="/mnt/ssd/QwenModels/Qwen3-8B_eagle3/pytorch_model.bin"):
    """Load FC weights from PyTorch model."""
    state_dict = torch.load(model_path, map_location='cpu')
    fc_weight = state_dict['fc.weight']
    print(f"FC weight shape: {fc_weight.shape}, dtype: {fc_weight.dtype}")
    print(f"FC weight first 4: {fc_weight.view(-1)[:4].tolist()}")
    return fc_weight


def main():
    # Load FC input from C++
    try:
        fc_input = load_binary_fc_input()
        print(f"FC input shape: {fc_input.shape}")
        print(f"FC input first 4: {fc_input[:4].tolist()}")
        print(f"FC input stats: min={fc_input.min():.4f}, max={fc_input.max():.4f}, mean={fc_input.mean():.4f}")
    except FileNotFoundError:
        print("FC input file not found! Run C++ code first to generate /tmp/cpp_fc_input.bin")
        return
    
    # Load FC weights
    fc_weight = load_eagle3_fc_weights()
    
    # Convert to FP16 for computation (matching C++)
    fc_input_fp16 = fc_input.to(torch.float16)
    fc_weight_fp16 = fc_weight.to(torch.float16)
    
    # Compute FC output: output = input @ weight.T
    # In PyTorch Linear: y = x @ W^T + b, W shape is (out_features, in_features)
    # FC weight shape: [4096, 12288] = [hidden_size, 3*hidden_size]
    # FC input shape: [12288] = [3*hidden_size]
    # FC output shape: [4096] = [hidden_size]
    
    fc_output = torch.nn.functional.linear(fc_input_fp16.unsqueeze(0), fc_weight_fp16)
    fc_output = fc_output.squeeze(0)
    
    print(f"\n=== FC Output (Python FP16) ===")
    print(f"FC output shape: {fc_output.shape}")
    print(f"FC output first 4: {fc_output[:4].tolist()}")
    print(f"FC output stats: min={fc_output.min():.4f}, max={fc_output.max():.4f}, mean={fc_output.mean():.4f}")
    
    # Compare with FP32 computation
    fc_output_fp32 = torch.nn.functional.linear(fc_input.unsqueeze(0), fc_weight.float())
    fc_output_fp32 = fc_output_fp32.squeeze(0)
    
    print(f"\n=== FC Output (Python FP32) ===")
    print(f"FC output first 4: {fc_output_fp32[:4].tolist()}")
    print(f"FC output stats: min={fc_output_fp32.min():.4f}, max={fc_output_fp32.max():.4f}")
    
    # Verify dimensions
    print(f"\n=== Dimension Check ===")
    print(f"Input dim: {fc_input.shape[0]} (expected 12288 = 3*4096)")
    print(f"Weight dim: {fc_weight.shape} (expected [4096, 12288])")
    print(f"Output dim: {fc_output.shape[0]} (expected 4096)")
    
    # Check if the large values are expected
    print(f"\n=== C++ Reported FC Output ===")
    print(f"C++ reported: [343.7500, 710.0000, -734.5000, -103.6250]")
    print(f"Match: {torch.allclose(fc_output[:4].float(), torch.tensor([343.75, 710.0, -734.5, -103.625]), rtol=0.1)}")


if __name__ == '__main__':
    main()
