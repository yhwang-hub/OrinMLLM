#!/usr/bin/env python3
"""Compare C++ logits output with Python reference."""

import numpy as np
import struct

def load_binary_float32(path, count):
    """Load binary float32 data."""
    with open(path, 'rb') as f:
        data = f.read(count * 4)
        values = struct.unpack(f'{count}f', data)
        return np.array(values, dtype=np.float32)

def main():
    # Load C++ logits
    cpp_logits_path = "/tmp/cpp_logits.bin"
    draft_vocab_size = 32000
    
    cpp_logits = load_binary_float32(cpp_logits_path, draft_vocab_size)
    print(f"C++ logits shape: {cpp_logits.shape}")
    print(f"C++ logits first 4: {cpp_logits[:4]}")
    print(f"C++ logits stats: min={cpp_logits.min():.4f}, max={cpp_logits.max():.4f}")
    
    # Find argmax
    cpp_argmax = np.argmax(cpp_logits)
    print(f"\nC++ argmax (draft index): {cpp_argmax}")
    print(f"C++ max logit value: {cpp_logits[cpp_argmax]:.4f}")
    
    # Load D2T mapping
    import torch
    d2t_data = torch.load("/mnt/ssd/QwenModels/Qwen3-8B_eagle3/eagle3.bin", map_location='cpu', weights_only=True)
    d2t = d2t_data['d2t'].numpy()
    
    target_token = cpp_argmax + d2t[cpp_argmax]
    print(f"C++ target token: {target_token}")
    
    # Top-5
    top5_indices = np.argsort(cpp_logits)[-5:][::-1]
    print("\nC++ Top-5:")
    for i, idx in enumerate(top5_indices):
        target = idx + d2t[idx]
        print(f"  {i}: draft={idx}, logit={cpp_logits[idx]:.4f}, target={target}")

    # Load C++ final_norm for comparison
    cpp_final_norm_path = "/tmp/cpp_final_norm.bin"
    hidden_size = 4096
    cpp_final_norm = load_binary_float32(cpp_final_norm_path, hidden_size)
    print(f"\nC++ final_norm first 4: {cpp_final_norm[:4]}")
    print(f"C++ final_norm stats: min={cpp_final_norm.min():.4f}, max={cpp_final_norm.max():.4f}")
    
    # Now compute Python logits using the same final_norm
    print("\n=== Python Comparison ===")
    lm_head_weight = d2t_data['lm_head.weight'].float().numpy()  # [32000, 4096]
    print(f"LM head weight shape: {lm_head_weight.shape}")
    print(f"LM head weight first 4: {lm_head_weight[0, :4]}")
    
    # Python logits computation
    py_logits = np.dot(lm_head_weight, cpp_final_norm)
    print(f"\nPython logits first 4: {py_logits[:4]}")
    print(f"Python logits stats: min={py_logits.min():.4f}, max={py_logits.max():.4f}")
    
    py_argmax = np.argmax(py_logits)
    print(f"Python argmax: {py_argmax}")
    
    # Compare
    diff = np.abs(cpp_logits - py_logits)
    print(f"\nLogits diff: max={diff.max():.6f}, mean={diff.mean():.6f}")
    
    if np.allclose(cpp_logits, py_logits, rtol=1e-2, atol=1e-2):
        print("✅ Logits match!")
    else:
        print("❌ Logits mismatch!")
        
        # Find where differences are largest
        worst_idx = np.argmax(diff)
        print(f"  Worst mismatch at index {worst_idx}:")
        print(f"    C++: {cpp_logits[worst_idx]:.6f}")
        print(f"    Python: {py_logits[worst_idx]:.6f}")

if __name__ == "__main__":
    main()
