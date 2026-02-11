#!/usr/bin/env python3
"""
Convert EAGLE-3 PyTorch model to KuiperLLama binary format.

Binary format:
- Header (44 bytes):
  - Magic: b'EGL3' (4 bytes)
  - Version: uint32 (4 bytes)
  - num_layers: uint32 (4 bytes)
  - hidden_size: uint32 (4 bytes)
  - num_heads: uint32 (4 bytes)
  - num_kv_heads: uint32 (4 bytes)
  - intermediate_size: uint32 (4 bytes)
  - target_vocab_size: uint32 (4 bytes)
  - draft_vocab_size: uint32 (4 bytes)
  - total_elements: uint64 (8 bytes)
- D2T mapping: int32[draft_vocab_size]
- Weights (all FP16):
  - fc.weight: [hidden_size, hidden_size * 3]
  - norm.weight: [hidden_size]
  - midlayer.input_layernorm.weight: [hidden_size]
  - midlayer.self_attn.q_proj.weight: [hidden_size, hidden_size * 2]
  - midlayer.self_attn.k_proj.weight: [kv_dim, hidden_size * 2]
  - midlayer.self_attn.v_proj.weight: [kv_dim, hidden_size * 2]
  - midlayer.self_attn.o_proj.weight: [hidden_size, hidden_size]
  - midlayer.post_attention_layernorm.weight: [hidden_size]
  - midlayer.mlp.gate_proj.weight: [intermediate_size, hidden_size]
  - midlayer.mlp.up_proj.weight: [intermediate_size, hidden_size]
  - midlayer.mlp.down_proj.weight: [hidden_size, intermediate_size]
  - lm_head.weight: [draft_vocab_size, hidden_size]
  - final_layernorm.weight: [hidden_size]
"""

import argparse
import struct
import torch
import numpy as np
import json
import os


def convert_eagle3(input_path, output_path, config_path=None):
    """Convert EAGLE-3 model to binary format."""
    
    print(f"Loading model from {input_path}")
    state_dict = torch.load(input_path, map_location='cpu')
    
    # Load or infer config
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        hidden_size = config.get('hidden_size', 4096)
        num_heads = config.get('num_attention_heads', 32)
        num_kv_heads = config.get('num_key_value_heads', 8)
        intermediate_size = config.get('intermediate_size', 12288)
        target_vocab_size = config.get('vocab_size', 151936)
        draft_vocab_size = config.get('draft_vocab_size', 32000)
    else:
        # Infer from weights
        hidden_size = 4096
        num_heads = 32
        num_kv_heads = 8
        intermediate_size = 12288
        target_vocab_size = 151936
        draft_vocab_size = 32000
    
    # Compute derived values
    head_dim = hidden_size // num_heads
    kv_dim = num_kv_heads * head_dim
    
    print(f"Config:")
    print(f"  hidden_size: {hidden_size}")
    print(f"  num_heads: {num_heads}")
    print(f"  num_kv_heads: {num_kv_heads}")
    print(f"  intermediate_size: {intermediate_size}")
    print(f"  target_vocab_size: {target_vocab_size}")
    print(f"  draft_vocab_size: {draft_vocab_size}")
    print(f"  head_dim: {head_dim}")
    print(f"  kv_dim: {kv_dim}")
    
    # Print available keys
    print("\nAvailable keys in state_dict:")
    for k in sorted(state_dict.keys()):
        v = state_dict[k]
        print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
    
    # Get d2t
    d2t = state_dict['d2t']
    print(f"\nd2t: shape={d2t.shape}, dtype={d2t.dtype}")
    print(f"  Range: {d2t.min()} to {d2t.max()}")
    print(f"  First 10: {d2t[:10].tolist()}")
    
    # Weight mapping
    # Note: In the original EAGLE-3 model:
    #   - norm.weight is for normalizing hidden states (hidden_norm)
    #   - midlayer.hidden_norm.weight is the same as norm.weight (some models have this)
    #   - There's no final_layernorm - output uses lm_head directly after decoder layer output
    weight_map = {
        'fc': 'fc.weight',
        'hidden_norm': 'midlayer.hidden_norm.weight',  # For normalizing hidden states after FC
        'input_layernorm': 'midlayer.input_layernorm.weight',  # For normalizing input embeddings
        'q_proj': 'midlayer.self_attn.q_proj.weight',
        'k_proj': 'midlayer.self_attn.k_proj.weight',
        'v_proj': 'midlayer.self_attn.v_proj.weight',
        'o_proj': 'midlayer.self_attn.o_proj.weight',
        'post_attention_layernorm': 'midlayer.post_attention_layernorm.weight',
        'gate_proj': 'midlayer.mlp.gate_proj.weight',
        'up_proj': 'midlayer.mlp.up_proj.weight',
        'down_proj': 'midlayer.mlp.down_proj.weight',
        'lm_head': 'lm_head.weight',
        'final_norm': 'norm.weight',  # Final norm before lm_head
    }
    
    # Verify all weights exist
    print("\nVerifying weights:")
    for name, key in weight_map.items():
        if key in state_dict:
            w = state_dict[key]
            print(f"  {name}: {key} -> {w.shape}")
        else:
            print(f"  {name}: {key} -> NOT FOUND!")
    
    # Calculate total elements
    total_elements = 0
    for name, key in weight_map.items():
        if key in state_dict:
            total_elements += state_dict[key].numel()
    print(f"\nTotal FP16 elements: {total_elements}")
    
    # Write binary file
    print(f"\nWriting to {output_path}")
    
    with open(output_path, 'wb') as f:
        # Write header
        f.write(b'EGL3')  # Magic
        f.write(struct.pack('I', 1))  # Version
        f.write(struct.pack('I', 1))  # num_layers
        f.write(struct.pack('I', hidden_size))
        f.write(struct.pack('I', num_heads))
        f.write(struct.pack('I', num_kv_heads))
        f.write(struct.pack('I', intermediate_size))
        f.write(struct.pack('I', target_vocab_size))
        f.write(struct.pack('I', draft_vocab_size))
        f.write(struct.pack('Q', total_elements))
        
        # Write d2t as int32
        d2t_int32 = d2t.to(torch.int32).numpy()
        print(f"Writing d2t ({len(d2t_int32)} int32 values)")
        f.write(d2t_int32.tobytes())
        
        # Write weights in order - must match C++ loading order in eagle3.cpp
        weight_order = [
            'fc', 'hidden_norm', 'input_layernorm',
            'q_proj', 'k_proj', 'v_proj', 'o_proj',
            'post_attention_layernorm',
            'gate_proj', 'up_proj', 'down_proj',
            'final_norm', 'lm_head'  # final_norm before lm_head to match C++ order
        ]
        
        for name in weight_order:
            key = weight_map[name]
            if key in state_dict:
                w = state_dict[key]
                w_fp16 = w.to(torch.float16).numpy()
                print(f"  Writing {name}: {w.shape} -> {w_fp16.nbytes} bytes")
                f.write(w_fp16.tobytes())
            else:
                print(f"  WARNING: {name} not found!")
    
    # Verify file
    file_size = os.path.getsize(output_path)
    expected_size = 44 + draft_vocab_size * 4 + total_elements * 2
    print(f"\nFile size: {file_size} bytes")
    print(f"Expected size: {expected_size} bytes")
    
    if file_size == expected_size:
        print("✓ File size matches!")
    else:
        print(f"✗ File size mismatch! Difference: {file_size - expected_size} bytes")
    
    print("Done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert EAGLE-3 model')
    parser.add_argument('input', help='Input pytorch_model.bin path')
    parser.add_argument('output', help='Output eagle3.bin path')
    parser.add_argument('--config', help='Optional config.json path')
    args = parser.parse_args()
    
    convert_eagle3(args.input, args.output, args.config)
