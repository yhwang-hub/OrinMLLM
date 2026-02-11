"""
This script exports Qwen3-8B model to FP16 .bin format for KuiperLLama.

Usage:
    cd /home/lvf6/disk/wyh/KuiperLLama && \
    conda activate KuiperLLama && \
    python tools/export_qwen3-8B-fp16.py Qwen3-8B-fp16.bin \
        --dtype=fp16 --hf=/home/lvf6/disk/wyh/QwenModels/Qwen3-8B/

This creates a FP16 model file for optimized inference on CUDA devices.

Note: Qwen3 differs from Qwen2.5 in the following ways:
    - No QKV biases (q_proj, k_proj, v_proj have bias=False)
    - Has q_norm and k_norm (RMSNorm applied to Q and K projections)
"""
import os
import struct
import argparse
import gc
from pathlib import Path

import numpy as np
import torch
from torch import nn


def serialize_fp32(file, tensor):
    """writes one fp32 tensor to file"""
    d = tensor.detach().cpu().view(-1).to(torch.float32).numpy()
    b = struct.pack(f'{len(d)}f', *d)
    file.write(b)


def serialize_fp16(file, tensor):
    """writes one fp16 tensor to file"""
    d = tensor.detach().cpu().view(-1).to(torch.float16).numpy()
    # Use numpy's tobytes for fp16
    file.write(d.tobytes())


def fp16_export(hf_dict, config, filepath):
    """
    Export the model weights in FP16 format.
    Header format (256 bytes):
    - magic: uint32 "ak47" (0x616b3437) - different magic for Qwen3
    - version: int32 = 4 (FP16 version for Qwen3)
    - dim: int32
    - hidden_dim: int32
    - n_layers: int32
    - n_heads: int32
    - n_kv_heads: int32
    - vocab_size: int32
    - max_seq_len: int32
    - shared_classifier: uint8
    - head_dim: int32 (Qwen3 specific)
    - padding to 256 bytes
    
    Weights order (all in FP16):
    1. attention_norm (input_layernorm) weights for all layers
    2. ffn_norm (post_attention_layernorm) weights for all layers
    3. final norm weight
    4. token embeddings
    5. wq weights for all layers
    6. wk weights for all layers
    7. wv weights for all layers
    8. wo weights for all layers
    9. w1 (gate_proj) weights for all layers
    10. w2 (down_proj) weights for all layers
    11. w3 (up_proj) weights for all layers
    12. output (lm_head) weights if not shared
    
    Qwen3 specific weights (in FP16, after main weights):
    13. q_norm weights for all layers
    14. k_norm weights for all layers
    """
    version = 4  # FP16 version for Qwen3
    
    out_file = open(filepath, 'wb')
    
    # Extract config values
    dim = config['hidden_size']
    hidden_dim = config['intermediate_size']
    n_layers = config['num_hidden_layers']
    n_heads = config['num_attention_heads']
    n_kv_heads = config['num_key_value_heads']
    vocab_size = config['vocab_size']
    max_seq_len = config['max_position_embeddings']
    head_dim = config.get('head_dim', dim // n_heads)
    
    # Check if classifier is shared
    shared_classifier = torch.equal(
        hf_dict['model.embed_tokens.weight'],
        hf_dict.get('lm_head.weight', hf_dict['model.embed_tokens.weight'])
    ) if 'lm_head.weight' in hf_dict else True
    
    # Write header (256 bytes)
    # 1) magic - use "ak47" (0x616b3437) for Qwen3
    out_file.write(struct.pack('I', 0x616b3437))
    # 2) version
    out_file.write(struct.pack('i', version))
    # 3) model params
    header = struct.pack('iiiiiii', dim, hidden_dim, n_layers, n_heads,
                         n_kv_heads, vocab_size, max_seq_len)
    out_file.write(header)
    # 4) shared classifier flag
    out_file.write(struct.pack('B', int(shared_classifier)))
    # 5) head_dim (Qwen3 specific)
    out_file.write(struct.pack('i', head_dim))
    # Pad to 256 bytes
    pad = 256 - out_file.tell()
    assert pad >= 0
    out_file.write(b'\0' * pad)
    
    print(f"Header written: version={version}, dim={dim}, hidden_dim={hidden_dim}")
    print(f"  n_layers={n_layers}, n_heads={n_heads}, n_kv_heads={n_kv_heads}")
    print(f"  vocab_size={vocab_size}, max_seq_len={max_seq_len}, head_dim={head_dim}")
    print(f"  shared_classifier={shared_classifier}")
    
    # Collect all weights
    weights = []
    weight_names = []
    
    # 1. attention_norm (input_layernorm) weights
    for i in range(n_layers):
        weights.append(hf_dict[f'model.layers.{i}.input_layernorm.weight'])
        weight_names.append(f"layer{i}.attention_norm")
    
    # 2. ffn_norm (post_attention_layernorm) weights
    for i in range(n_layers):
        weights.append(hf_dict[f'model.layers.{i}.post_attention_layernorm.weight'])
        weight_names.append(f"layer{i}.ffn_norm")
    
    # 3. final norm
    weights.append(hf_dict['model.norm.weight'])
    weight_names.append("final_norm")
    
    # 4. token embeddings
    weights.append(hf_dict['model.embed_tokens.weight'])
    weight_names.append("tok_embeddings")
    
    # 5-8. attention weights (no bias in Qwen3)
    for i in range(n_layers):
        weights.append(hf_dict[f'model.layers.{i}.self_attn.q_proj.weight'])
        weight_names.append(f"layer{i}.wq")
    for i in range(n_layers):
        weights.append(hf_dict[f'model.layers.{i}.self_attn.k_proj.weight'])
        weight_names.append(f"layer{i}.wk")
    for i in range(n_layers):
        weights.append(hf_dict[f'model.layers.{i}.self_attn.v_proj.weight'])
        weight_names.append(f"layer{i}.wv")
    for i in range(n_layers):
        weights.append(hf_dict[f'model.layers.{i}.self_attn.o_proj.weight'])
        weight_names.append(f"layer{i}.wo")
    
    # 9-11. FFN weights
    for i in range(n_layers):
        weights.append(hf_dict[f'model.layers.{i}.mlp.gate_proj.weight'])
        weight_names.append(f"layer{i}.w1")
    for i in range(n_layers):
        weights.append(hf_dict[f'model.layers.{i}.mlp.down_proj.weight'])
        weight_names.append(f"layer{i}.w2")
    for i in range(n_layers):
        weights.append(hf_dict[f'model.layers.{i}.mlp.up_proj.weight'])
        weight_names.append(f"layer{i}.w3")
    
    # 12. output weights if not shared
    if not shared_classifier:
        weights.append(hf_dict['lm_head.weight'])
        weight_names.append("output")
    
    # Write all weights in FP16
    print(f"\nWriting {len(weights)} weight tensors in FP16...")
    total_params = 0
    for i, (w, name) in enumerate(zip(weights, weight_names)):
        serialize_fp16(out_file, w)
        total_params += w.numel()
        if i % 20 == 0 or i == len(weights) - 1:
            print(f"  {i+1}/{len(weights)}: {name} shape={tuple(w.shape)}")
    
    # Qwen3 specific: Write q_norm and k_norm weights
    print(f"\nWriting Qwen3 specific q_norm and k_norm weights in FP16...")
    qk_norm_weights = []
    qk_norm_names = []
    
    # 13. q_norm weights for all layers
    for i in range(n_layers):
        qk_norm_weights.append(hf_dict[f'model.layers.{i}.self_attn.q_norm.weight'])
        qk_norm_names.append(f"layer{i}.q_norm")
    
    # 14. k_norm weights for all layers
    for i in range(n_layers):
        qk_norm_weights.append(hf_dict[f'model.layers.{i}.self_attn.k_norm.weight'])
        qk_norm_names.append(f"layer{i}.k_norm")
    
    for i, (w, name) in enumerate(zip(qk_norm_weights, qk_norm_names)):
        serialize_fp16(out_file, w)
        total_params += w.numel()
    
    print(f"  Written {len(qk_norm_weights)} qk_norm tensors")
    
    out_file.close()
    
    file_size = os.path.getsize(filepath)
    print(f"\nExport complete!")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Expected size (FP16): {total_params * 2:,} bytes + 256 header")
    print(f"  Actual file size: {file_size:,} bytes ({file_size / 1024 / 1024 / 1024:.2f} GB)")
    print(f"  Wrote {filepath}")
    
    # Verify file size
    expected_size = total_params * 2 + 256
    if file_size != expected_size:
        print(f"\n⚠️  WARNING: File size mismatch!")
        print(f"  Expected: {expected_size:,} bytes")
        print(f"  Actual:   {file_size:,} bytes")
        print(f"  Difference: {file_size - expected_size:,} bytes")
    else:
        print(f"\n✅ File size verified: matches expected size")


def load_hf_weights(model_path):
    """Load Qwen3-8B model weights from HuggingFace format."""
    try:
        from transformers import AutoConfig
    except ImportError:
        print("Error: transformers package required")
        return None, None

    model_path = Path(model_path)
    hf_config = AutoConfig.from_pretrained(model_path)
    
    print(f"Model config:")
    print(f"  hidden_size: {hf_config.hidden_size}")
    print(f"  num_hidden_layers: {hf_config.num_hidden_layers}")
    print(f"  num_attention_heads: {hf_config.num_attention_heads}")
    print(f"  num_key_value_heads: {hf_config.num_key_value_heads}")
    print(f"  intermediate_size: {hf_config.intermediate_size}")
    print(f"  vocab_size: {hf_config.vocab_size}")
    print(f"  max_position_embeddings: {hf_config.max_position_embeddings}")
    if hasattr(hf_config, 'head_dim'):
        print(f"  head_dim: {hf_config.head_dim}")
    
    config = {
        'hidden_size': hf_config.hidden_size,
        'num_hidden_layers': hf_config.num_hidden_layers,
        'num_attention_heads': hf_config.num_attention_heads,
        'num_key_value_heads': hf_config.num_key_value_heads,
        'intermediate_size': hf_config.intermediate_size,
        'vocab_size': hf_config.vocab_size,
        'max_position_embeddings': hf_config.max_position_embeddings,
        'rms_norm_eps': hf_config.rms_norm_eps,
    }
    
    # Qwen3 has head_dim as a separate config
    if hasattr(hf_config, 'head_dim'):
        config['head_dim'] = hf_config.head_dim
    else:
        config['head_dim'] = hf_config.hidden_size // hf_config.num_attention_heads
    
    # Load weights
    safetensor_files = sorted(list(model_path.glob("*.safetensors")))
    hf_dict = {}
    
    if safetensor_files:
        from safetensors import safe_open
        for sf_file in safetensor_files:
            print(f"Loading from {sf_file}")
            with safe_open(sf_file, framework="pt", device="cpu") as f:
                for key in f.keys():
                    hf_dict[key] = f.get_tensor(key)
    else:
        # Try loading from pytorch files
        pytorch_files = sorted(list(model_path.glob("*.bin")))
        if pytorch_files:
            for pt_file in pytorch_files:
                print(f"Loading from {pt_file}")
                state_dict = torch.load(pt_file, map_location="cpu")
                hf_dict.update(state_dict)
                del state_dict
                gc.collect()
        else:
            from transformers import AutoModelForCausalLM
            print("Loading model using AutoModelForCausalLM...")
            hf_model = AutoModelForCausalLM.from_pretrained(
                str(model_path), torch_dtype=torch.float32, low_cpu_mem_usage=True)
            hf_dict = hf_model.state_dict()
            del hf_model
            gc.collect()
    
    # Print some loaded keys for verification
    print(f"\nLoaded {len(hf_dict)} tensors")
    print("Sample keys:")
    sample_keys = list(hf_dict.keys())[:10]
    for key in sample_keys:
        print(f"  {key}: {hf_dict[key].shape}")
    
    # Verify Qwen3 specific keys exist
    print("\nVerifying Qwen3 specific keys (q_norm, k_norm)...")
    if 'model.layers.0.self_attn.q_norm.weight' in hf_dict:
        print("  ✅ q_norm found")
    else:
        print("  ❌ q_norm NOT found - this may not be a Qwen3 model!")
    
    if 'model.layers.0.self_attn.k_norm.weight' in hf_dict:
        print("  ✅ k_norm found")
    else:
        print("  ❌ k_norm NOT found - this may not be a Qwen3 model!")
    
    # Verify no QKV bias (Qwen3 specific)
    if 'model.layers.0.self_attn.q_proj.bias' in hf_dict:
        print("  ⚠️  q_proj.bias found - Qwen3 should not have QKV biases")
    else:
        print("  ✅ No q_proj.bias (expected for Qwen3)")
    
    return hf_dict, config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Qwen3-8B to FP16 bin format")
    parser.add_argument("filepath", type=str, help="output filepath")
    parser.add_argument("--dtype", type=str, default="fp16", help="dtype (fp16)")
    parser.add_argument("--hf", type=str, required=True, help="huggingface model path")
    args = parser.parse_args()

    print(f"Loading model weights from {args.hf}...")
    hf_dict, config = load_hf_weights(args.hf)

    if hf_dict is None:
        parser.error("Can't load input model!")

    total_params = sum(t.numel() for t in hf_dict.values())
    print(f"\nModel loaded: {total_params:,} parameters ({total_params / 1e9:.2f}B)")
    
    print(f"\nExporting to {args.filepath} in FP16 format...")
    fp16_export(hf_dict, config, args.filepath)
    
    print("\nDone!")
