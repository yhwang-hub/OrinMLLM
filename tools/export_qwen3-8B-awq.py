"""
This script exports Qwen3-8B AWQ quantized model to .bin format for KuiperLLama.

Usage:
    cd /mnt/ssd/workspace/KuiperLLama_20260120_fp16_awq && \
    python tools/export_qwen3-8B-awq.py /mnt/ssd/QwenModels/Qwen3-8B-awq.bin \
        --hf=/mnt/ssd/QwenModels/Qwen3-8B-awq/

This creates an AWQ INT4 model file for optimized inference on CUDA devices.

AWQ Format:
- qweight: [in_features, out_features/8] INT32 (8 INT4 values packed per INT32)
- qzeros: [num_groups, out_features/8] INT32
- scales: [num_groups, out_features] FP16

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


def serialize_fp16(file, tensor):
    """writes one fp16 tensor to file"""
    d = tensor.detach().cpu().view(-1).to(torch.float16).numpy()
    file.write(d.tobytes())


def serialize_int32(file, tensor):
    """writes one int32 tensor to file"""
    d = tensor.detach().cpu().view(-1).to(torch.int32).numpy()
    file.write(d.tobytes())


def awq_export(hf_dict, config, filepath, group_size=128):
    """
    Export the AWQ quantized model weights.
    
    Header format (256 bytes):
    - magic: uint32 "ak48" (0x616b3438) - different magic for Qwen3 AWQ
    - version: int32 = 5 (AWQ INT4 version for Qwen3)
    - dim: int32
    - hidden_dim: int32 (intermediate_size)
    - n_layers: int32
    - n_heads: int32
    - n_kv_heads: int32
    - vocab_size: int32
    - max_seq_len: int32
    - shared_classifier: uint8
    - head_dim: int32 (Qwen3 specific)
    - group_size: int32 (AWQ group size)
    - padding to 256 bytes
    
    Weights order:
    == FP16 weights (non-quantized) ==
    1. attention_norm (input_layernorm) for all layers - FP16
    2. ffn_norm (post_attention_layernorm) for all layers - FP16
    3. final norm weight - FP16
    4. token embeddings - FP16
    
    == AWQ quantized weights (for each layer) ==
    For each linear layer (wq, wk, wv, wo, w1, w2, w3):
      - qweight: [in_features, out_features/8] INT32
      - qzeros: [num_groups, out_features/8] INT32
      - scales: [num_groups, out_features] FP16
    
    == FP16 weights (non-quantized) ==
    12. output (lm_head) weights - FP16 (if not shared)
    13. q_norm weights for all layers - FP16
    14. k_norm weights for all layers - FP16
    """
    version = 5  # AWQ INT4 version for Qwen3
    
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
    # 1) magic - use "ak48" (0x616b3438) for Qwen3 AWQ
    out_file.write(struct.pack('I', 0x616b3438))
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
    # 6) group_size (AWQ specific)
    out_file.write(struct.pack('i', group_size))
    # Pad to 256 bytes
    pad = 256 - out_file.tell()
    assert pad >= 0
    out_file.write(b'\0' * pad)
    
    print(f"Header written: version={version}, dim={dim}, hidden_dim={hidden_dim}")
    print(f"  n_layers={n_layers}, n_heads={n_heads}, n_kv_heads={n_kv_heads}")
    print(f"  vocab_size={vocab_size}, max_seq_len={max_seq_len}, head_dim={head_dim}")
    print(f"  shared_classifier={shared_classifier}, group_size={group_size}")
    
    total_bytes = 256  # header
    
    # 1. attention_norm (input_layernorm) weights - FP16
    print("\nWriting attention_norm weights (FP16)...")
    for i in range(n_layers):
        w = hf_dict[f'model.layers.{i}.input_layernorm.weight']
        serialize_fp16(out_file, w)
        total_bytes += w.numel() * 2
    
    # 2. ffn_norm (post_attention_layernorm) weights - FP16
    print("Writing ffn_norm weights (FP16)...")
    for i in range(n_layers):
        w = hf_dict[f'model.layers.{i}.post_attention_layernorm.weight']
        serialize_fp16(out_file, w)
        total_bytes += w.numel() * 2
    
    # 3. final norm - FP16
    print("Writing final norm weight (FP16)...")
    w = hf_dict['model.norm.weight']
    serialize_fp16(out_file, w)
    total_bytes += w.numel() * 2
    
    # 4. token embeddings - FP16
    print("Writing token embeddings (FP16)...")
    w = hf_dict['model.embed_tokens.weight']
    serialize_fp16(out_file, w)
    total_bytes += w.numel() * 2
    
    # Helper function to write AWQ weights
    def write_awq_weights(layer_name, prefix=""):
        nonlocal total_bytes
        qweight = hf_dict[f'{layer_name}.qweight']
        qzeros = hf_dict[f'{layer_name}.qzeros']
        scales = hf_dict[f'{layer_name}.scales']
        
        serialize_int32(out_file, qweight)
        total_bytes += qweight.numel() * 4
        
        serialize_int32(out_file, qzeros)
        total_bytes += qzeros.numel() * 4
        
        serialize_fp16(out_file, scales)
        total_bytes += scales.numel() * 2
        
        if prefix:
            print(f"  {prefix}: qweight={tuple(qweight.shape)}, qzeros={tuple(qzeros.shape)}, scales={tuple(scales.shape)}")
    
    # 5-11. AWQ quantized weights for all layers
    print("\nWriting AWQ quantized weights...")
    
    # wq (q_proj) for all layers
    print("Writing wq weights (AWQ)...")
    for i in range(n_layers):
        write_awq_weights(f'model.layers.{i}.self_attn.q_proj', f"layer{i}.wq")
    
    # wk (k_proj) for all layers
    print("Writing wk weights (AWQ)...")
    for i in range(n_layers):
        write_awq_weights(f'model.layers.{i}.self_attn.k_proj', f"layer{i}.wk")
    
    # wv (v_proj) for all layers
    print("Writing wv weights (AWQ)...")
    for i in range(n_layers):
        write_awq_weights(f'model.layers.{i}.self_attn.v_proj', f"layer{i}.wv")
    
    # wo (o_proj) for all layers
    print("Writing wo weights (AWQ)...")
    for i in range(n_layers):
        write_awq_weights(f'model.layers.{i}.self_attn.o_proj', f"layer{i}.wo")
    
    # w1 (gate_proj) for all layers
    print("Writing w1 (gate_proj) weights (AWQ)...")
    for i in range(n_layers):
        write_awq_weights(f'model.layers.{i}.mlp.gate_proj', f"layer{i}.w1")
    
    # w2 (down_proj) for all layers
    print("Writing w2 (down_proj) weights (AWQ)...")
    for i in range(n_layers):
        write_awq_weights(f'model.layers.{i}.mlp.down_proj', f"layer{i}.w2")
    
    # w3 (up_proj) for all layers
    print("Writing w3 (up_proj) weights (AWQ)...")
    for i in range(n_layers):
        write_awq_weights(f'model.layers.{i}.mlp.up_proj', f"layer{i}.w3")
    
    # 12. output weights (lm_head) - FP16 (not quantized in this AWQ model)
    if not shared_classifier:
        print("\nWriting lm_head weights (FP16)...")
        w = hf_dict['lm_head.weight']
        serialize_fp16(out_file, w)
        total_bytes += w.numel() * 2
    
    # 13. q_norm weights for all layers - FP16
    print("\nWriting q_norm weights (FP16)...")
    for i in range(n_layers):
        w = hf_dict[f'model.layers.{i}.self_attn.q_norm.weight']
        serialize_fp16(out_file, w)
        total_bytes += w.numel() * 2
    
    # 14. k_norm weights for all layers - FP16
    print("Writing k_norm weights (FP16)...")
    for i in range(n_layers):
        w = hf_dict[f'model.layers.{i}.self_attn.k_norm.weight']
        serialize_fp16(out_file, w)
        total_bytes += w.numel() * 2
    
    out_file.close()
    
    file_size = os.path.getsize(filepath)
    print(f"\nExport complete!")
    print(f"  Expected size: {total_bytes:,} bytes ({total_bytes / 1024 / 1024 / 1024:.2f} GB)")
    print(f"  Actual file size: {file_size:,} bytes ({file_size / 1024 / 1024 / 1024:.2f} GB)")
    print(f"  Wrote {filepath}")
    
    # Verify file size
    if file_size != total_bytes:
        print(f"\n⚠️  WARNING: File size mismatch!")
        print(f"  Expected: {total_bytes:,} bytes")
        print(f"  Actual:   {file_size:,} bytes")
        print(f"  Difference: {file_size - total_bytes:,} bytes")
    else:
        print(f"\n✅ File size verified: matches expected size")


def load_hf_weights(model_path):
    """Load Qwen3-8B AWQ model weights from HuggingFace format."""
    try:
        from transformers import AutoConfig
    except ImportError:
        print("Error: transformers package required")
        return None, None, None

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
    
    # Check quantization config
    group_size = 128
    if hasattr(hf_config, 'quantization_config'):
        quant_config = hf_config.quantization_config
        print(f"\nQuantization config:")
        print(f"  quant_method: {quant_config.get('quant_method', 'unknown')}")
        print(f"  bits: {quant_config.get('bits', 'unknown')}")
        print(f"  group_size: {quant_config.get('group_size', 128)}")
        print(f"  zero_point: {quant_config.get('zero_point', True)}")
        group_size = quant_config.get('group_size', 128)
    
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
            print("Error: No model files found!")
            return None, None, None
    
    # Print some loaded keys for verification
    print(f"\nLoaded {len(hf_dict)} tensors")
    print("Sample keys:")
    sample_keys = list(hf_dict.keys())[:20]
    for key in sample_keys:
        print(f"  {key}: {hf_dict[key].shape}, dtype={hf_dict[key].dtype}")
    
    # Verify AWQ keys exist
    print("\nVerifying AWQ quantization keys...")
    if 'model.layers.0.self_attn.q_proj.qweight' in hf_dict:
        print("  ✅ AWQ qweight found")
    else:
        print("  ❌ AWQ qweight NOT found - this may not be an AWQ model!")
        return None, None, None
    
    if 'model.layers.0.self_attn.q_proj.qzeros' in hf_dict:
        print("  ✅ AWQ qzeros found")
    else:
        print("  ❌ AWQ qzeros NOT found")
        return None, None, None
    
    if 'model.layers.0.self_attn.q_proj.scales' in hf_dict:
        print("  ✅ AWQ scales found")
    else:
        print("  ❌ AWQ scales NOT found")
        return None, None, None
    
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
    
    return hf_dict, config, group_size


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Qwen3-8B AWQ to bin format")
    parser.add_argument("filepath", type=str, help="output filepath")
    parser.add_argument("--hf", type=str, required=True, help="huggingface model path")
    args = parser.parse_args()

    print(f"Loading model weights from {args.hf}...")
    hf_dict, config, group_size = load_hf_weights(args.hf)

    if hf_dict is None:
        parser.error("Can't load input model!")

    print(f"\nModel loaded with group_size={group_size}")
    
    print(f"\nExporting to {args.filepath} in AWQ INT4 format...")
    awq_export(hf_dict, config, args.filepath, group_size)
    
    print("\nDone!")
