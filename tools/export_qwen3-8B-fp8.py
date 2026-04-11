"""
This script exports Qwen3-4B FP8 (E4M3 block-quantized) model to .bin format for OrinMLLM.

Usage:
    cd /home/wangyh/OrinMLLM && \
    source /home/wangyh/miniconda3/etc/profile.d/conda.sh && \
    conda activate deeplearning && \
    python tools/export_qwen3-8B-fp8.py /home/wangyh/Qwen3-4B-FP8/Qwen3-8B-fp8.bin \
        --hf=/home/wangyh/Qwen3-4B-FP8/

FP8 Block-Quantized Format:
- weight: [out_features, in_features] FP8 E4M3 (1 byte per element)
- weight_scale_inv: [ceil(out/block_size), ceil(in/block_size)] BF16 -> stored as FP16
- block_size: 128 (from quantization_config.weight_block_size)
- Dequantize: fp16_val = fp8_val * weight_scale_inv[row//block_size, col//block_size]

Note: Qwen3 has q_norm, k_norm (no QKV biases).
"""
import os
import struct
import argparse
import gc
import math
from pathlib import Path

import numpy as np
import torch


def serialize_fp16(file, tensor):
    """writes one fp16 tensor to file"""
    d = tensor.detach().cpu().view(-1).to(torch.float16).numpy()
    file.write(d.tobytes())


def serialize_fp8_raw(file, tensor):
    """writes one fp8 tensor as raw bytes (1 byte per element)"""
    # FP8 E4M3 tensors: just write raw bytes
    d = tensor.detach().cpu().view(-1)
    # Convert to uint8 view of the fp8 bits
    raw = d.view(torch.uint8).numpy()
    file.write(raw.tobytes())


def fp8_export(hf_dict, config, filepath):
    """
    Export the FP8 block-quantized model weights.
    
    Header format (256 bytes):
    - magic: uint32 "fp88" (0x66703838)
    - version: int32 = 7 (FP8 block-quantized version for Qwen3)
    - dim: int32
    - hidden_dim: int32 (intermediate_size)
    - n_layers: int32
    - n_heads: int32
    - n_kv_heads: int32
    - vocab_size: int32
    - max_seq_len: int32
    - shared_classifier: uint8
    - head_dim: int32 (Qwen3 specific)
    - block_size: int32 (block quantization size, default 128)
    - padding to 256 bytes
    
    Weights order:
    == FP16 weights (non-quantized) ==
    1. attention_norm (input_layernorm) for all layers - FP16
    2. ffn_norm (post_attention_layernorm) for all layers - FP16
    3. final norm weight - FP16
    4. token embeddings - FP16
    
    == FP8 block-quantized weights ==
    For each linear layer group (wq, wk, wv, wo, w1, w2, w3):
      For each layer i in [0, n_layers):
        - weight: [out_features, in_features] FP8 (1 byte each)
        - weight_scale_inv: [scale_rows, scale_cols] FP16 (2 bytes each)
          where scale_rows = ceil(out_features/block_size), scale_cols = ceil(in_features/block_size)
    
    == FP16 weights (non-quantized) ==
    12. output (lm_head) weights - FP16 (if not shared)
    13. q_norm weights for all layers - FP16
    14. k_norm weights for all layers - FP16
    """
    version = 7
    block_size = config.get('block_size', 128)
    
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
        hf_dict['model.embed_tokens.weight'].to(torch.float32),
        hf_dict.get('lm_head.weight', hf_dict['model.embed_tokens.weight']).to(torch.float32)
    ) if 'lm_head.weight' in hf_dict else True
    
    # Write header (256 bytes)
    out_file.write(struct.pack('I', 0x66703838))  # magic "fp88"
    out_file.write(struct.pack('i', version))       # version 7
    header = struct.pack('iiiiiii', dim, hidden_dim, n_layers, n_heads,
                         n_kv_heads, vocab_size, max_seq_len)
    out_file.write(header)
    out_file.write(struct.pack('B', int(shared_classifier)))  # shared_classifier
    out_file.write(struct.pack('i', head_dim))                # head_dim
    out_file.write(struct.pack('i', block_size))              # block_size
    # Pad to 256 bytes
    pad = 256 - out_file.tell()
    assert pad >= 0
    out_file.write(b'\0' * pad)
    
    print(f"Header written: magic=0x66703838, version={version}")
    print(f"  dim={dim}, hidden_dim={hidden_dim}, n_layers={n_layers}")
    print(f"  n_heads={n_heads}, n_kv_heads={n_kv_heads}, vocab_size={vocab_size}")
    print(f"  max_seq_len={max_seq_len}, head_dim={head_dim}, block_size={block_size}")
    print(f"  shared_classifier={shared_classifier}")
    
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
    
    # Helper function to write FP8 block-quantized linear layer
    def write_fp8_weights(layer_name, prefix=""):
        nonlocal total_bytes
        weight = hf_dict[f'{layer_name}.weight']           # FP8 E4M3
        scale_inv = hf_dict[f'{layer_name}.weight_scale_inv']  # BF16

        out_features, in_features = weight.shape
        expected_scale_rows = math.ceil(out_features / block_size)
        expected_scale_cols = math.ceil(in_features / block_size)
        
        assert scale_inv.shape == (expected_scale_rows, expected_scale_cols), \
            f"{layer_name}: scale shape {scale_inv.shape} != expected ({expected_scale_rows}, {expected_scale_cols})"
        
        # Write FP8 weight (1 byte per element)
        serialize_fp8_raw(out_file, weight)
        total_bytes += weight.numel() * 1
        
        # Write scale_inv as FP16 (convert from BF16)
        serialize_fp16(out_file, scale_inv)
        total_bytes += scale_inv.numel() * 2
        
        if prefix:
            print(f"  {prefix}: weight={tuple(weight.shape)} FP8, scale_inv={tuple(scale_inv.shape)} FP16")
    
    # 5-11. FP8 quantized weights for all layers
    print("\nWriting FP8 block-quantized weights...")
    
    # wq (q_proj) for all layers
    print("Writing wq weights (FP8)...")
    for i in range(n_layers):
        write_fp8_weights(f'model.layers.{i}.self_attn.q_proj', f"layer{i}.wq")
    
    # wk (k_proj) for all layers
    print("Writing wk weights (FP8)...")
    for i in range(n_layers):
        write_fp8_weights(f'model.layers.{i}.self_attn.k_proj', f"layer{i}.wk")
    
    # wv (v_proj) for all layers
    print("Writing wv weights (FP8)...")
    for i in range(n_layers):
        write_fp8_weights(f'model.layers.{i}.self_attn.v_proj', f"layer{i}.wv")
    
    # wo (o_proj) for all layers
    print("Writing wo weights (FP8)...")
    for i in range(n_layers):
        write_fp8_weights(f'model.layers.{i}.self_attn.o_proj', f"layer{i}.wo")
    
    # w1 (gate_proj) for all layers
    print("Writing w1 (gate_proj) weights (FP8)...")
    for i in range(n_layers):
        write_fp8_weights(f'model.layers.{i}.mlp.gate_proj', f"layer{i}.w1")
    
    # w2 (down_proj) for all layers
    print("Writing w2 (down_proj) weights (FP8)...")
    for i in range(n_layers):
        write_fp8_weights(f'model.layers.{i}.mlp.down_proj', f"layer{i}.w2")
    
    # w3 (up_proj) for all layers
    print("Writing w3 (up_proj) weights (FP8)...")
    for i in range(n_layers):
        write_fp8_weights(f'model.layers.{i}.mlp.up_proj', f"layer{i}.w3")
    
    # 12. output weights (lm_head) - FP16
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
    
    if file_size != total_bytes:
        print(f"\n⚠️  WARNING: File size mismatch!")
        print(f"  Expected: {total_bytes:,} bytes")
        print(f"  Actual:   {file_size:,} bytes")
        print(f"  Difference: {file_size - total_bytes:,} bytes")
    else:
        print(f"\n✅ File size verified: matches expected size")


def load_hf_weights(model_path):
    """Load Qwen3-4B FP8 model weights from HuggingFace format."""
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
    
    # Check FP8 quantization config
    block_size = 128  # default
    if hasattr(hf_config, 'quantization_config'):
        quant_config = hf_config.quantization_config
        print(f"\nQuantization config:")
        print(f"  quant_method: {quant_config.get('quant_method', 'unknown')}")
        print(f"  fmt: {quant_config.get('fmt', 'unknown')}")
        print(f"  activation_scheme: {quant_config.get('activation_scheme', 'unknown')}")
        wbs = quant_config.get('weight_block_size', [128, 128])
        print(f"  weight_block_size: {wbs}")
        if isinstance(wbs, (list, tuple)) and len(wbs) >= 2:
            assert wbs[0] == wbs[1], f"Non-square block size not supported: {wbs}"
            block_size = wbs[0]
    
    config = {
        'hidden_size': hf_config.hidden_size,
        'num_hidden_layers': hf_config.num_hidden_layers,
        'num_attention_heads': hf_config.num_attention_heads,
        'num_key_value_heads': hf_config.num_key_value_heads,
        'intermediate_size': hf_config.intermediate_size,
        'vocab_size': hf_config.vocab_size,
        'max_position_embeddings': hf_config.max_position_embeddings,
        'rms_norm_eps': hf_config.rms_norm_eps,
        'block_size': block_size,
    }
    
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
            return None, None
    
    print(f"\nLoaded {len(hf_dict)} tensors")
    print("Sample keys:")
    sample_keys = list(hf_dict.keys())[:20]
    for key in sample_keys:
        print(f"  {key}: {hf_dict[key].shape}, dtype={hf_dict[key].dtype}")
    
    # Verify FP8 keys
    print("\nVerifying FP8 quantization keys...")
    if 'model.layers.0.self_attn.q_proj.weight' in hf_dict:
        w = hf_dict['model.layers.0.self_attn.q_proj.weight']
        print(f"  ✅ q_proj.weight found: dtype={w.dtype}")
        if w.dtype != torch.float8_e4m3fn:
            print(f"  ⚠️  Expected float8_e4m3fn, got {w.dtype}")
    else:
        print("  ❌ q_proj.weight NOT found!")
        return None, None
    
    if 'model.layers.0.self_attn.q_proj.weight_scale_inv' in hf_dict:
        s = hf_dict['model.layers.0.self_attn.q_proj.weight_scale_inv']
        print(f"  ✅ q_proj.weight_scale_inv found: shape={tuple(s.shape)}, dtype={s.dtype}")
    else:
        print("  ❌ q_proj.weight_scale_inv NOT found!")
        return None, None
    
    # Verify Qwen3 specific keys
    print("\nVerifying Qwen3 specific keys...")
    for key_check in ['q_norm.weight', 'k_norm.weight']:
        full_key = f'model.layers.0.self_attn.{key_check}'
        if full_key in hf_dict:
            print(f"  ✅ {key_check} found")
        else:
            print(f"  ❌ {key_check} NOT found!")
    
    if 'model.layers.0.self_attn.q_proj.bias' in hf_dict:
        print("  ⚠️  q_proj.bias found - unexpected for Qwen3")
    else:
        print("  ✅ No q_proj.bias (expected for Qwen3)")
    
    return hf_dict, config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Qwen3-4B FP8 to bin format")
    parser.add_argument("filepath", type=str, help="output filepath")
    parser.add_argument("--hf", type=str, required=True, help="huggingface model path")
    args = parser.parse_args()

    print(f"Loading model weights from {args.hf}...")
    hf_dict, config = load_hf_weights(args.hf)

    if hf_dict is None:
        parser.error("Can't load input model!")

    print(f"\nExporting to {args.filepath} in FP8 block-quantized format...")
    fp8_export(hf_dict, config, args.filepath)
    
    print("\nDone!")
