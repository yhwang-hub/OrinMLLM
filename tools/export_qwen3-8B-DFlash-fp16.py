"""
This script exports Qwen3-8B DFlash draft model to FP16 .bin format for KuiperLLama.

Usage:
    cd /mnt/ssd/workspace/OrinMLLM && \
    source .venv/bin/activate && \
    python tools/export_qwen3-8B-DFlash-fp16.py /mnt/ssd/QwenModels/Qwen3-8B-DFlash-fp16.bin \
        --hf=/mnt/ssd/QwenModels/Qwen3-8B-DFlash-b16

DFlash is a block-diffusion speculative decoding draft model that uses
cross-attention with target model hidden states. It has:
  - 5 transformer layers (vs 36 in target Qwen3-8B)
  - fc layer (20480 -> 4096) fusing 5 target layer hidden states
  - hidden_norm (RMSNorm)
  - No embedding or lm_head (shared with target model)
  - Non-causal (bidirectional) attention

Binary format header (256 bytes):
  magic: 0x64663136 ("df16") - DFlash FP16 magic
  version: int32 = 7
  dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, max_seq_len: int32 x 7
  shared_classifier: uint8
  head_dim: int32
  block_size: int32
  n_target_layers: int32
  target_layer_ids: int32 x 5
  padding to 256 bytes

Weight order (all in FP16):
  1. fc.weight                               [dim, n_target_layers * dim]
  2. hidden_norm.weight                       [dim]
  3. attention_norm (input_layernorm) x layers [dim]
  4. ffn_norm (post_attention_layernorm) x layers [dim]
  5. final norm.weight                        [dim]
  6. wq (q_proj) x layers                    [dim, dim]
  7. wk (k_proj) x layers                    [kv_dim, dim]
  8. wv (v_proj) x layers                    [kv_dim, dim]
  9. wo (o_proj) x layers                    [dim, dim]
  10. w1 (gate_proj) x layers                [intermediate, dim]
  11. w2 (down_proj) x layers                [dim, intermediate]
  12. w3 (up_proj) x layers                  [intermediate, dim]
  13. q_norm x layers                        [head_dim]
  14. k_norm x layers                        [head_dim]
"""
import os
import struct
import argparse
import gc
import json
from pathlib import Path

import numpy as np


def serialize_fp16(file, data_bytes):
    """writes raw bytes (already FP16 or converted) to file"""
    file.write(data_bytes)


def bf16_to_fp16_bytes(bf16_numpy_array):
    """Convert BF16 raw bytes to FP16 via FP32 intermediate.
    bf16_numpy_array is a numpy array of dtype uint16 representing BF16 values."""
    # BF16 -> FP32: shift left by 16 bits
    fp32_bits = bf16_numpy_array.astype(np.uint32) << 16
    fp32_values = fp32_bits.view(np.float32)
    # FP32 -> FP16
    fp16_values = fp32_values.astype(np.float16)
    return fp16_values.tobytes()


def load_safetensors_raw(filepath):
    """Load safetensors file and return dict of {name: (shape, dtype, raw_bytes)}"""
    with open(filepath, 'rb') as f:
        header_size = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(header_size))
        data_start = 8 + header_size
        
        tensors = {}
        for key, info in header.items():
            if key == '__metadata__':
                continue
            shape = info['shape']
            dtype = info['dtype']
            offsets = info['data_offsets']
            start, end = offsets
            
            f.seek(data_start + start)
            raw_data = f.read(end - start)
            tensors[key] = (shape, dtype, raw_data)
    
    return tensors


def get_tensor_fp16_bytes(tensors, key):
    """Get a tensor's data as FP16 bytes, converting from BF16 if needed."""
    shape, dtype, raw_data = tensors[key]
    num_elements = 1
    for s in shape:
        num_elements *= s
    
    if dtype == 'BF16':
        # Parse as uint16 array (raw BF16 bits), then convert
        bf16_array = np.frombuffer(raw_data, dtype=np.uint16)
        assert len(bf16_array) == num_elements, f"Shape mismatch for {key}: expected {num_elements}, got {len(bf16_array)}"
        return bf16_to_fp16_bytes(bf16_array), num_elements
    elif dtype == 'F16':
        return raw_data, num_elements
    elif dtype == 'F32':
        fp32_array = np.frombuffer(raw_data, dtype=np.float32)
        fp16_array = fp32_array.astype(np.float16)
        return fp16_array.tobytes(), num_elements
    else:
        raise ValueError(f"Unsupported dtype: {dtype} for tensor {key}")


def dflash_fp16_export(tensors, config, filepath):
    """Export DFlash model weights in FP16 format."""
    version = 7  # DFlash FP16 version
    
    out_file = open(filepath, 'wb')
    
    dim = config['hidden_size']
    hidden_dim = config['intermediate_size']
    n_layers = config['num_hidden_layers']
    n_heads = config['num_attention_heads']
    n_kv_heads = config['num_key_value_heads']
    vocab_size = config['vocab_size']
    max_seq_len = config['max_position_embeddings']
    head_dim = config.get('head_dim', dim // n_heads)
    block_size = config.get('block_size', 16)
    n_target_layers = config.get('num_target_layers', 36)
    
    dflash_config = config.get('dflash_config', {})
    target_layer_ids = dflash_config.get('target_layer_ids', [1, 9, 17, 25, 33])
    mask_token_id = dflash_config.get('mask_token_id', 151669)
    
    # Write header (256 bytes)
    # 1) magic "df16" = 0x64663136
    out_file.write(struct.pack('I', 0x64663136))
    # 2) version
    out_file.write(struct.pack('i', version))
    # 3) model params
    header = struct.pack('iiiiiii', dim, hidden_dim, n_layers, n_heads,
                         n_kv_heads, vocab_size, max_seq_len)
    out_file.write(header)
    # 4) shared classifier flag (DFlash has no lm_head, always shared with target)
    out_file.write(struct.pack('B', 1))
    # 5) head_dim
    out_file.write(struct.pack('i', head_dim))
    # 6) block_size
    out_file.write(struct.pack('i', block_size))
    # 7) n_target_layers
    out_file.write(struct.pack('i', n_target_layers))
    # 8) target_layer_ids (5 ints)
    for lid in target_layer_ids:
        out_file.write(struct.pack('i', lid))
    # 9) mask_token_id
    out_file.write(struct.pack('i', mask_token_id))
    # Pad to 256 bytes
    pad = 256 - out_file.tell()
    assert pad >= 0, f"Header too large: {out_file.tell()} bytes"
    out_file.write(b'\0' * pad)
    
    print(f"Header written: version={version}, dim={dim}, hidden_dim={hidden_dim}")
    print(f"  n_layers={n_layers}, n_heads={n_heads}, n_kv_heads={n_kv_heads}")
    print(f"  vocab_size={vocab_size}, max_seq_len={max_seq_len}, head_dim={head_dim}")
    print(f"  block_size={block_size}, n_target_layers={n_target_layers}")
    print(f"  target_layer_ids={target_layer_ids}, mask_token_id={mask_token_id}")
    
    total_params = 0
    weights_written = 0
    
    def write_weight(key, expected_elements=None):
        nonlocal total_params, weights_written
        data, num_elements = get_tensor_fp16_bytes(tensors, key)
        if expected_elements is not None:
            assert num_elements == expected_elements, \
                f"Size mismatch for {key}: expected {expected_elements}, got {num_elements}"
        out_file.write(data)
        total_params += num_elements
        weights_written += 1
        return num_elements
    
    # 1. fc.weight [dim, n_target_layers * dim]
    n_fc = write_weight('fc.weight', dim * len(target_layer_ids) * dim)
    print(f"  1. fc.weight: {n_fc} elements")
    
    # 2. hidden_norm.weight [dim]
    n_hn = write_weight('hidden_norm.weight', dim)
    print(f"  2. hidden_norm.weight: {n_hn} elements")
    
    # 3. attention_norm (input_layernorm) for all layers
    for i in range(n_layers):
        write_weight(f'layers.{i}.input_layernorm.weight', dim)
    print(f"  3. attention_norm x {n_layers}")
    
    # 4. ffn_norm (post_attention_layernorm) for all layers
    for i in range(n_layers):
        write_weight(f'layers.{i}.post_attention_layernorm.weight', dim)
    print(f"  4. ffn_norm x {n_layers}")
    
    # 5. final norm
    write_weight('norm.weight', dim)
    print(f"  5. final norm")
    
    kv_dim = (dim * n_kv_heads) // n_heads
    
    # 6. wq (q_proj) for all layers
    for i in range(n_layers):
        write_weight(f'layers.{i}.self_attn.q_proj.weight', dim * dim)
    print(f"  6. wq x {n_layers}")
    
    # 7. wk (k_proj) for all layers
    for i in range(n_layers):
        write_weight(f'layers.{i}.self_attn.k_proj.weight', kv_dim * dim)
    print(f"  7. wk x {n_layers}")
    
    # 8. wv (v_proj) for all layers
    for i in range(n_layers):
        write_weight(f'layers.{i}.self_attn.v_proj.weight', kv_dim * dim)
    print(f"  8. wv x {n_layers}")
    
    # 9. wo (o_proj) for all layers
    for i in range(n_layers):
        write_weight(f'layers.{i}.self_attn.o_proj.weight', dim * dim)
    print(f"  9. wo x {n_layers}")
    
    # 10. w1 (gate_proj) for all layers
    for i in range(n_layers):
        write_weight(f'layers.{i}.mlp.gate_proj.weight', hidden_dim * dim)
    print(f"  10. w1 (gate) x {n_layers}")
    
    # 11. w2 (down_proj) for all layers
    for i in range(n_layers):
        write_weight(f'layers.{i}.mlp.down_proj.weight', dim * hidden_dim)
    print(f"  11. w2 (down) x {n_layers}")
    
    # 12. w3 (up_proj) for all layers
    for i in range(n_layers):
        write_weight(f'layers.{i}.mlp.up_proj.weight', hidden_dim * dim)
    print(f"  12. w3 (up) x {n_layers}")
    
    # 13. q_norm for all layers
    for i in range(n_layers):
        write_weight(f'layers.{i}.self_attn.q_norm.weight', head_dim)
    print(f"  13. q_norm x {n_layers}")
    
    # 14. k_norm for all layers
    for i in range(n_layers):
        write_weight(f'layers.{i}.self_attn.k_norm.weight', head_dim)
    print(f"  14. k_norm x {n_layers}")
    
    out_file.close()
    
    file_size = os.path.getsize(filepath)
    print(f"\nExport complete!")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Weight tensors: {weights_written}")
    print(f"  Expected size (FP16): {total_params * 2:,} bytes + 256 header")
    print(f"  Actual file size: {file_size:,} bytes ({file_size / 1024 / 1024:.2f} MB)")
    
    expected_size = total_params * 2 + 256
    if file_size != expected_size:
        print(f"\n  WARNING: File size mismatch!")
        print(f"  Expected: {expected_size:,} bytes")
        print(f"  Actual:   {file_size:,} bytes")
        print(f"  Difference: {file_size - expected_size:,} bytes")
    else:
        print(f"\n  File size verified: matches expected size")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Qwen3-8B DFlash draft model to FP16 bin format")
    parser.add_argument("filepath", type=str, help="output filepath")
    parser.add_argument("--hf", type=str, required=True, help="HuggingFace DFlash model path")
    args = parser.parse_args()

    model_path = Path(args.hf)
    
    # Load config
    config_path = model_path / "config.json"
    if not config_path.exists():
        print(f"Error: config.json not found at {config_path}")
        exit(1)
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print(f"DFlash config:")
    print(f"  hidden_size: {config['hidden_size']}")
    print(f"  num_hidden_layers: {config['num_hidden_layers']}")
    print(f"  num_attention_heads: {config['num_attention_heads']}")
    print(f"  num_key_value_heads: {config['num_key_value_heads']}")
    print(f"  intermediate_size: {config['intermediate_size']}")
    print(f"  block_size: {config.get('block_size', 16)}")
    print(f"  num_target_layers: {config.get('num_target_layers', 36)}")
    print(f"  dflash_config: {config.get('dflash_config', {})}")
    
    # Load weights
    safetensor_files = sorted(list(model_path.glob("*.safetensors")))
    if not safetensor_files:
        print(f"Error: No safetensors files found in {model_path}")
        exit(1)
    
    print(f"\nLoading weights from {safetensor_files[0]}...")
    tensors = load_safetensors_raw(str(safetensor_files[0]))
    
    print(f"Loaded {len(tensors)} tensors")
    
    # Verify key weights exist
    required_keys = ['fc.weight', 'hidden_norm.weight', 'norm.weight']
    for key in required_keys:
        if key not in tensors:
            print(f"Error: Required weight '{key}' not found!")
            exit(1)
    
    print(f"\nExporting to {args.filepath} in FP16 format...")
    dflash_fp16_export(tensors, config, args.filepath)
    
    print("\nDone!")
