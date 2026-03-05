"""
This script exports Qwen3-8B SmoothQuant (INT8) quantized model to .bin format for KuiperLLama.

Usage:
    cd /home/wangyh/OrinMLLM && \
    python tools/export_qwen3-8B-sq.py /mnt/ssd/QwenModels/Qwen3-8B-sq.bin \
        --hf=/mnt/ssd/QwenModels/Qwen3-8B-sq/

SmoothQuant Format (per-tensor quantization):
- qweight: [out_features, in_features] INT8
- weight_scale: scalar BF16 (per-tensor weight scale)
- input_scale: scalar FP32 (per-tensor input activation scale)

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


def serialize_int8(file, tensor):
    """writes one int8 tensor to file"""
    d = tensor.detach().cpu().view(-1).to(torch.int8).numpy()
    file.write(d.tobytes())


def serialize_fp32_scalar(file, value):
    """writes one fp32 scalar to file"""
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().float().item()
    file.write(struct.pack('f', value))


def serialize_fp16_scalar(file, value):
    """writes one fp16 scalar to file (stored as 2 bytes)"""
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().to(torch.float16)
    else:
        value = torch.tensor(value, dtype=torch.float16)
    file.write(value.numpy().tobytes())


def sq_export(hf_dict, config, filepath):
    """
    Export the SmoothQuant INT8 quantized model weights.
    
    Header format (256 bytes):
    - magic: uint32 "sq48" (0x73713438) - magic for Qwen3 SmoothQuant
    - version: int32 = 6 (SQ INT8 version for Qwen3)
    - dim: int32
    - hidden_dim: int32 (intermediate_size)
    - n_layers: int32
    - n_heads: int32
    - n_kv_heads: int32
    - vocab_size: int32
    - max_seq_len: int32
    - shared_classifier: uint8
    - head_dim: int32 (Qwen3 specific)
    - padding to 256 bytes
    
    Weights order:
    == FP16 weights (non-quantized) ==
    1. attention_norm (input_layernorm) for all layers - FP16
    2. ffn_norm (post_attention_layernorm) for all layers - FP16
    3. final norm weight - FP16
    4. token embeddings - FP16
    
    == SQ quantized weights (for each layer group) ==
    For each linear layer (wq, wk, wv, wo, w1, w2, w3):
      For each layer i in [0, n_layers):
        - qweight: [out_features, in_features] INT8
        - weight_scale: FP16 scalar
        - input_scale: FP32 scalar
    
    == FP16 weights (non-quantized) ==
    12. output (lm_head) weights - FP16 (if not shared)
    13. q_norm weights for all layers - FP16
    14. k_norm weights for all layers - FP16
    """
    version = 6  # SQ INT8 version for Qwen3
    
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
    # 1) magic - use "sq48" (0x73713438) for Qwen3 SmoothQuant
    out_file.write(struct.pack('I', 0x73713438))
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
    
    # Helper function to write SQ quantized weights
    def write_sq_weights(layer_name, prefix=""):
        nonlocal total_bytes
        qweight = hf_dict[f'{layer_name}.qweight']
        weight_scale = hf_dict[f'{layer_name}.weight_scale']
        input_scale = hf_dict[f'{layer_name}.input_scale']
        
        # Write INT8 quantized weight [out_features, in_features]
        serialize_int8(out_file, qweight)
        total_bytes += qweight.numel() * 1  # INT8 = 1 byte
        
        # Write weight_scale as FP16 scalar (2 bytes)
        serialize_fp16_scalar(out_file, weight_scale)
        total_bytes += 2
        
        # Write input_scale as FP32 scalar (4 bytes)
        serialize_fp32_scalar(out_file, input_scale)
        total_bytes += 4
        
        if prefix:
            ws_val = weight_scale.item() if isinstance(weight_scale, torch.Tensor) else weight_scale
            is_val = input_scale.item() if isinstance(input_scale, torch.Tensor) else input_scale
            print(f"  {prefix}: qweight={tuple(qweight.shape)}, weight_scale={ws_val:.6f}, input_scale={is_val:.6f}")
    
    # 5-11. SQ quantized weights for all layers
    print("\nWriting SQ INT8 quantized weights...")
    
    # wq (q_proj) for all layers
    print("Writing wq weights (SQ INT8)...")
    for i in range(n_layers):
        write_sq_weights(f'model.layers.{i}.self_attn.q_proj', f"layer{i}.wq")
    
    # wk (k_proj) for all layers
    print("Writing wk weights (SQ INT8)...")
    for i in range(n_layers):
        write_sq_weights(f'model.layers.{i}.self_attn.k_proj', f"layer{i}.wk")
    
    # wv (v_proj) for all layers
    print("Writing wv weights (SQ INT8)...")
    for i in range(n_layers):
        write_sq_weights(f'model.layers.{i}.self_attn.v_proj', f"layer{i}.wv")
    
    # wo (o_proj) for all layers
    print("Writing wo weights (SQ INT8)...")
    for i in range(n_layers):
        write_sq_weights(f'model.layers.{i}.self_attn.o_proj', f"layer{i}.wo")
    
    # w1 (gate_proj) for all layers
    print("Writing w1 (gate_proj) weights (SQ INT8)...")
    for i in range(n_layers):
        write_sq_weights(f'model.layers.{i}.mlp.gate_proj', f"layer{i}.w1")
    
    # w2 (down_proj) for all layers
    print("Writing w2 (down_proj) weights (SQ INT8)...")
    for i in range(n_layers):
        write_sq_weights(f'model.layers.{i}.mlp.down_proj', f"layer{i}.w2")
    
    # w3 (up_proj) for all layers
    print("Writing w3 (up_proj) weights (SQ INT8)...")
    for i in range(n_layers):
        write_sq_weights(f'model.layers.{i}.mlp.up_proj', f"layer{i}.w3")
    
    # 12. output weights (lm_head) - FP16 (not quantized)
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
    """Load Qwen3-8B SmoothQuant model weights from HuggingFace format."""
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
    
    # Check quantization config
    if hasattr(hf_config, 'quantization_config'):
        quant_config = hf_config.quantization_config
        print(f"\nQuantization config:")
        print(f"  quant_method: {quant_config.get('quant_method', 'unknown')}")
        print(f"  bits: {quant_config.get('bits', 'unknown')}")
        print(f"  per_tensor: {quant_config.get('per_tensor', False)}")
        print(f"  zero_point: {quant_config.get('zero_point', True)}")
        print(f"  modules_to_not_convert: {quant_config.get('modules_to_not_convert', [])}")
    
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
            return None, None
    
    # Print some loaded keys for verification
    print(f"\nLoaded {len(hf_dict)} tensors")
    print("Sample keys:")
    sample_keys = list(hf_dict.keys())[:20]
    for key in sample_keys:
        print(f"  {key}: {hf_dict[key].shape}, dtype={hf_dict[key].dtype}")
    
    # Verify SQ keys exist
    print("\nVerifying SmoothQuant quantization keys...")
    if 'model.layers.0.self_attn.q_proj.qweight' in hf_dict:
        print("  ✅ SQ qweight found")
    else:
        print("  ❌ SQ qweight NOT found - this may not be an SQ model!")
        return None, None
    
    if 'model.layers.0.self_attn.q_proj.weight_scale' in hf_dict:
        print("  ✅ SQ weight_scale found")
    else:
        print("  ❌ SQ weight_scale NOT found")
        return None, None
    
    if 'model.layers.0.self_attn.q_proj.input_scale' in hf_dict:
        print("  ✅ SQ input_scale found")
    else:
        print("  ❌ SQ input_scale NOT found")
        return None, None
    
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
    
    # Verify lm_head is not quantized
    if 'lm_head.weight' in hf_dict:
        print(f"\n  lm_head.weight: dtype={hf_dict['lm_head.weight'].dtype} (should be bf16/fp16, NOT quantized)")
    
    return hf_dict, config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Qwen3-8B SmoothQuant to bin format")
    parser.add_argument("filepath", type=str, help="output filepath")
    parser.add_argument("--hf", type=str, required=True, help="huggingface model path")
    args = parser.parse_args()

    print(f"Loading model weights from {args.hf}...")
    hf_dict, config = load_hf_weights(args.hf)

    if hf_dict is None:
        parser.error("Can't load input model!")

    print(f"\nExporting to {args.filepath} in SmoothQuant INT8 format...")
    sq_export(hf_dict, config, args.filepath)
    
    print("\nDone!")
