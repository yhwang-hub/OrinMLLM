"""
This script exports Qwen2.5-7B-Instruct model to FP16 .bin format for KuiperLLama.

Usage:
    cd /home/lvf6/disk/wyh/KuiperLLama && \
    conda activate KuiperLLama && \
    python tools/export_qwen2.5-7B-fp16.py Qwen2.5-7B-fp16.bin \
        --dtype=fp16 --hf=/home/lvf6/disk/wyh/QwenModels/Qwen2.5-7B-Instruct/

This creates a FP16 model file for optimized inference on CUDA devices.
"""
import os
import struct
import argparse
import gc
from pathlib import Path

import numpy as np
import torch
from torch import nn

from model_qwen2 import ModelArgs, Transformer


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


def fp16_export(model, filepath):
    """
    Export the model weights in FP16 format.
    Header format (256 bytes):
    - magic: uint32 "ak42" (0x616b3432)
    - version: int32 = 3 (FP16 version)
    - dim: int32
    - hidden_dim: int32
    - n_layers: int32
    - n_heads: int32
    - n_kv_heads: int32
    - vocab_size: int32
    - max_seq_len: int32
    - shared_classifier: uint8
    - padding to 256 bytes
    
    Weights order (all in FP16):
    1. attention_norm weights for all layers
    2. ffn_norm weights for all layers
    3. final norm weight
    4. token embeddings
    5. wq weights for all layers
    6. wk weights for all layers
    7. wv weights for all layers
    8. wo weights for all layers
    9. w1 (gate) weights for all layers
    10. w2 (down) weights for all layers
    11. w3 (up) weights for all layers
    12. output (lm_head) weights if not shared
    
    Biases (in FP16, after weights):
    13. wq biases for all layers
    14. wk biases for all layers
    15. wv biases for all layers
    """
    version = 3  # FP16 version
    
    out_file = open(filepath, 'wb')
    
    # Write header (256 bytes)
    # 1) magic
    out_file.write(struct.pack('I', 0x616b3432))
    # 2) version
    out_file.write(struct.pack('i', version))
    # 3) model params
    p = model.params
    hidden_dim = model.layers[0].feed_forward.w1.weight.shape[0]
    n_kv_heads = p.n_heads if p.n_kv_heads is None else p.n_kv_heads
    header = struct.pack('iiiiiii', p.dim, hidden_dim, p.n_layers, p.n_heads,
                         n_kv_heads, p.vocab_size, p.max_seq_len)
    out_file.write(header)
    # 4) shared classifier flag
    shared_classifier = torch.equal(model.tok_embeddings.weight, model.output.weight)
    out_file.write(struct.pack('B', int(shared_classifier)))
    # Pad to 256 bytes
    pad = 256 - out_file.tell()
    assert pad >= 0
    out_file.write(b'\0' * pad)
    
    print(f"Header written: version={version}, dim={p.dim}, hidden_dim={hidden_dim}")
    print(f"  n_layers={p.n_layers}, n_heads={p.n_heads}, n_kv_heads={n_kv_heads}")
    print(f"  vocab_size={p.vocab_size}, max_seq_len={p.max_seq_len}")
    print(f"  shared_classifier={shared_classifier}")
    
    # Collect all weights
    weights = []
    weight_names = []
    
    # 1. attention_norm weights
    for i, layer in enumerate(model.layers):
        weights.append(layer.attention_norm.weight)
        weight_names.append(f"layer{i}.attention_norm")
    
    # 2. ffn_norm weights
    for i, layer in enumerate(model.layers):
        weights.append(layer.ffn_norm.weight)
        weight_names.append(f"layer{i}.ffn_norm")
    
    # 3. final norm
    weights.append(model.norm.weight)
    weight_names.append("final_norm")
    
    # 4. token embeddings
    weights.append(model.tok_embeddings.weight)
    weight_names.append("tok_embeddings")
    
    # 5-8. attention weights
    for i, layer in enumerate(model.layers):
        weights.append(layer.attention.wq.weight)
        weight_names.append(f"layer{i}.wq")
    for i, layer in enumerate(model.layers):
        weights.append(layer.attention.wk.weight)
        weight_names.append(f"layer{i}.wk")
    for i, layer in enumerate(model.layers):
        weights.append(layer.attention.wv.weight)
        weight_names.append(f"layer{i}.wv")
    for i, layer in enumerate(model.layers):
        weights.append(layer.attention.wo.weight)
        weight_names.append(f"layer{i}.wo")
    
    # 9-11. FFN weights
    for i, layer in enumerate(model.layers):
        weights.append(layer.feed_forward.w1.weight)
        weight_names.append(f"layer{i}.w1")
    for i, layer in enumerate(model.layers):
        weights.append(layer.feed_forward.w2.weight)
        weight_names.append(f"layer{i}.w2")
    for i, layer in enumerate(model.layers):
        weights.append(layer.feed_forward.w3.weight)
        weight_names.append(f"layer{i}.w3")
    
    # 12. output weights if not shared
    if not shared_classifier:
        weights.append(model.output.weight)
        weight_names.append("output")
    
    # Write all weights in FP16
    print(f"\nWriting {len(weights)} weight tensors in FP16...")
    total_params = 0
    for i, (w, name) in enumerate(zip(weights, weight_names)):
        serialize_fp16(out_file, w)
        total_params += w.numel()
        if i % 20 == 0 or i == len(weights) - 1:
            print(f"  {i+1}/{len(weights)}: {name} shape={tuple(w.shape)}")
    
    # Write biases in FP16 (Qwen2.5 has biases for q, k, v projections)
    print(f"\nWriting attention biases in FP16...")
    biases = []
    bias_names = []
    
    for i, layer in enumerate(model.layers):
        biases.append(layer.attention.wq.bias)
        bias_names.append(f"layer{i}.wq_bias")
    for i, layer in enumerate(model.layers):
        biases.append(layer.attention.wk.bias)
        bias_names.append(f"layer{i}.wk_bias")
    for i, layer in enumerate(model.layers):
        biases.append(layer.attention.wv.bias)
        bias_names.append(f"layer{i}.wv_bias")
    
    for i, (b, name) in enumerate(zip(biases, bias_names)):
        serialize_fp16(out_file, b)
        total_params += b.numel()
    
    out_file.close()
    
    file_size = os.path.getsize(filepath)
    print(f"\nExport complete!")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Expected size (FP16): {total_params * 2:,} bytes + 256 header")
    print(f"  Actual file size: {file_size:,} bytes ({file_size / 1024 / 1024 / 1024:.2f} GB)")
    print(f"  Wrote {filepath}")


def load_hf_model(model_path):
    """Load Qwen2.5-7B-Instruct model from HuggingFace format."""
    try:
        from transformers import AutoConfig
    except ImportError:
        print("Error: transformers package required")
        return None

    model_path = Path(model_path)
    hf_config = AutoConfig.from_pretrained(model_path)
    
    print(f"Model config:")
    print(f"  hidden_size: {hf_config.hidden_size}")
    print(f"  num_hidden_layers: {hf_config.num_hidden_layers}")
    print(f"  num_attention_heads: {hf_config.num_attention_heads}")
    print(f"  num_key_value_heads: {hf_config.num_key_value_heads}")
    print(f"  intermediate_size: {hf_config.intermediate_size}")
    print(f"  vocab_size: {hf_config.vocab_size}")
    
    config = ModelArgs()
    config.dim = hf_config.hidden_size
    config.n_layers = hf_config.num_hidden_layers
    config.n_heads = hf_config.num_attention_heads
    config.n_kv_heads = hf_config.num_key_value_heads
    config.vocab_size = hf_config.vocab_size
    config.hidden_dim = hf_config.intermediate_size
    config.norm_eps = hf_config.rms_norm_eps
    config.max_seq_len = hf_config.max_position_embeddings

    model = Transformer(config)
    
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
        from transformers import AutoModelForCausalLM
        hf_model = AutoModelForCausalLM.from_pretrained(
            str(model_path), torch_dtype=torch.float32, low_cpu_mem_usage=True)
        hf_dict = hf_model.state_dict()
        del hf_model
        gc.collect()

    # Set weights
    model.tok_embeddings.weight = nn.Parameter(hf_dict['model.embed_tokens.weight'])
    model.norm.weight = nn.Parameter(hf_dict['model.norm.weight'])

    for layer in model.layers:
        i = layer.layer_id
        layer.attention_norm.weight = nn.Parameter(hf_dict[f'model.layers.{i}.input_layernorm.weight'])
        layer.attention.wq.weight = nn.Parameter(hf_dict[f'model.layers.{i}.self_attn.q_proj.weight'])
        layer.attention.wq.bias = nn.Parameter(hf_dict[f'model.layers.{i}.self_attn.q_proj.bias'])
        layer.attention.wk.weight = nn.Parameter(hf_dict[f'model.layers.{i}.self_attn.k_proj.weight'])
        layer.attention.wk.bias = nn.Parameter(hf_dict[f'model.layers.{i}.self_attn.k_proj.bias'])
        layer.attention.wv.weight = nn.Parameter(hf_dict[f'model.layers.{i}.self_attn.v_proj.weight'])
        layer.attention.wv.bias = nn.Parameter(hf_dict[f'model.layers.{i}.self_attn.v_proj.bias'])
        layer.attention.wo.weight = nn.Parameter(hf_dict[f'model.layers.{i}.self_attn.o_proj.weight'])
        layer.ffn_norm.weight = nn.Parameter(hf_dict[f'model.layers.{i}.post_attention_layernorm.weight'])
        layer.feed_forward.w1.weight = nn.Parameter(hf_dict[f'model.layers.{i}.mlp.gate_proj.weight'])
        layer.feed_forward.w2.weight = nn.Parameter(hf_dict[f'model.layers.{i}.mlp.down_proj.weight'])
        layer.feed_forward.w3.weight = nn.Parameter(hf_dict[f'model.layers.{i}.mlp.up_proj.weight'])
        print(f"  Loaded layer {i}/{config.n_layers-1}")

    if 'lm_head.weight' in hf_dict:
        model.output.weight = nn.Parameter(hf_dict['lm_head.weight'])
        print("Using separate lm_head.weight")
    else:
        model.output.weight = model.tok_embeddings.weight
    
    model.eval()
    del hf_dict
    gc.collect()
    
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Qwen2.5-7B to FP16 bin format")
    parser.add_argument("filepath", type=str, help="output filepath")
    parser.add_argument("--dtype", type=str, default="fp16", help="dtype (fp16)")
    parser.add_argument("--hf", type=str, required=True, help="huggingface model path")
    args = parser.parse_args()

    print(f"Loading model from {args.hf}...")
    model = load_hf_model(args.hf)

    if model is None:
        parser.error("Can't load input model!")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel loaded: {total_params:,} parameters ({total_params / 1e9:.2f}B)")
    
    print(f"\nExporting to {args.filepath} in FP16 format...")
    fp16_export(model, args.filepath)
    
    print("\nDone!")
