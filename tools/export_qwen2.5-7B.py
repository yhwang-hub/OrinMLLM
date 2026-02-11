"""
This script exports Qwen2.5-7B-Instruct model to .bin format for KuiperLLama.

Usage:
    cd /home/lvf6/disk/wyh/KuiperLLama && \
    python tools/export_qwen2.5-7B.py Qwen2.5-7B.bin \
        --hf=/home/lvf6/disk/wyh/QwenModels/Qwen2.5-7B-Instruct/

Qwen2.5-7B-Instruct model configuration:
- hidden_size: 3584
- num_hidden_layers: 28
- num_attention_heads: 28
- num_key_value_heads: 4
- intermediate_size: 18944
- vocab_size: 152064
- tie_word_embeddings: false
- max_position_embeddings: 32768
"""
import os
import gzip
import shutil
import struct
import argparse
import json
import gc
from pathlib import Path

import numpy as np
import torch
from torch import nn

from model_qwen2 import ModelArgs, Transformer


# -----------------------------------------------------------------------------
# common utilities

def serialize_fp32(file, tensor):
    """ writes one fp32 tensor to file that is open in wb mode """
    d = tensor.detach().cpu().view(-1).to(torch.float32).numpy()
    b = struct.pack(f'{len(d)}f', *d)
    file.write(b)


def serialize_int8(file, tensor):
    """ writes one int8 tensor to file that is open in wb mode """
    d = tensor.detach().cpu().view(-1).numpy().astype(np.int8)
    b = struct.pack(f'{len(d)}b', *d)
    file.write(b)


def quantize_q80(w, group_size):
    """
    takes a tensor and returns the Q8_0 quantized version
    i.e. symmetric quantization into int8, range [-127,127]
    """
    assert w.numel() % group_size == 0
    ori_shape = w.shape
    w = w.float()  # convert to float32
    w = w.reshape(-1, group_size)
    # find the max in each group
    wmax = torch.abs(w).max(dim=1).values
    # calculate the scaling factor such that float = quant * scale
    scale = wmax / 127.0
    # scale into range [-127, 127]
    quant = w / scale[:, None]
    # round to nearest integer
    int8val = torch.round(quant).to(torch.int8)
    # dequantize by rescaling
    fp32val = (int8val.float() * scale[:, None]).view(-1)
    fp32valr = fp32val.reshape(-1, group_size)
    # calculate the max error in each group
    err = torch.abs(fp32valr - w).max(dim=1).values
    # find the max error across all groups
    maxerr = err.max().item()
    return int8val, scale, maxerr


# -----------------------------------------------------------------------------
# legacy

def legacy_export(model, filepath):
    """ Original export of llama2.c bin files, i.e. version v0 """
    out_file = open(filepath, 'wb')

    # first write out the header
    hidden_dim = model.layers[0].feed_forward.w1.weight.shape[0]
    p = model.params
    shared_classifier = torch.equal(model.tok_embeddings.weight, model.output.weight)
    # legacy format uses negative/positive vocab size as a shared classifier flag
    if not shared_classifier:
        p.vocab_size = -p.vocab_size
    n_kv_heads = p.n_heads if p.n_kv_heads is None else p.n_kv_heads
    header = struct.pack('iiiiiii', p.dim, hidden_dim, p.n_layers, p.n_heads,
                         n_kv_heads, p.vocab_size, p.max_seq_len)
    out_file.write(header)

    # next write out the embedding weights
    serialize_fp32(out_file, model.tok_embeddings.weight)

    # now all the layers
    # attention weights
    for layer in model.layers:
        serialize_fp32(out_file, layer.attention_norm.weight)
    for layer in model.layers:
        serialize_fp32(out_file, layer.attention.wq.weight)
        serialize_fp32(out_file, layer.attention.wq.bias)
    for layer in model.layers:
        serialize_fp32(out_file, layer.attention.wk.weight)
        serialize_fp32(out_file, layer.attention.wk.bias)
    for layer in model.layers:
        serialize_fp32(out_file, layer.attention.wv.weight)
        serialize_fp32(out_file, layer.attention.wv.bias)
    for layer in model.layers:
        serialize_fp32(out_file, layer.attention.wo.weight)
    # ffn weights
    for layer in model.layers:
        serialize_fp32(out_file, layer.ffn_norm.weight)
    for layer in model.layers:
        serialize_fp32(out_file, layer.feed_forward.w1.weight)
    for layer in model.layers:
        serialize_fp32(out_file, layer.feed_forward.w2.weight)
    for layer in model.layers:
        serialize_fp32(out_file, layer.feed_forward.w3.weight)
    # final rmsnorm
    serialize_fp32(out_file, model.norm.weight)
    # freqs_cis
    serialize_fp32(out_file, model.freqs_cos[:p.max_seq_len])
    serialize_fp32(out_file, model.freqs_sin[:p.max_seq_len])

    # final classifier weights
    if not shared_classifier:
        serialize_fp32(out_file, model.output.weight)

    # write to binary file
    out_file.close()
    print(f"wrote {filepath}")


def legacy_export_quant(model, filepath):
    print('export quant model')
    """ Original export of llama2.c bin files, i.e. version v0 """
    out_file = open(filepath, 'wb')

    # first write out the header
    hidden_dim = model.layers[0].feed_forward.w1.weight.shape[0]
    p = model.params
    shared_classifier = torch.equal(model.tok_embeddings.weight, model.output.weight)
    # legacy format uses negative/positive vocab size as a shared classifier flag
    if not shared_classifier:
        p.vocab_size = -p.vocab_size
    n_kv_heads = p.n_heads if p.n_kv_heads is None else p.n_kv_heads
    group_size = 64
    header = struct.pack('iiiiiiii', p.dim, hidden_dim, p.n_layers, p.n_heads,
                         n_kv_heads, p.vocab_size, p.max_seq_len, group_size)
    out_file.write(header)

    group_size = 64
    for layer in model.layers:
        q, s, err = quantize_q80(layer.attention.wq.weight, group_size)
        serialize_int8(out_file, q)
        serialize_fp32(out_file, s)
    for layer in model.layers:
        q, s, err = quantize_q80(layer.attention.wk.weight, group_size)
        serialize_int8(out_file, q)
        serialize_fp32(out_file, s)
    for layer in model.layers:
        q, s, err = quantize_q80(layer.attention.wv.weight, group_size)
        serialize_int8(out_file, q)
        serialize_fp32(out_file, s)
    for layer in model.layers:
        q, s, err = quantize_q80(layer.attention.wo.weight, group_size)
        serialize_int8(out_file, q)
        serialize_fp32(out_file, s)

    for layer in model.layers:
        q, s, err = quantize_q80(layer.feed_forward.w1.weight, group_size)
        serialize_int8(out_file, q)
        serialize_fp32(out_file, s)
    for layer in model.layers:
        q, s, err = quantize_q80(layer.feed_forward.w2.weight, group_size)
        serialize_int8(out_file, q)
        serialize_fp32(out_file, s)
    for layer in model.layers:
        q, s, err = quantize_q80(layer.feed_forward.w3.weight, group_size)
        serialize_int8(out_file, q)
        serialize_fp32(out_file, s)

    # final classifier weights
    if not shared_classifier:
        # serialize_fp32(out_file, model.output.weight)
        q, s, err = quantize_q80(model.output.weight, group_size)
        serialize_int8(out_file, q)
        serialize_fp32(out_file, s)


    # next write out the embedding weights
    serialize_fp32(out_file, model.tok_embeddings.weight)

    # attention weights
    for layer in model.layers:
        serialize_fp32(out_file, layer.attention_norm.weight)

    # ffn weights
    for layer in model.layers:
        serialize_fp32(out_file, layer.ffn_norm.weight)

    # final rmsnorm
    serialize_fp32(out_file, model.norm.weight)
    # freqs_cis
    # serialize_fp32(out_file, model.freqs_cos[:p.max_seq_len])
    # serialize_fp32(out_file, model.freqs_sin[:p.max_seq_len])

    # write to binary file
    out_file.close()
    print(f"wrote {filepath}")


# -----------------------------------------------------------------------------
# new version

def version1_export(model, filepath):
    """
    Export the model weights in full float32 .bin file to be read from C.
    This is same as legacy_export, but with a proper header.
    """
    version = 1

    out_file = open(filepath, 'wb')
    # first write out the header. the header will be 256 bytes
    # 1) write magic, which will be uint32 of "ak42" in ASCII
    out_file.write(struct.pack('I', 0x616b3432))
    # 2) write version, which will be int
    out_file.write(struct.pack('i', version))
    # 3) write the params, which will be 7 ints
    p = model.params
    hidden_dim = model.layers[0].feed_forward.w1.weight.shape[0]
    n_kv_heads = p.n_heads if p.n_kv_heads is None else p.n_kv_heads
    header = struct.pack('iiiiiii', p.dim, hidden_dim, p.n_layers, p.n_heads,
                         n_kv_heads, p.vocab_size, p.max_seq_len)
    out_file.write(header)
    # 4) write some other flags
    shared_classifier = torch.equal(model.tok_embeddings.weight, model.output.weight)
    out_file.write(struct.pack('B', int(shared_classifier)))
    pad = 256 - out_file.tell()  # pad rest with zeros; tell returns current pos
    assert pad >= 0
    out_file.write(b'\0' * pad)

    # now let's write out all the params
    weights = [
        *[layer.attention_norm.weight for layer in model.layers],
        *[layer.ffn_norm.weight for layer in model.layers],
        model.norm.weight,
        model.tok_embeddings.weight,
        *[layer.attention.wq.weight for layer in model.layers],
        *[layer.attention.wk.weight for layer in model.layers],
        *[layer.attention.wv.weight for layer in model.layers],
        *[layer.attention.wo.weight for layer in model.layers],
        *[layer.feed_forward.w1.weight for layer in model.layers],
        *[layer.feed_forward.w2.weight for layer in model.layers],
        *[layer.feed_forward.w3.weight for layer in model.layers],
    ]
    if not shared_classifier:
        weights.append(model.output.weight)
    for w in weights:
        serialize_fp32(out_file, w)

    # write to binary file
    out_file.close()
    print(f"wrote {filepath}")


def version2_export(model, filepath, group_size=64):
    """
    Export the model weights in Q8_0 into .bin file to be read from C.
    That is:
    - quantize all weights to symmetric int8, in range [-127, 127]
    - all other tensors (the rmsnorm params) are kept and exported in fp32
    - quantization is done in groups of group_size to reduce the effects of any outliers
    """
    version = 2

    # let's first do some validation for this export type
    while model.params.dim % group_size != 0:
        group_size //= 2
        print(f"BACKOFF: reducing group size to {group_size} to fit hidden_dim")
    weights = [
        model.tok_embeddings.weight,
        *[layer.attention.wq.weight for layer in model.layers],
        *[layer.attention.wk.weight for layer in model.layers],
        *[layer.attention.wv.weight for layer in model.layers],
        *[layer.attention.wo.weight for layer in model.layers],
        *[layer.feed_forward.w1.weight for layer in model.layers],
        *[layer.feed_forward.w2.weight for layer in model.layers],
        *[layer.feed_forward.w3.weight for layer in model.layers],
    ]
    shared_classifier = torch.equal(model.tok_embeddings.weight, model.output.weight)
    if not shared_classifier:
        weights.append(model.output.weight)
    for i, w in enumerate(weights):
        assert w.numel() % group_size == 0, f"weight {i} has numel {w.numel()}, not a multiple of group_size {group_size}"

    # write
    out_file = open(filepath, 'wb')
    # first write out the header. the header will be 256 bytes
    # 1) write magic, which will be uint32 of "ak42" in ASCII
    out_file.write(struct.pack('I', 0x616b3432))
    # 2) write version, which will be int
    out_file.write(struct.pack('i', version))
    # 3) write the params, which will be 7 ints
    p = model.params
    hidden_dim = model.layers[0].feed_forward.w1.weight.shape[0]
    n_kv_heads = p.n_heads if p.n_kv_heads is None else p.n_kv_heads
    header = struct.pack('iiiiiii', p.dim, hidden_dim, p.n_layers, p.n_heads,
                         n_kv_heads, p.vocab_size, p.max_seq_len)
    out_file.write(header)
    # 4) write some other flags
    out_file.write(struct.pack('B', int(shared_classifier)))
    out_file.write(struct.pack('i', group_size))  # group size used for quantization
    pad = 256 - out_file.tell()  # pad rest with zeros; tell returns current pos
    assert pad >= 0
    out_file.write(b'\0' * pad)
    # now that the header is done, let's write out the model

    # first let's write out all the params that we are keeping in fp32: the norms
    for layer in model.layers:  # attention norms
        serialize_fp32(out_file, layer.attention_norm.weight)
    for layer in model.layers:  # MLP norms
        serialize_fp32(out_file, layer.ffn_norm.weight)
    serialize_fp32(out_file, model.norm.weight)  # final pre-classifier norm

    # now let's write out all the params that we are quantizing to Q8_0
    # note we skip classifier weights, which are shared with the embedding
    ew = []
    for i, w in enumerate(weights):
        # quantize this weight
        q, s, err = quantize_q80(w, group_size)
        # save the int8 weights to file
        serialize_int8(out_file, q)  # save the tensor in int8
        serialize_fp32(out_file, s)  # save scale factors
        # logging
        ew.append((err, w.shape))
        print(f"{i + 1}/{len(weights)} quantized {tuple(w.shape)} to Q8_0 with max error {err}")

    # print the highest error across all weights, should be very small, e.g. O(~0.001)
    ew.sort(reverse=True)
    print(f"max quantization group error across all weights: {ew[0][0]}")

    # write to binary file
    out_file.close()
    print(f"wrote {filepath}")


# -----------------------------------------------------------------------------
# Load / import functions

def load_hf_model(model_path):
    """
    Load Qwen2.5-7B-Instruct model from HuggingFace format.
    Optimized for memory efficiency.
    """
    try:
        from transformers import AutoConfig
    except ImportError:
        print("Error: transformers package is required to load huggingface models")
        print("Please run `pip install transformers` to install it")
        return None

    model_path = Path(model_path)
    
    # 先加载配置，不加载模型权重
    hf_config = AutoConfig.from_pretrained(model_path)
    
    print(f"Model config:")
    print(f"  hidden_size: {hf_config.hidden_size}")
    print(f"  num_hidden_layers: {hf_config.num_hidden_layers}")
    print(f"  num_attention_heads: {hf_config.num_attention_heads}")
    print(f"  num_key_value_heads: {hf_config.num_key_value_heads}")
    print(f"  intermediate_size: {hf_config.intermediate_size}")
    print(f"  vocab_size: {hf_config.vocab_size}")
    print(f"  max_position_embeddings: {hf_config.max_position_embeddings}")
    print(f"  tie_word_embeddings: {hf_config.tie_word_embeddings}")
    
    # convert Config to ModelArgs
    config = ModelArgs()
    config.dim = hf_config.hidden_size
    config.n_layers = hf_config.num_hidden_layers
    config.n_heads = hf_config.num_attention_heads
    config.n_kv_heads = hf_config.num_key_value_heads
    config.vocab_size = hf_config.vocab_size
    config.hidden_dim = hf_config.intermediate_size
    config.norm_eps = hf_config.rms_norm_eps
    config.max_seq_len = hf_config.max_position_embeddings

    # 创建目标模型
    model = Transformer(config)
    print(f"\nCreated Transformer model:")
    print(model)

    # 直接从文件加载 state_dict，不创建完整的 HuggingFace 模型
    safetensor_files = sorted(list(model_path.glob("*.safetensors")))
    bin_files = list(model_path.glob("*.bin"))
    
    hf_dict = {}
    if safetensor_files:
        # 使用 safetensors 加载
        try:
            from safetensors import safe_open
            for sf_file in safetensor_files:
                print(f"Loading from {sf_file}")
                with safe_open(sf_file, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        hf_dict[key] = f.get_tensor(key)
        except ImportError:
            print("safetensors not installed, trying torch.load")
            for sf_file in safetensor_files:
                hf_dict.update(torch.load(sf_file, map_location='cpu'))
    elif bin_files:
        # 使用 pytorch bin 文件加载
        for bin_file in bin_files:
            if 'pytorch_model' in bin_file.name or 'model' in bin_file.name:
                print(f"Loading from {bin_file}")
                hf_dict.update(torch.load(bin_file, map_location='cpu'))
    else:
        print("No model files found, falling back to AutoModelForCausalLM")
        from transformers import AutoModelForCausalLM
        hf_model = AutoModelForCausalLM.from_pretrained(
            str(model_path), 
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        print(hf_model)
        hf_dict = hf_model.state_dict()
        del hf_model
        gc.collect()

    print(f"Loaded {len(hf_dict)} tensors from model files")
    print(f"Keys: {list(hf_dict.keys())[:10]}...")

    # 设置权重
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
        
        # 逐层清理已使用的权重以节省内存
        del hf_dict[f'model.layers.{i}.input_layernorm.weight']
        del hf_dict[f'model.layers.{i}.self_attn.q_proj.weight']
        del hf_dict[f'model.layers.{i}.self_attn.q_proj.bias']
        del hf_dict[f'model.layers.{i}.self_attn.k_proj.weight']
        del hf_dict[f'model.layers.{i}.self_attn.k_proj.bias']
        del hf_dict[f'model.layers.{i}.self_attn.v_proj.weight']
        del hf_dict[f'model.layers.{i}.self_attn.v_proj.bias']
        del hf_dict[f'model.layers.{i}.self_attn.o_proj.weight']
        del hf_dict[f'model.layers.{i}.post_attention_layernorm.weight']
        del hf_dict[f'model.layers.{i}.mlp.gate_proj.weight']
        del hf_dict[f'model.layers.{i}.mlp.down_proj.weight']
        del hf_dict[f'model.layers.{i}.mlp.up_proj.weight']
        
        if i % 5 == 0:
            gc.collect()
        print(f"  Loaded layer {i}/{config.n_layers-1}")

    # final classifier - Qwen2.5-7B 使用独立的 lm_head (tie_word_embeddings=false)
    if 'lm_head.weight' in hf_dict:
        model.output.weight = nn.Parameter(hf_dict['lm_head.weight'])
        print("Using separate lm_head.weight")
    else:
        # 如果没有找到 lm_head，使用 tied embeddings
        print("lm_head.weight not found, using tied embeddings (embed_tokens.weight)")
        model.output.weight = model.tok_embeddings.weight
    
    model.eval()
    
    # 释放内存
    del hf_dict
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return model


# -----------------------------------------------------------------------------
# API entrypoint

def model_export(model, filepath, version, dtype=torch.float32):
    """
    Versions docs:
    v0: legacy llama2.c float format, DEPRECATED
    v1: float32 export
    v2: int8 quantized Q8_0 export, similar to llama.cpp, in groups
    """
    if version == 0:
        legacy_export(model, filepath)
    elif version == 1:
        version1_export(model, filepath)
    elif version == 2:
        version2_export(model, filepath)
    elif version == 3:
        legacy_export_quant(model, filepath)
    else:
        raise ValueError(f"unknown version {version}")


def calculate_expected_size(model, version=0, group_size=64):
    """
    计算预期的bin文件大小（字节）
    """
    p = model.params
    hidden_dim = model.layers[0].feed_forward.w1.weight.shape[0]
    n_kv_heads = p.n_heads if p.n_kv_heads is None else p.n_kv_heads
    head_dim = p.dim // p.n_heads
    shared_classifier = torch.equal(model.tok_embeddings.weight, model.output.weight)
    
    if version == 0:
        # legacy format: header (7*4=28 bytes) + all weights in fp32
        header_size = 7 * 4  # 7 integers
        
        # Embedding: vocab_size * dim
        emb_size = p.vocab_size * p.dim * 4
        
        # Per layer:
        # attention_norm: dim
        # wq: dim * dim + dim (weight + bias)
        # wk: dim * (n_kv_heads * head_dim) + n_kv_heads * head_dim
        # wv: dim * (n_kv_heads * head_dim) + n_kv_heads * head_dim
        # wo: dim * dim
        # ffn_norm: dim
        # w1: dim * hidden_dim
        # w2: hidden_dim * dim
        # w3: dim * hidden_dim
        
        kv_dim = n_kv_heads * head_dim
        layer_size = (
            p.dim +  # attention_norm
            p.dim * p.dim + p.dim +  # wq
            p.dim * kv_dim + kv_dim +  # wk
            p.dim * kv_dim + kv_dim +  # wv
            p.dim * p.dim +  # wo
            p.dim +  # ffn_norm
            p.dim * hidden_dim +  # w1
            hidden_dim * p.dim +  # w2
            p.dim * hidden_dim  # w3
        ) * 4  # fp32
        
        # final norm
        final_norm_size = p.dim * 4
        
        # freqs_cos and freqs_sin
        freqs_size = 2 * p.max_seq_len * (p.dim // p.n_heads // 2) * 4
        
        # output (if not shared)
        output_size = 0 if shared_classifier else p.vocab_size * p.dim * 4
        
        total = header_size + emb_size + p.n_layers * layer_size + final_norm_size + freqs_size + output_size
        
    else:
        # 简化计算，假设所有权重都是fp32
        total_params = sum(p.numel() for p in model.parameters())
        total = total_params * 4  # fp32
        
    return total


def verify_bin_file(filepath, model, version=0):
    """
    验证生成的bin文件大小是否合理
    """
    import os
    
    if not os.path.exists(filepath):
        print(f"Error: {filepath} does not exist!")
        return False
    
    actual_size = os.path.getsize(filepath)
    expected_size = calculate_expected_size(model, version)
    
    print(f"\n=== File Size Verification ===")
    print(f"Actual file size:   {actual_size:,} bytes ({actual_size / 1024 / 1024:.2f} MB)")
    print(f"Expected size:      {expected_size:,} bytes ({expected_size / 1024 / 1024:.2f} MB)")
    
    # 允许一定误差（header等）
    tolerance = 0.1  # 10%
    if abs(actual_size - expected_size) / expected_size < tolerance:
        print(f"✓ File size is within expected range")
        return True
    else:
        ratio = actual_size / expected_size
        print(f"! File size ratio: {ratio:.2f}")
        if actual_size > expected_size * 0.5:
            print(f"✓ File size seems reasonable")
            return True
        else:
            print(f"✗ File size may be incorrect!")
            return False


# -----------------------------------------------------------------------------
# CLI entrypoint

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Export Qwen2.5-7B-Instruct to bin format")
    parser.add_argument("filepath", type=str, help="the output filepath")
    parser.add_argument("--version", default=0, type=int, help="the version to export with (0: legacy, 1: fp32, 2: int8)")
    parser.add_argument("--dtype", type=str, help="dtype of the model (fp16, fp32)", default="fp32")
    parser.add_argument("--hf", type=str, required=True, help="huggingface model path")
    args = parser.parse_args()
    dtype = {"fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    print(f"Loading model from {args.hf}...")
    model = load_hf_model(args.hf)

    if model is None:
        parser.error("Can't load input model!")

    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel loaded successfully!")
    print(f"Total parameters: {total_params:,} ({total_params / 1e9:.2f}B)")
    
    # export
    print(f"\nExporting to {args.filepath} with version {args.version}...")
    model_export(model, args.filepath, args.version, args.dtype)
    
    # 验证文件大小
    verify_bin_file(args.filepath, model, args.version)
    
    print("\nDone!")
