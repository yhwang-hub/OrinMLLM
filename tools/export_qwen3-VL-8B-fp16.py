"""
This script exports Qwen3-VL-8B model to FP16 .bin format for KuiperLLama.

Usage:
    cd /mnt/ssd/workspace/KuiperLLama_20260202_fp16_vlm && \
    conda activate KuiperLLama && \
    python tools/export_qwen3-VL-8B-fp16.py /mnt/ssd/QwenModels/Qwen3-VL-8B-fp16.bin \
        --dtype=fp16 --hf=/mnt/ssd/QwenModels/Qwen3-VL-8B-Instruct

This creates a FP16 model file for optimized inference on CUDA devices.

Model Structure (Qwen3-VL-8B):
==============================

1. Vision Encoder (ViT):
   - patch_embed: Conv3d for patch embedding
   - pos_embed: Position embedding (num_position_embeddings x hidden_size)
   - blocks: 27 transformer blocks with:
     - norm1, norm2: LayerNorm with bias
     - attn: QKV fused attention with bias + output projection with bias
     - mlp: linear_fc1 + GELU + linear_fc2 (all with bias)
   - merger: Projection from vision hidden to LLM hidden
   - deepstack_merger_list: 3 additional mergers for deepstack features
   
2. Language Model (LLM):
   - embed_tokens: Token embeddings
   - layers: 36 transformer layers with:
     - input_layernorm, post_attention_layernorm: RMSNorm
     - self_attn: q_proj, k_proj, v_proj, o_proj + q_norm, k_norm
     - mlp: gate_proj, up_proj, down_proj
   - norm: Final RMSNorm
   - lm_head: Output projection

3. Special tokens:
   - image_token_id: 151655
   - video_token_id: 151656  
   - vision_start_token_id: 151652
   - vision_end_token_id: 151653
"""
import os
import struct
import argparse
import gc
import json
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
    file.write(d.tobytes())


def serialize_int32(file, value):
    """writes one int32 value to file"""
    file.write(struct.pack('i', value))


def serialize_uint32(file, value):
    """writes one uint32 value to file"""
    file.write(struct.pack('I', value))


def serialize_float32(file, value):
    """writes one float32 value to file"""
    file.write(struct.pack('f', value))


class Qwen3VLExporter:
    """Exports Qwen3-VL model to binary format"""
    
    # Magic number for Qwen3-VL model: "qw3v" in ASCII
    MAGIC = 0x71773376
    VERSION = 1
    
    def __init__(self, hf_dict, config):
        self.hf_dict = hf_dict
        self.config = config
        
    def export(self, filepath):
        """Export model to binary file"""
        out_file = open(filepath, 'wb')
        
        # Write header
        self._write_header(out_file)
        
        # Write Vision Encoder weights
        self._write_vision_encoder(out_file)
        
        # Write Language Model weights
        self._write_language_model(out_file)
        
        out_file.close()
        
        # Verify file size
        file_size = os.path.getsize(filepath)
        print(f"\n✅ Export complete!")
        print(f"  File size: {file_size:,} bytes ({file_size / 1024 / 1024 / 1024:.2f} GB)")
        print(f"  Wrote {filepath}")
        
    def _write_header(self, out_file):
        """Write file header (512 bytes)"""
        # Vision config
        vision_config = self.config['vision_config']
        text_config = self.config['text_config']
        
        # 1) Magic number (4 bytes)
        serialize_uint32(out_file, self.MAGIC)
        
        # 2) Version (4 bytes)
        serialize_int32(out_file, self.VERSION)
        
        # 3) Vision config (48 bytes)
        serialize_int32(out_file, vision_config['hidden_size'])        # vit_hidden_size: 1152
        serialize_int32(out_file, vision_config['intermediate_size'])  # vit_intermediate_size: 4304
        serialize_int32(out_file, vision_config['num_heads'])          # vit_num_heads: 16
        serialize_int32(out_file, vision_config['depth'])              # vit_depth: 27
        serialize_int32(out_file, vision_config['patch_size'])         # patch_size: 16
        serialize_int32(out_file, vision_config['temporal_patch_size']) # temporal_patch_size: 2
        serialize_int32(out_file, vision_config['in_channels'])        # in_channels: 3
        serialize_int32(out_file, vision_config['spatial_merge_size']) # spatial_merge_size: 2
        serialize_int32(out_file, vision_config['out_hidden_size'])    # out_hidden_size: 4096
        serialize_int32(out_file, vision_config['num_position_embeddings']) # num_position_embeddings: 2304
        
        # deepstack_visual_indexes (3 values)
        deepstack_indexes = vision_config.get('deepstack_visual_indexes', [8, 16, 24])
        for idx in deepstack_indexes:
            serialize_int32(out_file, idx)
        
        # 4) Text/LLM config (56 bytes)
        serialize_int32(out_file, text_config['hidden_size'])          # llm_dim: 4096
        serialize_int32(out_file, text_config['intermediate_size'])    # llm_hidden_dim: 12288
        serialize_int32(out_file, text_config['num_hidden_layers'])    # llm_n_layers: 36
        serialize_int32(out_file, text_config['num_attention_heads'])  # llm_n_heads: 32
        serialize_int32(out_file, text_config['num_key_value_heads'])  # llm_n_kv_heads: 8
        serialize_int32(out_file, text_config['vocab_size'])           # vocab_size: 151936
        serialize_int32(out_file, text_config['max_position_embeddings']) # max_seq_len: 262144
        serialize_int32(out_file, text_config.get('head_dim', 128))    # head_dim: 128
        serialize_float32(out_file, text_config.get('rms_norm_eps', 1e-6)) # rms_norm_eps
        serialize_float32(out_file, text_config.get('rope_theta', 5000000)) # rope_theta
        
        # 5) Special tokens (20 bytes)
        serialize_int32(out_file, self.config.get('image_token_id', 151655))
        serialize_int32(out_file, self.config.get('video_token_id', 151656))
        serialize_int32(out_file, self.config.get('vision_start_token_id', 151652))
        serialize_int32(out_file, self.config.get('vision_end_token_id', 151653))
        serialize_int32(out_file, text_config.get('eos_token_id', 151645))
        
        # 6) Flags (4 bytes)
        shared_classifier = torch.equal(
            self.hf_dict['model.language_model.embed_tokens.weight'],
            self.hf_dict.get('lm_head.weight', self.hf_dict['model.language_model.embed_tokens.weight'])
        ) if 'lm_head.weight' in self.hf_dict else True
        serialize_int32(out_file, int(not shared_classifier))  # has_lm_head flag
        
        # Pad to 512 bytes
        current_pos = out_file.tell()
        pad = 512 - current_pos
        assert pad >= 0, f"Header too large: {current_pos} bytes"
        out_file.write(b'\0' * pad)
        
        print(f"Header written: {out_file.tell()} bytes")
        print(f"  Vision: hidden={vision_config['hidden_size']}, depth={vision_config['depth']}, patch={vision_config['patch_size']}")
        print(f"  LLM: dim={text_config['hidden_size']}, layers={text_config['num_hidden_layers']}, heads={text_config['num_attention_heads']}")
        print(f"  Vocab: {text_config['vocab_size']}, shared_classifier={shared_classifier}")
        
    def _write_vision_encoder(self, out_file):
        """Write Vision Encoder (ViT) weights"""
        print("\n=== Writing Vision Encoder ===")
        
        vision_config = self.config['vision_config']
        vit_depth = vision_config['depth']  # 27
        deepstack_indexes = vision_config.get('deepstack_visual_indexes', [8, 16, 24])
        
        total_params = 0
        
        # 1. Patch embedding: Conv3d weight and bias
        # Shape: [hidden_size, in_channels, temporal_patch_size, patch_size, patch_size]
        #        [1152, 3, 2, 16, 16]
        print("  Writing patch_embed...")
        weight = self.hf_dict['model.visual.patch_embed.proj.weight']
        bias = self.hf_dict['model.visual.patch_embed.proj.bias']
        serialize_fp16(out_file, weight)
        serialize_fp16(out_file, bias)
        total_params += weight.numel() + bias.numel()
        print(f"    patch_embed.weight: {tuple(weight.shape)}")
        print(f"    patch_embed.bias: {tuple(bias.shape)}")
        
        # 2. Position embedding (if exists in weights, otherwise skip)
        # Note: Qwen3-VL uses learned position embedding or interpolation
        # Check if pos_embed exists
        if 'model.visual.pos_embed.weight' in self.hf_dict:
            print("  Writing pos_embed...")
            pos_embed = self.hf_dict['model.visual.pos_embed.weight']
            serialize_fp16(out_file, pos_embed)
            total_params += pos_embed.numel()
            print(f"    pos_embed: {tuple(pos_embed.shape)}")
        else:
            print("  pos_embed not found (will use learned embedding layer)")
        
        # 3. Transformer blocks
        print(f"  Writing {vit_depth} transformer blocks...")
        for i in range(vit_depth):
            prefix = f'model.visual.blocks.{i}'
            
            # norm1 (LayerNorm with bias)
            serialize_fp16(out_file, self.hf_dict[f'{prefix}.norm1.weight'])
            serialize_fp16(out_file, self.hf_dict[f'{prefix}.norm1.bias'])
            
            # norm2 (LayerNorm with bias)
            serialize_fp16(out_file, self.hf_dict[f'{prefix}.norm2.weight'])
            serialize_fp16(out_file, self.hf_dict[f'{prefix}.norm2.bias'])
            
            # attn.qkv (fused QKV projection)
            serialize_fp16(out_file, self.hf_dict[f'{prefix}.attn.qkv.weight'])
            serialize_fp16(out_file, self.hf_dict[f'{prefix}.attn.qkv.bias'])
            
            # attn.proj (output projection)
            serialize_fp16(out_file, self.hf_dict[f'{prefix}.attn.proj.weight'])
            serialize_fp16(out_file, self.hf_dict[f'{prefix}.attn.proj.bias'])
            
            # mlp.linear_fc1
            serialize_fp16(out_file, self.hf_dict[f'{prefix}.mlp.linear_fc1.weight'])
            serialize_fp16(out_file, self.hf_dict[f'{prefix}.mlp.linear_fc1.bias'])
            
            # mlp.linear_fc2
            serialize_fp16(out_file, self.hf_dict[f'{prefix}.mlp.linear_fc2.weight'])
            serialize_fp16(out_file, self.hf_dict[f'{prefix}.mlp.linear_fc2.bias'])
            
            # Count parameters
            for suffix in ['.norm1.weight', '.norm1.bias', '.norm2.weight', '.norm2.bias',
                          '.attn.qkv.weight', '.attn.qkv.bias', '.attn.proj.weight', '.attn.proj.bias',
                          '.mlp.linear_fc1.weight', '.mlp.linear_fc1.bias',
                          '.mlp.linear_fc2.weight', '.mlp.linear_fc2.bias']:
                total_params += self.hf_dict[f'{prefix}{suffix}'].numel()
            
            if i % 9 == 0 or i == vit_depth - 1:
                print(f"    Block {i}: written")
        
        # 4. Main merger (vision to LLM projection)
        print("  Writing merger...")
        prefix = 'model.visual.merger'
        serialize_fp16(out_file, self.hf_dict[f'{prefix}.norm.weight'])
        serialize_fp16(out_file, self.hf_dict[f'{prefix}.norm.bias'])
        serialize_fp16(out_file, self.hf_dict[f'{prefix}.linear_fc1.weight'])
        serialize_fp16(out_file, self.hf_dict[f'{prefix}.linear_fc1.bias'])
        serialize_fp16(out_file, self.hf_dict[f'{prefix}.linear_fc2.weight'])
        serialize_fp16(out_file, self.hf_dict[f'{prefix}.linear_fc2.bias'])
        
        for suffix in ['.norm.weight', '.norm.bias', '.linear_fc1.weight', '.linear_fc1.bias',
                      '.linear_fc2.weight', '.linear_fc2.bias']:
            total_params += self.hf_dict[f'{prefix}{suffix}'].numel()
        
        # 5. Deepstack mergers (3 additional mergers)
        print(f"  Writing {len(deepstack_indexes)} deepstack mergers...")
        for idx in range(len(deepstack_indexes)):
            prefix = f'model.visual.deepstack_merger_list.{idx}'
            serialize_fp16(out_file, self.hf_dict[f'{prefix}.norm.weight'])
            serialize_fp16(out_file, self.hf_dict[f'{prefix}.norm.bias'])
            serialize_fp16(out_file, self.hf_dict[f'{prefix}.linear_fc1.weight'])
            serialize_fp16(out_file, self.hf_dict[f'{prefix}.linear_fc1.bias'])
            serialize_fp16(out_file, self.hf_dict[f'{prefix}.linear_fc2.weight'])
            serialize_fp16(out_file, self.hf_dict[f'{prefix}.linear_fc2.bias'])
            
            for suffix in ['.norm.weight', '.norm.bias', '.linear_fc1.weight', '.linear_fc1.bias',
                          '.linear_fc2.weight', '.linear_fc2.bias']:
                total_params += self.hf_dict[f'{prefix}{suffix}'].numel()
        
        print(f"  Vision encoder total parameters: {total_params:,}")
        
    def _write_language_model(self, out_file):
        """Write Language Model (LLM) weights"""
        print("\n=== Writing Language Model ===")
        
        text_config = self.config['text_config']
        n_layers = text_config['num_hidden_layers']  # 36
        
        total_params = 0
        
        # 1. RMSNorm weights (input_layernorm + post_attention_layernorm for each layer + final norm)
        print("  Writing RMSNorm weights...")
        for i in range(n_layers):
            serialize_fp16(out_file, self.hf_dict[f'model.language_model.layers.{i}.input_layernorm.weight'])
            total_params += self.hf_dict[f'model.language_model.layers.{i}.input_layernorm.weight'].numel()
        
        for i in range(n_layers):
            serialize_fp16(out_file, self.hf_dict[f'model.language_model.layers.{i}.post_attention_layernorm.weight'])
            total_params += self.hf_dict[f'model.language_model.layers.{i}.post_attention_layernorm.weight'].numel()
        
        # Final norm
        serialize_fp16(out_file, self.hf_dict['model.language_model.norm.weight'])
        total_params += self.hf_dict['model.language_model.norm.weight'].numel()
        print(f"    RMSNorm: {2 * n_layers + 1} tensors")
        
        # 2. Token embeddings
        print("  Writing token embeddings...")
        embed_weight = self.hf_dict['model.language_model.embed_tokens.weight']
        serialize_fp16(out_file, embed_weight)
        total_params += embed_weight.numel()
        print(f"    embed_tokens: {tuple(embed_weight.shape)}")
        
        # 3. Attention weights (Q, K, V, O projections)
        print("  Writing attention weights...")
        for proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
            for i in range(n_layers):
                weight = self.hf_dict[f'model.language_model.layers.{i}.self_attn.{proj_name}.weight']
                serialize_fp16(out_file, weight)
                total_params += weight.numel()
        print(f"    Q/K/V/O projections: {4 * n_layers} tensors")
        
        # 4. FFN weights (gate_proj, down_proj, up_proj)
        print("  Writing FFN weights...")
        for proj_name in ['gate_proj', 'down_proj', 'up_proj']:
            for i in range(n_layers):
                weight = self.hf_dict[f'model.language_model.layers.{i}.mlp.{proj_name}.weight']
                serialize_fp16(out_file, weight)
                total_params += weight.numel()
        print(f"    Gate/Down/Up projections: {3 * n_layers} tensors")
        
        # 5. LM head (output projection)
        print("  Writing lm_head...")
        if 'lm_head.weight' in self.hf_dict:
            lm_head_weight = self.hf_dict['lm_head.weight']
            serialize_fp16(out_file, lm_head_weight)
            total_params += lm_head_weight.numel()
            print(f"    lm_head: {tuple(lm_head_weight.shape)}")
        
        # 6. Qwen3 specific: q_norm and k_norm
        print("  Writing q_norm and k_norm...")
        for i in range(n_layers):
            serialize_fp16(out_file, self.hf_dict[f'model.language_model.layers.{i}.self_attn.q_norm.weight'])
            total_params += self.hf_dict[f'model.language_model.layers.{i}.self_attn.q_norm.weight'].numel()
        
        for i in range(n_layers):
            serialize_fp16(out_file, self.hf_dict[f'model.language_model.layers.{i}.self_attn.k_norm.weight'])
            total_params += self.hf_dict[f'model.language_model.layers.{i}.self_attn.k_norm.weight'].numel()
        print(f"    q_norm/k_norm: {2 * n_layers} tensors")
        
        print(f"  Language model total parameters: {total_params:,}")


def load_hf_weights(model_path):
    """Load Qwen3-VL model weights from HuggingFace format."""
    try:
        from transformers import AutoConfig
    except ImportError:
        print("Error: transformers package required")
        return None, None

    model_path = Path(model_path)
    
    # Load config
    config_path = model_path / "config.json"
    with open(config_path, 'r') as f:
        full_config = json.load(f)
    
    print(f"Model config:")
    print(f"  Architecture: {full_config.get('architectures', ['Unknown'])}")
    print(f"  Model type: {full_config.get('model_type', 'Unknown')}")
    
    vision_config = full_config.get('vision_config', {})
    text_config = full_config.get('text_config', {})
    
    print(f"\nVision config:")
    print(f"  hidden_size: {vision_config.get('hidden_size')}")
    print(f"  depth: {vision_config.get('depth')}")
    print(f"  num_heads: {vision_config.get('num_heads')}")
    print(f"  patch_size: {vision_config.get('patch_size')}")
    print(f"  out_hidden_size: {vision_config.get('out_hidden_size')}")
    print(f"  deepstack_visual_indexes: {vision_config.get('deepstack_visual_indexes')}")
    
    print(f"\nText config:")
    print(f"  hidden_size: {text_config.get('hidden_size')}")
    print(f"  num_hidden_layers: {text_config.get('num_hidden_layers')}")
    print(f"  num_attention_heads: {text_config.get('num_attention_heads')}")
    print(f"  num_key_value_heads: {text_config.get('num_key_value_heads')}")
    print(f"  intermediate_size: {text_config.get('intermediate_size')}")
    print(f"  vocab_size: {text_config.get('vocab_size')}")
    
    config = {
        'vision_config': vision_config,
        'text_config': text_config,
        'image_token_id': full_config.get('image_token_id', 151655),
        'video_token_id': full_config.get('video_token_id', 151656),
        'vision_start_token_id': full_config.get('vision_start_token_id', 151652),
        'vision_end_token_id': full_config.get('vision_end_token_id', 151653),
    }
    
    # Load weights from safetensors
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
            print("Error: No safetensors or pytorch files found")
            return None, None
    
    # Print sample keys
    print(f"\nLoaded {len(hf_dict)} tensors")
    print("\nSample keys (vision):")
    vision_keys = [k for k in hf_dict.keys() if 'visual' in k][:10]
    for key in vision_keys:
        print(f"  {key}: {hf_dict[key].shape}")
    
    print("\nSample keys (language model):")
    lang_keys = [k for k in hf_dict.keys() if 'language_model' in k][:10]
    for key in lang_keys:
        print(f"  {key}: {hf_dict[key].shape}")
    
    # Verify key tensors exist
    print("\nVerifying key tensors...")
    required_keys = [
        'model.visual.patch_embed.proj.weight',
        'model.visual.blocks.0.attn.qkv.weight',
        'model.visual.merger.linear_fc1.weight',
        'model.language_model.embed_tokens.weight',
        'model.language_model.layers.0.self_attn.q_proj.weight',
        'model.language_model.layers.0.self_attn.q_norm.weight',
        'lm_head.weight',
    ]
    
    all_found = True
    for key in required_keys:
        if key in hf_dict:
            print(f"  ✅ {key}: {hf_dict[key].shape}")
        else:
            print(f"  ❌ {key}: NOT FOUND")
            all_found = False
    
    if not all_found:
        print("\n⚠️  Some required keys are missing!")
    
    return hf_dict, config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Qwen3-VL-8B to FP16 bin format")
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
    exporter = Qwen3VLExporter(hf_dict, config)
    exporter.export(args.filepath)
    
    print("\nDone!")
