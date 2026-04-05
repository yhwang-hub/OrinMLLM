"""
This script exports Qwen3.5-9B model to FP16 .bin format for KuiperLLama.

Usage:
    cd /home/wangyh/OrinMLLM && \
    python tools/export_qwen3-5-9B-fp16.py /mnt/ssd/QwenModels/Qwen3.5-9B-fp16.bin \
        --dtype=fp16 --hf=/mnt/ssd/QwenModels/Qwen3.5-9B

This creates a FP16 model file for optimized inference on CUDA devices.

Model Structure (Qwen3.5-9B):
==============================

1. Vision Encoder (ViT) - same structure as Qwen3-VL minus deepstack:
   - patch_embed: Conv3d for patch embedding
   - pos_embed: Position embedding (num_position_embeddings x hidden_size)
   - blocks: 27 transformer blocks with:
     - norm1, norm2: LayerNorm with bias
     - attn: QKV fused attention with bias + output projection with bias
     - mlp: linear_fc1 + GELU + linear_fc2 (all with bias)
   - merger: Projection from vision hidden to LLM hidden
   - NO deepstack mergers (deepstack_visual_indexes = [])

2. Language Model (Hybrid Linear+Full Attention):
   - embed_tokens: Token embeddings [248320, 4096]
   - 32 Transformer layers:
     - 24 linear attention layers (layers 0,1,2,4,5,6,...,28,29,30):
       - input_layernorm, post_attention_layernorm: RMSNorm
       - linear_attn: in_proj_qkv, in_proj_z, conv1d, A_log, dt_bias,
                       in_proj_a, in_proj_b, norm, out_proj
       - mlp: gate_proj, up_proj, down_proj
     - 8 full attention layers (layers 3,7,11,15,19,23,27,31):
       - input_layernorm, post_attention_layernorm: RMSNorm
       - self_attn: q_proj (includes output gate), k_proj, v_proj, o_proj + q_norm, k_norm
       - mlp: gate_proj, up_proj, down_proj
   - norm: Final RMSNorm
   - lm_head: Output projection [248320, 4096]

3. Special tokens:
   - image_token_id: 248056
   - video_token_id: 248057
   - vision_start_token_id: 248053
   - vision_end_token_id: 248054
   - eos_token_id: 248044

Binary Layout:
==============
Header (512 bytes):
  [0:4]     Magic: 0x71333539 ("q359")
  [4:8]     Version: 1
  [8:48]    Vision config (10 int32 + 3 unused int32 for deepstack padding)
  [48:92]   Text config (8 int32 + 2 float32 + 4 additional ints for hybrid config)
  [92:112]  Special tokens (5 int32)
  [112:116] Flags (has_lm_head)
  [116:120] num_full_attn_layers (8)
  [120:152] full_attn_layer_indices (8 int32)
  [152:160] linear_attn config (conv_kernel_dim=4, linear_key_head_dim=128)
  [160:168] linear_attn config (linear_num_key_heads=16, linear_num_value_heads=32)
  [168:176] linear_attn config (linear_value_head_dim=128, partial_rotary_factor as float32)
  [176:180] mrope_interleaved flag
  [180:512] padding

Vision Weights (same order as Qwen3-VL):
  patch_embed_weight, patch_embed_bias
  pos_embed_weight
  27 blocks: norm1_w, norm1_b, norm2_w, norm2_b, qkv_w, qkv_b, proj_w, proj_b,
             mlp_fc1_w, mlp_fc1_b, mlp_fc2_w, mlp_fc2_b
  merger: norm_w, norm_b, fc1_w, fc1_b, fc2_w, fc2_b

LLM Weights (per-type grouping for efficient mmap loading):
  -- RMSNorm weights --
  input_layernorm for all 32 layers
  post_attention_layernorm for all 32 layers
  final_norm
  -- Token embeddings --
  embed_tokens [248320, 4096]
  -- Full attention weights (8 layers) --
  q_proj for full_attn layers
  k_proj for full_attn layers
  v_proj for full_attn layers
  o_proj for full_attn layers
  q_norm for full_attn layers
  k_norm for full_attn layers
  -- Linear attention weights (24 layers) --
  in_proj_qkv for linear_attn layers
  in_proj_z for linear_attn layers
  in_proj_a for linear_attn layers
  in_proj_b for linear_attn layers
  A_log for linear_attn layers (float32!)
  dt_bias for linear_attn layers
  conv1d for linear_attn layers
  norm for linear_attn layers (float32!)
  out_proj for linear_attn layers
  -- FFN weights (all 32 layers) --
  gate_proj for all layers
  down_proj for all layers
  up_proj for all layers
  -- LM Head --
  lm_head [248320, 4096]
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


class Qwen35Exporter:
    """Exports Qwen3.5-9B model to binary format"""
    
    # Magic number for Qwen3.5 model: "q359"
    MAGIC = 0x71333539
    VERSION = 1
    
    # Layer type classification
    FULL_ATTN_LAYERS = [3, 7, 11, 15, 19, 23, 27, 31]
    LINEAR_ATTN_LAYERS = [i for i in range(32) if i not in [3, 7, 11, 15, 19, 23, 27, 31]]
    
    def __init__(self, hf_dict, config):
        self.hf_dict = hf_dict
        self.config = config
        
        # Build layer type lists from config
        layer_types = config['text_config'].get('layer_types', [])
        if layer_types:
            self.FULL_ATTN_LAYERS = [i for i, t in enumerate(layer_types) if t == 'full_attention']
            self.LINEAR_ATTN_LAYERS = [i for i, t in enumerate(layer_types) if t == 'linear_attention']
        
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
        vision_config = self.config['vision_config']
        text_config = self.config['text_config']
        
        # 1) Magic number (4 bytes)
        serialize_uint32(out_file, self.MAGIC)
        
        # 2) Version (4 bytes)
        serialize_int32(out_file, self.VERSION)
        
        # 3) Vision config (52 bytes = 13 int32)
        serialize_int32(out_file, vision_config['hidden_size'])        # 1152
        serialize_int32(out_file, vision_config['intermediate_size'])  # 4304
        serialize_int32(out_file, vision_config['num_heads'])          # 16
        serialize_int32(out_file, vision_config['depth'])              # 27
        serialize_int32(out_file, vision_config['patch_size'])         # 16
        serialize_int32(out_file, vision_config.get('temporal_patch_size', 2))
        serialize_int32(out_file, vision_config.get('in_channels', 3))
        serialize_int32(out_file, vision_config.get('spatial_merge_size', 2))
        serialize_int32(out_file, vision_config.get('out_hidden_size', 4096))
        serialize_int32(out_file, vision_config.get('num_position_embeddings', 2304))
        
        # deepstack_visual_indexes: 3 padding zeros (no deepstack in Qwen3.5)
        deepstack_indexes = vision_config.get('deepstack_visual_indexes', [])
        for i in range(3):
            if i < len(deepstack_indexes):
                serialize_int32(out_file, deepstack_indexes[i])
            else:
                serialize_int32(out_file, 0)
        
        # 4) Text/LLM config (44 bytes = 8 int32 + 2 float32)
        serialize_int32(out_file, text_config['hidden_size'])          # 4096
        serialize_int32(out_file, text_config['intermediate_size'])    # 12288
        serialize_int32(out_file, text_config['num_hidden_layers'])    # 32
        serialize_int32(out_file, text_config['num_attention_heads'])  # 16
        serialize_int32(out_file, text_config['num_key_value_heads'])  # 4
        serialize_int32(out_file, text_config['vocab_size'])           # 248320
        serialize_int32(out_file, text_config.get('max_position_embeddings', 262144))
        serialize_int32(out_file, text_config.get('head_dim', 256))
        serialize_float32(out_file, text_config.get('rms_norm_eps', 1e-6))
        rope_params = text_config.get('rope_parameters', {})
        serialize_float32(out_file, rope_params.get('rope_theta', 10000000))
        
        # 5) Special tokens (20 bytes = 5 int32)
        serialize_int32(out_file, self.config.get('image_token_id', 248056))
        serialize_int32(out_file, self.config.get('video_token_id', 248057))
        serialize_int32(out_file, self.config.get('vision_start_token_id', 248053))
        serialize_int32(out_file, self.config.get('vision_end_token_id', 248054))
        serialize_int32(out_file, text_config.get('eos_token_id', 248044))
        
        # 6) Flags (4 bytes)
        shared_classifier = self.config.get('tie_word_embeddings', False)
        serialize_int32(out_file, int(not shared_classifier))  # has_lm_head flag
        
        # 7) Hybrid attention config (new for Qwen3.5)
        # Number of full attention layers and their indices
        serialize_int32(out_file, len(self.FULL_ATTN_LAYERS))
        for idx in self.FULL_ATTN_LAYERS:
            serialize_int32(out_file, idx)
        # Pad to 8 full_attn indices (max)
        for _ in range(8 - len(self.FULL_ATTN_LAYERS)):
            serialize_int32(out_file, -1)
        
        # Linear attention config
        serialize_int32(out_file, text_config.get('linear_conv_kernel_dim', 4))
        serialize_int32(out_file, text_config.get('linear_key_head_dim', 128))
        serialize_int32(out_file, text_config.get('linear_num_key_heads', 16))
        serialize_int32(out_file, text_config.get('linear_num_value_heads', 32))
        serialize_int32(out_file, text_config.get('linear_value_head_dim', 128))
        serialize_float32(out_file, rope_params.get('partial_rotary_factor', 0.25))
        
        # MRoPE config
        mrope_interleaved = rope_params.get('mrope_interleaved', True)
        serialize_int32(out_file, int(mrope_interleaved))
        mrope_section = rope_params.get('mrope_section', [11, 11, 10])
        for s in mrope_section:
            serialize_int32(out_file, s)
        
        # Full attention interval
        serialize_int32(out_file, text_config.get('full_attention_interval', 4))
        
        # attn_output_gate flag
        serialize_int32(out_file, int(text_config.get('attn_output_gate', True)))
        
        # Number of deepstack mergers (0 for Qwen3.5)
        serialize_int32(out_file, len(vision_config.get('deepstack_visual_indexes', [])))
        
        # Pad to 512 bytes
        current_pos = out_file.tell()
        pad = 512 - current_pos
        assert pad >= 0, f"Header too large: {current_pos} bytes"
        out_file.write(b'\0' * pad)
        
        print(f"Header written: {out_file.tell()} bytes")
        print(f"  Vision: hidden={vision_config['hidden_size']}, depth={vision_config['depth']}, patch={vision_config['patch_size']}")
        print(f"  LLM: dim={text_config['hidden_size']}, layers={text_config['num_hidden_layers']}, heads={text_config['num_attention_heads']}")
        print(f"  Vocab: {text_config['vocab_size']}, shared_classifier={shared_classifier}")
        print(f"  Full attn layers: {self.FULL_ATTN_LAYERS}")
        print(f"  Linear attn layers: {self.LINEAR_ATTN_LAYERS}")
        
    def _write_vision_encoder(self, out_file):
        """Write Vision Encoder (ViT) weights"""
        print("\n=== Writing Vision Encoder ===")
        
        vision_config = self.config['vision_config']
        vit_depth = vision_config['depth']  # 27
        
        total_params = 0
        
        # 1. Patch embedding: Conv3d weight and bias
        print("  Writing patch_embed...")
        weight = self.hf_dict['model.visual.patch_embed.proj.weight']
        bias = self.hf_dict['model.visual.patch_embed.proj.bias']
        serialize_fp16(out_file, weight)
        serialize_fp16(out_file, bias)
        total_params += weight.numel() + bias.numel()
        print(f"    patch_embed.weight: {tuple(weight.shape)}")
        print(f"    patch_embed.bias: {tuple(bias.shape)}")
        
        # 2. Position embedding
        if 'model.visual.pos_embed.weight' in self.hf_dict:
            print("  Writing pos_embed...")
            pos_embed = self.hf_dict['model.visual.pos_embed.weight']
            serialize_fp16(out_file, pos_embed)
            total_params += pos_embed.numel()
            print(f"    pos_embed: {tuple(pos_embed.shape)}")
        
        # 3. Transformer blocks (same as Qwen3-VL)
        print(f"  Writing {vit_depth} transformer blocks...")
        for i in range(vit_depth):
            prefix = f'model.visual.blocks.{i}'
            
            serialize_fp16(out_file, self.hf_dict[f'{prefix}.norm1.weight'])
            serialize_fp16(out_file, self.hf_dict[f'{prefix}.norm1.bias'])
            serialize_fp16(out_file, self.hf_dict[f'{prefix}.norm2.weight'])
            serialize_fp16(out_file, self.hf_dict[f'{prefix}.norm2.bias'])
            serialize_fp16(out_file, self.hf_dict[f'{prefix}.attn.qkv.weight'])
            serialize_fp16(out_file, self.hf_dict[f'{prefix}.attn.qkv.bias'])
            serialize_fp16(out_file, self.hf_dict[f'{prefix}.attn.proj.weight'])
            serialize_fp16(out_file, self.hf_dict[f'{prefix}.attn.proj.bias'])
            serialize_fp16(out_file, self.hf_dict[f'{prefix}.mlp.linear_fc1.weight'])
            serialize_fp16(out_file, self.hf_dict[f'{prefix}.mlp.linear_fc1.bias'])
            serialize_fp16(out_file, self.hf_dict[f'{prefix}.mlp.linear_fc2.weight'])
            serialize_fp16(out_file, self.hf_dict[f'{prefix}.mlp.linear_fc2.bias'])
            
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
        
        # 5. NO deepstack mergers for Qwen3.5
        print("  No deepstack mergers (Qwen3.5)")
        
        print(f"  Vision encoder total parameters: {total_params:,}")
        
    def _write_language_model(self, out_file):
        """Write Language Model (LLM) weights"""
        print("\n=== Writing Language Model ===")
        
        text_config = self.config['text_config']
        n_layers = text_config['num_hidden_layers']  # 32
        
        total_params = 0
        
        # 1. RMSNorm weights (input_layernorm for all layers)
        print("  Writing input_layernorm weights...")
        for i in range(n_layers):
            w = self.hf_dict[f'model.language_model.layers.{i}.input_layernorm.weight']
            serialize_fp16(out_file, w)
            total_params += w.numel()
        
        # 2. RMSNorm weights (post_attention_layernorm for all layers)
        print("  Writing post_attention_layernorm weights...")
        for i in range(n_layers):
            w = self.hf_dict[f'model.language_model.layers.{i}.post_attention_layernorm.weight']
            serialize_fp16(out_file, w)
            total_params += w.numel()
        
        # 3. Final norm
        print("  Writing final norm...")
        w = self.hf_dict['model.language_model.norm.weight']
        serialize_fp16(out_file, w)
        total_params += w.numel()
        print(f"    RMSNorm: {2 * n_layers + 1} tensors")
        
        # 4. Token embeddings
        print("  Writing token embeddings...")
        embed_weight = self.hf_dict['model.language_model.embed_tokens.weight']
        serialize_fp16(out_file, embed_weight)
        total_params += embed_weight.numel()
        print(f"    embed_tokens: {tuple(embed_weight.shape)}")
        
        # 5. Full attention weights (8 layers: 3,7,11,15,19,23,27,31)
        print(f"  Writing full attention weights ({len(self.FULL_ATTN_LAYERS)} layers)...")
        
        # q_proj for full_attn layers (includes output gate: [2*q_dim, dim])
        for i in self.FULL_ATTN_LAYERS:
            w = self.hf_dict[f'model.language_model.layers.{i}.self_attn.q_proj.weight']
            serialize_fp16(out_file, w)
            total_params += w.numel()
        print(f"    q_proj: {len(self.FULL_ATTN_LAYERS)} tensors")
        
        # k_proj for full_attn layers
        for i in self.FULL_ATTN_LAYERS:
            w = self.hf_dict[f'model.language_model.layers.{i}.self_attn.k_proj.weight']
            serialize_fp16(out_file, w)
            total_params += w.numel()
        print(f"    k_proj: {len(self.FULL_ATTN_LAYERS)} tensors")
        
        # v_proj for full_attn layers
        for i in self.FULL_ATTN_LAYERS:
            w = self.hf_dict[f'model.language_model.layers.{i}.self_attn.v_proj.weight']
            serialize_fp16(out_file, w)
            total_params += w.numel()
        print(f"    v_proj: {len(self.FULL_ATTN_LAYERS)} tensors")
        
        # o_proj for full_attn layers
        for i in self.FULL_ATTN_LAYERS:
            w = self.hf_dict[f'model.language_model.layers.{i}.self_attn.o_proj.weight']
            serialize_fp16(out_file, w)
            total_params += w.numel()
        print(f"    o_proj: {len(self.FULL_ATTN_LAYERS)} tensors")
        
        # q_norm for full_attn layers
        for i in self.FULL_ATTN_LAYERS:
            w = self.hf_dict[f'model.language_model.layers.{i}.self_attn.q_norm.weight']
            serialize_fp16(out_file, w)
            total_params += w.numel()
        print(f"    q_norm: {len(self.FULL_ATTN_LAYERS)} tensors")
        
        # k_norm for full_attn layers
        for i in self.FULL_ATTN_LAYERS:
            w = self.hf_dict[f'model.language_model.layers.{i}.self_attn.k_norm.weight']
            serialize_fp16(out_file, w)
            total_params += w.numel()
        print(f"    k_norm: {len(self.FULL_ATTN_LAYERS)} tensors")
        
        # 6. Linear attention weights (24 layers)
        print(f"  Writing linear attention weights ({len(self.LINEAR_ATTN_LAYERS)} layers)...")
        
        # in_proj_qkv for linear_attn layers
        for i in self.LINEAR_ATTN_LAYERS:
            w = self.hf_dict[f'model.language_model.layers.{i}.linear_attn.in_proj_qkv.weight']
            serialize_fp16(out_file, w)
            total_params += w.numel()
        print(f"    in_proj_qkv: {len(self.LINEAR_ATTN_LAYERS)} tensors")
        
        # in_proj_z for linear_attn layers
        for i in self.LINEAR_ATTN_LAYERS:
            w = self.hf_dict[f'model.language_model.layers.{i}.linear_attn.in_proj_z.weight']
            serialize_fp16(out_file, w)
            total_params += w.numel()
        print(f"    in_proj_z: {len(self.LINEAR_ATTN_LAYERS)} tensors")
        
        # in_proj_a for linear_attn layers
        for i in self.LINEAR_ATTN_LAYERS:
            w = self.hf_dict[f'model.language_model.layers.{i}.linear_attn.in_proj_a.weight']
            serialize_fp16(out_file, w)
            total_params += w.numel()
        print(f"    in_proj_a: {len(self.LINEAR_ATTN_LAYERS)} tensors")
        
        # in_proj_b for linear_attn layers
        for i in self.LINEAR_ATTN_LAYERS:
            w = self.hf_dict[f'model.language_model.layers.{i}.linear_attn.in_proj_b.weight']
            serialize_fp16(out_file, w)
            total_params += w.numel()
        print(f"    in_proj_b: {len(self.LINEAR_ATTN_LAYERS)} tensors")
        
        # A_log for linear_attn layers (FLOAT32!)
        for i in self.LINEAR_ATTN_LAYERS:
            w = self.hf_dict[f'model.language_model.layers.{i}.linear_attn.A_log']
            serialize_fp32(out_file, w)
            total_params += w.numel()
        print(f"    A_log (fp32): {len(self.LINEAR_ATTN_LAYERS)} tensors")
        
        # dt_bias for linear_attn layers
        for i in self.LINEAR_ATTN_LAYERS:
            w = self.hf_dict[f'model.language_model.layers.{i}.linear_attn.dt_bias']
            serialize_fp16(out_file, w)
            total_params += w.numel()
        print(f"    dt_bias: {len(self.LINEAR_ATTN_LAYERS)} tensors")
        
        # conv1d for linear_attn layers
        for i in self.LINEAR_ATTN_LAYERS:
            w = self.hf_dict[f'model.language_model.layers.{i}.linear_attn.conv1d.weight']
            serialize_fp16(out_file, w)
            total_params += w.numel()
        print(f"    conv1d: {len(self.LINEAR_ATTN_LAYERS)} tensors")
        
        # norm for linear_attn layers (FLOAT32!)
        for i in self.LINEAR_ATTN_LAYERS:
            w = self.hf_dict[f'model.language_model.layers.{i}.linear_attn.norm.weight']
            serialize_fp32(out_file, w)
            total_params += w.numel()
        print(f"    norm (fp32): {len(self.LINEAR_ATTN_LAYERS)} tensors")
        
        # out_proj for linear_attn layers
        for i in self.LINEAR_ATTN_LAYERS:
            w = self.hf_dict[f'model.language_model.layers.{i}.linear_attn.out_proj.weight']
            serialize_fp16(out_file, w)
            total_params += w.numel()
        print(f"    out_proj: {len(self.LINEAR_ATTN_LAYERS)} tensors")
        
        # 7. FFN weights (all 32 layers)
        print("  Writing FFN weights...")
        for proj_name in ['gate_proj', 'down_proj', 'up_proj']:
            for i in range(n_layers):
                w = self.hf_dict[f'model.language_model.layers.{i}.mlp.{proj_name}.weight']
                serialize_fp16(out_file, w)
                total_params += w.numel()
        print(f"    Gate/Down/Up projections: {3 * n_layers} tensors")
        
        # 8. LM head
        print("  Writing lm_head...")
        if 'lm_head.weight' in self.hf_dict:
            lm_head_weight = self.hf_dict['lm_head.weight']
            serialize_fp16(out_file, lm_head_weight)
            total_params += lm_head_weight.numel()
            print(f"    lm_head: {tuple(lm_head_weight.shape)}")
        
        print(f"  Language model total parameters: {total_params:,}")


def load_hf_weights(model_path):
    """Load Qwen3.5-9B model weights from HuggingFace format."""
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
    print(f"  head_dim: {text_config.get('head_dim')}")
    print(f"  layer_types: {text_config.get('layer_types')}")
    print(f"  linear_conv_kernel_dim: {text_config.get('linear_conv_kernel_dim')}")
    print(f"  attn_output_gate: {text_config.get('attn_output_gate')}")
    
    config = {
        'vision_config': vision_config,
        'text_config': text_config,
        'image_token_id': full_config.get('image_token_id', 248056),
        'video_token_id': full_config.get('video_token_id', 248057),
        'vision_start_token_id': full_config.get('vision_start_token_id', 248053),
        'vision_end_token_id': full_config.get('vision_end_token_id', 248054),
        'tie_word_embeddings': full_config.get('tie_word_embeddings', False),
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
    
    print(f"\nLoaded {len(hf_dict)} tensors")
    
    # Print sample keys
    print("\nSample keys (vision):")
    vision_keys = sorted([k for k in hf_dict.keys() if 'visual' in k])[:10]
    for key in vision_keys:
        print(f"  {key}: {hf_dict[key].shape} {hf_dict[key].dtype}")
    
    print("\nSample keys (language model):")
    lang_keys = sorted([k for k in hf_dict.keys() if 'language_model' in k])[:15]
    for key in lang_keys:
        print(f"  {key}: {hf_dict[key].shape} {hf_dict[key].dtype}")
    
    print("\nSample keys (linear_attn):")
    lin_keys = sorted([k for k in hf_dict.keys() if 'linear_attn' in k and 'layers.0.' in k])
    for key in lin_keys:
        print(f"  {key}: {hf_dict[key].shape} {hf_dict[key].dtype}")
    
    print("\nSample keys (self_attn):")
    sa_keys = sorted([k for k in hf_dict.keys() if 'self_attn' in k and 'layers.3.' in k])
    for key in sa_keys:
        print(f"  {key}: {hf_dict[key].shape} {hf_dict[key].dtype}")
    
    # Verify key tensors exist
    print("\nVerifying key tensors...")
    required_keys = [
        'model.visual.patch_embed.proj.weight',
        'model.visual.blocks.0.attn.qkv.weight',
        'model.visual.merger.linear_fc1.weight',
        'model.language_model.embed_tokens.weight',
        # Full attention layer (layer 3)
        'model.language_model.layers.3.self_attn.q_proj.weight',
        'model.language_model.layers.3.self_attn.q_norm.weight',
        # Linear attention layer (layer 0)
        'model.language_model.layers.0.linear_attn.in_proj_qkv.weight',
        'model.language_model.layers.0.linear_attn.A_log',
        'model.language_model.layers.0.linear_attn.conv1d.weight',
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
    parser = argparse.ArgumentParser(description="Export Qwen3.5-9B to FP16 bin format")
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
    exporter = Qwen35Exporter(hf_dict, config)
    exporter.export(args.filepath)
    
    print("\nDone!")
