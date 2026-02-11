"""
Script for Converting Qwen3 .pth weights to FP16 Binary Format
Author: Bound
Date: May 30, 2025
Version: 1.1
"""
import struct
import torch
import argparse
import numpy as np
import os


def serialize_fp16(file, tensor):
    """Writes one fp16 tensor to file that is open in wb mode"""
    d = tensor.detach().cpu().view(-1).to(torch.float16).numpy()
    b = struct.pack(f'{len(d)}e', *d)  # 'e' is for float16 (half precision)
    file.write(b)


def parse_args():
    parser = argparse.ArgumentParser(description="Convert Qwen3 model to FP16 binary format.")
    parser.add_argument("-n", "--model_name", type=str, 
                        default="/home/lvf6/disk/wyh/QwenModels/Qwen3-8B",
                        help="HuggingFace model name or local path")
    parser.add_argument("-o", "--output", type=str, 
                        default="/home/lvf6/disk/wyh/KuiperLLama/Qwen3-8B-fp16.bin",
                        help="Output binary file path")
    return parser.parse_args()


def main():
    args = parse_args()
    
    print(f"Loading model from: {args.model_name}")
    
    # Load model using HuggingFace transformers
    from transformers import AutoModelForCausalLM
    
    # Load model on CPU to get all weights
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="cpu",  # Load on CPU to avoid meta tensors
        low_cpu_mem_usage=False
    )
    
    config = model.config
    
    # Extract model configuration
    hidden_size = config.hidden_size
    intermediate_size = config.intermediate_size
    num_hidden_layers = config.num_hidden_layers
    num_attention_heads = config.num_attention_heads
    num_key_value_heads = config.num_key_value_heads
    vocab_size = config.vocab_size
    max_position_embeddings = config.max_position_embeddings
    head_dim = getattr(config, 'head_dim', hidden_size // num_attention_heads)
    
    dim = num_attention_heads * head_dim
    
    print(f"Model configuration:")
    print(f"  dim (hidden_size): {dim}")
    print(f"  hidden_dim: {hidden_size}")
    print(f"  n_layers: {num_hidden_layers}")
    print(f"  n_heads: {num_attention_heads}")
    print(f"  n_kv_heads: {num_key_value_heads}")
    print(f"  vocab_size: {vocab_size}")
    print(f"  max_seq_len: {max_position_embeddings}")
    print(f"  intermediate_size: {intermediate_size}")
    
    # Build weights list in the correct order
    weights = [
        # 1. input_layernorm weights for all layers
        *[layer.input_layernorm.weight for layer in model.model.layers],
        # 2. post_attention_layernorm weights for all layers
        *[layer.post_attention_layernorm.weight for layer in model.model.layers],
        # 3. final norm weight
        model.model.norm.weight,
        # 4. embed_tokens weight
        model.model.embed_tokens.weight,
        # 5. q_proj weights for all layers
        *[layer.self_attn.q_proj.weight for layer in model.model.layers],
        # 6. q_norm weights for all layers
        *[layer.self_attn.q_norm.weight for layer in model.model.layers],
        # 7. k_proj weights for all layers
        *[layer.self_attn.k_proj.weight for layer in model.model.layers],
        # 8. k_norm weights for all layers
        *[layer.self_attn.k_norm.weight for layer in model.model.layers],
        # 9. v_proj weights for all layers
        *[layer.self_attn.v_proj.weight for layer in model.model.layers],
        # 10. o_proj weights for all layers
        *[layer.self_attn.o_proj.weight for layer in model.model.layers],
        # 11. gate_proj weights for all layers
        *[layer.mlp.gate_proj.weight for layer in model.model.layers],
        # 12. down_proj weights for all layers
        *[layer.mlp.down_proj.weight for layer in model.model.layers],
        # 13. up_proj weights for all layers
        *[layer.mlp.up_proj.weight for layer in model.model.layers],
        # 14. lm_head weight
        model.lm_head.weight
    ]
    
    print(f"\nTotal weight tensors: {len(weights)}")
    
    # Calculate expected file size
    total_params = sum(w.numel() for w in weights)
    header_size = 8 * 4  # 8 integers, 4 bytes each
    data_size = total_params * 2  # fp16 = 2 bytes per element
    expected_size = header_size + data_size
    print(f"Total parameters: {total_params:,}")
    print(f"Expected file size: {expected_size:,} bytes ({expected_size / (1024**3):.2f} GB)")
    
    # Export to binary file
    file_path = args.output
    out_file = open(file_path, 'wb')
    
    # Write header: 8 integers
    header = struct.pack('iiiiiiii', dim, hidden_size, num_hidden_layers, num_attention_heads,
                         num_key_value_heads, vocab_size, max_position_embeddings, intermediate_size)
    out_file.write(header)
    
    # Write weights in fp16 format
    print("\nWriting weights to binary file...")
    for idx, w in enumerate(weights):
        serialize_fp16(out_file, w)
        if (idx + 1) % 50 == 0:
            print(f"  Progress: {idx + 1}/{len(weights)} tensors written")
    
    out_file.close()
    
    # Verify file size
    actual_size = os.path.getsize(file_path)
    print(f"\n✅ Wrote {file_path}")
    print(f"   Actual file size: {actual_size:,} bytes ({actual_size / (1024**3):.2f} GB)")
    print(f"   Expected size: {expected_size:,} bytes")
    
    if actual_size == expected_size:
        print("   ✅ File size verification: PASSED")
    else:
        print(f"   ❌ File size mismatch! Difference: {actual_size - expected_size} bytes")
    
    # Clean up
    del model
    del weights
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


if __name__ == "__main__":
    main()
