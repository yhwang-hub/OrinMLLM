#!/usr/bin/env python3
"""Check which tokens are reachable by D2T mapping."""

import torch

def main():
    # Load EAGLE-3 model
    eagle_path = "/mnt/ssd/QwenModels/Qwen3-8B_eagle3/pytorch_model.bin"
    data = torch.load(eagle_path, map_location='cpu', weights_only=False)
    
    # Print available keys
    print(f"Keys in model: {list(data.keys())[:20]}...")
    
    # Look for d2t
    if 'd2t' in data:
        d2t = data['d2t'].numpy()
    elif 'model.d2t' in data:
        d2t = data['model.d2t'].numpy()
    else:
        # Search for d2t
        d2t_key = None
        for k in data.keys():
            if 'd2t' in k.lower():
                d2t_key = k
                break
        if d2t_key:
            d2t = data[d2t_key].numpy()
        else:
            print("No d2t found!")
            return
            
    print(f"\nFound d2t!")
    
    print(f"D2T shape: {d2t.shape}")
    print(f"D2T dtype: {d2t.dtype}")
    
    # Calculate reachable tokens: target = draft_idx + d2t[draft_idx]
    reachable = set()
    min_target = 10000000
    max_target = 0
    for draft_idx in range(len(d2t)):
        target = draft_idx + d2t[draft_idx]
        reachable.add(target)
        min_target = min(min_target, target)
        max_target = max(max_target, target)
    
    print(f"\nReachable tokens: {len(reachable)}")
    print(f"Min target: {min_target}")
    print(f"Max target: {max_target}")
    
    # Check specific tokens
    test_tokens = [59975, 151645, 2585, 25310, 100, 200, 1000, 5000, 10000, 30000, 50000]
    print("\nReachability check:")
    for t in test_tokens:
        print(f"  Token {t}: {'reachable' if t in reachable else 'NOT reachable'}")
    
    # Distribution of reachable tokens
    ranges = [
        (0, 10000),
        (10000, 30000),
        (30000, 50000),
        (50000, 100000),
        (100000, 150000),
        (150000, 160000)
    ]
    print("\nReachable token distribution:")
    for low, high in ranges:
        count = sum(1 for t in reachable if low <= t < high)
        print(f"  [{low}, {high}): {count} tokens")
    
    # Check Chinese token ranges (typically higher IDs in Qwen3)
    chinese_like = sum(1 for t in reachable if t >= 50000)
    print(f"\nTokens >= 50000 (likely CJK): {chinese_like}")
    
    # Sample some reachable tokens around 59975
    nearby = sorted([t for t in reachable if 55000 <= t <= 65000])
    print(f"\nReachable tokens near 59975 (55000-65000): {len(nearby)} tokens")
    if nearby:
        print(f"  Sample: {nearby[:10]}...")
    
    # First and last 20 reachable tokens
    sorted_reachable = sorted(reachable)
    print(f"\nFirst 20 reachable: {sorted_reachable[:20]}")
    print(f"Last 20 reachable: {sorted_reachable[-20:]}")

if __name__ == "__main__":
    main()
