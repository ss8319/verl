#!/usr/bin/env python3
"""
Check image resolutions in parquet dataset files and estimate token budget for Qwen3-VL.

Qwen3-VL-2B-Instruct uses:
- patch_size = 16
- spatial_merge_size = 2
- Token count ≈ ceil(H / (patch_size * merge_size)) * ceil(W / (patch_size * merge_size))
- Token count ≈ ceil(H / 32) * ceil(W / 32)
"""

import pandas as pd
from PIL import Image
import io
import os
import sys
from collections import defaultdict

def get_image_from_bytes(img_bytes):
    """Load image from bytes."""
    return Image.open(io.BytesIO(img_bytes))

def estimate_qwen3vl_tokens(width, height, patch_size=16, merge_size=2):
    """
    Estimate token count for Qwen3-VL-2B-Instruct.
    
    Qwen3-VL-2B uses:
    - patch_size = 16
    - merge_size = 2 (spatial merge)
    - Tokens = ceil(H/16/2) * ceil(W/16/2) = ceil(H/32) * ceil(W/32)
    """
    import math
    h_patches = math.ceil(height / (patch_size * merge_size))
    w_patches = math.ceil(width / (patch_size * merge_size))
    return h_patches * w_patches

def analyze_parquet(parquet_path, image_key='images', max_samples=None, base_dir=None):
    """Analyze image resolutions in a parquet file."""
    print(f"\n{'='*60}")
    print(f"Analyzing: {parquet_path}")
    print(f"{'='*60}")
    
    if not os.path.exists(parquet_path):
        print(f"  ERROR: File not found!")
        return None
    
    df = pd.read_parquet(parquet_path)
    print(f"  Total samples: {len(df)}")
    
    if image_key not in df.columns:
        print(f"  ERROR: Column '{image_key}' not found. Available columns: {list(df.columns)}")
        return None
    
    resolutions = []
    token_counts = []
    errors = 0
    
    samples_to_check = min(len(df), max_samples) if max_samples else len(df)
    
    import numpy as np
    
    for idx in range(samples_to_check):
        try:
            img_data = df.iloc[idx][image_key]
            
            # Handle numpy array of dicts (file paths)
            if isinstance(img_data, np.ndarray) and img_data.dtype == object:
                # Array of image dicts, take first one
                if len(img_data) == 0:
                    errors += 1
                    continue
                img_data = img_data[0]
            
            # Handle list of images (take first one)
            if isinstance(img_data, list):
                if len(img_data) == 0:
                    errors += 1
                    continue
                img_data = img_data[0]
            
            # Handle dict with 'image' key (file path)
            if isinstance(img_data, dict) and 'image' in img_data:
                img_path = img_data['image']
                # Remove file:// prefix if present
                if img_path.startswith('file://'):
                    img_path = img_path[7:]
                # Make path absolute
                if base_dir and not os.path.isabs(img_path):
                    img_path = os.path.join(base_dir, img_path)
                if not os.path.exists(img_path):
                    if errors < 3:
                        print(f"  WARNING: Image not found at idx {idx}: {img_path}")
                    errors += 1
                    continue
                img = Image.open(img_path)
                w, h = img.size
            # Handle various other image formats
            elif isinstance(img_data, bytes):
                img = get_image_from_bytes(img_data)
                w, h = img.size
            elif isinstance(img_data, Image.Image):
                img = img_data
                w, h = img.size
            elif isinstance(img_data, dict) and 'bytes' in img_data:
                img = get_image_from_bytes(img_data['bytes'])
                w, h = img.size
            elif isinstance(img_data, np.ndarray):
                # numpy array: shape is (H, W, C) or (H, W)
                if len(img_data.shape) == 3:
                    h, w = img_data.shape[:2]
                elif len(img_data.shape) == 2:
                    h, w = img_data.shape
                else:
                    errors += 1
                    continue
            else:
                if errors < 3:
                    print(f"  WARNING: Unknown image format at idx {idx}: {type(img_data)}")
                errors += 1
                continue
            resolutions.append((w, h))
            tokens = estimate_qwen3vl_tokens(w, h)
            token_counts.append(tokens)
            
        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"  WARNING: Error at idx {idx}: {e}")
    
    if not resolutions:
        print(f"  ERROR: No valid images found!")
        return None
    
    # Statistics
    widths = [r[0] for r in resolutions]
    heights = [r[1] for r in resolutions]
    
    print(f"\n  Analyzed {len(resolutions)} images ({errors} errors)")
    print(f"\n  Width statistics:")
    print(f"    Min: {min(widths)}, Max: {max(widths)}, Mean: {sum(widths)/len(widths):.0f}")
    print(f"\n  Height statistics:")
    print(f"    Min: {min(heights)}, Max: {max(heights)}, Mean: {sum(heights)/len(heights):.0f}")
    
    print(f"\n  Estimated Qwen3-VL token counts (patch=16, merge=2):")
    print(f"    Min: {min(token_counts)}, Max: {max(token_counts)}, Mean: {sum(token_counts)/len(token_counts):.0f}")
    
    # Distribution
    print(f"\n  Resolution distribution (top 10):")
    res_counts = defaultdict(int)
    for r in resolutions:
        res_counts[r] += 1
    sorted_res = sorted(res_counts.items(), key=lambda x: -x[1])[:10]
    for res, count in sorted_res:
        tokens = estimate_qwen3vl_tokens(res[0], res[1])
        print(f"    {res[0]}x{res[1]}: {count} samples (~{tokens} tokens)")
    
    # Token budget recommendations
    print(f"\n  Token budget recommendations:")
    p95_tokens = sorted(token_counts)[int(len(token_counts) * 0.95)]
    p99_tokens = sorted(token_counts)[int(len(token_counts) * 0.99)]
    print(f"    95th percentile: {p95_tokens} tokens")
    print(f"    99th percentile: {p99_tokens} tokens")
    print(f"    Max observed: {max(token_counts)} tokens")
    
    # Suggest min/max pixels
    print(f"\n  Suggested Qwen3-VL settings for ~1024 tokens max:")
    print(f"    max_pixels = 32*32*1024 = 1048576 (~1024x1024 effective)")
    print(f"    min_pixels = 32*32*4 = 4096 (~64x64 effective)")
    
    return {
        'resolutions': resolutions,
        'token_counts': token_counts,
        'max_tokens': max(token_counts),
        'p95_tokens': p95_tokens,
    }

def main():
    # Dataset paths
    dataset_base = "/fs04/scratch2/ub62/ssim0070/SSL4RL/our_datasets/dermogpt"
    # Base dir for resolving relative image paths (file://datasets/...)
    image_base = "/fs04/scratch2/ub62/ssim0070/SSL4RL"
    tasks = ['rotation', 'contrastive', 'position', 'jigsaw', 'jigsaw_small']
    splits = ['train', 'valid', 'test']
    
    all_results = {}
    
    for task in tasks:
        task_dir = os.path.join(dataset_base, task)
        if not os.path.exists(task_dir):
            print(f"\nSkipping {task} (directory not found)")
            continue
            
        for split in splits:
            parquet_path = os.path.join(task_dir, f"{split}.parquet")
            if os.path.exists(parquet_path):
                result = analyze_parquet(parquet_path, max_samples=500, base_dir=image_base)
                if result:
                    all_results[f"{task}/{split}"] = result
    
    # Summary
    if all_results:
        print(f"\n{'='*60}")
        print("OVERALL SUMMARY")
        print(f"{'='*60}")
        all_max_tokens = [r['max_tokens'] for r in all_results.values()]
        all_p95_tokens = [r['p95_tokens'] for r in all_results.values()]
        print(f"  Max tokens across all datasets: {max(all_max_tokens)}")
        print(f"  Max 95th percentile: {max(all_p95_tokens)}")
        print(f"\n  Recommendation: Set max_pixels to limit tokens to ~{max(all_p95_tokens) + 256}")

if __name__ == "__main__":
    main()
