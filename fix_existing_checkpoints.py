#!/usr/bin/env python3
"""
Fix existing checkpoint files to include the api_calls field.
"""

import json
from pathlib import Path

def fix_checkpoints():
    """Add missing api_calls field to existing checkpoints."""
    
    checkpoint_dir = Path("data/output/checkpoints")
    
    # Find all checkpoint files (not records files)
    checkpoint_files = list(checkpoint_dir.glob("checkpoint_*.json"))
    
    if not checkpoint_files:
        print("[INFO] No checkpoint files found")
        return
    
    for checkpoint_file in checkpoint_files:
        print(f"[INFO] Checking {checkpoint_file.name}...")
        
        try:
            # Load checkpoint
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Check if api_calls field is missing
            if 'api_calls' not in data:
                # Estimate api_calls based on chunk_index
                # Each chunk typically makes 1 API call
                estimated_calls = data.get('chunk_index', 0) + 1
                data['api_calls'] = estimated_calls
                
                # Save updated checkpoint
                with open(checkpoint_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                
                print(f"   [FIXED] Added api_calls={estimated_calls}")
            else:
                print(f"   [OK] api_calls field already exists: {data['api_calls']}")
                
        except Exception as e:
            print(f"   [ERROR] Failed to process: {e}")

if __name__ == "__main__":
    print("=" * 60)
    print("Fixing Existing Checkpoints")
    print("=" * 60)
    fix_checkpoints()
    print("=" * 60)
    print("[DONE] Checkpoint fix complete")