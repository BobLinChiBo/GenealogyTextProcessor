#!/usr/bin/env python3
"""
Cleanup script to remove stuck checkpoints and restart the pipeline.
"""

import os
import shutil
from pathlib import Path

def cleanup_checkpoints():
    """Remove all checkpoint files to start fresh."""
    print("Cleaning up stuck checkpoints...")
    
    # Remove checkpoint files
    checkpoint_dir = Path("data/output/checkpoints")
    if checkpoint_dir.exists():
        for file in checkpoint_dir.glob("checkpoint_*.json"):
            try:
                file.unlink()
                print(f"   Removed: {file}")
            except Exception as e:
                print(f"   Failed to remove {file}: {e}")
        
        for file in checkpoint_dir.glob("records_*.json"):
            try:
                file.unlink()
                print(f"   Removed: {file}")
            except Exception as e:
                print(f"   Failed to remove {file}: {e}")
    
    # Remove intermediate chunk files
    output_dir = Path("data/output")
    if output_dir.exists():
        for file in output_dir.glob("genealogy_data_*_chunk_*.json"):
            try:
                file.unlink()
                print(f"   Removed: {file}")
            except Exception as e:
                print(f"   Failed to remove {file}: {e}")
    
    print("[OK] Cleanup complete!")

def test_gpt5_fix():
    """Test the GPT-5 fix before running the pipeline."""
    print("\nTesting GPT-5 fix...")
    
    try:
        from test_gpt5_fix import test_gpt5_connection, test_gpt5_without_function_calling
        
        # Test both methods
        success1 = test_gpt5_connection()
        success2 = test_gpt5_without_function_calling()
        
        if success1 or success2:
            print("[OK] GPT-5 fix test passed!")
            return True
        else:
            print("[ERROR] GPT-5 fix test failed!")
            return False
            
    except Exception as e:
        print(f"[ERROR] Error testing GPT-5 fix: {e}")
        return False

def main():
    print("=" * 60)
    print("Pipeline Cleanup and Restart")
    print("=" * 60)
    
    # Step 1: Cleanup
    cleanup_checkpoints()
    
    # Step 2: Test GPT-5 fix
    if test_gpt5_fix():
        print("\n[OK] Ready to restart pipeline!")
        print("\nTo restart the pipeline, run:")
        print("   python run_pipeline.py --skip merge clean")
        print("\nOr to run the full pipeline:")
        print("   python run_pipeline.py")
    else:
        print("\n[ERROR] GPT-5 fix test failed!")
        print("Consider switching to a different model:")
        print("   - gpt-4o (recommended)")
        print("   - gpt-4o-mini (faster)")
        print("   - gpt-3.5-turbo (cheapest)")
        
        # Update config to use gpt-4o instead
        print("\nUpdating config to use gpt-4o...")
        try:
            config_file = Path("config/config.yaml")
            if config_file.exists():
                content = config_file.read_text(encoding='utf-8')
                content = content.replace('model: "gpt-5"', 'model: "gpt-4o"')
                content = content.replace('use_function_calling: false', 'use_function_calling: true')
                content = content.replace('chunk_size: 5', 'chunk_size: 10')
                content = content.replace('context_size: 5', 'context_size: 10')
                config_file.write_text(content, encoding='utf-8')
                print("[OK] Updated config to use gpt-4o")
        except Exception as e:
            print(f"[ERROR] Failed to update config: {e}")
    
    print("=" * 60)

if __name__ == "__main__":
    main()

