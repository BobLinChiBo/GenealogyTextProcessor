#!/usr/bin/env python3
"""
Test script to verify the resume functionality is working correctly.
"""

import sys
import json
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from pipeline.genealogy_parser_v3 import GenealogyParser

def test_checkpoint_loading():
    """Test that checkpoint loading works with the fixed field names."""
    
    # Check if checkpoint files exist
    checkpoint_dir = Path("data/output/checkpoints")
    checkpoint_files = list(checkpoint_dir.glob("checkpoint_*.json"))
    
    if not checkpoint_files:
        print("[WARNING] No checkpoint files found to test")
        return False
    
    # Test loading the first checkpoint
    checkpoint_file = checkpoint_files[0]
    print(f"[INFO] Testing checkpoint: {checkpoint_file}")
    
    try:
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            checkpoint_data = json.load(f)
        
        print(f"[OK] Checkpoint loaded successfully")
        print(f"   Stage: {checkpoint_data.get('stage')}")
        print(f"   Chunk index: {checkpoint_data.get('chunk_index')}")
        print(f"   Total chunks: {checkpoint_data.get('total_chunks')}")
        print(f"   API calls: {checkpoint_data.get('api_calls', 'Not found - will use 0')}")
        print(f"   Records so far: {checkpoint_data.get('totals', {}).get('records_so_far')}")
        
        # Check if the required fields exist
        if 'chunk_index' not in checkpoint_data:
            print("[ERROR] Missing 'chunk_index' field")
            return False
            
        print("\n[OK] Checkpoint structure is valid!")
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to load checkpoint: {e}")
        return False

def test_resume_initialization():
    """Test that the parser can initialize with resume=True."""
    
    print("\n[INFO] Testing parser initialization with resume=True...")
    
    try:
        # Create a parser instance with resume enabled
        # Using openai provider since it doesn't require extra packages
        parser = GenealogyParser(
            input_file="data/intermediate/cleaned_text.txt",
            output_file="data/output/test_resume.json",
            provider="openai",
            model_name="gpt-4o-mini",
            api_key="test_key",  # Just for testing initialization
            resume=True
        )
        
        print("[OK] Parser initialized successfully with resume=True")
        
        # Check if checkpoint files are detected
        if parser.checkpoint_file.exists():
            print(f"[OK] Checkpoint file found: {parser.checkpoint_file}")
        else:
            print(f"[INFO] No checkpoint file at: {parser.checkpoint_file}")
            
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to initialize parser: {e}")
        return False

def main():
    print("=" * 60)
    print("Testing Resume Functionality Fix")
    print("=" * 60)
    
    # Test 1: Check checkpoint structure
    test1_passed = test_checkpoint_loading()
    
    # Test 2: Check parser initialization
    test2_passed = test_resume_initialization()
    
    print("\n" + "=" * 60)
    if test1_passed and test2_passed:
        print("[SUCCESS] All tests passed! Resume functionality should work.")
        print("\nYou can now run:")
        print("   python run_pipeline.py --skip merge clean --resume")
    else:
        print("[WARNING] Some tests failed. Check the errors above.")
    print("=" * 60)

if __name__ == "__main__":
    main()