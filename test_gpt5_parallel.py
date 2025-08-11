#!/usr/bin/env python3
"""
Test script for GPT-5 with parallel processing.
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Try to load .env file
try:
    from dotenv import load_dotenv
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        load_dotenv(env_file)
        print(f"[OK] Loaded .env file from: {env_file}")
    else:
        print("⚠️ No .env file found, using environment variables")
except ImportError:
    print("⚠️ python-dotenv not installed, using environment variables")

def test_gpt5_parallel():
    """Test GPT-5 with parallel processing."""
    print("=" * 60)
    print("GPT-5 Parallel Processing Test")
    print("=" * 60)
    
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("[ERROR] OPENAI_API_KEY not found!")
        return False
    
    print("[OK] API key found")
    
    try:
        from pipeline.genealogy_parser_parallel import ParallelGenealogyParser
        
        # Create a small test file
        test_file = Path("test_parallel_input.txt")
        test_content = """
王大明，字德明，生於康熙年間。
子：王小明，字德小，生於雍正年間。
王小明，字德小，生於雍正年間。
子：王小華，字德華，生於乾隆年間。
王小華，字德華，生於乾隆年間。
子：王小強，字德強，生於嘉慶年間。
        """
        
        test_file.write_text(test_content.strip(), encoding='utf-8')
        print(f"[OK] Created test file: {test_file}")
        
        # Create parallel parser with GPT-5
        print("\n[PROCESSING] Creating GPT-5 parallel parser...")
        parser = ParallelGenealogyParser(
            input_file=str(test_file),
            output_file="test_parallel_output.json",
            provider="openai",
            model_name="gpt-5",
            api_key=api_key,
            temperature=0.2,
            max_workers=2,  # Small number for testing
            context_size=2,  # Renamed from overlap_lines for consistency
            use_function_calling=True,  # Will be disabled automatically for GPT-5
            max_retries=2,
            retry_delay=3,
            requests_per_minute=20,  # Reduced for GPT-5
            auto_adjust_workers=True,
            slow_start=True
        )
        
        print("[PROCESSING] Running parallel parsing with GPT-5...")
        print("   (This may take 1-2 minutes due to GPT-5's longer response times)")
        
        # Run the parsing
        result = parser.parse_genealogy(chunk_size=2)  # Small chunks for testing
        
        if result:
            print(f"\n[OK] GPT-5 parallel test successful!")
            print(f"   Lines processed: {result.get('lines_processed', 0)}")
            print(f"   Records created: {result.get('records_created', 0)}")
            print(f"   API calls: {result.get('api_calls', 0)}")
            print(f"   Chunks processed: {result.get('chunks_processed', 0)}")
            print(f"   Chunks failed: {result.get('chunks_failed', 0)}")
            
            # Check if output file was created
            output_file = Path("test_parallel_output.json")
            if output_file.exists():
                print(f"   Output file: {output_file}")
                print(f"   Output size: {output_file.stat().st_size} bytes")
            else:
                print("   ⚠️ No output file created")
            
            return True
        else:
            print(f"\n[ERROR] GPT-5 parallel test failed")
            return False
            
    except Exception as e:
        print(f"\n[ERROR] GPT-5 parallel test error: {e}")
        print(f"   Error type: {type(e).__name__}")
        return False
    finally:
        # Clean up test files
        try:
            if test_file.exists():
                test_file.unlink()
                print(f"   Cleaned up test file")
        except:
            pass

def main():
    success = test_gpt5_parallel()
    
    print("\n" + "=" * 60)
    if success:
        print("[OK] GPT-5 parallel processing is working!")
        print("You can now use parallel mode with GPT-5:")
        print("   python run_pipeline.py --parallel")
    else:
        print("[ERROR] GPT-5 parallel test failed.")
    print("=" * 60)

if __name__ == "__main__":
    main()

