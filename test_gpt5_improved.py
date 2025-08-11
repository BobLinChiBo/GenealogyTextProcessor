#!/usr/bin/env python3
"""
Test script to verify improved GPT-5 handling.
"""

import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from llm import get_provider
from prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE

def test_gpt5_with_function_calling():
    """Test GPT-5 with function calling (now enabled)."""
    print("="*60)
    print("Testing GPT-5 with function calling (improved handling)...")
    print("="*60)
    
    try:
        # Get API key from environment
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("[ERROR] OPENAI_API_KEY not found in environment")
            return False
        
        # Create provider
        provider = get_provider(
            provider_name="openai",
            model_name="gpt-5",
            api_key=api_key,
            temperature=0.2,
            max_retries=2,
            retry_delay=5
        )
        
        # Test with genealogy text
        test_text = """
        王大明，字德明，號東山，生於康熙十年。
        配李氏，生三子。
        長子：王小明，字德小，生於雍正五年。
        次子：王中明，字德中，生於雍正八年。
        三子：王少明，字德少，生於乾隆元年。
        """
        
        print("\nMaking API call to GPT-5 with function calling...")
        response = provider.parse_genealogy(
            text=test_text,
            system_prompt=SYSTEM_PROMPT,
            user_prompt_template=USER_PROMPT_TEMPLATE,
            use_function_calling=True  # Now allowed for GPT-5
        )
        
        if response.success:
            print(f"\n[SUCCESS] GPT-5 with function calling worked!")
            print(f"Records found: {len(response.records)}")
            if response.records:
                print("\nSample record:")
                try:
                    print(f"  {response.records[0]}")
                except UnicodeEncodeError:
                    # Windows console encoding issue - just show we got data
                    print(f"  [Record with Chinese characters - {len(str(response.records[0]))} chars]")
            if response.usage:
                print(f"\nTokens used: {response.usage.get('total_tokens', 'unknown')}")
            return True
        else:
            print(f"\n[WARNING] GPT-5 call returned error: {response.error}")
            print("The fallback mechanisms should have handled this...")
            return False
            
    except Exception as e:
        print(f"\n[ERROR] Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_gpt5_json_mode():
    """Test GPT-5 with JSON mode (fallback)."""
    print("\n" + "="*60)
    print("Testing GPT-5 with JSON mode (fallback)...")
    print("="*60)
    
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("[ERROR] OPENAI_API_KEY not found in environment")
            return False
        
        provider = get_provider(
            provider_name="openai",
            model_name="gpt-5",
            api_key=api_key,
            temperature=0.2,
            max_retries=2,
            retry_delay=5
        )
        
        test_text = """
        李小華，字德華，生於嘉慶年間。
        妻張氏，育有二子。
        """
        
        print("\nMaking API call to GPT-5 without function calling...")
        response = provider.parse_genealogy(
            text=test_text,
            system_prompt=SYSTEM_PROMPT,
            user_prompt_template=USER_PROMPT_TEMPLATE,
            use_function_calling=False  # Force JSON mode
        )
        
        if response.success:
            print(f"\n[SUCCESS] GPT-5 JSON mode worked!")
            print(f"Records found: {len(response.records)}")
            if response.records:
                print("\nSample record:")
                try:
                    print(f"  {response.records[0]}")
                except UnicodeEncodeError:
                    # Windows console encoding issue - just show we got data
                    print(f"  [Record with Chinese characters - {len(str(response.records[0]))} chars]")
            return True
        else:
            print(f"\n[WARNING] GPT-5 JSON mode failed: {response.error}")
            return False
            
    except Exception as e:
        print(f"\n[ERROR] Test failed with exception: {e}")
        return False

def test_gpt4o_comparison():
    """Test GPT-4o for comparison."""
    print("\n" + "="*60)
    print("Testing GPT-4o for comparison...")
    print("="*60)
    
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("[ERROR] OPENAI_API_KEY not found in environment")
            return False
        
        provider = get_provider(
            provider_name="openai",
            model_name="gpt-4o",
            api_key=api_key,
            temperature=0.2,
            max_retries=2,
            retry_delay=5
        )
        
        test_text = """
        陳大成，字成功，生於道光年間。
        妻林氏，生一子一女。
        """
        
        print("\nMaking API call to GPT-4o...")
        response = provider.parse_genealogy(
            text=test_text,
            system_prompt=SYSTEM_PROMPT,
            user_prompt_template=USER_PROMPT_TEMPLATE,
            use_function_calling=True
        )
        
        if response.success:
            print(f"\n[SUCCESS] GPT-4o worked as expected!")
            print(f"Records found: {len(response.records)}")
            if response.usage:
                print(f"Tokens used: {response.usage.get('total_tokens', 'unknown')}")
            return True
        else:
            print(f"\n[ERROR] GPT-4o failed: {response.error}")
            return False
            
    except Exception as e:
        print(f"\n[ERROR] Test failed with exception: {e}")
        return False

if __name__ == "__main__":
    print("\n" + "="*70)
    print("      GPT-5 Improved Handling Test Suite")
    print("="*70)
    
    # Test 1: GPT-5 with function calling (now enabled)
    gpt5_fc_success = test_gpt5_with_function_calling()
    
    # Test 2: GPT-5 with JSON mode
    gpt5_json_success = test_gpt5_json_mode()
    
    # Test 3: GPT-4o for comparison
    gpt4o_success = test_gpt4o_comparison()
    
    # Summary
    print("\n" + "="*70)
    print("                    TEST SUMMARY")
    print("="*70)
    print(f"GPT-5 with function calling: {'PASSED' if gpt5_fc_success else 'FAILED'}")
    print(f"GPT-5 with JSON mode:        {'PASSED' if gpt5_json_success else 'FAILED'}")
    print(f"GPT-4o (baseline):           {'PASSED' if gpt4o_success else 'FAILED'}")
    
    if gpt5_fc_success or gpt5_json_success:
        print("\n[RESULT] GPT-5 improvements are working!")
        print("You can now use GPT-5 with the pipeline.")
        if not gpt5_fc_success:
            print("Note: Function calling may still be unstable.")
    elif gpt4o_success:
        print("\n[RESULT] GPT-5 still has issues, but GPT-4o works fine.")
        print("Consider using GPT-4o until GPT-5 stabilizes.")
    else:
        print("\n[RESULT] Both models have issues. Check your API key and quota.")
    
    print("="*70)