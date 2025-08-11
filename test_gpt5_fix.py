#!/usr/bin/env python3
"""
Test script to verify GPT-5 fixes work properly.
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from llm import get_provider
from prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE

def test_gpt5_connection():
    """Test if GPT-5 can connect and respond properly."""
    print("Testing GPT-5 connection...")
    
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
            max_retries=1,
            retry_delay=2
        )
        
        # Test with a simple genealogy text
        test_text = """
        王大明，字德明，生於康熙年間。
        子：王小明，字德小，生於雍正年間。
        """
        
        print("Making test API call to GPT-5...")
        response = provider.parse_genealogy(
            text=test_text,
            system_prompt=SYSTEM_PROMPT,
            user_prompt_template=USER_PROMPT_TEMPLATE,
            use_function_calling=True
        )
        
        if response.success:
            print(f"[OK] GPT-5 test successful!")
            print(f"   Records found: {len(response.records)}")
            if response.usage:
                print(f"   Tokens used: {response.usage.get('total_tokens', 'unknown')}")
            return True
        else:
            print(f"[ERROR] GPT-5 test failed: {response.error}")
            return False
            
    except Exception as e:
        print(f"[ERROR] GPT-5 test error: {e}")
        return False

def test_gpt5_without_function_calling():
    """Test GPT-5 without function calling as fallback."""
    print("\nTesting GPT-5 without function calling...")
    
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
            max_retries=1,
            retry_delay=2
        )
        
        test_text = """
        李小明，字德明，生於乾隆年間。
        子：李小華，字德華，生於嘉慶年間。
        """
        
        print("Making test API call to GPT-5 (no function calling)...")
        response = provider.parse_genealogy(
            text=test_text,
            system_prompt=SYSTEM_PROMPT,
            user_prompt_template=USER_PROMPT_TEMPLATE,
            use_function_calling=False
        )
        
        if response.success:
            print(f"[OK] GPT-5 test without function calling successful!")
            print(f"   Records found: {len(response.records)}")
            return True
        else:
            print(f"[ERROR] GPT-5 test without function calling failed: {response.error}")
            return False
            
    except Exception as e:
        print(f"[ERROR] GPT-5 test error: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("GPT-5 Fix Test")
    print("=" * 60)
    
    # Test 1: With function calling
    success1 = test_gpt5_connection()
    
    # Test 2: Without function calling
    success2 = test_gpt5_without_function_calling()
    
    print("\n" + "=" * 60)
    if success1 or success2:
        print("[OK] GPT-5 fixes appear to be working!")
        print("You can now run the pipeline with GPT-5.")
    else:
        print("[ERROR] GPT-5 tests failed.")
        print("Consider using a different model like gpt-4o or gpt-4o-mini.")
    print("=" * 60)

