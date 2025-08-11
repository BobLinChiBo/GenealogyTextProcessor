#!/usr/bin/env python3
"""
Test script for GPT-5 with context parsing.
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

def test_gpt5_context():
    """Test GPT-5 with context parsing."""
    print("=" * 60)
    print("GPT-5 Context Parsing Test")
    print("=" * 60)
    
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("[ERROR] OPENAI_API_KEY not found!")
        return False
    
    print("[OK] API key found")
    
    try:
        from llm import get_provider
        from prompts import SYSTEM_PROMPT, USER_PROMPT_WITH_CONTEXT
        
        # Create provider with GPT-5
        print("\n[PROCESSING] Creating GPT-5 provider...")
        provider = get_provider(
            provider_name="openai",
            model_name="gpt-5",
            api_key=api_key,
            temperature=0.2,
            max_retries=2,
            retry_delay=3
        )
        
        # Test with context and new text
        context_text = """
        王大明，字德明，生於康熙年間。
        子：王小明，字德小，生於雍正年間。
        """
        
        new_text = """
        王小明，字德小，生於雍正年間。
        子：王小華，字德華，生於乾隆年間。
        """
        
        print("[PROCESSING] Making API call to GPT-5 with context...")
        print("   (This may take 30-60 seconds due to GPT-5's longer response times)")
        
        response = provider.parse_with_context(
            context_text=context_text,
            new_text=new_text,
            system_prompt=SYSTEM_PROMPT,
            user_prompt_template=USER_PROMPT_WITH_CONTEXT,
            use_function_calling=False  # Disabled for GPT-5
        )
        
        if response.success:
            print(f"\n[OK] GPT-5 context test successful!")
            print(f"   Records found: {len(response.records)}")
            if response.usage:
                print(f"   Tokens used: {response.usage.get('total_tokens', 'unknown')}")
            
            # Debug: Show the raw response
            if hasattr(response, 'raw_response') and response.raw_response:
                print(f"\n🔍 Raw response from GPT-5:")
                print(f"   {response.raw_response[:500]}...")
            
            # Show the records if any
            if response.records:
                print(f"\n📋 Records found:")
                for i, record in enumerate(response.records):
                    print(f"   Record {i+1}:")
                    print(f"     Name: {record.get('name', 'N/A')}")
                    print(f"     Father: {record.get('father', 'N/A')}")
                    print(f"     Info: {record.get('info', 'N/A')}")
            else:
                print(f"\n⚠️ No records found in response")
            
            return True
        else:
            print(f"\n[ERROR] GPT-5 context test failed: {response.error}")
            return False
            
    except Exception as e:
        print(f"\n[ERROR] GPT-5 context test error: {e}")
        print(f"   Error type: {type(e).__name__}")
        return False

def main():
    success = test_gpt5_context()
    
    print("\n" + "=" * 60)
    if success:
        print("[OK] GPT-5 context parsing is working!")
        print("The pipeline should now work with all chunks.")
    else:
        print("[ERROR] GPT-5 context test failed.")
    print("=" * 60)

if __name__ == "__main__":
    main()

