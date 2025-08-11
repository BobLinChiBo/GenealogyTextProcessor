#!/usr/bin/env python3
"""
Test script to verify birth and death time extraction works correctly.
"""

import json
import sys
import os
from pathlib import Path

# Set UTF-8 encoding for Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from llm import get_provider
from prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE

def test_extraction():
    """Test the birth and death time extraction."""
    
    # Test text with various birth/death time formats
    test_text = """錢氏 蜴言旦左誥明嘉靖二十年辛丑十月十二日寅時生萬歷二十七年己亥八月十九日午時歿葬許家術祖山
張三公長子李四字德明明成化三年丁亥二月初五日卯時生正德十年乙亥十一月廿三日申時卒
王氏嘉靖五年丙戌正月初一日子時生
趙五萬歷元年癸酉歿"""

    print("Testing birth and death time extraction...")
    print("=" * 60)
    print("Test text:")
    print(test_text)
    print("=" * 60)
    
    # Try OpenAI first, then fall back to other providers
    try:
        # Check if we have an API key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("No OPENAI_API_KEY found. Please set it in your environment or .env file")
            print("\nYou can test manually by running:")
            print("  python src/pipeline/genealogy_parser_v3.py data/test_birth_death.txt -o data/test_birth_death_output.json")
            return
            
        provider = get_provider(
            provider_name="openai",
            model_name="gpt-4o-mini",
            temperature=0.2
        )
        
        # Parse the text
        response = provider.parse_genealogy(
            text=test_text,
            system_prompt=SYSTEM_PROMPT,
            user_prompt_template=USER_PROMPT_TEMPLATE,
            use_function_calling=False  # Use JSON mode for testing
        )
        
        if response.success and response.records:
            print("\nExtracted records:")
            print("-" * 60)
            for i, record in enumerate(response.records, 1):
                print(f"\nRecord {i}:")
                print(f"  Name: {record.get('name')}")
                print(f"  Sex: {record.get('sex')}")
                print(f"  Birth Time: {record.get('birth_time')}")
                print(f"  Death Time: {record.get('death_time')}")
                print(f"  Original: {record.get('original_text')}")
            
            # Save to file for inspection
            output_file = Path("data/test_birth_death_output.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(response.records, f, ensure_ascii=False, indent=2)
            print(f"\n✓ Full results saved to: {output_file}")
            
            # Verify specific extractions
            print("\n" + "=" * 60)
            print("Verification:")
            
            # Check first record (錢氏)
            first = response.records[0] if response.records else {}
            expected_birth = "明嘉靖二十年辛丑十月十二日寅時"
            expected_death = "萬歷二十七年己亥八月十九日午時"
            
            if first.get('birth_time') == expected_birth:
                print(f"✓ Birth time correctly extracted: {expected_birth}")
            else:
                print(f"✗ Birth time mismatch. Expected: {expected_birth}, Got: {first.get('birth_time')}")
            
            if first.get('death_time') == expected_death:
                print(f"✓ Death time correctly extracted: {expected_death}")
            else:
                print(f"✗ Death time mismatch. Expected: {expected_death}, Got: {first.get('death_time')}")
                
        else:
            print(f"\n✗ Parsing failed: {response.error}")
            
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_extraction()