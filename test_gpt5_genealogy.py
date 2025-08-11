#!/usr/bin/env python3
"""Test GPT-5 genealogy parsing with new parameters"""

from dotenv import load_dotenv
import os
import sys
import json

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from llm.openai_provider import OpenAIProvider

load_dotenv()

# Test configuration with GPT-5 parameters
config = {
    'reasoning_effort': 'minimal',  # Use minimal for faster testing
    'verbosity': 'low'  # Low verbosity for concise output
}

# Initialize provider
provider = OpenAIProvider(
    model_name='gpt-5',
    temperature=0.2,
    config=config
)

# Test text from the genealogy document
test_text = """
王氏世系谱
第一世：王安，字子安，明洪武年间自江西迁居福建
第二世：王德，字明德，生三子
第三世：王仁、王义、王礼
"""

# System and user prompts
system_prompt = """You are a genealogy data extraction specialist. Extract structured genealogy records from Chinese text.
Output a JSON array of person records with fields: name, generation, courtesy_name, birth_year, death_year, father, mother, spouse, children, notes."""

user_prompt = f"""Extract genealogy records from this text:

{test_text}

Return only a JSON array of person records."""

print("Testing GPT-5 with genealogy text...")
print(f"Model: gpt-5")
print(f"Reasoning effort: {config['reasoning_effort']}")
print(f"Verbosity: {config['verbosity']}")
print("-" * 60)

# Test the parse_genealogy method
response = provider.parse_genealogy(
    text=test_text,
    system_prompt=system_prompt,
    user_prompt_template=user_prompt,
    use_function_calling=False  # Start with JSON mode
)

if response.success:
    print("[SUCCESS] GPT-5 parsed the genealogy text!")
    print(f"Records extracted: {len(response.records)}")
    if response.records:
        print("\nExtracted records:")
        for i, record in enumerate(response.records, 1):
            print(f"\n{i}. {record.get('name', 'Unknown')}")
            if 'generation' in record:
                print(f"   Generation: {record['generation']}")
            if 'courtesy_name' in record:
                print(f"   Courtesy name: {record['courtesy_name']}")
            if 'notes' in record:
                print(f"   Notes: {record['notes'][:100]}...")
else:
    print(f"[ERROR] Failed to parse: {response.error}")

# Show usage stats if available
if response.usage:
    print(f"\nToken usage:")
    print(f"  Input: {response.usage.get('prompt_tokens', 'N/A')}")
    print(f"  Output: {response.usage.get('completion_tokens', 'N/A')}")
    print(f"  Total: {response.usage.get('total_tokens', 'N/A')}")