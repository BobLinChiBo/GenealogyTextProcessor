#!/usr/bin/env python3
"""Test the OpenAI provider directly to see the actual error."""

import os
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from dotenv import load_dotenv
from llm.openai_provider import OpenAIProvider

# Load environment variables
load_dotenv()

# Set up detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Initialize provider
provider = OpenAIProvider(
    model_name="gpt-5-nano",
    temperature=0.2,  # This should be ignored for GPT-5
    max_retries=1,
    config={}
)

# Test text
test_text = "張三 字子明 號東山 生於清朝"

# Test system and user prompts
system_prompt = "Extract genealogy records from Chinese text. Return JSON with 'records' array."
user_prompt_template = "Extract records from: {text}"

# Test the provider
print("Testing OpenAI provider with gpt-5-nano...")
print("-" * 50)

response = provider.parse_genealogy(
    text=test_text,
    system_prompt=system_prompt,
    user_prompt_template=user_prompt_template,
    use_function_calling=True
)

if response.success:
    print(f"SUCCESS! Found {len(response.records)} records")
    print(f"Records: {response.records}")
else:
    print(f"FAILED: {response.error}")
    print(f"Raw response: {response.raw_response}")