#!/usr/bin/env python3
"""Test GPT-5-nano API call directly without temperature."""

import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Test prompt
system_prompt = "You are a helpful assistant that extracts genealogy records from Chinese text."
user_prompt = "Extract genealogy records from: 張三 字子明 號東山 生於清朝"

# Test 1: Simple completion without temperature
print("Test 1: Simple completion (no temperature)")
try:
    response = client.chat.completions.create(
        model="gpt-5-nano",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        max_completion_tokens=1000
        # No temperature parameter
    )
    print("Success! Simple completion works!")
    print(f"Response: {response.choices[0].message.content[:200]}...")
except Exception as e:
    print(f"Failed: {e}")
    print(f"Error type: {type(e).__name__}")

print("\n" + "="*50 + "\n")

# Test 2: Function calling without temperature
print("Test 2: Function calling (no temperature)")
try:
    tools = [{
        "type": "function",
        "function": {
            "name": "extract_genealogy_records",
            "description": "Extract genealogy records from text",
            "parameters": {
                "type": "object",
                "properties": {
                    "records": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "info": {"type": "string"}
                            }
                        }
                    }
                },
                "required": ["records"]
            }
        }
    }]
    
    response = client.chat.completions.create(
        model="gpt-5-nano",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        tools=tools,
        tool_choice={"type": "function", "function": {"name": "extract_genealogy_records"}},
        max_completion_tokens=1000
        # No temperature parameter
    )
    print("Success! Function calling works!")
    if response.choices[0].message.tool_calls:
        print(f"Function response: {response.choices[0].message.tool_calls[0].function.arguments[:200]}...")
    else:
        print(f"Response content: {response.choices[0].message.content[:200] if response.choices[0].message.content else 'No content'}...")
except Exception as e:
    print(f"Failed: {e}")
    print(f"Error type: {type(e).__name__}")
    
    # Try to get more error details
    import traceback
    print("\nFull traceback:")
    traceback.print_exc()