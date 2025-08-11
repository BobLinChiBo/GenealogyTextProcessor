#!/usr/bin/env python3
"""Test GPT-5-nano API call directly to debug the issue."""

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

# Test 1: Simple completion without function calling
print("Test 1: Simple completion")
try:
    response = client.chat.completions.create(
        model="gpt-5-nano",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        max_completion_tokens=1000,
        temperature=0.2
    )
    print("[OK] Simple completion works!")
    print(f"Response: {response.choices[0].message.content[:200]}...")
except Exception as e:
    print(f"[ERROR] Simple completion failed: {e}")
    print(f"Error type: {type(e).__name__}")

print("\n" + "="*50 + "\n")

# Test 2: JSON mode
print("Test 2: JSON mode")
try:
    response = client.chat.completions.create(
        model="gpt-5-nano",
        messages=[
            {"role": "system", "content": system_prompt + " Return a JSON object with a 'records' array."},
            {"role": "user", "content": user_prompt}
        ],
        response_format={"type": "json_object"},
        max_completion_tokens=1000,
        temperature=0.2
    )
    print("[OK] JSON mode works!")
    print(f"Response: {response.choices[0].message.content[:200]}...")
except Exception as e:
    print(f"[ERROR] JSON mode failed: {e}")
    print(f"Error type: {type(e).__name__}")

print("\n" + "="*50 + "\n")

# Test 3: Function calling
print("Test 3: Function calling")
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
                }
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
        max_completion_tokens=1000,
        temperature=0.2
    )
    print("[OK] Function calling works!")
    if response.choices[0].message.tool_calls:
        print(f"Function response: {response.choices[0].message.tool_calls[0].function.arguments[:200]}...")
    else:
        print(f"Response content: {response.choices[0].message.content[:200]}...")
except Exception as e:
    print(f"[ERROR] Function calling failed: {e}")
    print(f"Error type: {type(e).__name__}")
    
    # Try to get more error details
    import traceback
    print("\nFull traceback:")
    traceback.print_exc()