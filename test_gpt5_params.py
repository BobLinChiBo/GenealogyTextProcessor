#!/usr/bin/env python3
"""Test what parameters GPT-5 actually accepts"""

from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Test 1: Basic call with standard parameters
print("Test 1: Standard parameters (without max_tokens)...")
try:
    response = client.chat.completions.create(
        model="gpt-5",
        messages=[{"role": "user", "content": "Say hello"}],
        temperature=0.2
    )
    print("[SUCCESS] Standard parameters work")
    print(f"Response: {response.choices[0].message.content}")
except Exception as e:
    print(f"[ERROR] {e}")

print("\n" + "-"*60 + "\n")

# Test 2: Try without token limits
print("Test 2: Without any token limit parameters...")
try:
    response = client.chat.completions.create(
        model="gpt-5",
        messages=[{"role": "user", "content": "Say hello"}],
        temperature=0.2
    )
    print("[SUCCESS] No token limits works")
    print(f"Response: {response.choices[0].message.content}")
except Exception as e:
    print(f"[ERROR] {e}")

print("\n" + "-"*60 + "\n")

# Test 3: Try with reasoning_effort
print("Test 3: Using reasoning_effort...")
try:
    response = client.chat.completions.create(
        model="gpt-5",
        messages=[{"role": "user", "content": "Say hello"}],
        reasoning_effort="minimal"
    )
    print("[SUCCESS] reasoning_effort works")
    print(f"Response: {response.choices[0].message.content}")
except Exception as e:
    print(f"[ERROR] {e}")

print("\n" + "-"*60 + "\n")

# Test 4: Try with verbosity
print("Test 4: Using verbosity...")
try:
    response = client.chat.completions.create(
        model="gpt-5",
        messages=[{"role": "user", "content": "Say hello"}],
        verbosity="low"
    )
    print("[SUCCESS] verbosity works")
    print(f"Response: {response.choices[0].message.content}")
except Exception as e:
    print(f"[ERROR] {e}")