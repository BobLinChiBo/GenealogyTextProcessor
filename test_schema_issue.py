#!/usr/bin/env python3
"""Test if the complex schema is causing issues with GPT-5-nano."""

import os
import json
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Test with actual genealogy text from the file
test_text = """王 晉至 一公宗派
晉 字子喬周靈王太子也好吹笙作鳳鳴游伊洛 之間遇道士接以上嵩高山三十餘年一日 遇桓良曰告我家七月七曰待我於緱氏山 頭至期往則晉乘白鶴舉手謝時人而去
晉公 子 宗敬 為司徒時人因王之子孫號曰王家成父敗賊 有功遂賜為氏後太原瑯琊為著
宗敬 公子 森 為上卿封安平侯長子彬次子術封武烈侯
森公 子 彬"""

system_prompt = """You are an expert in Chinese genealogy extraction. Extract structured genealogy records from Chinese family tree text.
IMPORTANT: Some fields may not be present in the text. Use null for missing information."""

# Test 1: Simple schema
print("Test 1: Simple schema (just name and info)")
print("-" * 50)
simple_tools = [{
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
                        },
                        "required": ["name"]
                    }
                }
            },
            "required": ["records"]
        }
    }
}]

try:
    response = client.chat.completions.create(
        model="gpt-5-nano",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Extract records from:\n{test_text}"}
        ],
        tools=simple_tools,
        tool_choice={"type": "function", "function": {"name": "extract_genealogy_records"}},
        max_completion_tokens=2000
    )
    
    if response.choices[0].message.tool_calls:
        result = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
        print(f"SUCCESS! Got {len(result['records'])} records")
        print("First record:", json.dumps(result['records'][0], ensure_ascii=False, indent=2))
    else:
        print("FAILED: No tool calls in response")
        print("Content:", response.choices[0].message.content)
except Exception as e:
    print(f"FAILED: {e}")

print("\n" + "="*50 + "\n")

# Test 2: Complex schema with nullable fields
print("Test 2: Complex schema (nullable fields)")
print("-" * 50)
complex_tools = [{
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
                            "name": {"type": ["string", "null"]},
                            "gender": {"type": ["string", "null"]},
                            "father": {"type": ["string", "null"]},
                            "birth_order": {"type": ["string", "null"]},
                            "courtesy": {"type": ["string", "null"]},
                            "children": {"type": ["array", "null"]},
                            "info": {"type": ["string", "null"]},
                            "original_text": {"type": ["string", "null"]},
                            "note": {"type": ["string", "null"]},
                            "is_update_for_previous": {"type": ["boolean", "null"]},
                            "skip": {"type": ["boolean", "null"]}
                        },
                        "required": ["name", "original_text"]
                    }
                }
            },
            "required": ["records"]
        }
    }
}]

try:
    response = client.chat.completions.create(
        model="gpt-5-nano",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Extract records from:\n{test_text}"}
        ],
        tools=complex_tools,
        tool_choice={"type": "function", "function": {"name": "extract_genealogy_records"}},
        max_completion_tokens=2000
    )
    
    if response.choices[0].message.tool_calls:
        result = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
        print(f"SUCCESS! Got {len(result['records'])} records")
        print("First record:", json.dumps(result['records'][0], ensure_ascii=False, indent=2))
    else:
        print("FAILED: No tool calls in response")
        print("Content:", response.choices[0].message.content)
except Exception as e:
    print(f"FAILED: {e}")

print("\n" + "="*50 + "\n")

# Test 3: The actual complex schema from the codebase
print("Test 3: Actual schema from codebase (11 required fields!)")
print("-" * 50)
actual_tools = [{
    "type": "function",
    "function": {
        "name": "extract_genealogy_records",
        "description": "Extract structured genealogy records from Chinese family tree text",
        "parameters": {
            "type": "object",
            "properties": {
                "records": {
                    "type": "array",
                    "description": "Array of genealogy records, one per line of text",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "The main person described in the line"},
                            "gender": {"type": "string", "description": "Gender of the person", "enum": ["male", "female", None]},
                            "father": {"type": "string", "description": "The father's name"},
                            "birth_order": {"type": "string", "description": "Birth order if mentioned"},
                            "courtesy": {"type": "string", "description": "The person's courtesy name (字)"},
                            "children": {"type": "array", "items": {"type": "object"}},
                            "info": {"type": "string", "description": "ALL biographical details"},
                            "original_text": {"type": "string", "description": "The original line of text"},
                            "note": {"type": "string", "description": "Reasoning for the interpretation"},
                            "is_update_for_previous": {"type": "boolean", "description": "True if this line only adds info"},
                            "skip": {"type": "boolean", "description": "True if this line should be skipped"}
                        },
                        "required": ["name", "gender", "father", "birth_order", "courtesy", 
                                   "children", "info", "original_text", "note", 
                                   "is_update_for_previous", "skip"]
                    }
                }
            },
            "required": ["records"]
        }
    }
}]

try:
    response = client.chat.completions.create(
        model="gpt-5-nano",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Extract records from:\n{test_text}"}
        ],
        tools=actual_tools,
        tool_choice={"type": "function", "function": {"name": "extract_genealogy_records"}},
        max_completion_tokens=2000
    )
    
    if response.choices[0].message.tool_calls:
        result = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
        print(f"SUCCESS! Got {len(result['records'])} records")
        print("First record:", json.dumps(result['records'][0], ensure_ascii=False, indent=2))
    else:
        print("FAILED: No tool calls in response")
        print("Content:", response.choices[0].message.content)
except Exception as e:
    print(f"FAILED: {e}")