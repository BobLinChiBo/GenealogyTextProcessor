#!/usr/bin/env python3
"""Test the nullable schema with GPT-5-nano."""

import os
import json
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Test with actual genealogy text
test_text = """王 晉至 一公宗派
晉 字子喬周靈王太子也好吹笙作鳳鳴游伊洛"""

system_prompt = """You are an expert in Chinese genealogy extraction. Extract structured genealogy records.
For each line of text, create a record with all fields. Use null for missing information.
Set skip=false for genealogy records, skip=true for titles/noise."""

# Schema with all fields required but nullable
nullable_schema = [{
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
                            "name": {"type": ["string", "null"], "description": "The main person described"},
                            "sex": {"type": ["string", "null"], "description": "Sex of the person"},
                            "father": {"type": ["string", "null"], "description": "The father's name"},
                            "birth_order": {"type": ["string", "null"], "description": "Birth order"},
                            "courtesy": {"type": ["string", "null"], "description": "Courtesy name (字)"},
                            "children": {"type": ["array", "null"], "description": "Array of children"},
                            "info": {"type": ["string", "null"], "description": "Biographical details"},
                            "original_text": {"type": "string", "description": "The original line of text"},
                            "note": {"type": ["string", "null"], "description": "Reasoning"},
                            "is_update_for_previous": {"type": ["boolean", "null"], "description": "True if updates previous"},
                            "skip": {"type": "boolean", "description": "True if should be skipped"}
                        },
                        "required": ["name", "sex", "father", "birth_order", "courtesy", 
                                   "children", "info", "original_text", "note", 
                                   "is_update_for_previous", "skip"]
                    }
                }
            },
            "required": ["records"]
        }
    }
}]

print("Testing schema with all fields required but nullable")
print("-" * 60)

try:
    response = client.chat.completions.create(
        model="gpt-5-nano",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Extract genealogy records from this text, one record per line. Use null for missing fields:\n\n{test_text}"}
        ],
        tools=nullable_schema,
        tool_choice={"type": "function", "function": {"name": "extract_genealogy_records"}},
        max_completion_tokens=4000
    )
    
    if response.choices[0].message.tool_calls:
        print("SUCCESS! Got tool calls")
        result = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
        records = result.get('records', [])
        print(f"Extracted {len(records)} records")
        
        for i, record in enumerate(records):
            print(f"\nRecord {i+1}:")
            print(f"  Original: {record.get('original_text', '')}")
            print(f"  Name: {record.get('name', 'MISSING')}")
            print(f"  Skip: {record.get('skip', 'MISSING')}")
            print(f"  Father: {record.get('father', 'MISSING')}")
    else:
        print("FAILED: No tool calls in response")
        if response.choices[0].message.content:
            print("Content returned instead:", response.choices[0].message.content[:500])
        else:
            print("No content returned either")
            
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()