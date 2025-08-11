#!/usr/bin/env python3
"""Test if the simplified schema works with GPT-5-nano."""

import os
import json
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Test with actual genealogy text from the file (first 5 lines)
test_text = """王 晉至 一公宗派
晉 字子喬周靈王太子也好吹笙作鳳鳴游伊洛 之間遇道士接以上嵩高山三十餘年一日 遇桓良曰告我家七月七曰待我於緱氏山 頭至期往則晉乘白鶴舉手謝時人而去
晉公 子 宗敬 為司徒時人因王之子孫號曰王家成父敗賊 有功遂賜為氏後太原瑯琊為著
宗敬 公子 森 為上卿封安平侯長子彬次子術封武烈侯
森公 子 彬"""

system_prompt = """You are an expert in Chinese genealogy extraction. Extract structured genealogy records from Chinese family tree text.
Each line should become a record. For lines that are genealogy records, set skip=false. For titles or noise, set skip=true.
Include whatever information you can extract - use null for missing fields."""

# Simplified schema - only original_text and skip are required
simplified_tools = [{
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
                            "sex": {"type": "string", "description": "Sex of the person", "enum": ["male", "female", None]},
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
                        "required": ["original_text", "skip"]  # Only these 2 are required now
                    }
                }
            },
            "required": ["records"]
        }
    }
}]

print("Testing simplified schema (only original_text and skip required)")
print("-" * 60)

try:
    response = client.chat.completions.create(
        model="gpt-5-nano",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Extract genealogy records from this text, one record per line:\n\n{test_text}"}
        ],
        tools=simplified_tools,
        tool_choice={"type": "function", "function": {"name": "extract_genealogy_records"}},
        max_completion_tokens=4000
    )
    
    if response.choices[0].message.tool_calls:
        print("SUCCESS! Got tool calls")
        result = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
        records = result.get('records', [])
        print(f"Extracted {len(records)} records")
        
        for i, record in enumerate(records[:3]):  # Show first 3 records
            print(f"\nRecord {i+1}:")
            print(f"  Original: {record.get('original_text', '')[:50]}...")
            print(f"  Name: {record.get('name', 'N/A')}")
            print(f"  Skip: {record.get('skip', 'N/A')}")
            print(f"  Info: {record.get('info', 'N/A')[:50] if record.get('info') else 'N/A'}...")
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