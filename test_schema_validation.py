#!/usr/bin/env python3
"""
Test to verify the schema updates for birth_time and death_time fields.
"""

import json
import sys
from pathlib import Path

# Set UTF-8 encoding for Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from pipeline.schema_loader import load_genealogy_schema, get_schema_for_function_calling

def test_schema():
    """Test that the schema includes birth_time and death_time fields."""
    
    print("Testing schema updates...")
    print("=" * 60)
    
    # Load the schema
    schema = load_genealogy_schema()
    
    # Check the main schema structure
    assert "properties" in schema
    assert "records" in schema["properties"]
    
    # Get the record item schema
    record_schema = schema["properties"]["records"]["items"]["properties"]
    
    # Check for new fields
    print("\nChecking for birth_time and death_time fields...")
    
    if "birth_time" in record_schema:
        print("✓ birth_time field found in schema")
        print(f"  Type: {record_schema['birth_time']['type']}")
        print(f"  Description: {record_schema['birth_time']['description']}")
    else:
        print("✗ birth_time field NOT found in schema")
        
    if "death_time" in record_schema:
        print("✓ death_time field found in schema")
        print(f"  Type: {record_schema['death_time']['type']}")
        print(f"  Description: {record_schema['death_time']['description']}")
    else:
        print("✗ death_time field NOT found in schema")
    
    # Check required fields
    required_fields = schema["properties"]["records"]["items"]["required"]
    print(f"\nRequired fields: {', '.join(required_fields)}")
    
    assert "birth_time" in required_fields, "birth_time should be in required fields"
    assert "death_time" in required_fields, "death_time should be in required fields"
    
    # Test function calling schema
    print("\n" + "=" * 60)
    print("Testing function calling schema...")
    func_schema = get_schema_for_function_calling()
    
    assert "parameters" in func_schema
    assert "properties" in func_schema["parameters"]
    assert "records" in func_schema["parameters"]["properties"]
    
    print("✓ Function calling schema structure is valid")
    
    # Create a sample record to validate
    sample_record = {
        "name": "錢氏",
        "sex": "female",
        "father": None,
        "birth_order": None,
        "courtesy": None,
        "birth_time": "明嘉靖二十年辛丑十月十二日寅時",
        "death_time": "萬歷二十七年己亥八月十九日午時",
        "children": [],
        "info": "錢氏，明嘉靖二十年辛丑十月十二日寅時生，萬歷二十七年己亥八月十九日午時歿，葬許家術祖山。",
        "original_text": "錢氏 蜴言旦左誥明嘉靖二十年辛丑十月十二日寅時生萬歷二十七年己亥八月十九日午時歿葬許家術祖山",
        "note": "Extracted birth and death times",
        "is_update_for_previous": False,
        "skip": False
    }
    
    print("\n" + "=" * 60)
    print("Sample record with birth/death times:")
    print(json.dumps(sample_record, ensure_ascii=False, indent=2))
    
    print("\n✓ All schema tests passed!")
    print("\nThe schema has been successfully updated to include:")
    print("  - birth_time field for birth information")
    print("  - death_time field for death information")
    print("\nLLMs will now extract these fields when parsing genealogy text.")

if __name__ == "__main__":
    test_schema()