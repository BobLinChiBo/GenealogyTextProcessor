"""
Schema loader utility for genealogy extraction.

This module provides a centralized way to load and cache the genealogy schema,
ensuring all providers and validators use the same schema definition.
"""

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, Any, Optional


@lru_cache(maxsize=1)
def load_genealogy_schema(path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load and cache the genealogy schema from JSON file.
    
    Args:
        path: Optional path to schema file. If not provided, uses default location.
        
    Returns:
        Dictionary containing the parsed JSON schema.
        
    Raises:
        FileNotFoundError: If schema file doesn't exist.
        json.JSONDecodeError: If schema file is invalid JSON.
    """
    if path is None:
        # Default path: project_root/schemas/genealogy.schema.json
        path = Path(__file__).resolve().parents[2] / "schemas" / "genealogy.schema.json"
    
    with open(path, "r", encoding="utf-8") as f:
        schema = json.load(f)
    
    return schema


def get_schema_for_function_calling() -> Dict[str, Any]:
    """
    Get the schema formatted for OpenAI function calling.
    
    This wraps the base schema in the function calling format expected
    by OpenAI and compatible providers.
    
    Returns:
        Dictionary with name, description, and parameters for function calling.
    """
    schema = load_genealogy_schema()
    
    # Extract just the records definition for the parameters
    records_schema = schema["properties"]["records"]
    
    return {
        "name": "extract_genealogy_records",
        "description": schema.get("description", "Extract structured genealogy records from Chinese family tree text"),
        "parameters": {
            "type": "object",
            "properties": {
                "records": records_schema
            },
            "required": ["records"],
            "additionalProperties": False
        },
        "strict": True  # For OpenAI structured output
    }


def get_schema_version() -> str:
    """
    Get the version of the current schema.
    
    Returns:
        Version string from the schema file.
    """
    schema = load_genealogy_schema()
    return schema.get("version", "unknown")


def validate_schema_version(expected_version: str) -> bool:
    """
    Check if the schema version matches the expected version.
    
    Args:
        expected_version: The version to check against.
        
    Returns:
        True if versions match, False otherwise.
    """
    return get_schema_version() == expected_version