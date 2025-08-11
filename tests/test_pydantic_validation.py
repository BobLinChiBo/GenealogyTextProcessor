#!/usr/bin/env python3
"""
Test suite for Pydantic validation in genealogy parser
"""

import pytest
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pipeline.genealogy_parser import (
    GenealogyParser, 
    PYDANTIC_AVAILABLE,
    PersonRecord,
    ChildRecord,
    GenealogyResponse
)

# Skip tests if Pydantic is not available
pytestmark = pytest.mark.skipif(
    not PYDANTIC_AVAILABLE,
    reason="Pydantic not installed"
)


class TestPydanticModels:
    """Test Pydantic model validation"""
    
    def test_child_record_valid(self):
        """Test valid child record creation"""
        child = ChildRecord(order=1, name="王大")
        assert child.order == 1
        assert child.name == "王大"
        
        # Test with dict conversion
        child_dict = child.model_dump()
        assert child_dict == {"order": 1, "name": "王大"}
    
    def test_child_record_invalid_order(self):
        """Test that negative order raises validation error"""
        with pytest.raises(Exception) as exc_info:
            ChildRecord(order=0, name="Test")
        assert "greater than or equal to 1" in str(exc_info.value).lower()
        
        with pytest.raises(Exception) as exc_info:
            ChildRecord(order=-1, name="Test")
        assert "greater than or equal to 1" in str(exc_info.value).lower()
    
    def test_child_record_empty_name(self):
        """Test that empty name raises validation error"""
        with pytest.raises(Exception) as exc_info:
            ChildRecord(order=1, name="")
        assert "at least 1 character" in str(exc_info.value).lower() or "cannot be empty" in str(exc_info.value).lower()
        
        with pytest.raises(Exception) as exc_info:
            ChildRecord(order=1, name="   ")
        assert "cannot be empty or whitespace" in str(exc_info.value).lower()
    
    def test_person_record_valid(self):
        """Test valid person record creation"""
        person = PersonRecord(
            name="寶二",
            father="成一",
            birth_order="次子",
            courtesy="天典",
            children=[
                ChildRecord(order=1, name="得才"),
                ChildRecord(order=2, name="得仁")
            ],
            info="為庠生，娶李氏",
            original_text="次子寶二字天典庠生娶李氏子二得才得仁",
            note="Second son identified",
            is_update_for_previous=False
        )
        
        assert person.name == "寶二"
        assert person.father == "成一"
        assert len(person.children) == 2
        assert person.children[0].name == "得才"
    
    def test_person_record_defaults(self):
        """Test person record with default values"""
        person = PersonRecord()
        assert person.name == ""
        assert person.father == ""
        assert person.birth_order == ""
        assert person.courtesy == ""
        assert person.children == []
        assert person.info == ""
        assert person.original_text == ""
        assert person.note == ""
        assert person.is_update_for_previous == False
    
    def test_person_record_children_conversion(self):
        """Test automatic conversion of children from dicts"""
        person = PersonRecord(
            name="Test",
            children=[
                {"order": 1, "name": "Child1"},
                {"order": 2, "name": "Child2"}
            ]
        )
        
        assert len(person.children) == 2
        assert all(isinstance(child, ChildRecord) for child in person.children)
        assert person.children[0].name == "Child1"
        assert person.children[1].order == 2
    
    def test_genealogy_response_valid(self):
        """Test valid genealogy response creation"""
        response = GenealogyResponse(
            records=[
                PersonRecord(name="Person1"),
                PersonRecord(name="Person2", father="Person1")
            ]
        )
        
        assert len(response.records) == 2
        assert response.records[0].name == "Person1"
        assert response.records[1].father == "Person1"
    
    def test_genealogy_response_from_dict(self):
        """Test creating response from dictionary"""
        data = {
            "records": [
                {
                    "name": "寶一",
                    "father": "成一",
                    "birth_order": "長子",
                    "courtesy": "全順",
                    "children": [],
                    "info": "公以功封軍門讚護",
                    "original_text": "成一公長子寶一字全順公以功封軍門讚護",
                    "note": "First son",
                    "is_update_for_previous": False
                }
            ]
        }
        
        response = GenealogyResponse(**data)
        assert len(response.records) == 1
        assert response.records[0].name == "寶一"
        assert response.records[0].birth_order == "長子"


class TestFunctionCallingIntegration:
    """Test function calling with Pydantic validation"""
    
    def test_parse_function_response(self):
        """Test parsing function calling response with Pydantic"""
        parser = GenealogyParser(
            input_file="dummy.txt",
            output_file="test_output.json"
        )
        
        # Simulate function calling response
        function_args = json.dumps({
            "records": [
                {
                    "name": "寶二",
                    "father": "成一",
                    "birth_order": "次子",
                    "courtesy": "天典",
                    "children": [
                        {"order": 1, "name": "得才"},
                        {"order": 2, "name": "得仁"}
                    ],
                    "info": "為庠生，娶李氏",
                    "original_text": "次子寶二字天典庠生娶李氏子二得才得仁",
                    "note": "Second son",
                    "is_update_for_previous": False
                }
            ]
        })
        
        # Test that it can be validated
        response = GenealogyResponse.model_validate_json(function_args)
        records = [record.model_dump() for record in response.records]
        
        assert len(records) == 1
        assert records[0]["name"] == "寶二"
        assert len(records[0]["children"]) == 2
    
    def test_invalid_function_response(self):
        """Test handling of invalid function response"""
        # Missing required field
        invalid_json = json.dumps({
            "records": [
                {
                    "name": "Test",
                    # Missing other required fields
                }
            ]
        })
        
        # Should still parse with defaults
        response = GenealogyResponse.model_validate_json(invalid_json)
        assert len(response.records) == 1
        assert response.records[0].name == "Test"
        assert response.records[0].father == ""  # Default value
    
    def test_extra_fields_rejected(self):
        """Test that extra fields are rejected"""
        with pytest.raises(Exception) as exc_info:
            PersonRecord(
                name="Test",
                father="TestFather",
                unknown_field="Should fail"
            )
        assert "extra" in str(exc_info.value).lower()


class TestPostProcessingWithPydantic:
    """Test post-processing with Pydantic models"""
    
    def test_post_process_with_validation(self):
        """Test post-processing records with Pydantic validation"""
        parser = GenealogyParser(
            input_file="dummy.txt",
            output_file="test_output.json"
        )
        
        raw_records = [
            {
                "name": "寶一",
                "father": "成一",
                "birth_order": "長子",
                "courtesy": "全順",
                "children": [],
                "info": "First info",
                "original_text": "Line 1",
                "note": "Note 1",
                "is_update_for_previous": False
            },
            {
                "name": "",
                "father": "",
                "birth_order": "",
                "courtesy": "",
                "children": [],
                "info": "Additional info",
                "original_text": "Line 2",
                "note": "Note 2",
                "is_update_for_previous": True
            }
        ]
        
        processed = parser._post_process_records(raw_records)
        
        # Should merge the second record into the first
        assert len(processed) == 1
        assert processed[0]["name"] == "寶一"
        assert "First info Additional info" in processed[0]["info"]
        assert "Line 1\nLine 2" in processed[0]["original_text"]
        assert "Note 1 | Note 2" in processed[0]["note"]
    
    def test_post_process_invalid_children(self):
        """Test post-processing with invalid children data"""
        parser = GenealogyParser(
            input_file="dummy.txt",
            output_file="test_output.json"
        )
        
        raw_records = [
            {
                "name": "Test",
                "father": "",
                "birth_order": "",
                "courtesy": "",
                "children": [
                    {"order": 1, "name": "Valid"},
                    {"order": -1, "name": "Invalid"},  # Invalid order
                    {"order": 2, "name": ""}  # Empty name
                ],
                "info": "",
                "original_text": "",
                "note": "",
                "is_update_for_previous": False
            }
        ]
        
        # Process should handle invalid data gracefully
        processed = parser._post_process_records(raw_records)
        assert len(processed) > 0  # Should still produce output


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])