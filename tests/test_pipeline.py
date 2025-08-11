#!/usr/bin/env python3
"""
Test suite for the genealogy text processor pipeline
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pipeline.file_merger import FileMerger
from pipeline.text_cleaner import TextCleaner
from utils.chinese_detector import is_chinese_char, chinese_ratio, contains_genealogy_keywords


class TestChineseDetector:
    """Test Chinese text detection utilities"""
    
    def test_is_chinese_char(self):
        """Test Chinese character detection"""
        assert is_chinese_char('中') == True
        assert is_chinese_char('A') == False
        assert is_chinese_char('1') == False
        assert is_chinese_char('。') == True
        
    def test_chinese_ratio(self):
        """Test Chinese ratio calculation"""
        assert chinese_ratio('中文測試') == 1.0
        assert chinese_ratio('ABC123') == 0.0
        assert chinese_ratio('中文ABC') == pytest.approx(0.4, 0.1)
        
    def test_genealogy_keywords(self):
        """Test genealogy keyword detection"""
        assert contains_genealogy_keywords('長子王大娶李氏') == True
        assert contains_genealogy_keywords('隨機文字內容') == False


class TestFileMerger:
    """Test file merger functionality"""
    
    def test_filename_parsing(self):
        """Test filename parsing"""
        merger = FileMerger("dummy")
        
        # Create test path
        test_path = Path("test_Page_1_right_col3.txt")
        file_info = merger.parse_filename(test_path)
        
        assert file_info is not None
        assert file_info.page == 1
        assert file_info.side == "right"
        assert file_info.column == 3
        
    def test_sort_order(self):
        """Test file sorting logic"""
        merger = FileMerger("dummy")
        
        # Create test file infos
        from pipeline.file_merger import FileInfo
        
        files = [
            FileInfo(Path("a"), page=1, side="left", column=2),
            FileInfo(Path("b"), page=1, side="right", column=3),
            FileInfo(Path("c"), page=2, side="right", column=1),
            FileInfo(Path("d"), page=1, side="right", column=2),
        ]
        
        # Sort using the merger's logic
        sorted_files = sorted(files, key=merger.get_sort_key)
        
        # Verify order: Page 1 right col3, Page 1 right col2, Page 1 left col2, Page 2 right col1
        assert sorted_files[0].column == 3 and sorted_files[0].side == "right" and sorted_files[0].page == 1
        assert sorted_files[1].column == 2 and sorted_files[1].side == "right" and sorted_files[1].page == 1
        assert sorted_files[2].column == 2 and sorted_files[2].side == "left" and sorted_files[2].page == 1
        assert sorted_files[3].page == 2


class TestTextCleaner:
    """Test text cleaning functionality"""
    
    def test_line_validation(self):
        """Test line validation logic"""
        cleaner = TextCleaner("dummy")
        
        # Valid genealogy line
        valid, reason = cleaner.is_valid_line("長子王大字仲文娶李氏")
        assert valid == True
        
        # Invalid line (mostly numbers)
        valid, reason = cleaner.is_valid_line("12345 67890")
        assert valid == False
        
        # Empty line
        valid, reason = cleaner.is_valid_line("")
        assert valid == False
        
        # Line with keyword but low Chinese ratio
        valid, reason = cleaner.is_valid_line("ABC 子 DEF")
        assert valid == True  # Should pass due to keyword


def test_imports():
    """Test that all modules can be imported"""
    try:
        from pipeline import file_merger, text_cleaner, genealogy_parser
        from utils import file_handler, chinese_detector, logger
        from config import Config
        assert True
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])