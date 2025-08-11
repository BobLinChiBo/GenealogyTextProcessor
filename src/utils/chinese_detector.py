#!/usr/bin/env python3
"""
chinese_detector.py

Utility functions for detecting and analyzing Chinese text.
"""

import re
from typing import Dict, List, Tuple, Set, Optional
from collections import Counter


# Unicode ranges for Chinese characters
CJK_UNIFIED_IDEOGRAPHS = (0x4E00, 0x9FFF)  # Main block
CJK_EXTENSION_A = (0x3400, 0x4DBF)
CJK_EXTENSION_B = (0x20000, 0x2A6DF)
CJK_EXTENSION_C = (0x2A700, 0x2B73F)
CJK_EXTENSION_D = (0x2B740, 0x2B81F)
CJK_EXTENSION_E = (0x2B820, 0x2CEAF)
CJK_RADICALS_SUPPLEMENT = (0x2E80, 0x2EFF)
CJK_SYMBOLS_AND_PUNCTUATION = (0x3000, 0x303F)


def is_chinese_char(char: str) -> bool:
    """
    Check if a character is Chinese.
    
    Args:
        char: Single character to check
        
    Returns:
        True if the character is Chinese
    """
    if len(char) != 1:
        return False
        
    code_point = ord(char)
    
    # Check main CJK block first (most common)
    if CJK_UNIFIED_IDEOGRAPHS[0] <= code_point <= CJK_UNIFIED_IDEOGRAPHS[1]:
        return True
        
    # Check other blocks
    ranges = [
        CJK_EXTENSION_A,
        CJK_RADICALS_SUPPLEMENT,
        CJK_SYMBOLS_AND_PUNCTUATION
    ]
    
    for start, end in ranges:
        if start <= code_point <= end:
            return True
            
    return False


def count_chinese_chars(text: str) -> int:
    """
    Count the number of Chinese characters in text.
    
    Args:
        text: Input text
        
    Returns:
        Number of Chinese characters
    """
    return sum(1 for char in text if is_chinese_char(char))


def chinese_ratio(text: str) -> float:
    """
    Calculate the ratio of Chinese characters in text.
    
    Args:
        text: Input text
        
    Returns:
        Ratio of Chinese characters (0.0 to 1.0)
    """
    if not text:
        return 0.0
        
    total_chars = len(text)
    chinese_chars = count_chinese_chars(text)
    
    return chinese_chars / total_chars


def extract_chinese_text(text: str) -> str:
    """
    Extract only Chinese characters from text.
    
    Args:
        text: Input text
        
    Returns:
        String containing only Chinese characters
    """
    return ''.join(char for char in text if is_chinese_char(char))


def analyze_chinese_content(text: str) -> Dict:
    """
    Analyze Chinese content in text.
    
    Args:
        text: Input text
        
    Returns:
        Dictionary with analysis results
    """
    total_chars = len(text)
    chinese_chars = count_chinese_chars(text)
    non_chinese_chars = total_chars - chinese_chars
    
    # Count specific types of characters
    digits = sum(1 for char in text if char.isdigit())
    letters = sum(1 for char in text if char.isalpha() and not is_chinese_char(char))
    spaces = sum(1 for char in text if char.isspace())
    punctuation = sum(1 for char in text if char in '，。、！？；：""''（）《》【】')
    
    return {
        'total_characters': total_chars,
        'chinese_characters': chinese_chars,
        'chinese_ratio': chinese_chars / total_chars if total_chars > 0 else 0,
        'non_chinese_characters': non_chinese_chars,
        'digits': digits,
        'latin_letters': letters,
        'spaces': spaces,
        'chinese_punctuation': punctuation,
        'other': total_chars - chinese_chars - digits - letters - spaces - punctuation
    }


def find_chinese_names(text: str, min_length: int = 2, max_length: int = 4) -> List[str]:
    """
    Find potential Chinese names in text.
    
    Args:
        text: Input text
        min_length: Minimum name length
        max_length: Maximum name length
        
    Returns:
        List of potential names
    """
    # Common surname characters
    common_surnames = set('王李張劉陳楊趙黃周吳徐孫朱馬胡郭林何高梁郑謝羅唐韋馮姜沈呂盧蔣蔡賈魏薛葉閻余潘杜戴夏鐘汪田任姚彭呂錢')
    
    # Pattern for potential names
    chinese_pattern = f'[\\u4e00-\\u9fff]{{{min_length},{max_length}}}'
    potential_names = re.findall(chinese_pattern, text)
    
    # Filter based on surname
    names = []
    for name in potential_names:
        if len(name) >= 2 and name[0] in common_surnames:
            names.append(name)
            
    return names


def detect_traditional_simplified(text: str) -> str:
    """
    Detect if text is primarily traditional or simplified Chinese.
    
    Args:
        text: Input text
        
    Returns:
        'traditional', 'simplified', or 'mixed'
    """
    # Sample characters that differ between traditional and simplified
    traditional_chars = set('國學習電腦機車書館圖說話語請謝對錯門開關愛樂觀點線紅藍綠黃銀錢財產業務員')
    simplified_chars = set('国学习电脑机车书馆图说话语请谢对错门开关爱乐观点线红蓝绿黄银钱财产业务员')
    
    trad_count = sum(1 for char in text if char in traditional_chars)
    simp_count = sum(1 for char in text if char in simplified_chars)
    
    if trad_count > simp_count * 2:
        return 'traditional'
    elif simp_count > trad_count * 2:
        return 'simplified'
    else:
        return 'mixed'


def get_character_frequency(text: str, top_n: int = 10) -> List[Tuple[str, int]]:
    """
    Get frequency of Chinese characters in text.
    
    Args:
        text: Input text
        top_n: Number of most frequent characters to return
        
    Returns:
        List of (character, count) tuples
    """
    chinese_chars = [char for char in text if is_chinese_char(char)]
    char_counts = Counter(chinese_chars)
    
    return char_counts.most_common(top_n)


def contains_genealogy_keywords(text: str, custom_keywords: Optional[Set[str]] = None) -> bool:
    """
    Check if text contains common genealogy keywords.
    
    Args:
        text: Input text
        custom_keywords: Additional keywords to check
        
    Returns:
        True if genealogy keywords are found
    """
    # Strong genealogy indicators (single occurrence is enough)
    strong_keywords = {
        '娶', '葬', '妣', '嗣', '承', '繼', '適', '配'
    }
    
    # Common words that need context (require at least 2 different ones)
    context_keywords = {
        '公', '子', '氏', '生', '卒', '字',
        '長', '次', '三', '四', '五',
        '養', '過', '房', '世', '代',
        '祖', '父', '孫', '曾', '玄', '來', '歸', '育',
        '男', '女', '妻', '夫', '兄', '弟', '姐', '妹'
    }
    
    if custom_keywords:
        context_keywords.update(custom_keywords)
    
    # Check for strong keywords
    for keyword in strong_keywords:
        if keyword in text:
            return True
    
    # Check for context keywords (need at least 2 different ones)
    found_keywords = sum(1 for keyword in context_keywords if keyword in text)
    return found_keywords >= 2