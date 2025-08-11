#!/usr/bin/env python3
"""
text_cleaner.py

Cleans merged genealogy text by filtering out OCR noise and invalid lines.
Uses multiple strategies including Chinese character ratio and keyword matching.
"""

import re
import logging
from pathlib import Path
from typing import List, Set, Optional, Dict
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class CleaningStats:
    """Statistics from the cleaning process"""
    total_lines: int = 0
    valid_lines: int = 0
    noise_threshold_passed: int = 0
    keyword_matched: int = 0
    empty_lines: int = 0
    noise_lines: int = 0


class TextCleaner:
    """Handles cleaning and filtering of merged genealogy text"""
    
    # Default genealogy keywords - MUST have at least one to be valid
    DEFAULT_KEYWORDS = [
        '公', '子', '氏', '娶', '葬', '字', '生', '卒', 
        '配', '適', '妣', '長', '次', '一', 
        '嗣', '承', '繼', '養', '過', '房', '世', '代',
        '祖', '父', '孫', '曾', '玄', '來', '歸', '育',
        '男', '女', '兄', '弟', '姐', '妹', '母', '妻',
        '夫', '叔', '伯', '姑', '姨', '舅', '侄', '甥'
    ]
    
    # Common OCR noise characters
    OCR_NOISE_CHARS = [
        # --- Common Radicals & Strokes (often OCR errors) ---
        '彳', '囗', '匚', '凵', '厂', '厶', '讠', '纟', 
        '钅', '攵', '刍', '匕', '丿', '丨', '亅', '亠',
        '冫', '冖', '卩', '阝', '廴', '廾', '弋', '弓',
        '彐', '彡', '夂', '夊', '夕', ' 寸', '小', '尢',

        # --- Visually Ambiguous or Rare Characters ---
        '凹', '囧', '圂', '囵', '凸', '乂', '卍', '豸',
        '壨', '匷', '囮', '㠯', 'ゐ', '幺', '爻', '疋',
        '耒', '肀', '聿', '舛', '艮', '韋', '飛', '頁',

        # --- Non-Standard or Private Use Area (PUA) Characters ---
        # These are very likely OCR errors creating non-existent characters.
        '𪉓', '𫏨', '𢪓', '𫎬', '𫄮', '𢀖', '𩙩'
    ]
    
    def __init__(self,
                 input_file: str,
                 output_file: str = "cleaned_text.txt",
                 noise_threshold: float = 0.5,
                 keywords: Optional[List[str]] = None,
                 keywords_file: Optional[str] = None,
                 encoding: str = "utf-8"):
        """
        Initialize the TextCleaner.
        
        Args:
            input_file: Path to the merged text file
            output_file: Path for the cleaned output
            noise_threshold: Minimum ratio of Chinese characters (0-1)
            keywords: List of genealogy keywords to check
            keywords_file: Path to file containing keywords (one per line)
            encoding: File encoding to use
        """
        self.input_file = Path(input_file)
        self.output_file = Path(output_file)
        self.noise_threshold = noise_threshold
        self.encoding = encoding
        
        # Initialize keywords
        self.keywords = set(self._load_keywords(keywords, keywords_file))
        
        # Initialize noise characters
        self.noise_chars = set(self.OCR_NOISE_CHARS)
        
        # Statistics
        self.stats = CleaningStats()
        
        # Store removed lines for detailed review
        self.removed_lines = []
        
    def _load_keywords(self, 
                      keywords: Optional[List[str]], 
                      keywords_file: Optional[str]) -> List[str]:
        """
        Load keywords from various sources.
        
        Args:
            keywords: Direct list of keywords
            keywords_file: Path to keywords file
            
        Returns:
            Combined list of keywords
        """
        result = set(self.DEFAULT_KEYWORDS)
        
        # Add provided keywords
        if keywords:
            result.update(keywords)
        
        # Load from file if provided
        if keywords_file:
            try:
                with open(keywords_file, 'r', encoding=self.encoding) as f:
                    file_keywords = [line.strip() for line in f if line.strip()]
                    result.update(file_keywords)
                    logger.info(f"Loaded {len(file_keywords)} keywords from {keywords_file}")
            except Exception as e:
                logger.warning(f"Failed to load keywords from {keywords_file}: {e}")
        
        return list(result)
    
    def calculate_chinese_ratio(self, text: str) -> float:
        """
        Calculate the ratio of Chinese characters in a text.
        
        Args:
            text: Input text
            
        Returns:
            Ratio of Chinese characters (0-1)
        """
        if not text:
            return 0.0
            
        chinese_chars = self.chinese_pattern.findall(text)
        return len(chinese_chars) / len(text)
    
    def contains_keywords(self, text: str) -> bool:
        """
        Check if text contains any genealogy keywords.
        
        Args:
            text: Input text
            
        Returns:
            True if any keyword is found
        """
        return any(keyword in text for keyword in self.keywords)
    
    def count_noise_chars(self, text: str) -> int:
        """
        Count OCR noise characters in text.
        
        Args:
            text: Input text
            
        Returns:
            Number of noise characters
        """
        return sum(1 for char in text if char in self.noise_chars)
    
    def is_valid_line(self, line: str) -> tuple[bool, str]:
        """
        Determine if a line is valid genealogy content.
        
        Args:
            line: Input line
            
        Returns:
            Tuple of (is_valid, reason)
        """
        stripped = line.strip()
        
        # Stage 1: Check for empty line
        if not stripped:
            self.stats.empty_lines += 1
            return False, "empty"
        
        # Stage 2: MUST contain genealogy keywords (critical check)
        if not self.contains_keywords(stripped):
            self.stats.noise_lines += 1
            return False, "no_keywords"
        
        # Stage 3: Check for excessive noise characters
        noise_count = self.count_noise_chars(stripped)
        noise_ratio = noise_count / len(stripped)
        
        if noise_ratio > self.noise_threshold:  # More than threshold% noise characters
            self.stats.noise_lines += 1
            return False, "high_noise_ratio"
        
        # Line passed all checks
        self.stats.keyword_matched += 1
        return True, "valid_genealogy"
    
    def clean_text(self, save_stats: bool = True) -> Dict:
        """
        Main method to clean the text file.
        
        Args:
            save_stats: Whether to save cleaning statistics
            
        Returns:
            Dictionary with cleaning statistics
        """
        # Validate input file
        if not self.input_file.exists():
            raise FileNotFoundError(f"Input file '{self.input_file}' not found")
        
        # Create output directory if needed
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting text cleaning from '{self.input_file}'")
        
        # Clear removed lines list
        self.removed_lines = []
        
        # Process file
        with open(self.input_file, 'r', encoding=self.encoding) as infile, \
             open(self.output_file, 'w', encoding=self.encoding) as outfile:
            
            for line_num, line in enumerate(infile, 1):
                self.stats.total_lines += 1
                
                # Show progress
                if line_num % 100 == 0:
                    logger.debug(f"Processing line {line_num}...")
                
                # Check if line is valid
                is_valid, reason = self.is_valid_line(line)
                
                if is_valid:
                    outfile.write(line)
                    self.stats.valid_lines += 1
                else:
                    # Store removed line with its line number and reason
                    self.removed_lines.append({
                        'line_number': line_num,
                        'reason': reason,
                        'text': line.rstrip('\n')
                    })
        
        # Save removed lines to file
        removed_lines_file = self.output_file.parent / f"{self.output_file.stem}_removed.txt"
        self._save_removed_lines(removed_lines_file)
        
        # Save statistics if requested
        if save_stats:
            stats_file = self.output_file.with_suffix('.stats.txt')
            self._save_statistics(stats_file)
        
        # Log summary
        logger.info(f"Cleaning complete. Kept {self.stats.valid_lines}/{self.stats.total_lines} lines")
        logger.info(f"Output saved to '{self.output_file}'")
        logger.info(f"Removed lines saved to '{removed_lines_file}'")
        
        return self._get_stats_dict()
    
    def _save_removed_lines(self, removed_lines_file: Path):
        """Save removed lines to a file for review"""
        with open(removed_lines_file, 'w', encoding=self.encoding) as f:
            f.write("Removed Lines During Text Cleaning\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Total lines removed: {len(self.removed_lines):,}\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Group removed lines by reason
            by_reason = {}
            for item in self.removed_lines:
                reason = item['reason']
                if reason not in by_reason:
                    by_reason[reason] = []
                by_reason[reason].append(item)
            
            # Write summary by reason
            f.write("Summary by Removal Reason:\n")
            f.write("-" * 40 + "\n")
            for reason, items in by_reason.items():
                reason_display = {
                    'empty': 'Empty lines',
                    'no_keywords': 'No genealogy keywords found',
                    'high_noise_ratio': 'High noise character ratio'
                }.get(reason, reason)
                f.write(f"{reason_display}: {len(items):,} lines\n")
            
            # Write detailed removed lines
            f.write("\n" + "=" * 70 + "\n")
            f.write("Detailed Removed Lines:\n")
            f.write("=" * 70 + "\n\n")
            
            for item in self.removed_lines:
                f.write(f"Line {item['line_number']} [{item['reason']}]: {item['text']}\n")
            
            f.write("\n" + "=" * 70 + "\n")
            f.write(f"End of removed lines report\n")
    
    def _save_statistics(self, stats_file: Path):
        """Save cleaning statistics to a file"""
        with open(stats_file, 'w', encoding=self.encoding) as f:
            f.write("Text Cleaning Statistics\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total lines processed: {self.stats.total_lines:,}\n")
            f.write(f"Valid lines kept: {self.stats.valid_lines:,}\n")
            f.write(f"Retention rate: {self.stats.valid_lines/self.stats.total_lines*100:.1f}%\n\n")
            f.write("Lines Filtered Out:\n")
            f.write(f"  - Empty lines: {self.stats.empty_lines:,}\n")
            f.write(f"  - Noise/Invalid lines: {self.stats.noise_lines:,}\n")
            f.write(f"    (no keywords or high noise ratio)\n\n")
            f.write("Valid Lines:\n")
            f.write(f"  - With genealogy keywords: {self.stats.keyword_matched:,}\n\n")
            f.write(f"Keywords used: {len(self.keywords)}\n")
            f.write(f"Noise characters defined: {len(self.noise_chars)}\n")
    
    def _get_stats_dict(self) -> Dict:
        """Convert statistics to dictionary"""
        return {
            "total_lines": self.stats.total_lines,
            "valid_lines": self.stats.valid_lines,
            "retention_rate": self.stats.valid_lines / self.stats.total_lines if self.stats.total_lines > 0 else 0,
            "keyword_matched": self.stats.keyword_matched,
            "empty_lines": self.stats.empty_lines,
            "noise_lines": self.stats.noise_lines,
            "output_file": str(self.output_file)
        }


def main():
    """Command-line interface for the text cleaner"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Clean merged genealogy text by filtering OCR noise",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "input_file",
        nargs="?",
        default="data/intermediate/merged_text.txt",
        help="Input file to clean (default: data/intermediate/merged_text.txt)"
    )
    parser.add_argument(
        "-o", "--output",
        default="data/intermediate/cleaned_text.txt",
        help="Output file path (default: data/intermediate/cleaned_text.txt)"
    )
    parser.add_argument(
        "-t", "--threshold",
        type=float,
        default=0.5,
        help="Chinese character ratio threshold (default: 0.5)"
    )
    parser.add_argument(
        "-k", "--keywords",
        nargs="+",
        help="Additional keywords to check"
    )
    parser.add_argument(
        "-f", "--keywords-file",
        help="File containing keywords (one per line)"
    )
    parser.add_argument(
        "-e", "--encoding",
        default="utf-8",
        help="File encoding (default: utf-8)"
    )
    parser.add_argument(
        "--no-stats",
        action="store_true",
        help="Don't save statistics file"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Create cleaner and run
        cleaner = TextCleaner(
            input_file=args.input_file,
            output_file=args.output,
            noise_threshold=args.threshold,
            keywords=args.keywords,
            keywords_file=args.keywords_file,
            encoding=args.encoding
        )
        
        stats = cleaner.clean_text(save_stats=not args.no_stats)
        
        # Print summary
        print("\nCleaning Summary:")
        print(f"  Total lines: {stats['total_lines']:,}")
        print(f"  Valid lines: {stats['valid_lines']:,}")
        print(f"  Retention rate: {stats['retention_rate']*100:.1f}%")
        print(f"  Output saved to: {stats['output_file']}")
        
    except Exception as e:
        logger.error(f"Error during cleaning: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())