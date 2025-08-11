#!/usr/bin/env python3
"""
file_merger.py

Merges individual character text files into a single consolidated file.
Handles the specific ordering requirements for Chinese genealogy texts:
- Page numbers in ascending order
- Within each page: right side before left side
- Within each side: columns from highest to lowest number
"""

import re
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class FileInfo:
    """Container for parsed file information"""
    path: Path
    page: int
    side: Optional[str] = None  # Optional for files without side info
    column: Optional[int] = None  # Optional for files without column info
    pattern_type: str = "page_only"  # Track which pattern matched


class FileMerger:
    """Handles merging of individual character files into consolidated text"""
    
    def __init__(self, 
                 input_dir: str,
                 output_file: str = "merged_text.txt",
                 filename_pattern: Optional[str] = None,
                 filename_patterns: Optional[List[Dict[str, str]]] = None,
                 encoding: str = "utf-8"):
        """
        Initialize the FileMerger.
        
        Args:
            input_dir: Directory containing input text files
            output_file: Path for the merged output file
            filename_pattern: Legacy single regex pattern for parsing filenames
            filename_patterns: List of pattern dictionaries with 'pattern' and 'type' keys
            encoding: File encoding to use
        """
        self.input_dir = Path(input_dir)
        self.output_file = Path(output_file)
        self.encoding = encoding
        
        # Set up filename patterns (support both legacy single pattern and new multi-pattern)
        if filename_patterns:
            self.filename_patterns = filename_patterns
        elif filename_pattern:
            # Convert legacy single pattern to new format
            self.filename_patterns = [{
                "pattern": filename_pattern,
                "type": "full",
                "description": "Legacy pattern"
            }]
        else:
            # Default patterns for different file formats
            self.filename_patterns = [
                {
                    "pattern": r".*_Page_(\d+)_([a-z]+).*_col(\d+)",
                    "type": "full",
                    "description": "Full pattern with page, side, and column"
                },
                {
                    "pattern": r".*_Page_(\d+)_([a-z]+)_",
                    "type": "page_side",
                    "description": "Page and side without column"
                },
                {
                    "pattern": r".*_Page_(\d+)_",
                    "type": "page_only",
                    "description": "Simple page number only"
                }
            ]
        
        # Compile all patterns
        self.compiled_patterns = [
            (re.compile(p["pattern"], re.IGNORECASE), p["type"], p.get("description", ""))
            for p in self.filename_patterns
        ]
        
        self.files_processed = 0
        self.total_characters = 0
        
    def parse_filename(self, filepath: Path) -> Optional[FileInfo]:
        """
        Extract page number, side, and column from filename using multiple patterns.
        Tries each pattern in order until one matches.
        
        Args:
            filepath: Path to the file
            
        Returns:
            FileInfo object or None if parsing fails
        """
        for regex, pattern_type, description in self.compiled_patterns:
            match = regex.search(filepath.name)
            if match:
                try:
                    # Always extract page number (group 1)
                    page = int(match.group(1))
                    
                    # Create FileInfo based on pattern type
                    if pattern_type == "full":
                        # Full pattern: page, side, column
                        side = match.group(2).lower()
                        column = int(match.group(3))
                        logger.debug(f"Matched '{filepath.name}' with full pattern: page={page}, side={side}, col={column}")
                        return FileInfo(path=filepath, page=page, side=side, column=column, pattern_type="full")
                    
                    elif pattern_type == "page_side":
                        # Page and side only
                        side = match.group(2).lower()
                        logger.debug(f"Matched '{filepath.name}' with page_side pattern: page={page}, side={side}")
                        return FileInfo(path=filepath, page=page, side=side, column=None, pattern_type="page_side")
                    
                    elif pattern_type == "page_only":
                        # Page only
                        logger.debug(f"Matched '{filepath.name}' with page_only pattern: page={page}")
                        return FileInfo(path=filepath, page=page, side=None, column=None, pattern_type="page_only")
                    
                    else:
                        # Unknown pattern type, treat as page_only
                        logger.warning(f"Unknown pattern type '{pattern_type}', treating as page_only")
                        return FileInfo(path=filepath, page=page, side=None, column=None, pattern_type="page_only")
                        
                except (ValueError, IndexError) as e:
                    logger.debug(f"Failed to parse '{filepath.name}' with pattern '{description}': {e}")
                    continue  # Try next pattern
        
        # No patterns matched
        logger.warning(f"Filename '{filepath.name}' doesn't match any expected patterns")
        return None
    
    def get_sort_key(self, file_info: FileInfo) -> Tuple[int, int, int]:
        """
        Generate sort key for a file based on ordering rules.
        Adapts to the available fields from the pattern that matched.
        
        Args:
            file_info: FileInfo object
            
        Returns:
            Tuple for sorting (page, side_priority, negative_column)
        """
        # Page number is always the primary sort key
        page = file_info.page
        
        # Side priority (if side info exists)
        if file_info.side:
            # Right side has priority 0, left side has priority 1
            side_priority = 0 if file_info.side.startswith("right") else 1
        else:
            # No side info, use neutral priority
            side_priority = 0
        
        # Column sorting (if column info exists)
        if file_info.column is not None:
            # Negative column for descending order within each side
            column_sort = -file_info.column
        else:
            # No column info, use neutral value
            column_sort = 0
        
        return (page, side_priority, column_sort)
    
    def read_characters(self, filepath: Path) -> str:
        """
        Read all characters from a file and join them into paragraphs.
        Preserves paragraph breaks (empty lines) in the output.
        Filters out header lines that match the pattern ======== filename ========
        
        Args:
            filepath: Path to the file
            
        Returns:
            String with paragraphs separated by newlines
        """
        try:
            with filepath.open("r", encoding=self.encoding, errors="ignore") as f:
                paragraphs = []
                current_paragraph = []
                
                for line in f:
                    # Remove BOM character if present
                    if line.startswith('\ufeff'):
                        line = line[1:]
                    
                    # Skip header lines that contain ======== pattern
                    if "========" in line and line.strip().startswith("========") and line.strip().endswith("========"):
                        continue
                    
                    line = line.strip()
                    
                    if not line:
                        # Empty line marks end of paragraph
                        if current_paragraph:
                            paragraphs.append(" ".join(current_paragraph))
                            current_paragraph = []
                    else:
                        # Add non-empty line to current paragraph
                        current_paragraph.append(line)
                
                # Don't forget the last paragraph if exists
                if current_paragraph:
                    paragraphs.append(" ".join(current_paragraph))
                
                # Join paragraphs with newlines
                return "\n".join(paragraphs)
        except Exception as e:
            logger.error(f"Error reading file '{filepath}': {e}")
            return ""
    
    def merge_files(self, show_progress: bool = True) -> dict:
        """
        Main method to merge all files according to the sorting rules.
        
        Args:
            show_progress: Whether to show progress messages
            
        Returns:
            Dictionary with merge statistics
        """
        # Validate input directory
        if not self.input_dir.exists() or not self.input_dir.is_dir():
            raise ValueError(f"Input directory '{self.input_dir}' doesn't exist or is not a directory")
        
        # Find all text files
        txt_files = list(self.input_dir.glob("*.txt"))
        if not txt_files:
            raise ValueError(f"No .txt files found in '{self.input_dir}'")
        
        logger.info(f"Found {len(txt_files)} text files to process")
        
        # Parse filenames and filter valid files
        file_infos: List[FileInfo] = []
        pattern_counts = {"full": 0, "page_side": 0, "page_only": 0}
        
        for filepath in txt_files:
            file_info = self.parse_filename(filepath)
            if file_info:
                file_infos.append(file_info)
                pattern_counts[file_info.pattern_type] = pattern_counts.get(file_info.pattern_type, 0) + 1
        
        # Log pattern matching statistics
        logger.info(f"Pattern matching results: {pattern_counts}")
        
        if not file_infos:
            raise ValueError("No files could be parsed with the current filename pattern")
        
        # Sort files according to rules
        file_infos.sort(key=self.get_sort_key)
        
        # Create output directory if needed
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Merge files
        with self.output_file.open("w", encoding=self.encoding) as output:
            for i, file_info in enumerate(file_infos):
                if show_progress and i % 10 == 0:
                    logger.info(f"Processing file {i+1}/{len(file_infos)}...")
                
                line_content = self.read_characters(file_info.path)
                if line_content:
                    output.write(line_content + "\n")
                    self.files_processed += 1
                    self.total_characters += len(line_content)
        
        # Prepare statistics
        stats = {
            "files_found": len(txt_files),
            "files_parsed": len(file_infos),
            "files_processed": self.files_processed,
            "total_characters": self.total_characters,
            "output_file": str(self.output_file)
        }
        
        logger.info(f"Successfully merged {self.files_processed} files into '{self.output_file}'")
        logger.info(f"Total characters written: {self.total_characters:,}")
        
        return stats


def main():
    """Command-line interface for the file merger"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Merge individual character text files into a consolidated file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ordering rules:
  - Pages are processed in ascending order (1, 2, 3, ...)
  - Within each page: right side before left side
  - Within each side: columns from highest to lowest (col5, col4, col3, ...)
        """
    )
    
    parser.add_argument(
        "input_dir",
        help="Directory containing the text files to merge"
    )
    parser.add_argument(
        "-o", "--output",
        default="data/intermediate/merged_text.txt",
        help="Output file path (default: data/intermediate/merged_text.txt)"
    )
    parser.add_argument(
        "-p", "--pattern",
        help="Custom regex pattern for parsing filenames"
    )
    parser.add_argument(
        "-e", "--encoding",
        default="utf-8",
        help="File encoding (default: utf-8)"
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
        # Create merger and run
        merger = FileMerger(
            input_dir=args.input_dir,
            output_file=args.output,
            filename_pattern=args.pattern,
            encoding=args.encoding
        )
        
        stats = merger.merge_files(show_progress=True)
        
        # Print summary
        print("\nMerge Summary:")
        print(f"  Files found: {stats['files_found']}")
        print(f"  Files parsed: {stats['files_parsed']}")
        print(f"  Files processed: {stats['files_processed']}")
        print(f"  Total characters: {stats['total_characters']:,}")
        print(f"  Output saved to: {stats['output_file']}")
        
    except Exception as e:
        logger.error(f"Error during merge: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())