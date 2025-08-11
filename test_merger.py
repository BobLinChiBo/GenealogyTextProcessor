#!/usr/bin/env python3
"""Test the FileMerger with new pattern fallback system"""

import logging
from src.pipeline.file_merger import FileMerger
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Create merger
merger = FileMerger(
    input_dir="data/input",
    output_file="data/intermediate/test_merged.txt"
)

# Test parsing a few filenames
test_files = [
    "Shu_Page_087_x2_enhanced_full_refined_deskewed.txt",  # Should match page_only
    "Wang2017_Page_001_right_border_col5_final.txt",  # Should match full
    "Doc_Page_100_left_enhanced.txt",  # Should match page_side
]

print("\n=== Testing filename parsing ===")
for filename in test_files:
    file_path = Path(filename)
    file_info = merger.parse_filename(file_path)
    if file_info:
        print(f"\n{filename}:")
        print(f"  Pattern type: {file_info.pattern_type}")
        print(f"  Page: {file_info.page}")
        print(f"  Side: {file_info.side}")
        print(f"  Column: {file_info.column}")
    else:
        print(f"\n{filename}: FAILED TO PARSE")

# Run the actual merge
print("\n=== Running merge ===")
stats = merger.merge_files(show_progress=True)

print("\n=== Merge Statistics ===")
print(f"Files found: {stats['files_found']}")
print(f"Files parsed: {stats['files_parsed']}")
print(f"Files processed: {stats['files_processed']}")
print(f"Total characters: {stats['total_characters']:,}")
print(f"Output file: {stats['output_file']}")