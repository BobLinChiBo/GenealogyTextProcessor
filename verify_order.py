#!/usr/bin/env python3
"""Verify the page ordering in the merged file"""

from pathlib import Path
from src.pipeline.file_merger import FileMerger

# Create merger
merger = FileMerger(
    input_dir="data/input",
    output_file="data/intermediate/test_merged.txt"
)

# Get all txt files and parse them
input_dir = Path("data/input")
txt_files = list(input_dir.glob("Shu_Page_*.txt"))

# Parse and sort
file_infos = []
for filepath in txt_files:
    file_info = merger.parse_filename(filepath)
    if file_info:
        file_infos.append(file_info)

# Sort using the merger's sort key
file_infos.sort(key=merger.get_sort_key)

# Print the sorted order
print("Page order after sorting:")
prev_page = None
for i, file_info in enumerate(file_infos):
    if prev_page and file_info.page != prev_page + 1:
        print(f"  ... gap from {prev_page} to {file_info.page}")
    if i < 10 or i >= len(file_infos) - 5:
        print(f"  {i+1:3d}. Page {file_info.page:3d}: {file_info.path.name}")
    elif i == 10:
        print(f"  ... ({len(file_infos) - 15} more files)")
    prev_page = file_info.page

# Verify ascending order
pages = [f.page for f in file_infos]
is_sorted = all(pages[i] <= pages[i+1] for i in range(len(pages)-1))
print(f"\n✓ Pages are in ascending order: {is_sorted}")
print(f"✓ Total files: {len(file_infos)}")
print(f"✓ Page range: {min(pages)} to {max(pages)}")