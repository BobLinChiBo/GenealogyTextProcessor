#!/usr/bin/env python3
"""Analyze missing pages in the genealogy document"""

from pathlib import Path
import re

# Get all Shu files
input_dir = Path("data/input")
shu_files = list(input_dir.glob("Shu_Page_*.txt"))

# Extract page numbers
page_numbers = []
pattern = re.compile(r"Shu_Page_(\d+)_")

for file in shu_files:
    match = pattern.search(file.name)
    if match:
        page_numbers.append(int(match.group(1)))

# Sort page numbers
page_numbers.sort()

# Find the complete range
min_page = min(page_numbers)
max_page = max(page_numbers)
total_expected = max_page - min_page + 1

print("=" * 60)
print("MISSING PAGES ANALYSIS")
print("=" * 60)
print(f"\nDataset Overview:")
print(f"  • First page: {min_page}")
print(f"  • Last page: {max_page}")
print(f"  • Total pages present: {len(page_numbers)}")
print(f"  • Expected pages (if continuous): {total_expected}")
print(f"  • Missing pages: {total_expected - len(page_numbers)}")

# Find all missing pages
all_pages = set(range(min_page, max_page + 1))
present_pages = set(page_numbers)
missing_pages = sorted(all_pages - present_pages)

# Group consecutive missing pages into ranges
if missing_pages:
    print(f"\n{len(missing_pages)} Missing Pages (grouped by ranges):")
    print("-" * 40)
    
    ranges = []
    start = missing_pages[0]
    end = missing_pages[0]
    
    for i in range(1, len(missing_pages)):
        if missing_pages[i] == missing_pages[i-1] + 1:
            end = missing_pages[i]
        else:
            ranges.append((start, end))
            start = missing_pages[i]
            end = missing_pages[i]
    ranges.append((start, end))
    
    # Print ranges with context
    for start, end in ranges:
        if start == end:
            # Single missing page
            before = start - 1 if start - 1 in present_pages else "—"
            after = start + 1 if start + 1 in present_pages else "—"
            print(f"  • Page {start:3d} (between {before} and {after})")
        else:
            # Range of missing pages
            count = end - start + 1
            before = start - 1 if start - 1 in present_pages else "—"
            after = end + 1 if end + 1 in present_pages else "—"
            print(f"  • Pages {start:3d}-{end:3d} ({count:3d} pages) (after page {before}, before page {after})")
    
    # Summary by size of gaps
    print(f"\nGap Analysis:")
    print("-" * 40)
    gap_sizes = {}
    for start, end in ranges:
        size = end - start + 1
        gap_sizes[size] = gap_sizes.get(size, 0) + 1
    
    for size in sorted(gap_sizes.keys()):
        count = gap_sizes[size]
        if size == 1:
            print(f"  • Single pages missing: {count} occurrence(s)")
        else:
            print(f"  • {size}-page gaps: {count} occurrence(s)")
    
    # Identify sections
    print(f"\nMissing Sections Detail:")
    print("-" * 40)
    
    major_gaps = [(s, e) for s, e in ranges if e - s >= 10]
    minor_gaps = [(s, e) for s, e in ranges if e - s < 10]
    
    if major_gaps:
        print("Major gaps (10+ pages):")
        for start, end in major_gaps:
            print(f"  • Pages {start:3d}-{end:3d}: {end-start+1} pages missing")
            print(f"    Last present before gap: Page {start-1}")
            print(f"    First present after gap: Page {end+1}")
    
    if minor_gaps:
        print("\nMinor gaps (<10 pages):")
        for start, end in minor_gaps:
            if start == end:
                print(f"  • Page {start:3d}")
            else:
                print(f"  • Pages {start:3d}-{end:3d} ({end-start+1} pages)")
else:
    print("\nNo missing pages - the dataset is complete within its range!")

# Check for potential book sections
print(f"\nPotential Book Structure:")
print("-" * 40)
sections = []
current_start = page_numbers[0]

for i in range(1, len(page_numbers)):
    if page_numbers[i] - page_numbers[i-1] > 20:  # Major gap indicates section break
        sections.append((current_start, page_numbers[i-1]))
        current_start = page_numbers[i]
sections.append((current_start, page_numbers[-1]))

if len(sections) > 1:
    print(f"The document appears to have {len(sections)} sections:")
    for i, (start, end) in enumerate(sections, 1):
        page_count = len([p for p in page_numbers if start <= p <= end])
        print(f"  Section {i}: Pages {start:3d}-{end:3d} ({page_count} pages)")
else:
    print("The document appears to be mostly continuous.")

print("\n" + "=" * 60)