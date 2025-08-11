#!/usr/bin/env python3
"""
Test script for file merger with specific genealogy text files
"""

import sys
import logging
from pathlib import Path
import shutil
import io

# Set up proper encoding for Windows console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from pipeline.file_merger import FileMerger

def setup_test_data():
    """Copy test files to appropriate test directory structure"""
    # Create test directories
    test_input_dir = Path("data/test_merger_input")
    test_output_dir = Path("data/test_merger_output")
    
    # Clean up old test directories if they exist
    if test_input_dir.exists():
        shutil.rmtree(test_input_dir)
    if test_output_dir.exists():
        shutil.rmtree(test_output_dir)
    
    # Create fresh directories
    test_input_dir.mkdir(parents=True, exist_ok=True)
    test_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy the test files
    source_files = [
        Path("data/Shu_Page_090_x2_enhanced_full_refined_deskewed.txt"),
        Path("data/Shu_Page_091_x2_enhanced_full_refined_deskewed.txt")
    ]
    
    for source_file in source_files:
        if source_file.exists():
            # Create a simulated column file structure for testing
            # Since these are full page files, we'll treat them as single column files
            page_num = source_file.name.split("_")[2]  # Extract page number
            
            # Create a test filename that matches the expected pattern
            # Format: PREFIX_Page_XXX_right_border_col01_ID.txt
            test_filename = f"Shu_Page_{page_num}_right_border_col01_test.txt"
            dest_file = test_input_dir / test_filename
            
            # Copy file content
            shutil.copy2(source_file, dest_file)
            print(f"Created test file: {dest_file}")
        else:
            print(f"Warning: Source file not found: {source_file}")
    
    return test_input_dir, test_output_dir

def test_basic_merger():
    """Test basic file merger functionality"""
    print("\n=== Testing Basic File Merger ===\n")
    
    # Setup test data
    input_dir, output_dir = setup_test_data()
    output_file = output_dir / "merged_output.txt"
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create and run merger
    merger = FileMerger(
        input_dir=str(input_dir),
        output_file=str(output_file),
        filename_pattern=r"_Page_(\d+)_([a-z]+)_.*?col(\d+)"
    )
    
    print(f"Input directory: {input_dir}")
    print(f"Output file: {output_file}")
    print("\nRunning merger...\n")
    
    # Run the merge
    result = merger.merge_files(show_progress=True)
    # Check if files were actually processed
    success = result.get('files_processed', 0) > 0
    
    if success:
        print(f"\n[SUCCESS] Merge completed successfully!")
        print(f"  - Files processed: {merger.files_processed}")
        print(f"  - Total characters: {merger.total_characters}")
        
        # Display merged content preview
        if output_file.exists():
            with open(output_file, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
                print(f"\n=== Merged Output Preview (first 30 lines) ===\n")
                for i, line in enumerate(lines[:30], 1):
                    if line.strip():
                        print(f"{i:3}: {line}")
                
                if len(lines) > 30:
                    print(f"\n... ({len(lines)} total lines)")
    else:
        print("\n[FAILED] Merge failed!")
    
    return success

def test_direct_files():
    """Test merging the original files directly without renaming"""
    print("\n=== Testing Direct File Merge (Original Names) ===\n")
    
    # Create test directories
    test_output_dir = Path("data/test_direct_output")
    test_output_dir.mkdir(parents=True, exist_ok=True)
    output_file = test_output_dir / "merged_direct.txt"
    
    # Use a custom pattern that matches the actual filenames
    # The files are named: Shu_Page_087_x2_enhanced_full_refined_deskewed.txt
    custom_pattern = r"Page_(\d+)"  # Simple pattern to extract just page number
    
    # Since these files don't have side/column info, we'll extend FileMerger
    # For now, let's just read and concatenate them in page order
    
    files = [
        Path("data/Shu_Page_090_x2_enhanced_full_refined_deskewed.txt"),
        Path("data/Shu_Page_091_x2_enhanced_full_refined_deskewed.txt")
    ]
    
    print("Processing files in page order:")
    
    merged_content = []
    for file_path in sorted(files):
        if file_path.exists():
            print(f"  - Reading: {file_path.name}")
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # Remove the header line if present
                lines = content.split('\n')
                if lines and lines[0].startswith('========'):
                    lines = lines[1:]
                merged_content.append('\n'.join(lines))
        else:
            print(f"  - File not found: {file_path}")
    
    # Write merged content
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(merged_content))
    
    print(f"\n[SUCCESS] Direct merge completed!")
    print(f"  Output file: {output_file}")
    
    # Display preview
    with open(output_file, 'r', encoding='utf-8') as f:
        content = f.read()
        lines = content.split('\n')
        print(f"\n=== Direct Merge Preview (first 40 lines) ===\n")
        for i, line in enumerate(lines[:40], 1):
            if line.strip():
                print(f"{i:3}: {line}")
        
        if len(lines) > 40:
            print(f"\n... ({len(lines)} total lines)")

if __name__ == "__main__":
    print("=" * 60)
    print("File Merger Test Script")
    print("=" * 60)
    
    # Test 1: Basic merger with renamed files
    test_basic_merger()
    
    # Test 2: Direct merge without renaming
    test_direct_files()
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)