# Sample Data Format

This directory contains sample input files demonstrating the expected format for the genealogy text processor.

## File Naming Convention

Files must follow this naming pattern:
```
prefix_Page_[PageNumber]_[Side]_...col[ColumnNumber].txt
```

Where:
- `[PageNumber]`: The page number (e.g., 1, 2, 3)
- `[Side]`: Either "right" or "left" 
- `[ColumnNumber]`: The column number (e.g., 1, 2, 3)

## File Content Format

Each file should contain:
- One Chinese character per line
- No empty lines between characters
- UTF-8 encoding

## Processing Order

Files are processed in the following order:
1. Page numbers: ascending (1, 2, 3...)
2. Within each page: right side before left side
3. Within each side: columns in descending order (col3, col2, col1)

## Example

For a genealogy book with 2 pages, each having 3 columns on both sides:

Processing order would be:
1. `Page_1_right_col3.txt`
2. `Page_1_right_col2.txt`
3. `Page_1_right_col1.txt`
4. `Page_1_left_col3.txt`
5. `Page_1_left_col2.txt`
6. `Page_1_left_col1.txt`
7. `Page_2_right_col3.txt`
8. ... and so on