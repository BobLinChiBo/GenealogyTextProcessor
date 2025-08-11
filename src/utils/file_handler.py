#!/usr/bin/env python3
"""
file_handler.py

Utility functions for file operations and path management.
"""

import json
import shutil
from pathlib import Path
from typing import Union, List, Optional, Any


def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Directory path
        
    Returns:
        Path object for the directory
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_text_file(filepath: Union[str, Path], 
                   encoding: str = "utf-8",
                   errors: str = "ignore") -> str:
    """
    Read entire text file content.
    
    Args:
        filepath: Path to the file
        encoding: File encoding
        errors: Error handling strategy
        
    Returns:
        File content as string
    """
    with open(filepath, 'r', encoding=encoding, errors=errors) as f:
        return f.read()


def read_lines(filepath: Union[str, Path],
               encoding: str = "utf-8",
               skip_empty: bool = True) -> List[str]:
    """
    Read file lines into a list.
    
    Args:
        filepath: Path to the file
        encoding: File encoding
        skip_empty: Whether to skip empty lines
        
    Returns:
        List of lines
    """
    with open(filepath, 'r', encoding=encoding) as f:
        lines = [line.rstrip('\n') for line in f]
        
    if skip_empty:
        lines = [line for line in lines if line.strip()]
        
    return lines


def write_text_file(filepath: Union[str, Path],
                    content: str,
                    encoding: str = "utf-8",
                    create_dir: bool = True):
    """
    Write text content to a file.
    
    Args:
        filepath: Path to the file
        content: Content to write
        encoding: File encoding
        create_dir: Whether to create parent directory
    """
    filepath = Path(filepath)
    
    if create_dir:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
    with open(filepath, 'w', encoding=encoding) as f:
        f.write(content)


def read_json(filepath: Union[str, Path],
              encoding: str = "utf-8") -> Any:
    """
    Read JSON file.
    
    Args:
        filepath: Path to JSON file
        encoding: File encoding
        
    Returns:
        Parsed JSON content
    """
    with open(filepath, 'r', encoding=encoding) as f:
        return json.load(f)


def write_json(filepath: Union[str, Path],
               data: Any,
               encoding: str = "utf-8",
               ensure_ascii: bool = False,
               indent: int = 2,
               create_dir: bool = True):
    """
    Write data to JSON file.
    
    Args:
        filepath: Path to JSON file
        data: Data to serialize
        encoding: File encoding
        ensure_ascii: Whether to escape non-ASCII characters
        indent: JSON indentation
        create_dir: Whether to create parent directory
    """
    filepath = Path(filepath)
    
    if create_dir:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
    with open(filepath, 'w', encoding=encoding) as f:
        json.dump(data, f, ensure_ascii=ensure_ascii, indent=indent)


def copy_file(source: Union[str, Path],
              destination: Union[str, Path],
              create_dir: bool = True):
    """
    Copy a file to a new location.
    
    Args:
        source: Source file path
        destination: Destination file path
        create_dir: Whether to create parent directory
    """
    source = Path(source)
    destination = Path(destination)
    
    if create_dir:
        destination.parent.mkdir(parents=True, exist_ok=True)
        
    shutil.copy2(source, destination)


def get_file_size(filepath: Union[str, Path]) -> int:
    """
    Get file size in bytes.
    
    Args:
        filepath: Path to the file
        
    Returns:
        File size in bytes
    """
    return Path(filepath).stat().st_size


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def find_files(directory: Union[str, Path],
               pattern: str = "*.txt",
               recursive: bool = True) -> List[Path]:
    """
    Find files matching a pattern.
    
    Args:
        directory: Directory to search
        pattern: Glob pattern
        recursive: Whether to search recursively
        
    Returns:
        List of matching file paths
    """
    directory = Path(directory)
    
    if recursive:
        return list(directory.rglob(pattern))
    else:
        return list(directory.glob(pattern))


def backup_file(filepath: Union[str, Path],
                suffix: str = ".bak") -> Path:
    """
    Create a backup of a file.
    
    Args:
        filepath: Path to the file
        suffix: Backup file suffix
        
    Returns:
        Path to the backup file
    """
    filepath = Path(filepath)
    backup_path = filepath.with_suffix(filepath.suffix + suffix)
    
    # Find unique backup name if already exists
    counter = 1
    while backup_path.exists():
        backup_path = filepath.with_suffix(f"{filepath.suffix}{suffix}.{counter}")
        counter += 1
    
    shutil.copy2(filepath, backup_path)
    return backup_path