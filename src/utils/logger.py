#!/usr/bin/env python3
"""
logger.py

Logging configuration and utilities for the genealogy processor.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Union


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for console output"""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        # Add color to level name
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.RESET}"
        
        # Format the message
        result = super().format(record)
        
        # Reset levelname for other handlers
        record.levelname = levelname
        
        return result


def setup_logging(name: str = "genealogy",
                  level: Union[str, int] = "INFO",
                  log_dir: Optional[Union[str, Path]] = None,
                  console: bool = True,
                  file: bool = True,
                  colored: bool = True) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        name: Logger name
        level: Logging level
        log_dir: Directory for log files
        console: Whether to log to console
        file: Whether to log to file
        colored: Whether to use colored console output
        
    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    if colored and sys.platform != 'win32':  # Colors might not work well on Windows
        console_formatter = ColoredFormatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
    else:
        console_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
    
    # Add console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # Add file handler
    if file:
        if log_dir is None:
            log_dir = Path("logs")
        else:
            log_dir = Path(log_dir)
            
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create log filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"{name}_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        # Also create a symlink to latest log
        latest_log = log_dir / f"{name}_latest.log"
        if latest_log.exists():
            latest_log.unlink()
        try:
            latest_log.symlink_to(log_file.name)
        except:
            # Symlinks might not work on Windows
            pass
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class LoggingContext:
    """Context manager for temporary logging configuration"""
    
    def __init__(self, logger: logging.Logger, level: Union[str, int]):
        self.logger = logger
        self.new_level = level
        self.old_level = logger.level
        
    def __enter__(self):
        self.logger.setLevel(self.new_level)
        return self.logger
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.setLevel(self.old_level)


def log_execution_time(func):
    """Decorator to log function execution time"""
    import functools
    import time
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.debug(f"{func.__name__} completed in {execution_time:.2f} seconds")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"{func.__name__} failed after {execution_time:.2f} seconds: {e}")
            raise
    
    return wrapper


def log_progress(iterable, logger: logging.Logger, 
                 message: str = "Processing",
                 log_interval: int = 100):
    """
    Log progress while iterating.
    
    Args:
        iterable: Iterable to process
        logger: Logger instance
        message: Progress message prefix
        log_interval: How often to log progress
        
    Yields:
        Items from the iterable
    """
    for i, item in enumerate(iterable):
        if i > 0 and i % log_interval == 0:
            logger.info(f"{message}: {i} items processed")
        yield item


def create_file_logger(name: str,
                      log_file: Union[str, Path],
                      level: Union[str, int] = "INFO") -> logging.Logger:
    """
    Create a logger that only logs to a specific file.
    
    Args:
        name: Logger name
        log_file: Path to log file
        level: Logging level
        
    Returns:
        File logger
    """
    logger = logging.getLogger(f"{name}_file")
    logger.setLevel(level)
    logger.handlers.clear()
    
    # Create file handler
    handler = logging.FileHandler(log_file, encoding='utf-8')
    handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    ))
    
    logger.addHandler(handler)
    return logger