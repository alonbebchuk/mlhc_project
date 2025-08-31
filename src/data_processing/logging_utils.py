"""
Logging Utilities for Data Processing Pipeline

This module provides specialized logging functionality with hierarchical nesting
to track function call flows and execution times throughout the data processing pipeline.

The NestedLogger class creates indented log output that visually represents the
call stack depth, making it easy to follow complex nested function executions
and identify performance bottlenecks.

Features:
- Automatic indentation based on call stack depth
- Timestamp logging for performance monitoring
- Simple start/end logging pattern for functions
- Visual hierarchy representation in console output
"""
from datetime import datetime


class NestedLogger:
    """
    A logger that provides hierarchical indentation to visualize function call nesting.
    
    This logger automatically manages indentation levels to create a visual representation
    of function call hierarchies. Each function start increases indentation, and each
    function end decreases it, creating a tree-like structure in the log output.
    
    The logger includes high-precision timestamps (millisecond accuracy) to help
    identify performance bottlenecks in the data processing pipeline.
    
    Attributes:
        _nesting_level (int): Current indentation level (0 = no indentation)
    """

    def __init__(self):
        """Initialize the logger with zero nesting level."""
        self._nesting_level = 0

    def _get_timestamp(self) -> str:
        """
        Get formatted timestamp with millisecond precision.
        
        Returns:
            str: Timestamp in format 'HH:MM:SS.mmm' (hours:minutes:seconds.milliseconds)
        """
        return datetime.now().strftime('%H:%M:%S.%f')[:-3]

    def _get_indent(self) -> str:
        """
        Get indentation string based on current nesting level.
        
        Returns:
            str: String of spaces (4 spaces per nesting level) for visual indentation
        """
        return "    " * self._nesting_level

    def log_start(self, function_name: str) -> None:
        """
        Log the start of a function execution with hierarchical indentation.
        
        Prints a timestamped start message with current indentation level,
        then increases the nesting level for any subsequent nested function calls.
        
        Args:
            function_name (str): Name of the function being started
            
        Example output:
            10:30:45.123 Started extract_data
                10:30:45.124 Started get_cohort_hadm_ids_and_targets
        """
        indent = self._get_indent()
        timestamp = self._get_timestamp()
        print(f"{indent}{timestamp} Started {function_name}")
        self._nesting_level += 1

    def log_end(self, function_name: str) -> None:
        """
        Log the end of a function execution with hierarchical indentation.
        
        Decreases the nesting level and prints a timestamped completion message
        with the updated indentation level.
        
        Args:
            function_name (str): Name of the function being completed
            
        Example output:
                10:30:46.789 Finished get_cohort_hadm_ids_and_targets
            10:30:47.234 Finished extract_data
        """
        # Decrease nesting level (with safety check)
        if self._nesting_level > 0:
            self._nesting_level -= 1
        
        indent = self._get_indent()
        timestamp = self._get_timestamp()
        print(f"{indent}{timestamp} Finished {function_name}")


# Global logger instance for use across the data processing pipeline
# This single instance maintains consistent nesting state across all modules
logger = NestedLogger()