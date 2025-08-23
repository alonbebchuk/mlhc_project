from datetime import datetime


class NestedLogger:
    """A logger that provides tab nesting to show function call hierarchy."""

    def __init__(self):
        self._nesting_level = 0

    def _get_timestamp(self) -> str:
        """Get formatted timestamp."""
        return datetime.now().strftime('%H:%M:%S.%f')[:-3]

    def _get_indent(self) -> str:
        """Get indentation string based on current nesting level."""
        return "    " * self._nesting_level

    def log_start(self, function_name: str) -> None:
        """Log the start of a function with indentation."""
        indent = self._get_indent()
        timestamp = self._get_timestamp()
        print(f"{indent}{timestamp} Started {function_name}")
        self._nesting_level += 1

    def log_end(self, function_name: str) -> None:
        """Log the end of a function with indentation."""
        if self._nesting_level > 0:
            self._nesting_level -= 1
        indent = self._get_indent()
        timestamp = self._get_timestamp()
        print(f"{indent}{timestamp} Finished {function_name}")


# Global logger instance
logger = NestedLogger()