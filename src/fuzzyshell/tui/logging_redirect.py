"""
Comprehensive logging redirection for TUI mode.
Captures ALL external library output and redirects to debug console.
"""

import logging
import sys
import io
import contextlib
from typing import Optional, List


class TUILogCapture:
    """Captures and redirects ALL logging output during TUI mode."""
    
    def __init__(self):
        self.original_handlers: dict = {}
        self.original_levels: dict = {}
        self.captured_loggers: List[str] = []
        self.debug_handler: Optional[logging.Handler] = None
        self.active = False
        
        # Create a debug handler that writes to our debug console
        self.debug_handler = logging.FileHandler('debug.log', mode='a')
        self.debug_handler.setFormatter(logging.Formatter(
            '%(asctime)s - TUI_REDIRECT - %(name)s - %(levelname)s - %(message)s'
        ))
    
    def start_capture(self):
        """Start capturing all external library logging."""
        if self.active:
            return
            
        self.active = True
        
        # Known external libraries that log during model loading
        external_loggers = [
            'transformers',
            'tokenizers', 
            'huggingface_hub',
            'onnxruntime',
            'urllib3',
            'requests',
            'filelock',
            'tqdm',
            # Root logger to catch anything else
            ''
        ]
        
        for logger_name in external_loggers:
            logger = logging.getLogger(logger_name)
            
            # Store original configuration
            self.original_handlers[logger_name] = logger.handlers.copy()
            self.original_levels[logger_name] = logger.level
            self.captured_loggers.append(logger_name)
            
            # Remove all existing handlers
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)
            
            # Add our debug handler
            logger.addHandler(self.debug_handler)
            
            # Set level to capture everything but don't let it bubble up
            logger.setLevel(logging.DEBUG)
            logger.propagate = False
    
    def stop_capture(self):
        """Restore original logging configuration."""
        if not self.active:
            return
            
        self.active = False
        
        # Restore all captured loggers
        for logger_name in self.captured_loggers:
            logger = logging.getLogger(logger_name)
            
            # Remove our handler
            if self.debug_handler in logger.handlers:
                logger.removeHandler(self.debug_handler)
            
            # Restore original handlers
            for handler in self.original_handlers.get(logger_name, []):
                logger.addHandler(handler)
            
            # Restore original level
            logger.setLevel(self.original_levels.get(logger_name, logging.WARNING))
            logger.propagate = True
        
        # Clear tracking
        self.captured_loggers.clear()
        self.original_handlers.clear()
        self.original_levels.clear()


class StdoutCapture:
    """Captures stdout/stderr during TUI mode."""
    
    def __init__(self):
        self.original_stdout = None
        self.original_stderr = None
        self.captured_output = None
        self.active = False
    
    def start_capture(self):
        """Start capturing stdout/stderr."""
        if self.active:
            return
            
        self.active = True
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        self.captured_output = io.StringIO()
        
        # Redirect both stdout and stderr to our capture
        sys.stdout = self.captured_output
        sys.stderr = self.captured_output
    
    def stop_capture(self):
        """Restore original stdout/stderr."""
        if not self.active:
            return
            
        self.active = False
        
        # Write captured output to debug log
        if self.captured_output:
            output = self.captured_output.getvalue()
            if output.strip():
                with open('debug.log', 'a') as f:
                    f.write(f"\n--- TUI STDOUT/STDERR CAPTURE ---\n{output}\n--- END CAPTURE ---\n")
        
        # Restore original streams
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        
        if self.captured_output:
            self.captured_output.close()
            self.captured_output = None


class ComprehensiveTUIRedirect:
    """Comprehensive redirection of ALL output during TUI mode."""
    
    def __init__(self):
        self.log_capture = TUILogCapture()
        self.stdout_capture = StdoutCapture()
        self.active = False
    
    def start_redirect(self):
        """Start comprehensive output redirection."""
        if self.active:
            return
            
        self.active = True
        
        # Start all capture mechanisms
        self.log_capture.start_capture()
        self.stdout_capture.start_capture()
        
        # Log that redirection is active
        with open('debug.log', 'a') as f:
            f.write(f"\n=== TUI MODE REDIRECTION STARTED ===\n")
    
    def stop_redirect(self):
        """Stop all output redirection."""
        if not self.active:
            return
            
        self.active = False
        
        # Stop all capture mechanisms
        self.stdout_capture.stop_capture()
        self.log_capture.stop_capture()
        
        # Log that redirection is stopped
        with open('debug.log', 'a') as f:
            f.write(f"=== TUI MODE REDIRECTION STOPPED ===\n\n")


# Global redirect instance
_tui_redirect = ComprehensiveTUIRedirect()


def start_tui_redirect():
    """Start comprehensive TUI output redirection."""
    _tui_redirect.start_redirect()


def stop_tui_redirect():
    """Stop comprehensive TUI output redirection."""
    _tui_redirect.stop_redirect()


@contextlib.contextmanager
def tui_redirect_context():
    """Context manager for automatic TUI redirection."""
    try:
        start_tui_redirect()
        yield
    finally:
        stop_tui_redirect()