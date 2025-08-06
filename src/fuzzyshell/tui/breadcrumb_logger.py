"""
Breadcrumb logging system for TUI screen transitions.
Provides visibility into the user journey through different screens.
"""

import logging
import time
from typing import Optional

# Dedicated logger for TUI breadcrumbs
breadcrumb_logger = logging.getLogger('FuzzyShell.TUI.Breadcrumbs')


class ScreenBreadcrumb:
    """Tracks screen transitions and provides breadcrumb logging."""
    
    def __init__(self):
        self.current_screen: Optional[str] = None
        self.screen_start_time: Optional[float] = None
        self.navigation_path = []
    
    def did_become_visible(self, screen_name: str, context: dict = None):
        """Log when a screen becomes visible."""
        now = time.time()
        
        # Log exit from previous screen
        if self.current_screen and self.screen_start_time:
            duration = now - self.screen_start_time
            breadcrumb_logger.info(f"Screen '{self.current_screen}' active for {duration:.2f}s")
        
        # Log entrance to new screen
        self.current_screen = screen_name
        self.screen_start_time = now
        self.navigation_path.append((screen_name, now))
        
        # Log the transition
        context_str = f" (context: {context})" if context else ""
        breadcrumb_logger.info(f"Screen transition: '{screen_name}' became visible{context_str}")
        
        # Keep navigation path manageable
        if len(self.navigation_path) > 10:
            self.navigation_path = self.navigation_path[-10:]
    
    def did_become_hidden(self, screen_name: str, reason: str = "navigation"):
        """Log when a screen becomes hidden."""
        if self.current_screen == screen_name:
            duration = time.time() - self.screen_start_time if self.screen_start_time else 0
            breadcrumb_logger.info(f"Screen '{screen_name}' hidden after {duration:.2f}s (reason: {reason})")
    
    def log_user_action(self, action: str, screen: str = None):
        """Log a user action within a screen."""
        current = screen or self.current_screen
        breadcrumb_logger.info(f"User action in '{current}': {action}")
    
    def get_navigation_summary(self) -> str:
        """Get a summary of the navigation path."""
        if not self.navigation_path:
            return "No navigation history"
        
        path_names = [item[0] for item in self.navigation_path]
        return " → ".join(path_names)


# Global breadcrumb tracker
_breadcrumb_tracker = ScreenBreadcrumb()


def screen_did_become_visible(screen_name: str, context: dict = None):
    """Global function to log screen transitions."""
    _breadcrumb_tracker.did_become_visible(screen_name, context)


def screen_did_become_hidden(screen_name: str, reason: str = "navigation"):
    """Global function to log screen hiding."""
    _breadcrumb_tracker.did_become_hidden(screen_name, reason)


def log_user_action(action: str, screen: str = None):
    """Global function to log user actions."""
    _breadcrumb_tracker.log_user_action(action, screen)


def get_navigation_summary() -> str:
    """Get the current navigation summary."""
    return _breadcrumb_tracker.get_navigation_summary()


# Context manager for automatic screen tracking
class ScreenContext:
    """Context manager that automatically tracks screen visibility."""
    
    def __init__(self, screen_name: str, context: dict = None):
        self.screen_name = screen_name
        self.context = context
    
    def __enter__(self):
        screen_did_become_visible(self.screen_name, self.context)
        return self
    
    def __exit__(self, exc_type, _exc_val, _exc_tb):
        reason = "error" if exc_type else "completion"
        screen_did_become_hidden(self.screen_name, reason)