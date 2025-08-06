"""
Command Description Generator for FuzzyShell.

Provides intelligent command descriptions using a fine-tuned CodeT5 model.
Extracted from the main TUI to provide clean separation of concerns.
"""

import time
import logging
import threading
from typing import List, Tuple, Optional

logger = logging.getLogger('FuzzyShell.CommandDescriptionGenerator')


class CommandDescriptionGenerator:
    """
    Generates intelligent descriptions for shell commands using ML models.
    
    Simplified implementation focusing on reliability over complexity.
    """
    
    def __init__(self):
        """Initialize the description generator."""
        self._description_handler = None
        self._init_lock = threading.Lock()
        self._current_command = None
        self.description = ""
        self._cache = {}  # Simple command -> description cache
        
    def get_display_text(self) -> List[Tuple[str, str]]:
        """
        Get formatted text for display in TUI widgets.
        
        Returns:
            List of (color, text) tuples for urwid display
        """
        if self.description:
            return [('light blue', "💡 "), ('white', self.description)]
        else:
            return [('dark gray', "💡 "), ('dark gray', "Select a command to see its description")]
    
    def update_command(self, command: str):
        """
        Update the current command and generate description immediately.
        
        Args:
            command: The command to generate description for
        """
        if not command or command == self._current_command:
            return
            
        self._current_command = command
        
        # Check cache first
        if command in self._cache:
            self.description = self._cache[command]
            logger.debug(f"Using cached description for: {command}")
            return
        
        # Generate description synchronously for simplicity
        self._generate_description(command)
    
    def _generate_description(self, command: str):
        """Generate description for a command."""
        try:
            handler = self._init_description_handler()
            if handler is None:
                # Fallback description
                self.description = f"Command: {command}"
            else:
                # Generate with model
                self.description = handler.generate_description(command)
                # Cache the result
                self._cache[command] = self.description
                
            logger.debug(f"Generated description for '{command}': {self.description[:50]}...")
            
        except Exception as e:
            logger.error("Error generating description: %s", str(e))
            # Fallback on error
            self.description = f"Command: {command}"
    
    def _init_description_handler(self):
        """Initialize the description handler on first use."""
        with self._init_lock:
            if self._description_handler is None:
                try:
                    from .model_handler import DescriptionHandler
                    self._description_handler = DescriptionHandler()
                    logger.debug("DescriptionHandler initialized successfully")
                except Exception as e:
                    logger.error("Failed to initialize DescriptionHandler: %s", str(e))
                    self._description_handler = False
        return self._description_handler if self._description_handler is not False else None
    
    def clear(self):
        """Clear current description."""
        self._current_command = None
        self.description = ""
    
    def get_cached_description(self, command: str) -> Optional[str]:
        """Get cached description if available."""
        return self._cache.get(command)
    
    def clear_cache(self):
        """Clear the description cache."""
        self._cache.clear()
        logger.debug("Description cache cleared")