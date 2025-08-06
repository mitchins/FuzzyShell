"""
Display widgets for FuzzyShell TUI.
"""

import urwid
import logging
from ...command_description_generator import CommandDescriptionGenerator

logger = logging.getLogger('FuzzyShell.TUI')


class CommandDescriptionPane(urwid.Text):
    """Widget that displays command descriptions with async generation."""
    
    def __init__(self):
        super().__init__("")
        self._generator = CommandDescriptionGenerator()
        self.update()

    def update(self):
        """Update the display based on current state."""
        text = self._generator.get_display_text()
        self.set_text(text)

    def generate_description_async(self, command: str, loop=None):
        """Generate description for a command (simplified synchronous interface)."""
        # Delegate to the generator and update display
        self._generator.update_command(command)
        self.update()
        
        # Draw screen immediately to show the change
        if loop:
            try:
                loop.draw_screen()
            except Exception as e:
                logger.debug(f"Could not draw screen: {e}")
    
    def clear_description(self):
        """Clear current description."""
        self._generator.clear()
        self.update()
    
    def set_description(self, description: str):
        """Set description directly."""
        self._generator.description = description
        self.update()


class SearchResult(urwid.WidgetWrap):
    """Widget representing a single search result with command and scores."""
    
    def __init__(self, command, score, search_mode, semantic_score, bm25_score, show_scores):
        self.command = command
        self.score = score
        self.search_mode = search_mode
        self.semantic_score = semantic_score
        self.bm25_score = bm25_score
        self.show_scores = show_scores
        
        # Create the actual widget content
        self._create_widget()
        super().__init__(self._widget)
    
    def _create_widget(self, focused=False):
        """Create the underlying widget with proper text content."""
        left_text, right_text = self._build_text_content(focused)
        
        # Create a columns layout with command on left, stats on right
        columns = urwid.Columns([
            ('weight', 1, urwid.Text(left_text, wrap='clip')),
            ('pack', urwid.Text(right_text, align='right'))
        ])
        
        # No background highlight - only arrow indicator shows selection
        self._widget = urwid.AttrMap(columns, None, focus_map=None)
        
    def selectable(self):
        """Make this widget selectable for navigation."""
        return True
        
    def keypress(self, size, key):
        """Handle key presses - return key to parent for handling."""
        return key
    
    def mouse_event(self, size, event, button, _col, row, focus):
        """Handle mouse events safely."""
        try:
            # Handle mouse click as selection
            if event == 'mouse press' and button == 1:  # Left click
                # Don't return 'enter' - let the parent handle it
                # Just indicate that the event was handled
                return True
            return False
        except Exception as e:
            logger.debug(f"Error in SearchResult.mouse_event: {e}")
            return False

    def _build_text_content(self, focus=False):
        """Build the text content for this search result."""
        # Left side: indicator + command
        left_text = []
        if focus:
            left_text.append(('light cyan', "▶ "))
        else:
            left_text.append(('dark gray', "  "))
        
        # Use default color for command text
        left_text.append(('default', self.command))
        
        # Right side: match statistics
        right_text = []
        if self.show_scores and self.score > 0:
            semantic_pct = int(self.semantic_score * 100)
            bm25_pct = int(self.bm25_score * 100)

            # Determine which score is dominant
            if bm25_pct > semantic_pct * 1.2:
                # Keyword dominant
                right_text.extend([
                    ('dark gray', "Key: "),
                    ('light green', f"{bm25_pct}%")
                ])
            elif semantic_pct > bm25_pct * 1.2:
                # Semantic dominant  
                right_text.extend([
                    ('dark gray', "Sem: "),
                    ('light blue', f"{semantic_pct}%")
                ])
            else:
                # Hybrid - show both
                right_text.extend([
                    ('dark gray', "Sem: "),
                    ('light blue', f"{semantic_pct}% "),
                    ('dark gray', "Key: "),
                    ('light green', f"{bm25_pct}%")
                ])
        
        return left_text, right_text

    def render(self, size, focus=False):
        """Render with focus handling through WidgetWrap."""
        try:
            # Update widget content based on focus state
            if focus != getattr(self, '_last_focus_state', False):
                logger.debug(f"SearchResult render: focus changed to {focus} for '{self.command[:20]}...'")
                self._create_widget(focus)
                self._last_focus_state = focus
                super().__init__(self._widget)  # Update the wrapped widget
            
            return super().render(size, focus)
        except Exception as e:
            logger.debug(f"Error in SearchResult.render: {e}")
            # Return a fallback simple text widget
            fallback = urwid.Text(f"Error: {self.command}")
            return fallback.render(size, focus)