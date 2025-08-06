"""
Custom edit widgets for FuzzyShell TUI.
"""

import urwid
import logging

logger = logging.getLogger('FuzzyShell.TUI')


class WatermarkEdit(urwid.Edit):
    """Custom Edit widget with watermark text when empty."""
    
    def __init__(self, caption="", edit_text="", watermark_text="", **kwargs):
        super().__init__(caption, edit_text, **kwargs)
        self.watermark_text = watermark_text
        self._has_been_edited = False
    
    def render(self, size, focus=False):
        """Override render to show watermark when empty."""
        try:
            if not self.edit_text and not self._has_been_edited and self.watermark_text:
                # Show watermark until user starts typing
                full_watermark = f"{self.caption}{self.watermark_text}"
                watermark_widget = urwid.Text([('placeholder_text', full_watermark)], align='left')
                return watermark_widget.render(size, focus)
            else:
                # Render normal edit widget
                return super().render(size, focus)
        except Exception as e:
            logger.debug(f"Error in WatermarkEdit.render: {e}")
            # Fallback to basic text widget
            fallback = urwid.Text(f"Search: {self.edit_text}")
            return fallback.render(size, focus)
    
    def keypress(self, size, key):
        """Track when user starts editing to hide watermark and handle global commands."""
        
        # Handle global commands that should work even when input is focused
        if key in ('ctrl s', '\x13', 'ctrl S', '\x1f'):  # Multiple representations of Ctrl+S
            # Let the parent handle this global command
            logger.debug(f"WatermarkEdit: Passing through ctrl s: '{key}'")
            return key
        elif key == 'ctrl r':
            # Let the parent handle this global command  
            return key
        elif key in ('esc', 'escape'):
            # Let the parent handle escape
            return key
        elif key in ('up', 'down', 'page up', 'page down', 'home', 'end'):
            # Let urwid handle navigation naturally - just pass through to unhandled_input
            logger.info(f"🔑 WatermarkEdit: Passing navigation key '{key}' to parent")
            return key
        
        # Mark as edited when user starts typing
        if key and len(key) == 1 and key.isprintable():
            self._has_been_edited = True
        elif key == 'backspace' and self.edit_text:
            self._has_been_edited = True
        
        return super().keypress(size, key)