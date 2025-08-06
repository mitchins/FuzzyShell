"""
Indicator widgets for FuzzyShell TUI.
"""

import time
import urwid


class SearchModeIndicator(urwid.Text):
    """Displays the current search mode with icon and color."""
    
    def __init__(self):
        super().__init__("")
        self.search_mode = "hybrid"
        self.update()

    def set_search_mode(self, mode):
        self.search_mode = mode
        self.update()

    def update(self):
        mode_icons = {
            "semantic": "🧠",
            "keyword": "🔍",
            "hybrid": "⚡"
        }
        mode_colors = {
            "semantic": "light blue",
            "keyword": "light green",
            "hybrid": "light magenta"
        }
        icon = mode_icons.get(self.search_mode, "⚡")
        color = mode_colors.get(self.search_mode, "light magenta")
        
        text = [
            (color, f"{icon} "),
            ('bold', (color, self.search_mode.upper()))
        ]
        self.set_text(text)


class LoadingIndicator(urwid.Text):
    """Shows loading state with animated spinner and progress information."""
    
    def __init__(self):
        super().__init__("")
        self.is_loading = False
        self.message = "Searching..."
        self._spinner_alarm = None
        self._spinner_chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        self._spinner_index = 0
        self._start_time = 0
        self._elapsed_time = 0
    
    def start_loading(self, loop, message="Searching..."):
        """Start the loading animation."""
        self.is_loading = True
        self.message = message
        self._spinner_index = 0
        self._start_time = time.time()
        self._elapsed_time = 0
        self._update_spinner()
        # Update spinner every 0.1 seconds
        self._spinner_alarm = loop.set_alarm_in(0.1, self._spinner_callback)
    
    def stop_loading(self, loop):
        """Stop the loading animation."""
        self.is_loading = False
        if self._spinner_alarm:
            loop.remove_alarm(self._spinner_alarm)
            self._spinner_alarm = None
        self.set_text("")
    
    def _spinner_callback(self, loop, _user_data):
        """Update spinner animation."""
        if self.is_loading:
            self._elapsed_time = time.time() - self._start_time
            self._update_spinner()
            self._spinner_alarm = loop.set_alarm_in(0.1, self._spinner_callback)
    
    def _update_spinner(self):
        """Update the spinner display with elapsed time."""
        if self.is_loading:
            spinner = self._spinner_chars[self._spinner_index]
            self._spinner_index = (self._spinner_index + 1) % len(self._spinner_chars)
            
            # Show elapsed time for longer searches
            if self._elapsed_time > 1.0:
                time_text = f" ({self._elapsed_time:.1f}s)"
                message_color = 'yellow' if self._elapsed_time > 3.0 else 'dark gray'
            else:
                time_text = ""
                message_color = 'dark gray'
                
            text = [
                ('dark cyan', f"{spinner} "), 
                (message_color, self.message + time_text)
            ]
            self.set_text(text)