"""
Status Display Manager - Manages status information and display formatting.

Consolidates status tracking and footer management for consistent display across the application.
"""

import urwid
from typing import Optional, Dict, Any
from fuzzyshell.tui.key_bindings import KEY_HELP, KEY_EXPERT_SCREEN

# Import version at module level to avoid circular imports
try:
    from fuzzyshell import __version__
except ImportError:
    # Fallback for development/circular import issues
    __version__ = "1.0.0"


class StatusInfo:
    """Data class to hold status information."""
    
    def __init__(self):
        self.item_count = 0
        self.embedding_model = "unknown"
        self.search_time = 0.0
        self.show_scores = False
        self.query = ""
        self.results_count = 0
        
    def update(self, **kwargs):
        """Update status fields from keyword arguments."""
        for key, value in kwargs.items():
            if hasattr(self, key) and value is not None:
                setattr(self, key, value)
                
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for easy access."""
        return {
            'item_count': self.item_count,
            'embedding_model': self.embedding_model,
            'search_time': self.search_time,
            'show_scores': self.show_scores,
            'query': self.query,
            'results_count': self.results_count
        }


class StatusDisplayManager:
    """Manages status information display and footer formatting."""
    
    def __init__(self):
        self.status_info = StatusInfo()
        self._footer_widget = None
        
    def update_status(self, **kwargs):
        """Update status information."""
        self.status_info.update(**kwargs)
        self._refresh_display()
        
    def get_status_info(self) -> StatusInfo:
        """Get current status information."""
        return self.status_info
        
    def create_footer_widget(self) -> urwid.Widget:
        """Create the footer widget with current status."""
        left_text_widget = urwid.Text("")
        right_text_widget = urwid.Text("", align='right')
        
        footer_columns = urwid.Columns([left_text_widget, right_text_widget])
        self._footer_widget = urwid.AttrMap(footer_columns, 'footer')
        
        # Store references for updates
        self._left_text_widget = left_text_widget
        self._right_text_widget = right_text_widget
        
        self._refresh_display()
        return self._footer_widget
        
    def _refresh_display(self):
        """Refresh the footer display with current status."""
        if not hasattr(self, '_left_text_widget'):
            return
            
        # Left side: version, item count, search time
        left_text = [('bold', ('dark cyan', f"FuzzyShell v{__version__}"))]
        
        if self.status_info.item_count > 0:
            left_text.append(('dark gray', f" • {self.status_info.item_count:,} items"))
            
        if self.status_info.search_time > 0:
            left_text.append(('dark green', f" • {self.status_info.search_time*1000:.0f}ms"))
            
        if self.status_info.results_count > 0 and self.status_info.query:
            left_text.append(('dark yellow', f" • {self.status_info.results_count} matches"))

        # Right side: key bindings
        right_text = self._generate_key_bindings()
        
        self._left_text_widget.set_text(left_text)
        self._right_text_widget.set_text(right_text)
        
    def _generate_key_bindings(self) -> list:
        """Generate key binding text for footer."""
        # Generate key labels from constants
        help_key = "^G" if KEY_HELP == 'ctrl g' else ("F1" if KEY_HELP == 'f1' else KEY_HELP.upper())
        expert_key = "^E" if KEY_EXPERT_SCREEN == 'ctrl e' else KEY_EXPERT_SCREEN.upper()
        
        scores_status = "scores(ON)" if self.status_info.show_scores else "scores(OFF)"
        
        return [
            ('bold', help_key), ('dark gray', " tips "),
            ('bold', expert_key), ('dark gray', " expert "),
            ('bold', "ESC"), ('dark gray', " quit "),
            ('bold', "↑↓"), ('dark gray', " navigate "),
            ('bold', "ENTER"), ('dark gray', " select "),
            ('bold', "^S/S"), ('dark gray', f" {scores_status}")
        ]
        
    def format_status_text(self, template: str = None) -> str:
        """Format status information as text string."""
        if template is None:
            template = "FuzzyShell v{version} • {item_count:,} items • {search_time:.0f}ms"
            
        return template.format(
            version=__version__,
            item_count=self.status_info.item_count,
            search_time=self.status_info.search_time * 1000,  # Convert to ms
            embedding_model=self.status_info.embedding_model,
            show_scores=self.status_info.show_scores,
            query=self.status_info.query,
            results_count=self.status_info.results_count
        )
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics for display."""
        return {
            'version': __version__,
            'total_items': self.status_info.item_count,
            'embedding_model': self.status_info.embedding_model,
            'last_search_time_ms': round(self.status_info.search_time * 1000, 1),
            'scores_visible': self.status_info.show_scores,
            'current_query': self.status_info.query,
            'current_results': self.status_info.results_count
        }
        
    def reset_search_status(self):
        """Reset search-related status information."""
        self.update_status(
            search_time=0.0,
            query="",
            results_count=0
        )


# Legacy compatibility class that wraps StatusDisplayManager
class StatusFooter(urwid.WidgetWrap):
    """Legacy StatusFooter that delegates to StatusDisplayManager."""
    
    def __init__(self):
        self._display_manager = StatusDisplayManager()
        footer_widget = self._display_manager.create_footer_widget()
        
        # Extract the columns widget from the AttrMap
        if isinstance(footer_widget, urwid.AttrMap):
            columns_widget = footer_widget.original_widget
        else:
            columns_widget = footer_widget
            
        super().__init__(columns_widget)
        
        # Expose properties for backward compatibility
        self.item_count = 0
        self.embedding_model = "unknown"
        self.search_time = 0.0
        self.show_scores = False
        
    def update(self, item_count=None, embedding_model=None, search_time=None, show_scores=None):
        """Update footer with new status information."""
        # Update local properties for backward compatibility
        if item_count is not None:
            self.item_count = item_count
        if embedding_model is not None:
            self.embedding_model = embedding_model
        if search_time is not None:
            self.search_time = search_time
        if show_scores is not None:
            self.show_scores = show_scores
            
        # Update the display manager
        self._display_manager.update_status(
            item_count=item_count,
            embedding_model=embedding_model,
            search_time=search_time,
            show_scores=show_scores
        )