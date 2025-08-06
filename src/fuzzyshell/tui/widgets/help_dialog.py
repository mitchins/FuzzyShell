"""
Help dialog widget for FuzzyShell TUI.
"""

import urwid
from ..key_bindings import KEY_HELP, KEY_QUIT

# Import version at module level to avoid circular imports
try:
    from fuzzyshell import __version__
except ImportError:
    # Fallback for development/circular import issues
    __version__ = "1.0.0"


class HelpDialog(urwid.WidgetWrap):
    """Help dialog showing keyboard shortcuts and about info"""
    
    def __init__(self):
        help_text = [
            ('bold', 'Search Results:\n'),
            ('bold', '• Poor relevance: '), 'Re-index with "fuzzy --ingest"\n',
            ('bold', '• Missing commands: '), 'Run ingestion to capture recent history\n',
            ('bold', '• Descriptions: '), 'AI-generated, may be inaccurate\n\n',
            
            ('bold', 'Performance:\n'),
            ('bold', '• Slow search: '), 'Update clusters with "fuzzy --update-clusters"\n',
            ('bold', '• Model issues: '), 'Check ~/.fuzzyshell/model/ exists\n',
            ('bold', '• Database corrupt: '), 'Delete fuzzyshell.db to reset\n\n',
            
            ('dark gray', 'Run "fuzzy --help" for advanced usage and model options\n\n'),
            ('dark gray', 'Press Tab or ESC to close')
        ]
        
        content = urwid.Text(help_text)
        padded = urwid.Padding(content, left=2, right=2)
        filled = urwid.Filler(padded, valign='top', top=1)
        
        # Create a box with border
        box = urwid.LineBox(filled, title='Help & Tips')
        
        # Make it an overlay that can be closed - use default styling
        super().__init__(box)
    
    def keypress(self, size, key):
        if key in (KEY_QUIT, KEY_HELP):
            return 'close_help'
        return super().keypress(size, key)