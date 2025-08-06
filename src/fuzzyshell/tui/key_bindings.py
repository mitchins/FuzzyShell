"""
Key binding constants for FuzzyShell TUI.
Centralized location for all keyboard shortcuts to maintain DRY principle.
"""

# Navigation and Selection
KEY_QUIT = 'esc'
KEY_SELECT = 'enter'
KEY_NAVIGATE_UP = 'up'
KEY_NAVIGATE_DOWN = 'down'
KEY_PAGE_UP = 'page up'
KEY_PAGE_DOWN = 'page down'
KEY_HOME = 'home'
KEY_END = 'end'

# Search and Display
KEY_TOGGLE_SCORES = 'ctrl s'
KEY_TOGGLE_SCORES_ALT = 's'  # Alternative without ctrl
KEY_REFRESH_SEARCH = 'ctrl r'

# Help and Information
KEY_HELP = 'ctrl g'  # Guide/help
KEY_EXPERT_SCREEN = 'ctrl e'  # Expert/technical info

# Global Navigation Keys (passed through from input)
NAVIGATION_KEYS = (KEY_NAVIGATE_UP, KEY_NAVIGATE_DOWN, KEY_PAGE_UP, KEY_PAGE_DOWN, KEY_HOME, KEY_END)

# Global Control Keys (work regardless of focus)
GLOBAL_KEYS = (KEY_TOGGLE_SCORES, KEY_TOGGLE_SCORES_ALT, KEY_REFRESH_SEARCH, KEY_QUIT)

# Input handling - keys that should be passed through to parent
INPUT_PASSTHROUGH_KEYS = NAVIGATION_KEYS + (KEY_QUIT, KEY_TOGGLE_SCORES, KEY_REFRESH_SEARCH)