#!/usr/bin/env python3
"""
Test script to check the UI layout and description pane.
"""

from src.fuzzyshell.fuzzyshell import FuzzyShell
from src.fuzzyshell.fuzzy_tui import FuzzyShellApp, CommandDescriptionPane
import logging

# Set up logging to see debug info
logging.basicConfig(level=logging.INFO)

def test_ui_layout():
    """Test that the UI has the description pane."""
    print("Testing UI layout...")
    
    # Create a FuzzyShell instance
    fuzzyshell = FuzzyShell()
    
    # Create the app
    def search_callback(query: str) -> list:
        if not query:
            return []
        return fuzzyshell.search(query, top_k=5)
    
    app = FuzzyShellApp(search_callback, fuzzyshell_instance=fuzzyshell)
    
    # Check the compose method includes our widgets
    widgets = list(app.compose())
    widget_ids = [getattr(w, 'id', None) for w in widgets]
    
    print(f"Found widgets: {[w.__class__.__name__ for w in widgets]}")
    print(f"Widget IDs: {widget_ids}")
    
    # Check for description pane
    description_panes = [w for w in widgets if isinstance(w, CommandDescriptionPane)]
    print(f"Found {len(description_panes)} CommandDescriptionPane widgets")
    
    if description_panes:
        print("✓ Description pane found in UI layout")
    else:
        print("❌ Description pane NOT found in UI layout")
        
    # Check CSS
    css = app.CSS
    if '#description-pane' in css:
        print("✓ Description pane CSS found")
    else:
        print("❌ Description pane CSS NOT found")
        
    return len(description_panes) > 0

def test_version_import():
    """Test version import."""
    print("\nTesting version import...")
    
    try:
        from src.fuzzyshell import __version__
        print(f"Version imported: {__version__}")
    except ImportError as e:
        print(f"❌ Failed to import version: {e}")
        return False
        
    # Test version in TUI
    try:
        from src.fuzzyshell.fuzzy_tui import __version__ as tui_version
        print(f"TUI version: {tui_version}")
        return True
    except ImportError as e:
        print(f"❌ Failed to import TUI version: {e}")
        return False

if __name__ == "__main__":
    print("FuzzyShell UI Test")
    print("=" * 30)
    
    version_ok = test_version_import()
    layout_ok = test_ui_layout()
    
    print("\n" + "=" * 30)
    if version_ok and layout_ok:
        print("✓ All tests passed")
        print("\nTry running: venv/bin/fuzzy")
        print("Then search for some commands (e.g., 'ls', 'cd', 'git')")
    else:
        print("❌ Some tests failed - investigating...")