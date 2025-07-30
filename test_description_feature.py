#!/usr/bin/env python3
"""
Test script to verify the command description feature works.
"""

from src.fuzzyshell.model_handler import DescriptionHandler
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)

def test_description_handler():
    """Test the DescriptionHandler class."""
    print("Testing DescriptionHandler...")
    
    try:
        # Initialize the handler (this will try to download the model)
        print("Initializing DescriptionHandler...")
        handler = DescriptionHandler()
        print("✓ DescriptionHandler initialized successfully")
        
        # Test description generation
        test_commands = [
            "ls -la",
            "cd /tmp", 
            "echo 'hello world'",
            "grep -r 'pattern' .",
            "find . -name '*.py'"
        ]
        
        print("\nGenerating descriptions for test commands:")
        for cmd in test_commands:
            try:
                desc = handler.generate_description(cmd)
                print(f"Command: {cmd}")
                print(f"Description: {desc}")
                print("-" * 50)
            except Exception as e:
                print(f"Failed to generate description for '{cmd}': {e}")
                
    except Exception as e:
        print(f"❌ Error initializing DescriptionHandler: {e}")
        print("This is expected if you don't have the T5-small model URL working yet.")
        print("The fallback functionality should still work in the TUI.")

def test_ui_components():
    """Test UI components can be imported."""
    print("\nTesting UI component imports...")
    
    try:
        from src.fuzzyshell.fuzzy_tui import CommandDescriptionPane, FuzzyShellApp
        print("✓ UI components imported successfully")
        
        # Test basic widget creation
        pane = CommandDescriptionPane()
        print("✓ CommandDescriptionPane created successfully")
        
    except Exception as e:
        print(f"❌ Error importing UI components: {e}")

if __name__ == "__main__":
    print("FuzzyShell Command Description Feature Test")
    print("=" * 50)
    
    test_ui_components()
    test_description_handler()
    
    print("\n" + "=" * 50)
    print("Test completed!")
    print("\nTo test the full UI, run:")
    print("  venv/bin/python -m fuzzyshell.fuzzyshell")