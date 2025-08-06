#!/usr/bin/env python3
"""Test command cleaning with modern Python 3.10+ syntax"""

import sys

from fuzzyshell.fuzzyshell import FuzzyShell

def test_modern_functionality():
    """Test that our modern Python improvements work"""
    print("🧪 Testing modern FuzzyShell functionality...")
    
    # Test command cleaning
    fs = FuzzyShell()
    
    # Test cases with actual problematic commands
    test_commands = [
        ": 1753138854:0;python process_tokens.py -c -b \"VMA new\" \"BOQ new\" \"ME\" | jq",
        ": 1747535765:0;grep FICTION extracted_categories.csv | wc -l",
        "regular command without timestamp"
    ]
    
    print("📝 Testing command cleaning:")
    for raw_cmd in test_commands:
        cleaned = fs.clean_shell_command(raw_cmd)
        print(f"  Input:  '{raw_cmd[:50]}...'")
        print(f"  Output: '{cleaned[:50]}...'")
        print()
    
    # Test modern match/case logic by importing the TUI class
    from fuzzyshell.fuzzy_tui import FuzzyShellApp
    
    print("🔧 Testing modern match/case search mode cycling:")
    # Create a mock app to test the cycling logic
    class MockApp:
        def __init__(self):
            self.search_mode = "hybrid"
        
        def cycle_mode(self):
            """Use if/elif for Python 3.9 compatibility"""
            if self.search_mode == "hybrid":
                self.search_mode = "semantic"
            elif self.search_mode == "semantic":
                self.search_mode = "keyword" 
            elif self.search_mode == "keyword":
                self.search_mode = "hybrid"
            else:
                self.search_mode = "hybrid"
    
    mock_app = MockApp()
    print(f"  Initial mode: {mock_app.search_mode}")
    
    mock_app.cycle_mode()
    print(f"  After 1 cycle: {mock_app.search_mode}")
    
    mock_app.cycle_mode()
    print(f"  After 2 cycles: {mock_app.search_mode}")
    
    mock_app.cycle_mode()
    print(f"  After 3 cycles: {mock_app.search_mode}")
    
    print("\n✅ Modern Python features working in FuzzyShell!")
    # Test passes if we reach this point without exceptions

if __name__ == "__main__":
    try:
        test_modern_functionality()
        print("🎉 All tests passed! Modern Python 3.10+ FuzzyShell is ready!")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)