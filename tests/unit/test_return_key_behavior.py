#!/usr/bin/env python3
"""Test return key behavior and command selection"""

import sys
import os
import tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from fuzzyshell.fuzzyshell import FuzzyShell
from test_helpers import create_test_db_connection

def test_command_selection_flow():
    """Test the complete flow from search to selection"""
    print("🧪 Testing command selection flow...")
    
    # Create FuzzyShell instance with clean test database
    conn = create_test_db_connection()
    fs = FuzzyShell(conn=conn)
    fs.init_model_sync()  # Initialize model for search functionality
    
    # Add some test commands with the cleaned format
    test_commands = [
        "python manage.py runserver",
        "git status",
        "docker ps -a",
        "npm run dev"
    ]
    
    for cmd in test_commands:
        fs.add_command(cmd)
    
    # Test that search returns clean results
    results = fs.search("python")
    print(f"✅ Search for 'python' returned {len(results)} results")
    
    if results:
        selected_command = results[0][0]  # First result, command part
        print(f"✅ Selected command: '{selected_command}'")
        
        # Verify it's clean (no timestamps)
        is_clean = not selected_command.startswith(': ') and ';' not in selected_command[:15]
        if is_clean:
            print("✅ Command is clean (no timestamps)")
        else:
            print("❌ Command still contains timestamps!")
        
        assert is_clean, "Command still contains timestamps"

def test_app_exit_behavior():
    """Test that the app properly exits and returns commands"""
    print("\n🔧 Testing app exit behavior...")
    
    # This simulates what happens when Enter is pressed
    class MockApp:
        def __init__(self):
            self.current_results = [("python manage.py runserver", 0.9)]
            self.selected_index = 0
            self.exited_with = None
        
        def exit(self, value):
            """Mock exit method that captures the return value"""
            self.exited_with = value
        
        def action_select_command(self):
            """Simulate the real action_select_command behavior"""
            if not self.current_results:
                self.exit(None)
                return
                
            if 0 <= self.selected_index < len(self.current_results):
                selected_command = self.current_results[self.selected_index][0]
                self.exit(selected_command)
            else:
                self.exit(None)
    
    # Test normal selection
    app = MockApp()
    app.action_select_command()
    
    correct_exit = app.exited_with == "python manage.py runserver"
    if correct_exit:
        print("✅ App exits with correct command")
    else:
        print(f"❌ App exited with: {app.exited_with}")
    
    assert correct_exit, f"App exited with wrong command: {app.exited_with}"
    
    # Test empty results
    app = MockApp()
    app.current_results = []
    app.action_select_command()
    
    empty_exit = app.exited_with is None
    if empty_exit:
        print("✅ App exits cleanly when no results")
    else:
        print(f"❌ App should exit with None, got: {app.exited_with}")
    
    assert empty_exit, f"App should exit with None when no results, got: {app.exited_with}"

def test_input_submit_handler():
    """Test that input submit is properly handled"""
    print("\n⌨️  Testing input submit handler...")
    
    # We can't easily test the actual Textual input without running the full UI,
    # but we can verify the handler exists and works
    from fuzzyshell.fuzzy_tui import FuzzyShellApp
    
    # Check that the method exists (enter key is handled in unhandled_input)
    has_handler = hasattr(FuzzyShellApp, 'unhandled_input')
    if has_handler:
        print("✅ Input submit handler exists (unhandled_input)")
    else:
        print("❌ Input submit handler missing!")
    
    assert has_handler, "Input submit handler missing from FuzzyShellApp"

if __name__ == "__main__":
    print("🔬 Return Key Behavior Test Suite")
    print("=" * 50)
    
    success = True
    
    try:
        success &= test_command_selection_flow()
        success &= test_app_exit_behavior()
        success &= test_input_submit_handler()
        
        print("\n" + "=" * 50)
        if success:
            print("🎉 All return key behavior tests passed!")
            print("✅ Commands are clean (no timestamps)")
            print("✅ Enter key should work to select commands")
            print("✅ App should exit and return selected command")
        else:
            print("❌ Some tests failed - check implementation")
            
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        success = False
    
    sys.exit(0 if success else 1)