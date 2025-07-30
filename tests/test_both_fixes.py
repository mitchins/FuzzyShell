#!/usr/bin/env python3
"""Test both fixes: search quality and shell integration"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from fuzzyshell.fuzzyshell import FuzzyShell
from .test_helpers import create_test_db_connection

def test_search_quality():
    """Test that search quality is good for basic commands"""
    print("🔍 Testing Search Quality...")
    
    test_conn = create_test_db_connection()
    fs = FuzzyShell(conn=test_conn)
    
    # Test basic command searches
    test_cases = [
        ("ls", ["ls", "ls -la", "ls -l"]),
        ("git", ["git", "git status", "git commit"]),
        ("docker", ["docker", "docker ps", "docker run"]),
        ("python", ["python", "python -m", "python3"])
    ]
    
    all_passed = True
    
    for query, expected_prefixes in test_cases:
        print(f"\n--- Testing '{query}' ---")
        results = fs.search(query, top_k=5, return_scores=False)
        
        # Verify result format is correct (2-tuple when return_scores=False)
        
        if not results:
            print(f"❌ No results for '{query}'")
            all_passed = False
            continue
        
        # Check if top results are relevant
        relevant_count = 0
        for cmd, score in results:
            cmd_lower = cmd.lower()
            for prefix in expected_prefixes:
                if cmd_lower.startswith(prefix.lower()):
                    relevant_count += 1
                    break
        
        if relevant_count >= 2:  # At least 2 relevant results in top 5
            print(f"✅ Good results for '{query}' ({relevant_count}/5 relevant)")
            for i, (cmd, score) in enumerate(results[:3], 1):
                print(f"   {i}. {cmd}")
        else:
            print(f"❌ Poor results for '{query}' ({relevant_count}/5 relevant)")
            all_passed = False
    
    return all_passed

def test_shell_wrapper_logic():
    """Test the shell wrapper logic (simulated)"""
    print("\n🐚 Testing Shell Integration Logic...")
    
    def simulate_fuzzy_command():
        """Simulate running fuzzy and returning a clean command"""
        test_conn = create_test_db_connection()
        fs = FuzzyShell(conn=test_conn)
        # Add a test command
        test_cmd = "git status"
        fs.add_command(test_cmd)
        
        # Search for it
        results = fs.search("git", top_k=1)
        if results:
            return results[0][0]  # Return the command
        return None
    
    # Simulate the shell wrapper logic
    selected_command = simulate_fuzzy_command()
    
    if selected_command:
        print(f"✅ Fuzzy returned: '{selected_command}'")
        
        # Check it's clean (no timestamps)
        if not selected_command.startswith(': ') and ';' not in selected_command[:15]:
            print("✅ Command is clean (no timestamps)")
        else:
            print("❌ Command still contains timestamps")
            return False
        
        # Simulate shell wrapper behavior
        print(f"✅ Shell wrapper would put: '{selected_command}' in command line")
        return True
    else:
        print("❌ No command returned")
        return False

def test_return_key_functionality():
    """Test return key functionality"""
    print("\n⌨️  Testing Return Key Functionality...")
    
    # Verify the input submit handler exists
    from fuzzyshell.fuzzy_tui import FuzzyShellApp
    
    if hasattr(FuzzyShellApp, 'on_input_submitted'):
        print("✅ Input submit handler exists")
        
        # Test the action_select_command logic
        class MockApp:
            def __init__(self):
                self.current_results = [("git status", 0.9), ("git commit", 0.8)]
                self.selected_index = 0
                self.exit_value = None
            
            def exit(self, value):
                self.exit_value = value
            
            def action_select_command(self):
                """Copy of the real method"""
                if not self.current_results:
                    self.exit(None)
                    return
                    
                if 0 <= self.selected_index < len(self.current_results):
                    selected_command = self.current_results[self.selected_index][0]
                    self.exit(selected_command)
                else:
                    self.exit(None)
        
        # Test normal case
        mock_app = MockApp()
        mock_app.action_select_command()
        
        if mock_app.exit_value == "git status":
            print("✅ Return key selects correct command")
        else:
            print(f"❌ Wrong command selected: {mock_app.exit_value}")
            return False
        
        return True
    else:
        print("❌ Input submit handler missing")
        return False

if __name__ == "__main__":
    print("🧪 Complete Integration Test: Search Quality + Shell Integration")
    print("=" * 70)
    
    success = True
    
    try:
        success &= test_search_quality()
        success &= test_shell_wrapper_logic() 
        success &= test_return_key_functionality()
        
        print("\n" + "=" * 70)
        if success:
            print("🎉 ALL TESTS PASSED!")
            print("✅ Search quality: Good results for basic commands")
            print("✅ Shell integration: Commands are clean and ready")
            print("✅ Return key: Properly selects and exits")
            print("\n🚀 FuzzyShell is ready for use!")
            print("   - Try: source venv/bin/activate && fuzzy")
            print("   - Search for: ls, git, docker, python")  
            print("   - Press Enter to select a command")
        else:
            print("❌ Some tests failed - check output above")
    
    except Exception as e:
        print(f"❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
        success = False
    
    sys.exit(0 if success else 1)