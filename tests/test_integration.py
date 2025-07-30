#!/usr/bin/env python3
"""Integration tests for complete FuzzyShell functionality"""

import sys
import os
import subprocess
import tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from .test_helpers import create_test_db_connection

def test_complete_workflow():
    """Test the complete workflow: ingest -> search -> select"""
    print("🔬 Testing complete workflow...")
    
    # Create a temporary shell history file
    with tempfile.NamedTemporaryFile(mode='w', suffix='_history', delete=False) as f:
        f.write(': 1753138854:0;python process_tokens.py -c -b "VMA new" "BOQ new" "ME" | jq\n')
        f.write(': 1747535765:0;git status\n')
        f.write(': 1743673275:0;docker ps -a\n')
        temp_history = f.name
    
    try:
        # Test command cleaning
        from fuzzyshell.fuzzyshell import FuzzyShell
        test_conn = create_test_db_connection()
        fs = FuzzyShell(conn=test_conn)
        
        # Test the cleaning function directly
        raw_command = ': 1753138854:0;python process_tokens.py -c -b "VMA new" "BOQ new" "ME" | jq'
        cleaned = fs.clean_shell_command(raw_command)
        
        expected = 'python process_tokens.py -c -b "VMA new" "BOQ new" "ME" | jq'
        
        if cleaned == expected:
            print("✅ Command cleaning works correctly")
        else:
            print(f"❌ Command cleaning failed:")
            print(f"   Expected: {expected}")
            print(f"   Got:      {cleaned}")
            return False
        
        # Test search functionality - use a unique search term
        fs.add_command(cleaned)
        # Search for a specific part that should match our cleaned command
        results = fs.search("process_tokens")
        
        # Look for our specific command in the results
        found_clean_command = False
        for cmd, score in results:
            if "process_tokens.py" in cmd and not cmd.startswith(": "):
                found_clean_command = True
                break
        
        if found_clean_command:
            print("✅ Search returns clean commands")
        else:
            print(f"❌ Search failed or returned dirty commands: {results}")
            return False
        
        print("✅ Complete workflow test passed")
        return True
        
    finally:
        # Clean up
        os.unlink(temp_history)

def test_shell_output_format():
    """Test that the app outputs commands in the correct format for shell integration"""
    print("\n📤 Testing shell output format...")
    
    # We can test the output handling logic
    from fuzzyshell.fuzzyshell import FuzzyShell
    
    test_conn = create_test_db_connection()
    fs = FuzzyShell(conn=test_conn)
    # Use a unique test command that won't conflict with existing database entries
    test_command = "unique_test_command_12345"
    fs.add_command(test_command)
    
    # Test the search callback functionality  
    def search_callback(query: str) -> list:
        if not query:
            return []
        return fs.search(query)
    
    results = search_callback("command")  # More generic search
    
    if results:
        # Check that all results are clean (no timestamps)
        all_clean = True
        for cmd, score in results:
            if cmd and cmd.startswith(": ") and ';' in cmd:
                all_clean = False
                break
        
        if all_clean:
            print("✅ All search results are clean and shell-ready")
            return True
        else:
            print("❌ Some results still contain timestamps")
            return False
    else:
        print("❌ No results from search callback")
        return False

if __name__ == "__main__":
    print("🔬 FuzzyShell Integration Test Suite")
    print("=" * 50)
    
    success = True
    
    try:
        success &= test_complete_workflow()
        success &= test_shell_output_format()
        
        print("\n" + "=" * 50)
        if success:
            print("🎉 All integration tests passed!")
            print("✅ Commands are properly cleaned")
            print("✅ Search returns clean results")
            print("✅ Output is ready for shell integration")
        else:
            print("❌ Some integration tests failed")
            
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        success = False
    
    sys.exit(0 if success else 1)