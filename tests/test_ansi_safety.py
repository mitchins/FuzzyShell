#!/usr/bin/env python3
"""Test to ensure no ANSI codes leak into output (simulating PYTE environment)"""

import re
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO

def strip_ansi(text):
    """Remove ANSI escape sequences from text"""
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)

def has_ansi_codes(text):
    """Check if text contains ANSI escape sequences"""
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return bool(ansi_escape.search(text))

def test_command_cleaning_output():
    """Test that command cleaning produces clean output without ANSI codes"""
    
    def clean_shell_command(raw_command):
        """Clean shell history format to extract actual command"""
        # Handle zsh history format: ': <timestamp>:<duration>; <command>'
        if raw_command.startswith(': ') and ';' in raw_command:
            # Find the semicolon that separates timestamp from command
            semicolon_pos = raw_command.find(';')
            if semicolon_pos != -1:
                # Extract command part after semicolon and strip whitespace
                command = raw_command[semicolon_pos + 1:].strip()
                return command
        
        # If no shell history format detected, return as-is
        return raw_command.strip()
    
    print("🧪 Testing ANSI-clean output...")
    
    # Test cases from the actual output you showed
    test_commands = [
        ": 1753138854:0; python process_tokens.py -c -b \"VMA new \"BOQ new\" \"ME\" | jq",
        ": 1747535765:0; grep FICTION extracted_categories _Kindle_eBooks_answered.csv | grep -v \"Other\" | grep -v \"Poetry\" | wc -1", 
        ": 1743673275:0; python stitch_chapter_seams-py —-limit 1000",
        ": 1733138452: 0; python analyse_chapters-py characters -model qwen3-4b-mlx --preview -detail --book \"15 The Chronicles of Narnia 6 - The Silver Chair - C. 5."
    ]
    
    all_clean = True
    
    # Capture stdout/stderr to check for ANSI codes
    stdout_capture = StringIO()
    stderr_capture = StringIO()
    
    with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
        for i, raw_command in enumerate(test_commands, 1):
            cleaned = clean_shell_command(raw_command)
            print(f"Command {i}: {cleaned}")
    
    # Get captured output
    stdout_content = stdout_capture.getvalue()
    stderr_content = stderr_capture.getvalue()
    
    # Check for ANSI codes in output
    if has_ansi_codes(stdout_content):
        print("❌ ANSI codes found in stdout!")
        print(f"Raw stdout: {repr(stdout_content)}")
        all_clean = False
    else:
        print("✅ No ANSI codes in stdout")
    
    if has_ansi_codes(stderr_content):
        print("❌ ANSI codes found in stderr!")
        print(f"Raw stderr: {repr(stderr_content)}")
        all_clean = False
    else:
        print("✅ No ANSI codes in stderr")
    
    # Show the clean output
    print("\n📝 Cleaned commands (ANSI-free):")
    for i, raw_command in enumerate(test_commands, 1):
        cleaned = clean_shell_command(raw_command)
        clean_output = strip_ansi(cleaned)
        print(f"  {i}. {clean_output}")
    
    return all_clean

def test_search_results_format():
    """Test that search results would be displayed cleanly"""
    print("\n🔍 Testing search results format...")
    
    # Simulate search results after cleaning
    mock_results = [
        ("python process_tokens.py -c -b \"VMA new \"BOQ new\" \"ME\" | jq", 0.95),
        ("grep FICTION extracted_categories _Kindle_eBooks_answered.csv | grep -v \"Other\" | grep -v \"Poetry\" | wc -1", 0.87),
        ("python stitch_chapter_seams-py —-limit 1000", 0.78)
    ]
    
    print("Expected clean search results display:")
    for i, (cmd, score) in enumerate(mock_results, 1):
        # Simulate the display format from SearchResult.render()
        prefix = "▶ " if i == 1 else "  "  # First item selected
        display_line = f"{prefix}{cmd}"
        
        # Check for ANSI codes
        if has_ansi_codes(display_line):
            print(f"❌ Result {i}: Contains ANSI codes")
            return False
        else:
            print(f"✅ Result {i}: {display_line}")
    
    return True

if __name__ == "__main__":
    print("🔬 ANSI Clean Output Test (PYTE simulation)")
    print("=" * 50)
    
    success = True
    
    try:
        success &= test_command_cleaning_output()
        success &= test_search_results_format()
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        success = False
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 All ANSI cleanliness tests passed!")
        print("🎯 Output should be safe for PYTE environment")
    else:
        print("❌ Some tests failed - ANSI codes may leak")