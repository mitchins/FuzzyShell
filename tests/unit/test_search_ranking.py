#!/usr/bin/env python3
"""Test search ranking order for command search scenarios"""

import pytest
import logging
from test_helpers import create_test_db_connection
from fuzzyshell.fuzzyshell import FuzzyShell

# Reduce logging noise during tests
logging.getLogger('FuzzyShell').setLevel(logging.CRITICAL)


@pytest.fixture
def fuzzyshell_with_commands():
    """Set up test database with representative commands"""
    conn = create_test_db_connection()
    fs = FuzzyShell(conn=conn)
    fs.init_model_sync()
    
    # Add realistic command corpus focusing on problematic cases
    test_commands = [
        # Core ls commands - should rank highly for "list files"
        "ls",
        "ls -l",
        "ls -lh", 
        "ls -la",
        "ls ./",
        "ls ..",
        "ls -lh *.txt",
        "ls -lh ~",
        "ls output",
        "ls util",
        "ls Assets",
        
        # Commands with "list" that aren't file listing 
        "ollama list",  # Lists AI models, not files
        "adb list",     # Lists Android devices
        "brew list",    # Lists installed packages
        "conda list",   # Lists conda packages
        "git stash list", # Lists git stashes
        
        # File operations that might compete semantically
        "find . -name '*.txt'",
        "find / -type f",
        "tree -L 2",
        "du -sh *",
        
        # Other common commands
        "git status",
        "git lfs ls-files",  # This one actually lists files
        "ps aux",
        "docker ps",
    ]
    
    for cmd in test_commands:
        fs.add_command(cmd)
    
    fs.update_corpus_stats()
    
    # Verify setup
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM commands")
    count = c.fetchone()[0]
    if count == 0:
        pytest.skip("Failed to add commands to test database")
    
    return fs


def test_list_files_ranking_order(fuzzyshell_with_commands):
    """
    Test that 'list files' query ranks actual file listing commands higher
    than commands that happen to contain 'list' but don't list files
    """
    fs = fuzzyshell_with_commands
    results = fs.search("list files")
    commands = [cmd for cmd, _ in results[:15]]  # Top 15 results
    
    # Core ls commands should dominate the top results
    core_ls_commands = {"ls", "ls -l", "ls -lh", "ls -la", "ls ./", "ls .."}
    
    # Count how many core ls commands appear in top 10
    top_10_commands = set(commands[:10])
    core_ls_in_top_10 = len(core_ls_commands.intersection(top_10_commands))
    
    assert core_ls_in_top_10 >= 4, (
        f"At least 4 core ls commands should be in top 10. "
        f"Found {core_ls_in_top_10} in top 10: {list(top_10_commands)}"
    )
    
    # Verify specific commands appear reasonably early
    ls_position = commands.index("ls") if "ls" in commands else 999
    ls_dash_l_position = commands.index("ls -l") if "ls -l" in commands else 999
    
    assert ls_position < 10, f"'ls' should be in top 10, found at position {ls_position + 1}"
    assert ls_dash_l_position < 10, f"'ls -l' should be in top 10, found at position {ls_dash_l_position + 1}"


def test_semantic_over_keyword_matching(fuzzyshell_with_commands):
    """
    Test that semantic relevance trumps simple keyword matching for common terms
    """
    fs = fuzzyshell_with_commands
    results = fs.search("list files") 
    commands = [cmd for cmd, _ in results[:20]]
    
    # Commands that list files but don't contain "list" should still rank well
    file_listing_commands = {"ls", "ls -l", "ls -lh", "find . -name '*.txt'", "tree -L 2"}
    
    # Commands that contain "list" but don't list files  
    non_file_list_commands = {"ollama list", "brew list", "conda list"}
    
    # Find positions
    file_listing_positions = []
    for cmd in file_listing_commands:
        if cmd in commands:
            file_listing_positions.append(commands.index(cmd))
    
    non_file_list_positions = []
    for cmd in non_file_list_commands:
        if cmd in commands:
            non_file_list_positions.append(commands.index(cmd))
    
    if file_listing_positions and non_file_list_positions:
        avg_file_listing_pos = sum(file_listing_positions) / len(file_listing_positions)
        avg_non_file_list_pos = sum(non_file_list_positions) / len(non_file_list_positions)
        
        # On average, actual file listing commands should rank better
        assert avg_file_listing_pos < avg_non_file_list_pos, (
            f"File listing commands should rank better on average. "
            f"File listing avg: {avg_file_listing_pos:.1f}, "
            f"Non-file list avg: {avg_non_file_list_pos:.1f}"
        )


def test_git_lfs_ls_files_appropriate_ranking(fuzzyshell_with_commands):
    """
    Test that 'git lfs ls-files' ranks appropriately for 'list files' 
    (it actually does list files, so should rank well but not dominate over basic ls)
    """
    fs = fuzzyshell_with_commands
    results = fs.search("list files")
    commands = [cmd for cmd, _ in results[:15]]
    
    if "git lfs ls-files" in commands:
        git_lfs_position = commands.index("git lfs ls-files")
        
        # Should be in top 15 since it actually lists files
        assert git_lfs_position < 15, (
            f"'git lfs ls-files' should rank in top 15 for 'list files', "
            f"found at position {git_lfs_position + 1}"
        )
        
        # But basic ls commands should generally rank higher
        basic_ls_positions = []
        for cmd in ["ls", "ls -l", "ls -lh"]:
            if cmd in commands:
                basic_ls_positions.append(commands.index(cmd))
        
        if basic_ls_positions:
            best_basic_ls_pos = min(basic_ls_positions)
            # Git lfs should not consistently beat all basic ls commands
            # (allowing some flexibility since git lfs does contain both query words)
            assert git_lfs_position >= best_basic_ls_pos, (
                f"Basic ls commands should generally rank higher than git lfs. "
                f"Best basic ls at {best_basic_ls_pos + 1}, "
                f"git lfs at {git_lfs_position + 1}"
            )


def test_directory_listing_variants(fuzzyshell_with_commands):
    """Test that different directory listing variants maintain reasonable relative order"""
    fs = fuzzyshell_with_commands
    results = fs.search("show directory contents")
    commands = [cmd for cmd, _ in results[:10]]
    
    # Directory listing commands should dominate
    dir_listing_commands = {"ls", "ls -l", "ls -lh", "ls -la", "tree -L 2"}
    
    dir_commands_in_top_10 = sum(1 for cmd in commands if cmd in dir_listing_commands)
    
    assert dir_commands_in_top_10 >= 3, (
        f"At least 3 directory listing commands should be in top 10 "
        f"for 'show directory contents'. Found {dir_commands_in_top_10}"
    )


def test_find_files_semantic_quality(fuzzyshell_with_commands):
    """Test that 'find files' query produces sensible results"""
    fs = fuzzyshell_with_commands
    results = fs.search("find files")
    commands = [cmd for cmd, _ in results[:10]]
    
    # Both find commands and ls commands should rank well
    file_finding_commands = {"find . -name '*.txt'", "find / -type f", "ls", "ls -l", "ls -lh"}
    
    relevant_commands_in_top_10 = sum(1 for cmd in commands if cmd in file_finding_commands)
    
    assert relevant_commands_in_top_10 >= 3, (
        f"At least 3 file-finding commands should be in top 10 "
        f"for 'find files'. Found {relevant_commands_in_top_10}"
    )


def test_ranking_stability(fuzzyshell_with_commands):
    """Test that search rankings are stable across multiple runs"""
    fs = fuzzyshell_with_commands
    query = "list files"
    
    # Run search multiple times
    results1 = fs.search(query)[:10]
    results2 = fs.search(query)[:10] 
    results3 = fs.search(query)[:10]
    
    commands1 = [cmd for cmd, _ in results1]
    commands2 = [cmd for cmd, _ in results2]
    commands3 = [cmd for cmd, _ in results3]
    
    # Top 5 should be identical across runs (allowing for tie-breaking variations)
    top5_1 = commands1[:5]
    top5_2 = commands2[:5]
    top5_3 = commands3[:5]
    
    # At least 4 out of 5 should be the same
    overlap_1_2 = len(set(top5_1).intersection(set(top5_2)))
    overlap_1_3 = len(set(top5_1).intersection(set(top5_3)))
    
    assert overlap_1_2 >= 4, (
        f"Search results should be stable. "
        f"Run 1 top 5: {top5_1}, Run 2 top 5: {top5_2}"
    )
    assert overlap_1_3 >= 4, (
        f"Search results should be stable. "
        f"Run 1 top 5: {top5_1}, Run 3 top 5: {top5_3}"
    )