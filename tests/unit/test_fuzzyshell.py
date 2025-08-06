import pytest
import os
import tempfile
import sqlite3
from fuzzyshell import FuzzyShell
import numpy as np
from test_helpers import create_test_db_connection


@pytest.fixture
def test_fuzzyshell():
    """Create FuzzyShell instance with test database and temp history file."""
    # Use stock model for unit tests (allows download)
    os.environ['FUZZYSHELL_MODEL'] = 'stock-minilm-l6'
    
    # Use dependency injection with in-memory SQLite database
    test_conn = create_test_db_connection()
    fuzzyshell = FuzzyShell(conn=test_conn)
    
    # Create a temporary history file
    test_commands = [
        "git push origin main",
        "docker build -t myapp .",
        "npm install express",
        "python manage.py runserver",
        "ls -la /var/log"
    ]
    
    # Write commands with a newline after each one and one at the end
    temp_history = tempfile.NamedTemporaryFile(mode='w', delete=False)
    temp_history.write('\n'.join(test_commands) + '\n')
    temp_history.flush()  # Ensure all data is written
    os.fsync(temp_history.fileno())  # Force filesystem sync
    temp_history.close()
    
    # Add attributes for access in tests
    fuzzyshell.test_conn = test_conn
    fuzzyshell.test_commands = test_commands
    fuzzyshell.temp_history_name = temp_history.name
    
    yield fuzzyshell
    
    # Clean up temporary history file (in-memory DB is automatically cleaned up)
    os.unlink(temp_history.name)

def test_init_db(test_fuzzyshell):
    """Test if database tables are created correctly"""
    # Use the same injected connection
    cursor = test_fuzzyshell.test_conn.cursor()
    
    # Check if commands table exists and has correct schema
    cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='commands'")
    commands_schema = cursor.fetchone()[0]
    assert "id INTEGER PRIMARY KEY" in commands_schema
    assert "command TEXT UNIQUE NOT NULL" in commands_schema

    # Check if embeddings table exists (virtual or regular depending on extensions)
    cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='embeddings'")
    embeddings_schema = cursor.fetchone()[0]
    # Should contain either VIRTUAL TABLE (with vss0) or regular TABLE
    assert "embeddings" in embeddings_schema.lower()
    assert "embedding" in embeddings_schema.lower()

def test_ingest_history(test_fuzzyshell):
    """Test history ingestion"""
    # Mock the get_shell_history_file method to return our temp file
    test_fuzzyshell.get_shell_history_file = lambda: test_fuzzyshell.temp_history_name
    
    # Ingest the test commands
    test_fuzzyshell.ingest_history()
    
    # Verify all commands were ingested using DAL
    with test_fuzzyshell.command_dal.connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT command FROM commands")
        ingested_commands = {row[0] for row in cursor.fetchall()}
    
    assert len(ingested_commands) == len(test_fuzzyshell.test_commands)
    for cmd in test_fuzzyshell.test_commands:
        assert cmd in ingested_commands


def test_search_exact_match(test_fuzzyshell):
    """Test search functionality with exact matches"""
    # Mock history and ingest test commands
    test_fuzzyshell.get_shell_history_file = lambda: test_fuzzyshell.temp_history_name
    test_fuzzyshell.ingest_history()
    
    # Test exact match
    results = test_fuzzyshell.search("git push origin main")
    assert len(results) > 0
    # Check that the exact match is in the results - debug output
    commands_in_results = [cmd for cmd, score in results]
    # Check that the exact match is found (any score)
    exact_match_found = any(cmd == "git push origin main"
                           for cmd, score in results)
    if not exact_match_found:
        print(f"Commands in results: {commands_in_results}")
        print(f"Test commands: {test_fuzzyshell.test_commands}")
    assert exact_match_found, f"Exact match 'git push origin main' should be found. Got: {results}"


def test_search_semantic_match(test_fuzzyshell):
    """Test search functionality with semantic matches"""
    # Mock history and ingest test commands
    test_fuzzyshell.get_shell_history_file = lambda: test_fuzzyshell.temp_history_name
    test_fuzzyshell.ingest_history()
    
    # Test semantic match
    results = test_fuzzyshell.search("start development server")
    matching_commands = [cmd for cmd, _ in results]
    assert "python manage.py runserver" in matching_commands


def test_search_no_results(test_fuzzyshell):
    """Test search with no matching results"""
    # Mock history and ingest test commands
    test_fuzzyshell.get_shell_history_file = lambda: test_fuzzyshell.temp_history_name
    test_fuzzyshell.ingest_history()
    
    # Test with unlikely query - check that results have very low scores
    results = test_fuzzyshell.search("xyzabc123nonexistent")
    # Either no results or very low scoring results (adjust threshold based on actual behavior)
    if results:
        max_score = max(score for _, score in results)
        assert max_score < 0.9, "Random query should not have very high similarity scores"


def test_embedding_consistency(test_fuzzyshell):
    """Test if embeddings are consistent across searches"""
    test_fuzzyshell.get_shell_history_file = lambda: test_fuzzyshell.temp_history_name
    test_fuzzyshell.ingest_history()
    
    # Perform same search twice
    results1 = test_fuzzyshell.search("git")
    results2 = test_fuzzyshell.search("git")
    
    # Check if results are consistent
    assert len(results1) == len(results2)
    for (cmd1, score1), (cmd2, score2) in zip(results1, results2):
        assert cmd1 == cmd2
        assert abs(score1 - score2) < 1e-5  # Use explicit tolerance instead of assertAlmostEqual


def test_hybrid_search(test_fuzzyshell):
    """Test that hybrid search balances exact and semantic matches"""
    test_fuzzyshell.get_shell_history_file = lambda: test_fuzzyshell.temp_history_name
    test_fuzzyshell.ingest_history()
    
    # Test with exact match term
    results = test_fuzzyshell.search("git push")
    assert any("git push" in cmd for cmd, _ in results)
    
    # Test with semantic match
    results = test_fuzzyshell.search("version control push")
    assert any("git push" in cmd for cmd, _ in results)
    
    # Test rare term matching
    results = test_fuzzyshell.search("runserver")
    assert any("runserver" in cmd for cmd, _ in results)
    
    # Test that exact matches score reasonably high
    exact_results = test_fuzzyshell.search("docker build")
    semantic_results = test_fuzzyshell.search("container create")
    
    # Find the score for "docker build" in both searches
    exact_score = next((score for cmd, score in exact_results 
                      if "docker build" in cmd), 0)
    semantic_score = next((score for cmd, score in semantic_results 
                         if "docker build" in cmd), 0)
    
    # Both should find the docker build command with decent scores
    assert exact_score > 0.5, "Exact match should have reasonable score"
    assert semantic_score > 0.05, "Semantic match should have some score"

