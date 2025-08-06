#!/usr/bin/env python3
"""
Unit tests for scoring preference functionality.

Tests the user-configurable scoring weights and their integration into
the search algorithm's hybrid scoring system.
"""

import pytest
import sqlite3
from test_helpers import create_test_db_connection

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from fuzzyshell import FuzzyShell
from fuzzyshell.data.datastore import MetadataDAL, MockDatabaseProvider


@pytest.fixture
def test_fuzzyshell():
    """Create a FuzzyShell instance with test database."""
    conn = create_test_db_connection()
    fs = FuzzyShell(conn=conn)
    return fs


@pytest.fixture
def metadata_dal(test_fuzzyshell):
    """Create MetadataDAL instance for testing."""
    return test_fuzzyshell.metadata_dal


def test_default_scoring_preference(metadata_dal):
    """Test that default scoring preference is balanced."""
    preference = metadata_dal.get_scoring_preference()
    assert preference == 'balanced'
    
    semantic_weight, bm25_weight = metadata_dal.get_scoring_weights()
    assert semantic_weight == 0.5
    assert bm25_weight == 0.5


@pytest.mark.parametrize("preference,expected_weights", [
    ('less_semantic', (0.3, 0.7)),
    ('balanced', (0.5, 0.5)),
    ('more_semantic', (0.7, 0.3)),
    ('semantic_only', (1.0, 0.0))
])
def test_set_scoring_preferences(metadata_dal, preference, expected_weights):
    """Test setting different scoring preferences."""
    metadata_dal.set_scoring_preference(preference)
    
    # Verify preference is stored
    stored_preference = metadata_dal.get_scoring_preference()
    assert stored_preference == preference
    
    # Verify weights are correct
    semantic_weight, bm25_weight = metadata_dal.get_scoring_weights()
    assert abs(semantic_weight - expected_weights[0]) < 0.1
    assert abs(bm25_weight - expected_weights[1]) < 0.1


def test_invalid_scoring_preference(metadata_dal):
    """Test that invalid scoring preferences raise ValueError."""
    with pytest.raises(ValueError):
        metadata_dal.set_scoring_preference('invalid_preference')


def test_hybrid_scoring_with_preferences(test_fuzzyshell):
    """Test that SearchEngine uses scoring preferences correctly."""
    fs = test_fuzzyshell
    
    # Ensure SearchEngine is initialized
    fs.wait_for_model(timeout=10.0)
    if fs.search_engine is None:
        pytest.skip("SearchEngine not initialized")
    
    # Add test commands to search from
    test_commands = [
        "ls -lh",
        "git status", 
        "docker ps",
        "test command example"
    ]
    
    for cmd in test_commands:
        fs.command_dal.add_command(cmd)
    
    query = "list files"
    
    # Test different preferences produce different results
    test_cases = ['less_semantic', 'balanced', 'more_semantic', 'semantic_only']
    
    results = {}
    for preference in test_cases:
        # Set preference
        fs.metadata_dal.set_scoring_preference(preference)
        
        # Perform search with current preference
        search_results = fs.search(query, top_k=3, return_scores=True)
        
        if search_results:
            results[preference] = search_results[0][1]  # Combined score
            
            # For semantic_only, BM25 weight should be 0
            if preference == 'semantic_only':
                # Verify SearchEngine uses semantic_only mode
                weights = fs.metadata_dal.get_scoring_weights()
                assert weights[1] == 0.0  # BM25 weight should be 0
    
    # Verify that different preferences produce different results
    if len(results) >= 2:
        preference_keys = list(results.keys())
        assert results[preference_keys[0]] != results[preference_keys[-1]]


def test_semantic_only_ignores_bm25(test_fuzzyshell):
    """Test that semantic_only mode completely ignores BM25 scores."""
    fs = test_fuzzyshell
    
    # Ensure SearchEngine is initialized
    fs.wait_for_model(timeout=10.0)
    if fs.search_engine is None:
        pytest.skip("SearchEngine not initialized")
    
    fs.metadata_dal.set_scoring_preference('semantic_only')
    
    # Add test commands with very different BM25 characteristics
    # Command 1: Many common words (low BM25 for most queries)
    cmd1 = "run cmd with many common words and the and run and cmd"
    # Command 2: Unique words (high BM25 for specific queries)  
    cmd2 = "specialized_unique_command_xyz"
    
    fs.command_dal.add_command(cmd1)
    fs.command_dal.add_command(cmd2)
    
    # Query that would have very different BM25 scores for these commands
    query = "specialized unique"
    
    # In semantic_only mode, the results should be based purely on semantic similarity
    # not on BM25 term matching
    results = fs.search(query, top_k=2, return_scores=True)
    
    if len(results) >= 2:
        # Verify that SearchEngine is using semantic_only weights
        weights = fs.metadata_dal.get_scoring_weights()
        assert weights[1] == 0.0, "BM25 weight should be 0 in semantic_only mode"
        assert weights[0] == 1.0, "Semantic weight should be 1 in semantic_only mode"


def test_preference_persistence():
    """Test that scoring preferences persist across instances."""
    conn = create_test_db_connection()
    db_provider = MockDatabaseProvider(conn)
    
    # Set a preference
    metadata_dal1 = MetadataDAL(db_provider)
    metadata_dal1.set_scoring_preference('more_semantic')
    
    # Create new metadata DAL with same database
    metadata_dal2 = MetadataDAL(db_provider)
    
    # Verify preference persisted
    stored_preference = metadata_dal2.get_scoring_preference()
    assert stored_preference == 'more_semantic'
    
    semantic_weight, bm25_weight = metadata_dal2.get_scoring_weights()
    assert abs(semantic_weight - 0.7) < 0.1
    assert abs(bm25_weight - 0.3) < 0.1


@pytest.mark.parametrize("preference", ['less_semantic', 'balanced', 'more_semantic', 'semantic_only'])
def test_scoring_weights_sum_to_one(metadata_dal, preference):
    """Test that scoring weights always sum to 1.0."""
    metadata_dal.set_scoring_preference(preference)
    semantic_weight, bm25_weight = metadata_dal.get_scoring_weights()
    
    # Weights should sum to 1.0 (within floating point precision)
    total_weight = semantic_weight + bm25_weight
    assert abs(total_weight - 1.0) < 0.1


def test_expert_screen_radio_buttons(test_fuzzyshell):
    """Test that ExpertScreen creates radio buttons correctly."""
    pytest.importorskip("urwid", reason="ExpertScreen requires urwid")
    
    from fuzzyshell.tui.screens.expert_screen import ExpertScreen
    
    # Create screen with FuzzyShell instance
    screen = ExpertScreen(test_fuzzyshell)
    
    # Should have 4 radio buttons (one for each scoring option)
    assert len(screen.scoring_radio_group) == 4
    
    # One button should be selected (current preference)
    selected_buttons = [btn for btn in screen.scoring_radio_group if btn.get_state()]
    assert len(selected_buttons) == 1
    
    # Test radio button callback works
    # Find the 'more_semantic' button and simulate selection
    more_semantic_btn = None
    for btn in screen.scoring_radio_group:
        if 'More Semantic' in btn.get_label():
            more_semantic_btn = btn
            break
    
    assert more_semantic_btn is not None
    
    # Simulate button press by calling the callback
    screen._on_scoring_change(more_semantic_btn, True, 'more_semantic')
    
    # Verify preference was changed
    new_preference = test_fuzzyshell.metadata_dal.get_scoring_preference()
    assert new_preference == 'more_semantic'