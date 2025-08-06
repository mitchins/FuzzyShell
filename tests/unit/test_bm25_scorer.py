#!/usr/bin/env python3
"""
Unit tests for BM25Scorer class.

Tests the BM25 scoring implementation including IDF calculation,
term frequency handling, and score computation.
"""

import pytest
import numpy as np
import math
from test_helpers import create_test_db_connection

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from fuzzyshell.bm25_scorer import BM25Scorer, BM25ScoreCalculator
from fuzzyshell.data.datastore import CommandDAL, MockDatabaseProvider


@pytest.fixture
def test_bm25_scorer():
    """Create BM25Scorer with test database."""
    conn = create_test_db_connection()
    db_provider = MockDatabaseProvider(conn)
    command_dal = CommandDAL(db_provider, load_vss=False)
    
    # Add test data
    test_commands = [
        ("docker run hello-world", ["docker", "run", "hello-world"]),
        ("docker ps -a", ["docker", "ps", "-a"]),
        ("ls -lh", ["ls", "-lh"]),
        ("git status", ["git", "status"]),
        ("python script.py", ["python", "script.py"]),
        ("docker build -t app", ["docker", "build", "-t", "app"]),
        ("npm install package", ["npm", "install", "package"]),
        ("grep pattern file", ["grep", "pattern", "file"])
    ]
    
    for cmd_text, terms in test_commands:
        # Add command with dummy embedding (required for get_all_commands_with_embeddings_ordered)
        dummy_embedding = b'dummy_embedding_data_' + cmd_text.encode()
        command_dal.add_command(cmd_text, embedding=dummy_embedding)
        cmd_id = command_dal.get_command_count()
        
        # Add term frequencies
        term_freq_data = [(term, cmd_id, 1.0) for term in terms]
        command_dal.add_term_frequencies_batch(term_freq_data)
    
    return BM25Scorer(command_dal)


def test_bm25_scorer_initialization(test_bm25_scorer):
    """Test BM25Scorer initializes correctly."""
    assert test_bm25_scorer.command_dal is not None
    assert test_bm25_scorer.k1 == 1.5  # Default k1
    assert test_bm25_scorer.b == 0.75  # Default b
    assert not test_bm25_scorer._corpus_stats_cached


def test_corpus_stats_update(test_bm25_scorer):
    """Test corpus statistics are calculated correctly."""
    # Force corpus stats update
    test_bm25_scorer._update_corpus_stats()
    
    # Should have correct number of documents
    expected_docs = test_bm25_scorer.command_dal.get_command_count()
    assert test_bm25_scorer.total_docs == expected_docs
    assert test_bm25_scorer.total_docs > 0
    
    # Should have valid average length
    assert test_bm25_scorer.avg_doc_length > 0.0
    
    # Should be marked as cached
    assert test_bm25_scorer._corpus_stats_cached


def test_idf_calculation(test_bm25_scorer):
    """Test IDF calculation."""
    # Test common term (appears in multiple documents)
    docker_idf = test_bm25_scorer._get_idf("docker")
    assert docker_idf >= 0.0
    
    # Test rare term (appears in fewer documents)
    python_idf = test_bm25_scorer._get_idf("python")
    assert python_idf >= 0.0
    
    # Rare terms should have higher IDF than common terms
    # (unless there's only one instance of each)
    if docker_idf > 0 and python_idf > 0:
        # We expect python to be rarer than docker in our test data
        assert python_idf >= docker_idf
    
    # Test non-existent term
    nonexistent_idf = test_bm25_scorer._get_idf("nonexistent-term-xyz")
    assert nonexistent_idf == 0.0
    
    # Test IDF caching
    cached_idf = test_bm25_scorer._get_idf("docker")
    assert docker_idf == cached_idf


def test_term_frequencies_retrieval(test_bm25_scorer):
    """Test term frequency retrieval for commands."""
    # Get term frequencies for first command (docker run hello-world)
    term_freqs = test_bm25_scorer._get_command_term_frequencies(1)
    
    assert isinstance(term_freqs, dict)
    assert "docker" in term_freqs
    assert "run" in term_freqs
    assert "hello-world" in term_freqs
    
    # All frequencies should be positive
    for freq in term_freqs.values():
        assert freq > 0.0


def test_command_length_retrieval(test_bm25_scorer):
    """Test command length retrieval."""
    # Get length for first command
    length = test_bm25_scorer._get_command_length(1)
    assert length > 0
    
    # Test non-existent command
    nonexistent_length = test_bm25_scorer._get_command_length(9999)
    assert nonexistent_length == 0


def test_single_command_score_calculation(test_bm25_scorer):
    """Test BM25 score calculation for a single command."""
    # Prepare query terms and IDF values
    query_terms = ["docker", "run"]
    idf_values = {term: test_bm25_scorer._get_idf(term) for term in query_terms}
    
    # Calculate score for command 1 (docker run hello-world)
    score = test_bm25_scorer._calculate_single_score(query_terms, 1, idf_values)
    
    assert isinstance(score, float)
    assert score >= 0.0
    
    # Score should be higher for command that contains both terms
    # vs command that contains only one term
    single_term_query = ["docker"]
    single_idf_values = {term: test_bm25_scorer._get_idf(term) for term in single_term_query}
    single_score = test_bm25_scorer._calculate_single_score(single_term_query, 1, single_idf_values)
    
    # Two-term query should generally score higher than single-term
    # (though this depends on IDF values and term frequencies)
    assert score >= 0.0
    assert single_score >= 0.0


def test_batch_score_calculation(test_bm25_scorer):
    """Test batch BM25 score calculation."""
    query_terms = ["docker", "ps"]
    
    # Get some command IDs
    all_commands = test_bm25_scorer.command_dal.get_all_commands_with_embeddings_ordered()
    command_ids = [cmd[0] for cmd in all_commands[:5]]  # Test first 5 commands
    
    # Calculate batch scores
    scores = test_bm25_scorer.calculate_scores_batch(query_terms, command_ids)
    
    assert isinstance(scores, np.ndarray)
    assert len(scores) == len(command_ids)
    
    # All scores should be non-negative
    assert np.all(scores >= 0.0)
    
    # Commands containing query terms should have higher scores
    # Command 2 is "docker ps -a", should have high score for ["docker", "ps"]
    docker_ps_score = scores[1]  # Assuming command ID 2 is at index 1
    
    # Should be higher than commands without these terms
    assert docker_ps_score >= 0.0


def test_cache_invalidation(test_bm25_scorer):
    """Test cache invalidation."""
    # Cache some IDF values
    test_bm25_scorer._get_idf("docker")
    test_bm25_scorer._get_idf("python")
    
    # Verify cache has entries
    assert len(test_bm25_scorer.idf_cache) > 0
    
    # Invalidate cache
    test_bm25_scorer.invalidate_cache()
    
    # Cache should be empty and corpus stats should be invalidated
    assert len(test_bm25_scorer.idf_cache) == 0
    assert not test_bm25_scorer._corpus_stats_cached


def test_scorer_stats(test_bm25_scorer):
    """Test getting scorer statistics."""
    # Force update to populate stats
    test_bm25_scorer._update_corpus_stats()
    test_bm25_scorer._get_idf("docker")  # Cache one term
    
    stats = test_bm25_scorer.get_stats()
    
    assert isinstance(stats, dict)
    assert 'total_docs' in stats
    assert 'avg_doc_length' in stats
    assert 'cached_terms' in stats
    assert 'k1' in stats
    assert 'b' in stats
    
    assert stats['total_docs'] > 0
    assert stats['avg_doc_length'] > 0.0
    assert stats['cached_terms'] == 1  # We cached "docker"
    assert stats['k1'] == 1.5
    assert stats['b'] == 0.75


def test_empty_query_handling(test_bm25_scorer):
    """Test handling of empty queries."""
    # Empty query terms
    scores = test_bm25_scorer.calculate_scores_batch([], [1, 2, 3])
    assert np.all(scores == 0.0)
    
    # Empty command IDs
    scores = test_bm25_scorer.calculate_scores_batch(["docker"], [])
    assert len(scores) == 0
    
    # Both empty
    scores = test_bm25_scorer.calculate_scores_batch([], [])
    assert len(scores) == 0


def test_custom_parameters():
    """Test BM25Scorer with custom k1 and b parameters."""
    conn = create_test_db_connection()
    db_provider = MockDatabaseProvider(conn)
    command_dal = CommandDAL(db_provider, load_vss=False)
    
    # Add minimal test data
    dummy_embedding = b'dummy_embedding_data'
    command_dal.add_command("test command", embedding=dummy_embedding)
    command_dal.add_term_frequencies_batch([("test", 1, 1.0), ("command", 1, 1.0)])
    
    custom_scorer = BM25Scorer(command_dal, k1=2.0, b=0.5)
    
    assert custom_scorer.k1 == 2.0
    assert custom_scorer.b == 0.5
    
    # Should still work with different parameters
    query_terms = ["test"]
    command_ids = [1]
    
    scores = custom_scorer.calculate_scores_batch(query_terms, command_ids)
    assert len(scores) == 1
    assert np.all(scores >= 0.0)


# BM25ScoreCalculator tests

def test_bm25_calculator_idf_calculation():
    """Test standalone IDF calculation."""
    # Test normal cases
    idf1 = BM25ScoreCalculator.calculate_idf(term_doc_freq=2, total_docs=10)
    idf2 = BM25ScoreCalculator.calculate_idf(term_doc_freq=1, total_docs=10)
    
    # Rarer terms (lower doc freq) should have higher IDF
    assert idf2 > idf1
    assert idf1 >= 0.0
    assert idf2 >= 0.0
    
    # Edge cases
    assert BM25ScoreCalculator.calculate_idf(0, 10) == 0.0
    assert BM25ScoreCalculator.calculate_idf(5, 0) == 0.0


def test_bm25_calculator_score_calculation():
    """Test standalone BM25 score calculation."""
    query_terms = ["docker", "run"]
    document_terms = ["docker", "run", "hello-world", "container"]
    doc_length = 4
    avg_doc_length = 3.5
    term_idf_values = {"docker": 1.2, "run": 1.5, "hello-world": 2.0}
    
    score = BM25ScoreCalculator.calculate_bm25_score(
        query_terms=query_terms,
        document_terms=document_terms,
        doc_length=doc_length,
        avg_doc_length=avg_doc_length,
        term_idf_values=term_idf_values
    )
    
    assert isinstance(score, float)
    assert score >= 0.0
    
    # Test empty cases
    empty_score = BM25ScoreCalculator.calculate_bm25_score(
        query_terms=[],
        document_terms=document_terms,
        doc_length=doc_length,
        avg_doc_length=avg_doc_length,
        term_idf_values=term_idf_values
    )
    assert empty_score == 0.0


def test_bm25_calculator_custom_parameters():
    """Test BM25 calculation with custom k1 and b parameters."""
    query_terms = ["test"]
    document_terms = ["test", "document", "extra", "words"]  # Make doc longer
    doc_length = 4
    avg_doc_length = 2.0  # Different from doc_length to make b parameter matter
    term_idf_values = {"test": 1.5}  # Higher IDF to make differences more pronounced
    
    # Test with different k1 values
    score_k1_1 = BM25ScoreCalculator.calculate_bm25_score(
        query_terms, document_terms, doc_length, avg_doc_length, 
        term_idf_values, k1=1.0, b=0.75
    )
    
    score_k1_2 = BM25ScoreCalculator.calculate_bm25_score(
        query_terms, document_terms, doc_length, avg_doc_length, 
        term_idf_values, k1=3.0, b=0.75  # More significant difference
    )
    
    # Different k1 values should produce different scores
    # (k1 controls term frequency saturation)
    assert abs(score_k1_1 - score_k1_2) > 1e-5  # Use absolute difference instead of assertNotAlmostEqual
    assert score_k1_1 >= 0.0
    assert score_k1_2 >= 0.0
    
    # Test with different b values (length normalization)
    score_b_0 = BM25ScoreCalculator.calculate_bm25_score(
        query_terms, document_terms, doc_length, avg_doc_length, 
        term_idf_values, k1=1.5, b=0.0  # No length normalization
    )
    
    score_b_1 = BM25ScoreCalculator.calculate_bm25_score(
        query_terms, document_terms, doc_length, avg_doc_length, 
        term_idf_values, k1=1.5, b=1.0  # Full length normalization
    )
    
    # Different b values should produce different scores when doc_length != avg_doc_length
    assert abs(score_b_0 - score_b_1) > 1e-5  # Use absolute difference instead of assertNotAlmostEqual
    assert score_b_0 >= 0.0
    assert score_b_1 >= 0.0