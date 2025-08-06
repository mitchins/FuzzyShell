#!/usr/bin/env python3
"""
Unit tests for BM25 score normalization to prevent regression.

This test suite ensures that BM25 scores are properly normalized to [0,1] range
before being returned to users, preventing the score inflation bug where scores
appeared as 981%, 776%, etc.
"""

import pytest
import numpy as np
from test_helpers import create_test_db_connection

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from fuzzyshell.search_engine import SearchEngine
from fuzzyshell.bm25_scorer import BM25Scorer
from fuzzyshell.data.datastore import CommandDAL, MetadataDAL, MockDatabaseProvider


class MockModelHandler:
    """Mock model handler for testing."""
    
    def encode(self, texts):
        """Mock encoding that returns consistent embeddings."""
        embeddings = []
        for text in texts:
            # Create deterministic embeddings based on text
            hash_val = hash(text) % 1000000
            embedding = np.array([hash_val / 1000000.0] * 384, dtype=np.float32)
            embeddings.append(embedding)
        return np.array(embeddings)


@pytest.fixture
def test_search_engine():
    """Create SearchEngine with test database and mock model."""
    conn = create_test_db_connection()
    db_provider = MockDatabaseProvider(conn)
    
    # Create DALs
    command_dal = CommandDAL(db_provider, load_vss=False)
    metadata_dal = MetadataDAL(db_provider)
    
    # Mock model handler
    model_handler = MockModelHandler()
    
    def quantize_embedding(embedding):
        """Simple quantization for testing."""
        return embedding.astype(np.float32)
    
    def dequantize_embedding(embedding_blob):
        """Simple dequantization for testing."""
        return np.frombuffer(embedding_blob, dtype=np.float32)
    
    # Create search engine
    search_engine = SearchEngine(
        model_handler=model_handler,
        command_dal=command_dal,
        metadata_dal=metadata_dal,
        cache_dal=None,
        quantize_func=quantize_embedding,
        dequantize_func=dequantize_embedding
    )
    
    # Add test commands with varied BM25 relevance
    test_commands = [
        # High BM25 relevance for "docker" queries
        "docker run hello-world",
        "docker ps -a",
        "docker build -t myapp .",
        "docker compose up -d",
        "docker logs container-name",
        # Medium BM25 relevance
        "ls -lh docker-compose.yml",
        "grep docker /var/log/syslog",
        # Low BM25 relevance for "docker" queries
        "ls -la",
        "git status",
        "python script.py",
        "npm install",
        "curl -X GET https://api.example.com",
        "find . -name '*.py'",
        "ssh user@server",
        "pytest tests/"
    ]
    
    for cmd in test_commands:
        # Generate embedding and quantize
        embedding = model_handler.encode([cmd])[0]
        quantized = quantize_embedding(embedding)
        
        # Add command with embedding
        command_dal.add_command(cmd, embedding=quantized.tobytes())
        
        # Add term frequencies for BM25
        terms = cmd.lower().split()
        cmd_id = command_dal.get_command_count()  # Simple way to get last ID
        term_freq_data = [(term, cmd_id, 1.0) for term in terms]
        command_dal.add_term_frequencies_batch(term_freq_data)
    
    return search_engine


def test_bm25_normalize_basic_functionality(test_search_engine):
    """Test basic BM25 score normalization functionality."""
    # Create raw BM25 scores that would cause inflation
    raw_scores = np.array([15.2, 8.7, 3.4, 1.2, 0.8, 0.0])
    
    # Normalize using SearchEngine method
    normalized = test_search_engine._normalize_bm25_scores(raw_scores)
    
    # All scores should be in [0,1] range
    assert np.all(normalized >= 0.0), f"All normalized scores should be >= 0, got: {normalized}"
    assert np.all(normalized <= 1.0), f"All normalized scores should be <= 1, got: {normalized}"
    
    # Highest score should be 1.0
    assert abs(np.max(normalized) - 1.0) < 1e-6, "Highest normalized score should be 1.0"
    
    # Relative ordering should be preserved
    assert normalized[0] > normalized[1] > normalized[2], "Relative ordering should be preserved after normalization"
    
    # Zero score should remain zero
    assert normalized[-1] == 0.0, "Zero scores should remain zero"


def test_bm25_normalize_edge_cases(test_search_engine):
    """Test BM25 normalization edge cases."""
    # Empty array
    empty_scores = np.array([])
    normalized_empty = test_search_engine._normalize_bm25_scores(empty_scores)
    assert len(normalized_empty) == 0, "Empty array should return empty"
    
    # Single value
    single_score = np.array([5.7])
    normalized_single = test_search_engine._normalize_bm25_scores(single_score)
    assert len(normalized_single) == 1, "Single value should return single value"
    assert abs(normalized_single[0] - 1.0) < 1e-6, "Single value should normalize to 1.0"
    
    # All zeros
    zero_scores = np.array([0.0, 0.0, 0.0])
    normalized_zeros = test_search_engine._normalize_bm25_scores(zero_scores)
    assert np.all(normalized_zeros == 0.0), "All zero scores should remain zero"
    
    # All same non-zero values
    same_scores = np.array([4.2, 4.2, 4.2])
    normalized_same = test_search_engine._normalize_bm25_scores(same_scores)
    assert np.allclose(normalized_same, 1.0), "Identical non-zero scores should all normalize to 1.0"


def test_bm25_normalize_handles_extreme_values(test_search_engine):
    """Test BM25 normalization with extreme values."""
    # Very large values that would cause massive inflation
    extreme_scores = np.array([1000.0, 750.0, 500.0, 0.1])
    normalized_extreme = test_search_engine._normalize_bm25_scores(extreme_scores)
    
    # Should still be in [0,1] range despite extreme inputs
    assert np.all(normalized_extreme >= 0.0), "Extreme values should still normalize to >= 0"
    assert np.all(normalized_extreme <= 1.0), "Extreme values should still normalize to <= 1"
    
    # Max should be 1.0
    assert abs(np.max(normalized_extreme) - 1.0) < 1e-6, "Max of extreme values should be 1.0"
    
    # NaN and Inf handling  
    problematic_scores = np.array([np.nan, np.inf, -np.inf, 5.0, 2.0])
    normalized_problematic = test_search_engine._normalize_bm25_scores(problematic_scores)
    
    # Should not contain NaN or Inf
    assert not np.any(np.isnan(normalized_problematic)), "Normalized scores should not contain NaN"
    assert not np.any(np.isinf(normalized_problematic)), "Normalized scores should not contain Inf"
    
    # Should be in valid range
    assert np.all(normalized_problematic >= 0.0), "Problematic values should normalize to >= 0"
    assert np.all(normalized_problematic <= 1.0), "Problematic values should normalize to <= 1"


def test_search_returns_normalized_bm25_scores(test_search_engine):
    """Test that search results contain normalized BM25 scores, not raw ones."""
    # Search with a query that should generate varied BM25 scores
    query = "docker"
    
    # Get candidates and calculate scores manually to simulate internal process
    candidates, _ = test_search_engine._get_search_candidates(query)
    query_terms = ["docker"]
    command_ids = [c[0] for c in candidates]
    
    # Calculate raw BM25 scores 
    raw_bm25_scores = test_search_engine._calculate_bm25_scores_batch(query_terms, command_ids)
    
    # Calculate normalized scores
    normalized_bm25_scores = test_search_engine._normalize_bm25_scores(raw_bm25_scores)
    
    # Test that raw scores could potentially be > 1.0 (the original problem)
    if np.any(raw_bm25_scores > 1.0):
        # If we have scores that would cause inflation, test the fix
        assert np.all(normalized_bm25_scores <= 1.0), f"Normalized scores should be <= 1.0, got max: {np.max(normalized_bm25_scores)}"
        
        # Test that normalization actually changed the scores
        max_inflation_factor = np.max(raw_bm25_scores) / np.max(normalized_bm25_scores)
        assert max_inflation_factor > 1.0, "Normalization should reduce inflated scores"
    
    # All normalized scores should be in valid range
    assert np.all(normalized_bm25_scores >= 0.0), "All BM25 scores should be >= 0"
    assert np.all(normalized_bm25_scores <= 1.0), "All BM25 scores should be <= 1 (no inflation)"


def test_score_inflation_regression_prevention(test_search_engine):
    """Regression test: ensure BM25 scores never show as percentages > 100%."""
    # This test specifically prevents the regression where BM25 scores
    # were returned as raw values like 9.81 (showing as 981% to users)
    
    # Create scenario that would generate high BM25 scores
    high_relevance_commands = [
        "docker run special-keyword unique-term",
        "special-keyword docker compose", 
        "unique-term special-keyword docker"
    ]
    
    # Add these commands to database
    model_handler = MockModelHandler()
    for cmd in high_relevance_commands:
        embedding = model_handler.encode([cmd])[0]
        quantized = embedding.astype(np.float32)
        test_search_engine.command_dal.add_command(cmd, embedding=quantized.tobytes())
        
        # Add term frequencies with high relevance
        terms = cmd.lower().split()
        cmd_id = test_search_engine.command_dal.get_command_count()
        term_freq_data = [(term, cmd_id, 1.0) for term in terms]
        test_search_engine.command_dal.add_term_frequencies_batch(term_freq_data)
    
    # Search for the special terms
    query_terms = ["special-keyword", "unique-term"]
    
    # Get all command IDs for testing
    all_candidates = test_search_engine.command_dal.get_all_commands_with_embeddings_ordered()
    all_command_ids = [c[0] for c in all_candidates]
    
    # Calculate BM25 scores
    raw_bm25_scores = test_search_engine._calculate_bm25_scores_batch(query_terms, all_command_ids)
    normalized_bm25_scores = test_search_engine._normalize_bm25_scores(raw_bm25_scores)
    
    # Regression prevention: no score should exceed 1.0 
    max_normalized_score = np.max(normalized_bm25_scores) if len(normalized_bm25_scores) > 0 else 0
    assert max_normalized_score <= 1.0, (f"REGRESSION: BM25 score exceeded 1.0: {max_normalized_score:.3f}. "
                                         f"This would show as {max_normalized_score*100:.0f}% to users!")
    
    # Ensure normalization is actually applied (scores should not all be identical)
    if len(normalized_bm25_scores) > 1:
        unique_scores = len(set(np.round(normalized_bm25_scores, decimals=6)))
        assert unique_scores > 1, "Normalization should preserve score differences"
    
    # Test that if raw scores were > 1, normalization fixed them
    if np.any(raw_bm25_scores > 1.0):
        max_raw_score = np.max(raw_bm25_scores)
        inflation_prevented = max_raw_score - max_normalized_score
        assert inflation_prevented > 0, (f"Normalization should prevent inflation. "
                                        f"Raw max: {max_raw_score:.3f}, Normalized max: {max_normalized_score:.3f}")


def test_normalization_preserves_relative_ordering(test_search_engine):
    """Test that BM25 normalization preserves the relative ordering of scores."""
    # Create a set of scores with clear ordering
    test_scores = np.array([12.5, 8.7, 5.2, 3.1, 1.4, 0.8, 0.2, 0.0])
    
    # Normalize them
    normalized = test_search_engine._normalize_bm25_scores(test_scores)
    
    # Check that relative ordering is preserved
    for i in range(len(normalized) - 1):
        assert normalized[i] >= normalized[i + 1], (f"Relative ordering not preserved at position {i}: "
                                                   f"{normalized[i]:.6f} < {normalized[i + 1]:.6f}")
    
    # Verify the transformation maintains proportional relationships
    # (within reasonable floating point precision)
    if len(test_scores) >= 3:
        # Test that the ratio between scores is approximately maintained
        # for the top scores (avoiding division by very small numbers)
        original_ratio = test_scores[0] / test_scores[1] if test_scores[1] > 0 else float('inf')
        normalized_ratio = normalized[0] / normalized[1] if normalized[1] > 0 else float('inf')
        
        # For normalized scores, the ratio should be smaller but still reflect the relationship
        if original_ratio != float('inf') and normalized_ratio != float('inf'):
            assert normalized_ratio > 1.0, "Top score should still be higher after normalization"