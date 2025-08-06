#!/usr/bin/env python3
"""
Unit tests for SearchEngine class.

Tests the extracted search functionality including semantic search,
BM25 scoring, and hybrid scoring.
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
        """Mock encoding that returns simple embeddings."""
        embeddings = []
        for text in texts:
            # Create a simple embedding based on text hash
            # This ensures consistent but different embeddings for different texts
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
        cache_dal=None,  # Not testing cache for now
        quantize_func=quantize_embedding,
        dequantize_func=dequantize_embedding
    )
    
    # Add some test commands
    test_commands = [
        "ls -lh",
        "git status", 
        "docker ps",
        "python script.py",
        "find . -name '*.py'",
        "grep -r 'pattern' src/",
        "curl -X GET https://api.example.com",
        "npm install",
        "pytest tests/",
        "ssh user@server"
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


def test_search_engine_initialization(test_search_engine):
    """Test that SearchEngine initializes correctly."""
    assert test_search_engine.model_handler is not None
    assert test_search_engine.command_dal is not None
    assert test_search_engine.metadata_dal is not None
    assert isinstance(test_search_engine.bm25_scorer, BM25Scorer)


def test_model_ready_check(test_search_engine):
    """Test model readiness check."""
    assert test_search_engine._is_model_ready()
    
    # Test with None model
    search_engine_no_model = SearchEngine(
        model_handler=None,
        command_dal=test_search_engine.command_dal,
        metadata_dal=test_search_engine.metadata_dal,
        cache_dal=None
    )
    assert not search_engine_no_model._is_model_ready()


def test_query_embedding_generation(test_search_engine):
    """Test query embedding generation."""
    query = "list files"
    embedding = test_search_engine._generate_query_embedding(query)
    
    assert isinstance(embedding, np.ndarray)
    assert len(embedding) == test_search_engine.model_output_dim


def test_get_search_candidates(test_search_engine):
    """Test getting search candidates."""
    candidates, is_filtered = test_search_engine._get_search_candidates("list")
    
    assert isinstance(candidates, list)
    assert isinstance(is_filtered, bool)
    assert len(candidates) > 0
    
    # Each candidate should be a tuple (id, command, embedding)
    for candidate in candidates:
        assert isinstance(candidate, tuple)
        assert len(candidate) == 3


def test_semantic_scores_calculation(test_search_engine):
    """Test semantic similarity score calculation."""
    # Get some test embeddings
    candidates, _ = test_search_engine._get_search_candidates("test")
    embeddings_blobs = [c[2] for c in candidates[:3]]  # Take first 3
    
    # Load embeddings
    candidate_embeddings = test_search_engine._load_embeddings_batch(
        tuple(embeddings_blobs), None, len(embeddings_blobs)
    )
    
    # Generate query embedding
    query_embedding = test_search_engine._generate_query_embedding("test query")
    
    # Calculate semantic scores
    scores = test_search_engine._calculate_semantic_scores(
        candidate_embeddings, query_embedding
    )
    
    assert isinstance(scores, np.ndarray)
    assert len(scores) == len(candidate_embeddings)
    
    # Scores should be in [0, 1] range
    assert np.all(scores >= 0.0)
    assert np.all(scores <= 1.0)


def test_bm25_scores_calculation(test_search_engine):
    """Test BM25 score calculation."""
    query_terms = ["ls", "git"]
    
    # Get some command IDs
    candidates, _ = test_search_engine._get_search_candidates("test")
    command_ids = [c[0] for c in candidates[:3]]
    
    # Calculate BM25 scores
    bm25_scores = test_search_engine._calculate_bm25_scores_batch(query_terms, command_ids)
    
    assert isinstance(bm25_scores, np.ndarray)
    assert len(bm25_scores) == len(command_ids)
    
    # Scores should be non-negative
    assert np.all(bm25_scores >= 0.0)


def test_bm25_score_normalization(test_search_engine):
    """Test BM25 score normalization prevents inflation regression."""
    # Test the normalization method directly
    raw_scores = np.array([10.5, 7.2, 3.8, 1.1, 0.0])
    normalized = test_search_engine._normalize_bm25_scores(raw_scores)
    
    # All normalized scores should be in [0,1] range
    assert np.all(normalized >= 0.0), "Normalized scores should be >= 0"
    assert np.all(normalized <= 1.0), "Normalized scores should be <= 1 (no inflation)"
    
    # Highest score should be 1.0
    assert abs(np.max(normalized) - 1.0) < 1e-6
    
    # Relative ordering should be preserved
    for i in range(len(normalized) - 1):
        if raw_scores[i] > raw_scores[i + 1]:
            assert normalized[i] > normalized[i + 1], "Relative ordering should be preserved"
    
    # Zero should remain zero
    assert normalized[-1] == 0.0, "Zero scores should remain zero"


def test_bm25_normalization_handles_edge_cases(test_search_engine):
    """Test BM25 normalization edge cases."""
    # Empty array
    empty_normalized = test_search_engine._normalize_bm25_scores(np.array([]))
    assert len(empty_normalized) == 0
    
    # Single value
    single_normalized = test_search_engine._normalize_bm25_scores(np.array([5.0]))
    assert abs(single_normalized[0] - 1.0) < 1e-6
    
    # All zeros
    zeros_normalized = test_search_engine._normalize_bm25_scores(np.array([0.0, 0.0, 0.0]))
    assert np.all(zeros_normalized == 0.0)
    
    # Handle NaN and Inf
    problematic = np.array([np.nan, np.inf, 5.0, 2.0])
    clean_normalized = test_search_engine._normalize_bm25_scores(problematic)
    
    # Should not contain NaN or Inf
    assert not np.any(np.isnan(clean_normalized))
    assert not np.any(np.isinf(clean_normalized))
    assert np.all(clean_normalized >= 0.0)
    assert np.all(clean_normalized <= 1.0)


@pytest.mark.parametrize("preference", [
    'balanced',
    'more_semantic',
    'semantic_only',
])
def test_dynamic_hybrid_scoring(test_search_engine, preference):
    """Test dynamic hybrid scoring with user preferences."""
    semantic_score = 0.8
    bm25_score = 0.6
    
    # Set preference
    test_search_engine.metadata_dal.set_scoring_preference(preference)
    
    # Calculate hybrid score
    hybrid_score = test_search_engine._dynamic_hybrid_score(
        semantic_score, bm25_score, "test command", "test query"
    )
    
    # Test that the scoring behaves according to preference
    # (Exact weights may vary due to normalization, but behavior should be correct)
    if preference == 'semantic_only':
        # Should be close to pure semantic score
        assert abs(hybrid_score - semantic_score) < 0.01
    elif preference == 'more_semantic':
        # Should be closer to semantic than balanced would be
        balanced_score = 0.5 * semantic_score + 0.5 * bm25_score
        assert hybrid_score > balanced_score - 0.05  # More semantic weight
    else:  # balanced
        # Should be a reasonable combination
        assert 0.6 < hybrid_score < 0.8  # Between our test scores


def test_cache_invalidation(test_search_engine):
    """Test cache invalidation."""
    # This should not raise an exception
    test_search_engine.invalidate_caches()
    
    # Verify BM25 cache was cleared
    stats = test_search_engine.bm25_scorer.get_stats()
    assert stats['cached_terms'] == 0


def test_search_integration(test_search_engine):
    """Test basic search integration (without full search due to complexity)."""
    # Test the components work together
    query = "list files"
    
    # Test query embedding generation
    query_embedding = test_search_engine._generate_query_embedding(query)
    assert isinstance(query_embedding, np.ndarray)
    
    # Test getting candidates
    candidates, _ = test_search_engine._get_search_candidates(query)
    assert len(candidates) > 0
    
    # Test components are initialized
    assert test_search_engine.bm25_scorer is not None
    assert test_search_engine._is_model_ready()