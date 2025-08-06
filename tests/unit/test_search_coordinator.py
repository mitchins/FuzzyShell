"""
Tests for SearchCoordinator and EmbeddingManager classes.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
import tempfile
import os

from fuzzyshell.search_coordinator import (
    SearchCoordinator, EmbeddingManager, SimilarityCalculator, SearchResult, SearchStats
)


class TestSearchResult:
    """Test SearchResult dataclass."""
    
    def test_search_result_creation(self):
        """Test SearchResult creation and attributes."""
        result = SearchResult(
            command="ls -la",
            semantic_score=0.95,
            bm25_score=0.80,
            combined_score=0.87,
            rank=1
        )
        
        assert result.command == "ls -la"
        assert result.semantic_score == 0.95
        assert result.bm25_score == 0.80
        assert result.combined_score == 0.87
        assert result.rank == 1


class TestSearchStats:
    """Test SearchStats dataclass."""
    
    def test_search_stats_creation(self):
        """Test SearchStats creation and attributes."""
        stats = SearchStats(
            total_time=1.23,
            embedding_time=0.45,
            ann_time=0.23,
            similarity_time=0.34,
            ranking_time=0.21,
            candidates_considered=150,
            results_returned=25
        )
        
        assert stats.total_time == 1.23
        assert stats.embedding_time == 0.45
        assert stats.ann_time == 0.23
        assert stats.similarity_time == 0.34
        assert stats.ranking_time == 0.21
        assert stats.candidates_considered == 150
        assert stats.results_returned == 25


class TestEmbeddingManager:
    """Test EmbeddingManager class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_model = MagicMock()
        self.mock_command_dal = MagicMock()
        self.mock_dequantize = MagicMock()
        self.mock_quantize = MagicMock()
        
        self.embedding_manager = EmbeddingManager(
            model_handler=self.mock_model,
            command_dal=self.mock_command_dal,
            dequantize_func=self.mock_dequantize,
            quantize_func=self.mock_quantize
        )
    
    def test_init(self):
        """Test EmbeddingManager initialization."""
        assert self.embedding_manager.model == self.mock_model
        assert self.embedding_manager.command_dal == self.mock_command_dal
        assert self.embedding_manager.dequantize_embedding == self.mock_dequantize
        assert self.embedding_manager.quantize_embedding == self.mock_quantize
    
    def test_get_fresh_embedding_with_quantization(self):
        """Test getting fresh embedding with quantization cycle."""
        # Setup mocks
        raw_embedding = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        quantized_embedding = np.array([10, 20, 30], dtype=np.int8)
        dequantized_embedding = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        
        self.mock_model.encode.return_value = [raw_embedding]
        self.mock_quantize.return_value = quantized_embedding
        self.mock_dequantize.return_value = dequantized_embedding
        
        # Test
        result = self.embedding_manager.get_fresh_embedding("test command")
        
        # Verify
        self.mock_model.encode.assert_called_once_with(["test command"])
        self.mock_quantize.assert_called_once_with(raw_embedding)
        self.mock_dequantize.assert_called_once_with(quantized_embedding)
        np.testing.assert_array_equal(result, dequantized_embedding)
    
    def test_get_fresh_embedding_without_quantization(self):
        """Test getting fresh embedding without quantization."""
        embedding_manager = EmbeddingManager(
            model_handler=self.mock_model,
            command_dal=self.mock_command_dal,
            dequantize_func=self.mock_dequantize,
            quantize_func=None
        )
        
        raw_embedding = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        self.mock_model.encode.return_value = [raw_embedding]
        
        result = embedding_manager.get_fresh_embedding("test command")
        
        # Should return raw embedding as float32
        np.testing.assert_array_equal(result, raw_embedding.astype(np.float32))
    
    def test_get_stored_embedding(self):
        """Test getting stored embedding from blob."""
        stored_blob = np.array([10, 20, 30], dtype=np.int8).tobytes()
        dequantized_embedding = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        
        self.mock_dequantize.return_value = dequantized_embedding
        
        result = self.embedding_manager.get_stored_embedding(123, stored_blob)
        
        # Should dequantize the blob
        self.mock_dequantize.assert_called_once()
        np.testing.assert_array_equal(result, dequantized_embedding)
    
    def test_verify_embedding_consistency_high(self):
        """Test embedding consistency verification with high consistency."""
        fresh_emb = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        stored_emb = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        
        self.mock_model.encode.return_value = [fresh_emb]
        self.mock_quantize.return_value = np.array([10, 20, 30], dtype=np.int8)
        self.mock_dequantize.return_value = fresh_emb
        
        is_consistent, consistency = self.embedding_manager.verify_embedding_consistency(
            "test command", stored_emb
        )
        
        assert is_consistent == True
        assert consistency > 0.99  # Should be very high for identical embeddings
    
    def test_verify_embedding_consistency_low(self):
        """Test embedding consistency verification with low consistency."""
        fresh_emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        stored_emb = np.array([0.0, 1.0, 0.0], dtype=np.float32)  # Orthogonal
        
        self.mock_model.encode.return_value = [fresh_emb]
        self.mock_quantize.return_value = np.array([100, 0, 0], dtype=np.int8)
        self.mock_dequantize.return_value = fresh_emb
        
        is_consistent, consistency = self.embedding_manager.verify_embedding_consistency(
            "test command", stored_emb
        )
        
        assert is_consistent == False
        assert consistency < 0.95  # Should be low for orthogonal embeddings
    
    def test_diagnose_corruption_all_consistent(self):
        """Test diagnosing embeddings when all are consistent."""
        # Setup mock data
        commands_data = [
            (1, "ls -la", b"embedding_blob_1"),
            (2, "git status", b"embedding_blob_2"),
            (3, "cd /home", b"embedding_blob_3")
        ]
        
        # Mock high consistency for all
        consistent_emb = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        self.mock_model.encode.return_value = [consistent_emb]
        self.mock_quantize.return_value = np.array([10, 20, 30], dtype=np.int8)
        self.mock_dequantize.return_value = consistent_emb
        
        # Mock get_stored_embedding to return consistent embedding
        self.embedding_manager.get_stored_embedding = MagicMock(return_value=consistent_emb)
        
        report = self.embedding_manager.diagnose_corruption(commands_data)
        
        assert report["total_tested"] == 3
        assert report["corrupted_count"] == 0
        assert report["avg_consistency"] > 0.99
        assert len(report["corrupted_commands"]) == 0
    
    def test_diagnose_corruption_some_corrupted(self):
        """Test diagnosing embeddings with some corruption."""
        commands_data = [
            (1, "ls -la", b"embedding_blob_1"),
            (2, "git status", b"embedding_blob_2")
        ]
        
        # Mock different consistency levels
        def mock_verify_consistency(command, stored_emb):
            if "ls" in command:
                return True, 0.98  # Consistent
            else:
                return False, 0.50  # Corrupted
        
        self.embedding_manager.verify_embedding_consistency = MagicMock(
            side_effect=mock_verify_consistency
        )
        self.embedding_manager.get_stored_embedding = MagicMock(
            return_value=np.array([0.1, 0.2, 0.3])
        )
        
        report = self.embedding_manager.diagnose_corruption(commands_data)
        
        assert report["total_tested"] == 2
        assert report["corrupted_count"] == 1
        assert 0.5 < report["avg_consistency"] < 1.0
        assert len(report["corrupted_commands"]) == 1
        assert report["corrupted_commands"][0]["command"] == "git status"


class TestSearchCoordinator:
    """Test SearchCoordinator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_model = MagicMock()
        self.mock_command_dal = MagicMock()
        self.mock_corpus_stats_dal = MagicMock()  
        self.mock_cache_dal = MagicMock()
        self.mock_ann_manager = MagicMock()
        self.mock_quantize = MagicMock()
        self.mock_dequantize = MagicMock()
        
        self.coordinator = SearchCoordinator(
            model_handler=self.mock_model,
            command_dal=self.mock_command_dal,
            ann_manager=self.mock_ann_manager,
            dequantize_func=self.mock_dequantize,
            corpus_stats_dal=self.mock_corpus_stats_dal,
            cache_dal=self.mock_cache_dal,
            quantize_func=self.mock_quantize
        )
    
    def test_init(self):
        """Test SearchCoordinator initialization."""
        assert self.coordinator.command_dal == self.mock_command_dal
        assert self.coordinator.ann_manager == self.mock_ann_manager
        assert self.coordinator.corpus_stats_dal == self.mock_corpus_stats_dal
        assert self.coordinator.cache_dal == self.mock_cache_dal
        assert isinstance(self.coordinator.embedding_manager, EmbeddingManager)
        assert isinstance(self.coordinator.similarity_calculator, SimilarityCalculator)
        assert self.coordinator.use_ann_search is True  # Should be True since ann_manager is provided
    
    def test_search_with_ann_enabled(self):
        """Test search method with ANN search enabled."""
        # Setup mocks
        query_embedding = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        self.mock_ann_manager.is_trained.return_value = True
        self.mock_ann_manager.search.return_value = [0, 1, 2]  # ANN candidate indices
        
        # Mock embedding manager
        self.coordinator.embedding_manager.get_fresh_embedding = MagicMock(return_value=query_embedding)
        
        # Mock command data
        all_commands_data = [
            (1, "ls -la", b"embedding_blob_1"),
            (2, "git status", b"embedding_blob_2"),
            (3, "cd /home", b"embedding_blob_3")
        ]
        self.mock_command_dal.get_all_commands_with_embeddings_for_clustering.return_value = all_commands_data
        
        # Mock stored embeddings
        stored_emb1 = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        stored_emb2 = np.array([0.4, 0.5, 0.6], dtype=np.float32)
        stored_emb3 = np.array([0.7, 0.8, 0.9], dtype=np.float32)
        self.coordinator.embedding_manager.get_stored_embedding = MagicMock(
            side_effect=[stored_emb1, stored_emb2, stored_emb3]
        )
        
        # Execute search
        results, stats = self.coordinator.search("test query", top_k=5)
        
        # Verify results
        assert isinstance(results, list)
        assert isinstance(stats, SearchStats)
        assert len(results) <= 5
        assert stats.total_time > 0
        assert stats.embedding_time >= 0
        assert stats.ann_time >= 0
        assert stats.similarity_time >= 0
        assert stats.ranking_time >= 0
        
        # Verify SearchResult objects
        for result in results:
            assert isinstance(result, SearchResult)
            assert isinstance(result.command, str)
            assert isinstance(result.semantic_score, float)
            assert isinstance(result.combined_score, float)
            assert isinstance(result.rank, int)
    
    def test_search_with_ann_disabled(self):
        """Test search method with ANN search disabled."""
        # Create coordinator without ANN manager
        coordinator = SearchCoordinator(
            model_handler=self.mock_model,
            command_dal=self.mock_command_dal,
            ann_manager=None,  # No ANN manager
            dequantize_func=self.mock_dequantize,
            quantize_func=self.mock_quantize
        )
        
        # Setup mocks
        query_embedding = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        coordinator.embedding_manager.get_fresh_embedding = MagicMock(return_value=query_embedding)
        
        # Mock command data for full scan
        all_commands_data = [
            (1, "ls -la", b"embedding_blob_1"),
            (2, "git status", b"embedding_blob_2")
        ]
        self.mock_command_dal.get_all_commands_with_embeddings_for_clustering.return_value = all_commands_data
        
        # Mock stored embeddings
        stored_emb1 = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        stored_emb2 = np.array([0.4, 0.5, 0.6], dtype=np.float32)
        coordinator.embedding_manager.get_stored_embedding = MagicMock(
            side_effect=[stored_emb1, stored_emb2]
        )
        
        # Execute search
        results, stats = coordinator.search("test query", top_k=5)
        
        # Should use full scan instead of ANN
        assert coordinator.use_ann_search is False
        assert isinstance(results, list)
        assert isinstance(stats, SearchStats)
    
    def test_load_candidate_embeddings(self):
        """Test loading embeddings for specific candidate indices."""
        # Mock command data
        all_commands_data = [
            (1, "ls -la", b"embedding_blob_1"),
            (2, "git status", b"embedding_blob_2"),
            (3, "cd /home", b"embedding_blob_3")
        ]
        self.mock_command_dal.get_all_commands_with_embeddings_for_clustering.return_value = all_commands_data
        
        # Mock embedding retrieval
        stored_emb1 = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        stored_emb3 = np.array([0.7, 0.8, 0.9], dtype=np.float32)
        self.coordinator.embedding_manager.get_stored_embedding = MagicMock(
            side_effect=[stored_emb1, stored_emb3]
        )
        
        # Load specific candidates (indices 0 and 2)
        candidates = self.coordinator._load_candidate_embeddings([0, 2])
        
        # Should return only the requested candidates
        assert len(candidates) == 2
        assert candidates[0] == (1, "ls -la", stored_emb1)
        assert candidates[1] == (3, "cd /home", stored_emb3)
    
    def test_load_all_embeddings(self):
        """Test loading all embeddings for full scan."""
        # Mock command data
        all_commands_data = [
            (1, "ls -la", b"embedding_blob_1"),
            (2, "git status", b"embedding_blob_2")
        ]
        self.mock_command_dal.get_all_commands_with_embeddings_for_clustering.return_value = all_commands_data
        
        # Mock embedding retrieval
        stored_emb1 = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        stored_emb2 = np.array([0.4, 0.5, 0.6], dtype=np.float32)
        self.coordinator.embedding_manager.get_stored_embedding = MagicMock(
            side_effect=[stored_emb1, stored_emb2]
        )
        
        # Load all candidates
        candidates = self.coordinator._load_all_embeddings()
        
        # Should return all candidates
        assert len(candidates) == 2
        assert candidates[0] == (1, "ls -la", stored_emb1)
        assert candidates[1] == (2, "git status", stored_emb2)
    
    def test_create_search_results(self):
        """Test creating SearchResult objects from ranked candidates."""
        ranked_candidates = [
            (1, "ls -la", 0.95),
            (2, "git status", 0.85),
            (3, "cd /home", 0.75)
        ]
        
        results = self.coordinator._create_search_results(ranked_candidates, "test query")
        
        assert len(results) == 3
        
        # Check first result
        assert results[0].command == "ls -la"
        assert results[0].semantic_score == 0.95
        assert results[0].bm25_score == 0.0  # TODO: Currently not implemented
        assert results[0].combined_score == 0.95  # Currently same as semantic
        assert results[0].rank == 1
        
        # Check second result
        assert results[1].command == "git status"
        assert results[1].semantic_score == 0.85
        assert results[1].rank == 2
        
        # Check third result
        assert results[2].command == "cd /home"
        assert results[2].semantic_score == 0.75
        assert results[2].rank == 3
    
    def test_diagnose_embeddings(self):
        """Test embedding corruption diagnosis."""
        # Mock command data
        sample_data = [
            (1, "ls -la", b"embedding_blob_1"),
            (2, "git status", b"embedding_blob_2")
        ]
        self.mock_command_dal.get_all_commands_with_embeddings_for_clustering.return_value = sample_data
        
        # Mock embedding manager diagnosis
        mock_report = {
            'total_tested': 2,
            'corrupted_count': 1,
            'corrupted_commands': [{'command': 'git status', 'consistency': 0.50}],
            'avg_consistency': 0.75,
            'min_consistency': 0.50,
            'max_consistency': 1.0
        }
        self.coordinator.embedding_manager.diagnose_corruption = MagicMock(return_value=mock_report)
        
        # Run diagnosis
        report = self.coordinator.diagnose_embeddings(sample_size=10)
        
        # Verify report structure
        assert report['total_tested'] == 2
        assert report['corrupted_count'] == 1
        assert report['avg_consistency'] == 0.75
        assert len(report['corrupted_commands']) == 1
        assert report['corrupted_commands'][0]['command'] == 'git status'


class TestSimilarityCalculator:
    """Test SimilarityCalculator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.calculator = SimilarityCalculator()
    
    def test_cosine_similarity(self):
        """Test cosine similarity calculation."""
        emb1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        emb2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        emb3 = np.array([1.0, 0.0, 0.0], dtype=np.float32)  # Same as emb1
        
        # Orthogonal vectors should have similarity 0
        similarity_orthogonal = SimilarityCalculator.cosine_similarity(emb1, emb2)
        assert abs(similarity_orthogonal - 0.0) < 0.001
        
        # Identical vectors should have similarity 1
        similarity_identical = SimilarityCalculator.cosine_similarity(emb1, emb3)
        assert abs(similarity_identical - 1.0) < 0.001
    
    def test_batch_cosine_similarity(self):
        """Test batch cosine similarity calculation."""
        query_emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        candidate_embs = np.array([
            [1.0, 0.0, 0.0],  # Same as query
            [0.0, 1.0, 0.0],  # Orthogonal to query
            [0.5, 0.5, 0.0],  # 45 degrees from query
        ], dtype=np.float32)
        
        similarities = SimilarityCalculator.batch_cosine_similarity(query_emb, candidate_embs)
        
        assert len(similarities) == 3
        assert abs(similarities[0] - 1.0) < 0.001  # Should be 1.0 (identical)
        assert abs(similarities[1] - 0.0) < 0.001  # Should be 0.0 (orthogonal)
        assert similarities[2] > 0.5  # Should be > 0.5 (positive correlation)
    
    def test_rank_candidates_empty(self):
        """Test ranking empty candidate list."""
        query_emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        candidates = []
        
        ranked = self.calculator.rank_candidates(query_emb, candidates)
        assert ranked == []
    
    def test_rank_candidates_single(self):
        """Test ranking single candidate."""
        query_emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        candidate_emb = np.array([0.5, 0.5, 0.0], dtype=np.float32)
        candidates = [(1, "test command", candidate_emb)]
        
        ranked = self.calculator.rank_candidates(query_emb, candidates)
        
        assert len(ranked) == 1
        assert ranked[0][0] == 1  # command_id
        assert ranked[0][1] == "test command"  # command text
        assert isinstance(ranked[0][2], float)  # similarity score
        assert ranked[0][2] > 0.5  # Should have positive similarity
    
    def test_rank_candidates_multiple(self):
        """Test ranking multiple candidates."""
        query_emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        
        # Create candidates with different similarities
        candidates = [
            (1, "orthogonal cmd", np.array([0.0, 1.0, 0.0], dtype=np.float32)),  # Low similarity
            (2, "identical cmd", np.array([1.0, 0.0, 0.0], dtype=np.float32)),   # High similarity
            (3, "partial cmd", np.array([0.7, 0.7, 0.0], dtype=np.float32)),     # Medium similarity
        ]
        
        ranked = self.calculator.rank_candidates(query_emb, candidates)
        
        assert len(ranked) == 3
        
        # Should be sorted by similarity descending
        assert ranked[0][1] == "identical cmd"  # Highest similarity first
        assert ranked[1][1] == "partial cmd"    # Medium similarity second
        assert ranked[2][1] == "orthogonal cmd" # Lowest similarity last
        
        # Verify scores are in descending order
        assert ranked[0][2] >= ranked[1][2] >= ranked[2][2]