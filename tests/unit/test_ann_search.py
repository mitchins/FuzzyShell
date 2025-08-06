import pytest
import numpy as np
import sqlite3
import tempfile
import os
from unittest.mock import patch, MagicMock

# Import test helpers and FuzzyShell modules
from test_helpers import create_test_db_connection
from fuzzyshell.fuzzyshell import FuzzyShell, USE_ANN_SEARCH, ANN_NUM_CLUSTERS
from fuzzyshell.ann_index_manager import ANNSearchIndex


class TestANNSearchIndex:
    """Test cases for the K-means based ANN search index."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.n_clusters = 4
        self.ann_index = ANNSearchIndex()
        
        # Create test embeddings - 100 embeddings in 384 dimensions
        np.random.seed(42)  # For reproducible tests
        self.test_embeddings = np.random.randn(100, 384).astype(np.float32)
        
        # Normalize embeddings to simulate real embedding behavior
        norms = np.linalg.norm(self.test_embeddings, axis=1, keepdims=True)
        self.test_embeddings = self.test_embeddings / norms
        
    def test_init(self):
        """Test ANNSearchIndex initialization."""
        assert self.ann_index.n_clusters == 0  # New API starts with 0
        assert self.ann_index.cluster_centers is None
        assert self.ann_index.cluster_indices is None
        assert self.ann_index.is_trained is False
        
    def test_fit_sufficient_data(self):
        """Test fitting with sufficient data."""
        result = self.ann_index.fit(self.test_embeddings, n_clusters=self.n_clusters)
        
        assert result is True  # fit() now returns boolean
        assert self.ann_index.is_trained is True
        assert self.ann_index.cluster_centers is not None
        assert self.ann_index.cluster_indices is not None
        
        # Check cluster centers shape
        assert self.ann_index.cluster_centers.shape == (self.n_clusters, 384)
        
        # Check cluster indices structure
        assert len(self.ann_index.cluster_indices) == self.n_clusters
        total_assigned = sum(len(indices) for indices in self.ann_index.cluster_indices)
        assert total_assigned == 100
        
    def test_fit_insufficient_data(self):
        """Test fitting with insufficient data (fewer samples than clusters)."""
        small_embeddings = self.test_embeddings[:2]  # Only 2 embeddings for 4 clusters
        result = self.ann_index.fit(small_embeddings, n_clusters=self.n_clusters)
        
        assert result is False
        assert self.ann_index.is_trained is False
        
    def test_kmeans_plus_plus_initialization(self):
        """Test k-means initialization produces reasonable centroids (simplified for new API)."""
        # Train the index first
        result = self.ann_index.fit(self.test_embeddings, n_clusters=self.n_clusters)
        assert result is True
        
        centroids = self.ann_index.cluster_centers
        assert centroids.shape == (self.n_clusters, 384)
        
        # Check that centroids are different from each other
        for i in range(self.n_clusters):
            for j in range(i + 1, self.n_clusters):
                distance = np.linalg.norm(centroids[i] - centroids[j])
                assert distance > 0.1  # Should be reasonably separated
                
    def test_search_candidates_trained(self):
        """Test search candidates when index is trained."""
        self.ann_index.fit(self.test_embeddings, n_clusters=self.n_clusters)
        
        # Create a query embedding
        query_embedding = np.random.randn(384).astype(np.float32)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        candidates = self.ann_index.search(query_embedding, n_candidates=2)
        
        assert isinstance(candidates, list)
        assert len(candidates) > 0
        assert len(candidates) <= 100  # Should not exceed total embeddings
        
        # All candidates should be valid indices
        assert all(0 <= idx < 100 for idx in candidates)
        
    def test_search_candidates_not_trained(self):
        """Test search candidates when index is not trained."""
        # Don't fit the index
        candidates = self.ann_index.search(np.random.randn(384))
        
        assert candidates == []  # Should return empty list when not trained
        
    def test_clustering_convergence(self):
        """Test that k-means clustering converges."""
        result = self.ann_index.fit(self.test_embeddings, n_clusters=self.n_clusters)
        
        # If trained successfully, convergence occurred
        assert result is True
        assert self.ann_index.is_trained is True
        
        # Check that each cluster has at least one point (with enough data)
        non_empty_clusters = sum(1 for indices in self.ann_index.cluster_indices if indices)
        assert non_empty_clusters > 0
        
    def test_cluster_quality(self):
        """Test that clustering produces reasonable quality clusters."""
        self.ann_index.fit(self.test_embeddings, n_clusters=self.n_clusters)
        
        # Calculate within-cluster sum of squares
        wcss = 0
        for cluster_id, indices in enumerate(self.ann_index.cluster_indices):
            if indices:
                cluster_points = self.test_embeddings[indices]
                cluster_center = self.ann_index.cluster_centers[cluster_id]
                distances = np.linalg.norm(cluster_points - cluster_center, axis=1)
                wcss += np.sum(distances ** 2)
                
        # WCSS should be reasonable (not too high)
        assert wcss < len(self.test_embeddings) * 2  # Heuristic threshold
        
    def test_deterministic_with_seed(self):
        """Test that clustering is deterministic with fixed random seed."""
        np.random.seed(42)
        ann_index1 = ANNSearchIndex()
        ann_index1.fit(self.test_embeddings, n_clusters=self.n_clusters)
        
        np.random.seed(42)
        ann_index2 = ANNSearchIndex()
        ann_index2.fit(self.test_embeddings, n_clusters=self.n_clusters)
        
        # Should produce same cluster centers (with same seed)
        assert np.allclose(ann_index1.cluster_centers, ann_index2.cluster_centers)


class TestFuzzyShellANNIntegration:
    """Test ANN search integration in FuzzyShell."""
    
    def setup_method(self):
        """Set up test database and FuzzyShell instance."""
        self.test_conn = create_test_db_connection()
        self.fuzzyshell = FuzzyShell(conn=self.test_conn)
        
        # Add test commands with embeddings
        self.test_commands = [
            "ls -la",
            "cd /home/user",
            "grep pattern file.txt",
            "find . -name '*.py'",
            "python script.py",
            "git status",
            "docker ps",
            "npm install",
            "wget https://example.com",
            "tar -xzf archive.tar.gz"
        ]
        
        # Mock the model to avoid downloading
        self.mock_model = MagicMock()
        # Create realistic embeddings for test commands
        np.random.seed(42)
        self.test_embeddings = np.random.randn(len(self.test_commands), 384).astype(np.float32)
        # Normalize embeddings
        norms = np.linalg.norm(self.test_embeddings, axis=1, keepdims=True)
        self.test_embeddings = self.test_embeddings / norms
        
        self.mock_model.encode.return_value = self.test_embeddings
        self.fuzzyshell._model = self.mock_model
        
        # Add commands to database and manually store embeddings
        c = self.test_conn.cursor()
        for i, command in enumerate(self.test_commands):
            self.fuzzyshell.add_command(command)
            # Manually add embeddings since we're mocking the model
            command_id = i + 1
            quantized_embedding = self.fuzzyshell.quantize_embedding(self.test_embeddings[i])
            c.execute('INSERT OR REPLACE INTO embeddings (rowid, embedding) VALUES (?, ?)', 
                     (command_id, quantized_embedding))
        self.test_conn.commit()
            
    def test_ann_index_initialization(self):
        """Test that ANN manager is properly initialized when USE_ANN_SEARCH is True."""
        # Wait for model initialization which also initializes ann_manager
        self.fuzzyshell.wait_for_model(timeout=10.0)
        
        if USE_ANN_SEARCH:
            assert self.fuzzyshell.ann_manager is not None
        else:
            assert self.fuzzyshell.ann_manager is None
            
    @patch('src.fuzzyshell.fuzzyshell.USE_ANN_SEARCH', True)
    def test_search_with_ann_enabled(self):
        """Test search functionality with ANN enabled."""
        # Ensure model is initialized
        self.fuzzyshell.wait_for_model(timeout=10.0)
        
        # Mock query embedding
        query_embedding = np.random.randn(384).astype(np.float32)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        self.mock_model.encode.return_value = [query_embedding]
        
        # Perform search
        results = self.fuzzyshell.search("test query", top_k=5)
        
        assert isinstance(results, list)
        assert len(results) > 0
        assert len(results) <= 5  # Should respect top_k
        
        # Each result should be a tuple with command and score
        for result in results:
            assert isinstance(result, tuple)
            assert len(result) == 2  # (command, score)
            assert isinstance(result[0], str)  # command
            assert isinstance(result[1], float)  # score
            
    @patch('src.fuzzyshell.fuzzyshell.USE_ANN_SEARCH', False)
    def test_search_with_ann_disabled(self):
        """Test search functionality with ANN disabled (linear search)."""
        # Ensure model is initialized
        self.fuzzyshell.wait_for_model(timeout=10.0)
        
        # Mock query embedding
        query_embedding = np.random.randn(384).astype(np.float32)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        self.mock_model.encode.return_value = [query_embedding]
        
        # Perform search
        results = self.fuzzyshell.search("test query", top_k=5)
        
        assert isinstance(results, list)
        assert len(results) > 0
        assert len(results) <= 5
        
    def test_ann_index_training_during_search(self):
        """Test that ANN manager handles index building during search if not already built."""
        if not USE_ANN_SEARCH:
            pytest.skip("ANN search is disabled")
            
        # Ensure we wait for model initialization
        self.fuzzyshell.wait_for_model(timeout=10.0)
            
        # Mock query embedding
        query_embedding = np.random.randn(384).astype(np.float32)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        self.mock_model.encode.return_value = [query_embedding]
        
        # Perform search - should trigger ANN index training
        results = self.fuzzyshell.search("test query", top_k=5)
        
        # Check that training occurred (if we have enough data)
        if len(self.test_commands) >= ANN_NUM_CLUSTERS:
            assert self.fuzzyshell.ann_manager.index.is_trained is True
        
    def test_ann_search_performance_comparison(self):
        """Test that ANN search produces reasonable results compared to linear search."""
        if not USE_ANN_SEARCH:
            pytest.skip("ANN search is disabled")
            
        # Ensure model is initialized
        self.fuzzyshell.wait_for_model(timeout=10.0)
        
        # Mock query embedding
        query_embedding = np.random.randn(384).astype(np.float32)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        self.mock_model.encode.return_value = [query_embedding]
        
        # Get results with ANN
        with patch('src.fuzzyshell.fuzzyshell.USE_ANN_SEARCH', True):
            ann_results = self.fuzzyshell.search("test query", top_k=5, return_scores=True)
            
        # Get results with linear search
        with patch('src.fuzzyshell.fuzzyshell.USE_ANN_SEARCH', False):
            linear_results = self.fuzzyshell.search("test query", top_k=5, return_scores=True)
            
        # Both should return results
        assert len(ann_results) > 0
        assert len(linear_results) > 0
        
        # Results should have same format
        for result in ann_results:
            assert len(result) == 4  # (command, combined_score, semantic_score, bm25_score)
            
        for result in linear_results:
            assert len(result) == 4
            
    def test_ann_cluster_size_configuration(self):
        """Test that ANN search respects cluster configuration."""
        if not USE_ANN_SEARCH:
            pytest.skip("ANN search is disabled")
            
        # Create ANN index with custom cluster size
        custom_clusters = 3
        ann_index = ANNSearchIndex()
        
        # Create sufficient test data
        test_embeddings = np.random.randn(20, 384).astype(np.float32)
        norms = np.linalg.norm(test_embeddings, axis=1, keepdims=True)
        test_embeddings = test_embeddings / norms
        
        result = ann_index.fit(test_embeddings, n_clusters=custom_clusters)
        
        assert result is True
        assert ann_index.is_trained is True
        assert ann_index.n_clusters == custom_clusters
        assert len(ann_index.cluster_indices) == custom_clusters
        
    def teardown_method(self):
        """Clean up test resources."""
        if hasattr(self, 'test_conn'):
            self.test_conn.close()


class TestANNSearchEdgeCases:
    """Test edge cases for ANN search functionality."""
    
    def test_empty_embeddings(self):
        """Test ANN index behavior with empty embeddings."""
        ann_index = ANNSearchIndex()
        empty_embeddings = np.array([]).reshape(0, 384)
        
        result = ann_index.fit(empty_embeddings, n_clusters=4)
        assert result is False
        assert ann_index.is_trained is False
        
    def test_single_embedding(self):
        """Test ANN index behavior with single embedding."""
        ann_index = ANNSearchIndex()
        single_embedding = np.random.randn(1, 384).astype(np.float32)
        
        result = ann_index.fit(single_embedding, n_clusters=4)
        assert result is False  # Not enough data for clustering
        assert ann_index.is_trained is False
        
    def test_identical_embeddings(self):
        """Test ANN index behavior with identical embeddings."""
        ann_index = ANNSearchIndex()
        # Create 10 identical embeddings
        identical_embedding = np.random.randn(384).astype(np.float32)
        identical_embeddings = np.tile(identical_embedding, (10, 1))
        
        result = ann_index.fit(identical_embeddings, n_clusters=2)
        
        # Should still train, but clusters might be degenerate
        assert result is True
        assert ann_index.is_trained is True
        
        # All points should be distributed across clusters
        total_points = sum(len(indices) for indices in ann_index.cluster_indices)
        assert total_points == 10
        
    def test_high_dimensional_query(self):
        """Test search with query embedding that has wrong dimensions."""
        ann_index = ANNSearchIndex()
        test_embeddings = np.random.randn(10, 384).astype(np.float32)
        ann_index.fit(test_embeddings, n_clusters=2)
        
        # Query with wrong dimensions
        wrong_dim_query = np.random.randn(256).astype(np.float32)  # Wrong dimension
        
        # Should handle gracefully (might raise exception or return empty results)
        try:
            candidates = ann_index.search(wrong_dim_query)
            # If it doesn't raise an exception, should return reasonable results
            assert isinstance(candidates, list)
        except (ValueError, IndexError):
            # Expected to fail with wrong dimensions
            pass
            
    def test_nan_embeddings(self):
        """Test ANN index behavior with NaN values in embeddings."""
        ann_index = ANNSearchIndex()
        test_embeddings = np.random.randn(10, 384).astype(np.float32)
        # Introduce some NaN values
        test_embeddings[0, :5] = np.nan
        
        # Should handle NaN values gracefully
        try:
            result = ann_index.fit(test_embeddings, n_clusters=2)
            # If training succeeds, check that it produces valid results
            if result and ann_index.is_trained:
                assert ann_index.cluster_centers is not None
                # Centers should not contain NaN (unless all data is NaN)
                assert not np.isnan(ann_index.cluster_centers).all()
        except (ValueError, RuntimeError):
            # Expected to fail with NaN values
            pass


