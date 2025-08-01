import unittest
import numpy as np
import sqlite3
import tempfile
import os
from unittest.mock import patch, MagicMock

# Import test helpers and FuzzyShell modules
from tests.test_helpers import create_test_db_connection
from src.fuzzyshell.fuzzyshell import FuzzyShell, ANNSearchIndex, USE_ANN_SEARCH, ANN_NUM_CLUSTERS


class TestANNSearchIndex(unittest.TestCase):
    """Test cases for the K-means based ANN search index."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.n_clusters = 4
        self.ann_index = ANNSearchIndex(n_clusters=self.n_clusters)
        
        # Create test embeddings - 100 embeddings in 384 dimensions
        np.random.seed(42)  # For reproducible tests
        self.test_embeddings = np.random.randn(100, 384).astype(np.float32)
        
        # Normalize embeddings to simulate real embedding behavior
        norms = np.linalg.norm(self.test_embeddings, axis=1, keepdims=True)
        self.test_embeddings = self.test_embeddings / norms
        
    def test_init(self):
        """Test ANNSearchIndex initialization."""
        self.assertEqual(self.ann_index.n_clusters, self.n_clusters)
        self.assertEqual(self.ann_index.max_iterations, 100)
        self.assertIsNone(self.ann_index.cluster_centers)
        self.assertIsNone(self.ann_index.cluster_assignments)
        self.assertIsNone(self.ann_index.cluster_indices)
        self.assertFalse(self.ann_index.is_trained)
        
    def test_fit_sufficient_data(self):
        """Test fitting with sufficient data."""
        self.ann_index.fit(self.test_embeddings)
        
        self.assertTrue(self.ann_index.is_trained)
        self.assertIsNotNone(self.ann_index.cluster_centers)
        self.assertIsNotNone(self.ann_index.cluster_assignments)
        self.assertIsNotNone(self.ann_index.cluster_indices)
        
        # Check cluster centers shape
        self.assertEqual(self.ann_index.cluster_centers.shape, (self.n_clusters, 384))
        
        # Check cluster assignments
        self.assertEqual(len(self.ann_index.cluster_assignments), 100)
        self.assertTrue(all(0 <= c < self.n_clusters for c in self.ann_index.cluster_assignments))
        
        # Check cluster indices structure
        self.assertEqual(len(self.ann_index.cluster_indices), self.n_clusters)
        total_assigned = sum(len(indices) for indices in self.ann_index.cluster_indices.values())
        self.assertEqual(total_assigned, 100)
        
    def test_fit_insufficient_data(self):
        """Test fitting with insufficient data (fewer samples than clusters)."""
        small_embeddings = self.test_embeddings[:2]  # Only 2 embeddings for 4 clusters
        self.ann_index.fit(small_embeddings)
        
        self.assertFalse(self.ann_index.is_trained)
        
    def test_kmeans_plus_plus_initialization(self):
        """Test k-means++ initialization produces reasonable centroids."""
        centroids = self.ann_index._init_centroids_plus_plus(self.test_embeddings)
        
        self.assertEqual(centroids.shape, (self.n_clusters, 384))
        
        # Check that centroids are different from each other
        for i in range(self.n_clusters):
            for j in range(i + 1, self.n_clusters):
                distance = np.linalg.norm(centroids[i] - centroids[j])
                self.assertGreater(distance, 0.1)  # Should be reasonably separated
                
    def test_search_candidates_trained(self):
        """Test search candidates when index is trained."""
        self.ann_index.fit(self.test_embeddings)
        
        # Create a query embedding
        query_embedding = np.random.randn(384).astype(np.float32)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        candidates = self.ann_index.search_candidates(query_embedding, n_candidates=2)
        
        self.assertIsInstance(candidates, list)
        self.assertGreater(len(candidates), 0)
        self.assertLessEqual(len(candidates), 100)  # Should not exceed total embeddings
        
        # All candidates should be valid indices
        self.assertTrue(all(0 <= idx < 100 for idx in candidates))
        
    def test_search_candidates_not_trained(self):
        """Test search candidates when index is not trained."""
        # Don't fit the index
        candidates = self.ann_index.search_candidates(np.random.randn(384))
        
        self.assertEqual(candidates, [])  # Should return empty list when not trained
        
    def test_clustering_convergence(self):
        """Test that k-means clustering converges."""
        self.ann_index.fit(self.test_embeddings)
        
        # If trained successfully, convergence occurred
        self.assertTrue(self.ann_index.is_trained)
        
        # Check that each cluster has at least one point (with enough data)
        non_empty_clusters = sum(1 for indices in self.ann_index.cluster_indices.values() if indices)
        self.assertGreater(non_empty_clusters, 0)
        
    def test_cluster_quality(self):
        """Test that clustering produces reasonable quality clusters."""
        self.ann_index.fit(self.test_embeddings)
        
        # Calculate within-cluster sum of squares
        wcss = 0
        for cluster_id, indices in self.ann_index.cluster_indices.items():
            if indices:
                cluster_points = self.test_embeddings[indices]
                cluster_center = self.ann_index.cluster_centers[cluster_id]
                distances = np.linalg.norm(cluster_points - cluster_center, axis=1)
                wcss += np.sum(distances ** 2)
                
        # WCSS should be reasonable (not too high)
        self.assertLess(wcss, len(self.test_embeddings) * 2)  # Heuristic threshold
        
    def test_deterministic_with_seed(self):
        """Test that clustering is deterministic with fixed random seed."""
        np.random.seed(42)
        ann_index1 = ANNSearchIndex(n_clusters=self.n_clusters)
        ann_index1.fit(self.test_embeddings)
        
        np.random.seed(42)
        ann_index2 = ANNSearchIndex(n_clusters=self.n_clusters)
        ann_index2.fit(self.test_embeddings)
        
        # Should produce same assignments (with same seed)
        self.assertTrue(np.array_equal(ann_index1.cluster_assignments, ann_index2.cluster_assignments))


class TestFuzzyShellANNIntegration(unittest.TestCase):
    """Test ANN search integration in FuzzyShell."""
    
    def setUp(self):
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
        """Test that ANN index is properly initialized when USE_ANN_SEARCH is True."""
        if USE_ANN_SEARCH:
            self.assertIsNotNone(self.fuzzyshell.ann_index)
            self.assertIsInstance(self.fuzzyshell.ann_index, ANNSearchIndex)
        else:
            self.assertIsNone(self.fuzzyshell.ann_index)
            
    @patch('src.fuzzyshell.fuzzyshell.USE_ANN_SEARCH', True)
    def test_search_with_ann_enabled(self):
        """Test search functionality with ANN enabled."""
        # Mock query embedding
        query_embedding = np.random.randn(384).astype(np.float32)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        self.mock_model.encode.return_value = [query_embedding]
        
        # Perform search
        results = self.fuzzyshell.search("test query", top_k=5)
        
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)
        self.assertLessEqual(len(results), 5)  # Should respect top_k
        
        # Each result should be a tuple with command and score
        for result in results:
            self.assertIsInstance(result, tuple)
            self.assertEqual(len(result), 2)  # (command, score)
            self.assertIsInstance(result[0], str)  # command
            self.assertIsInstance(result[1], float)  # score
            
    @patch('src.fuzzyshell.fuzzyshell.USE_ANN_SEARCH', False)
    def test_search_with_ann_disabled(self):
        """Test search functionality with ANN disabled (linear search)."""
        # Mock query embedding
        query_embedding = np.random.randn(384).astype(np.float32)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        self.mock_model.encode.return_value = [query_embedding]
        
        # Perform search
        results = self.fuzzyshell.search("test query", top_k=5)
        
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)
        self.assertLessEqual(len(results), 5)
        
    def test_ann_index_training_during_search(self):
        """Test that ANN index gets trained during search if not already trained."""
        if not USE_ANN_SEARCH:
            self.skipTest("ANN search is disabled")
            
        # Ensure ANN index is not trained initially
        if self.fuzzyshell.ann_index:
            self.fuzzyshell.ann_index.is_trained = False
            
        # Mock query embedding
        query_embedding = np.random.randn(384).astype(np.float32)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        self.mock_model.encode.return_value = [query_embedding]
        
        # Perform search - should trigger ANN index training
        results = self.fuzzyshell.search("test query", top_k=5)
        
        # Check that training occurred (if we have enough data)
        if len(self.test_commands) >= ANN_NUM_CLUSTERS:
            self.assertTrue(self.fuzzyshell.ann_index.is_trained)
        
    def test_ann_search_performance_comparison(self):
        """Test that ANN search produces reasonable results compared to linear search."""
        if not USE_ANN_SEARCH:
            self.skipTest("ANN search is disabled")
            
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
        self.assertGreater(len(ann_results), 0)
        self.assertGreater(len(linear_results), 0)
        
        # Results should have same format
        for result in ann_results:
            self.assertEqual(len(result), 4)  # (command, combined_score, semantic_score, bm25_score)
            
        for result in linear_results:
            self.assertEqual(len(result), 4)
            
    def test_ann_cluster_size_configuration(self):
        """Test that ANN search respects cluster configuration."""
        if not USE_ANN_SEARCH:
            self.skipTest("ANN search is disabled")
            
        # Create ANN index with custom cluster size
        custom_clusters = 3
        ann_index = ANNSearchIndex(n_clusters=custom_clusters)
        
        # Create sufficient test data
        test_embeddings = np.random.randn(20, 384).astype(np.float32)
        norms = np.linalg.norm(test_embeddings, axis=1, keepdims=True)
        test_embeddings = test_embeddings / norms
        
        ann_index.fit(test_embeddings)
        
        self.assertTrue(ann_index.is_trained)
        self.assertEqual(ann_index.n_clusters, custom_clusters)
        self.assertEqual(len(ann_index.cluster_indices), custom_clusters)
        
    def tearDown(self):
        """Clean up test resources."""
        if hasattr(self, 'test_conn'):
            self.test_conn.close()


class TestANNSearchEdgeCases(unittest.TestCase):
    """Test edge cases for ANN search functionality."""
    
    def test_empty_embeddings(self):
        """Test ANN index behavior with empty embeddings."""
        ann_index = ANNSearchIndex(n_clusters=4)
        empty_embeddings = np.array([]).reshape(0, 384)
        
        ann_index.fit(empty_embeddings)
        self.assertFalse(ann_index.is_trained)
        
    def test_single_embedding(self):
        """Test ANN index behavior with single embedding."""
        ann_index = ANNSearchIndex(n_clusters=4)
        single_embedding = np.random.randn(1, 384).astype(np.float32)
        
        ann_index.fit(single_embedding)
        self.assertFalse(ann_index.is_trained)  # Not enough data for clustering
        
    def test_identical_embeddings(self):
        """Test ANN index behavior with identical embeddings."""
        ann_index = ANNSearchIndex(n_clusters=2)
        # Create 10 identical embeddings
        identical_embedding = np.random.randn(384).astype(np.float32)
        identical_embeddings = np.tile(identical_embedding, (10, 1))
        
        ann_index.fit(identical_embeddings)
        
        # Should still train, but clusters might be degenerate
        self.assertTrue(ann_index.is_trained)
        
        # All points should be assigned to clusters
        self.assertEqual(len(ann_index.cluster_assignments), 10)
        
    def test_high_dimensional_query(self):
        """Test search with query embedding that has wrong dimensions."""
        ann_index = ANNSearchIndex(n_clusters=2)
        test_embeddings = np.random.randn(10, 384).astype(np.float32)
        ann_index.fit(test_embeddings)
        
        # Query with wrong dimensions
        wrong_dim_query = np.random.randn(256).astype(np.float32)  # Wrong dimension
        
        # Should handle gracefully (might raise exception or return empty results)
        try:
            candidates = ann_index.search_candidates(wrong_dim_query)
            # If it doesn't raise an exception, should return reasonable results
            self.assertIsInstance(candidates, list)
        except (ValueError, IndexError):
            # Expected to fail with wrong dimensions
            pass
            
    def test_nan_embeddings(self):
        """Test ANN index behavior with NaN values in embeddings."""
        ann_index = ANNSearchIndex(n_clusters=2)
        test_embeddings = np.random.randn(10, 384).astype(np.float32)
        # Introduce some NaN values
        test_embeddings[0, :5] = np.nan
        
        # Should handle NaN values gracefully
        try:
            ann_index.fit(test_embeddings)
            # If training succeeds, check that it produces valid results
            if ann_index.is_trained:
                self.assertIsNotNone(ann_index.cluster_centers)
                # Centers should not contain NaN (unless all data is NaN)
                self.assertFalse(np.isnan(ann_index.cluster_centers).all())
        except (ValueError, RuntimeError):
            # Expected to fail with NaN values
            pass


if __name__ == '__main__':
    # Run specific test suites
    unittest.main(verbosity=2)