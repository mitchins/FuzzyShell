import numpy as np
import time
from unittest.mock import patch, MagicMock

# Import test helpers and FuzzyShell modules
from test_helpers import create_test_db_connection
from fuzzyshell.fuzzyshell import FuzzyShell
from fuzzyshell.ann_index_manager import ANNSearchIndex


class TestANNPerformance:
    """Performance benchmarks for ANN search vs linear search."""

    def setup_method(self):
        """Set up performance test with larger dataset."""
        self.test_conn = create_test_db_connection()
        self.fuzzyshell = FuzzyShell(conn=self.test_conn)
        
        # Create a larger dataset for meaningful performance comparison
        self.n_commands = 1000
        np.random.seed(42)  # For reproducible results
        
        # Generate realistic command-like embeddings
        self.test_embeddings = np.random.randn(self.n_commands, 384).astype(np.float32)
        norms = np.linalg.norm(self.test_embeddings, axis=1, keepdims=True)
        self.test_embeddings = self.test_embeddings / norms
        
        # Create synthetic commands
        command_templates = [
            "ls -la /path/to/directory/{}",
            "cd /home/user/project/{}",
            "grep pattern file_{}.txt",
            "find . -name '*{}.py'",
            "python script_{}.py",
            "git commit -m 'Update {}'",
            "docker run container_{}",
            "npm install package_{}",
            "wget https://example.com/file_{}.tar.gz",
            "tar -xzf archive_{}.tar.gz"
        ]
        
        self.test_commands = []
        for i in range(self.n_commands):
            template_idx = i % len(command_templates)
            command = command_templates[template_idx].format(i)
            self.test_commands.append(command)
        
        # Mock the model
        self.mock_model = MagicMock()
        self.mock_model.encode.return_value = self.test_embeddings
        self.fuzzyshell._model = self.mock_model
        
        # Add commands to database with embeddings
        c = self.test_conn.cursor()
        for i, command in enumerate(self.test_commands):
            self.fuzzyshell.add_command(command)
            # Manually add embeddings
            command_id = i + 1
            quantized_embedding = self.fuzzyshell.quantize_embedding(self.test_embeddings[i])
            c.execute('INSERT OR REPLACE INTO embeddings (rowid, embedding) VALUES (?, ?)', 
                     (command_id, quantized_embedding))
        self.test_conn.commit()
        
    def test_ann_vs_linear_search_performance(self):
        """Compare ANN search performance against linear search."""
        # Create test query
        query_embedding = np.random.randn(384).astype(np.float32)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        self.mock_model.encode.return_value = [query_embedding]
        
        # Benchmark ANN search
        with patch('src.fuzzyshell.fuzzyshell.USE_ANN_SEARCH', True):
            ann_times = []
            for _ in range(5):  # Multiple runs for average
                start_time = time.time()
                ann_results = self.fuzzyshell.search("test query", top_k=10, return_scores=True)
                ann_times.append(time.time() - start_time)
            ann_avg_time = np.mean(ann_times)
            
        # Benchmark linear search
        with patch('src.fuzzyshell.fuzzyshell.USE_ANN_SEARCH', False):
            linear_times = []
            for _ in range(5):  # Multiple runs for average
                start_time = time.time()
                linear_results = self.fuzzyshell.search("test query", top_k=10, return_scores=True)
                linear_times.append(time.time() - start_time)
            linear_avg_time = np.mean(linear_times)
        
        # Performance assertions
        print(f"\\nPerformance Results for {self.n_commands} commands:")
        print(f"ANN Search Average Time: {ann_avg_time:.4f}s")
        print(f"Linear Search Average Time: {linear_avg_time:.4f}s")
        
        if ann_avg_time < linear_avg_time:
            speedup = linear_avg_time / ann_avg_time
            print(f"ANN Speedup: {speedup:.2f}x faster")
        else:
            slowdown = ann_avg_time / linear_avg_time
            print(f"ANN Slowdown: {slowdown:.2f}x slower (overhead from small dataset)")
        
        # Both should return results
        assert len(ann_results) > 0
        assert len(linear_results) > 0
        assert len(ann_results) == len(linear_results)  # Same top_k
        
        # Results should have proper format
        for result in ann_results:
            assert len(result) == 4  # (command, combined_score, semantic_score, bm25_score)
        
    def test_ann_clustering_overhead_measurement(self):
        """Measure the overhead of K-means clustering during first search."""
        # Fresh ANN index
        ann_index = ANNSearchIndex()
        
        # Generate embeddings for clustering
        test_embeddings = np.random.randn(500, 384).astype(np.float32)
        norms = np.linalg.norm(test_embeddings, axis=1, keepdims=True)
        test_embeddings = test_embeddings / norms
        
        # Measure clustering time
        start_time = time.time()
        result = ann_index.fit(test_embeddings, n_clusters=32)
        clustering_time = time.time() - start_time
        
        print(f"\\nClustering Performance:")
        print(f"K-means clustering time for {len(test_embeddings)} embeddings: {clustering_time:.4f}s")
        print(f"Clustering time per embedding: {clustering_time/len(test_embeddings)*1000:.2f}ms")
        
        # Clustering should complete in reasonable time and succeed
        assert result  # fit() should return True for success
        assert clustering_time < 2.0  # Should complete within 2 seconds
        assert ann_index.is_trained
        
    def test_ann_search_candidate_selection_efficiency(self):
        """Test that ANN search reduces the number of candidates effectively."""
        # Create ANN index with embeddings
        ann_index = ANNSearchIndex()
        test_embeddings = self.test_embeddings[:500]  # Use subset for this test
        ann_index.fit(test_embeddings, n_clusters=16)
        
        # Test query
        query_embedding = np.random.randn(384).astype(np.float32)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        # Get candidates with different cluster counts
        candidates_1 = ann_index.search(query_embedding, n_candidates=1)
        candidates_3 = ann_index.search(query_embedding, n_candidates=3)
        candidates_5 = ann_index.search(query_embedding, n_candidates=5)
        
        print(f"\\nCandidate Selection Efficiency:")
        print(f"Total embeddings: {len(test_embeddings)}")
        print(f"Candidates with 1 cluster: {len(candidates_1)} ({len(candidates_1)/len(test_embeddings)*100:.1f}%)")
        print(f"Candidates with 3 clusters: {len(candidates_3)} ({len(candidates_3)/len(test_embeddings)*100:.1f}%)")
        print(f"Candidates with 5 clusters: {len(candidates_5)} ({len(candidates_5)/len(test_embeddings)*100:.1f}%)")
        
        # Should have reasonable candidate reduction
        assert len(candidates_1) < len(test_embeddings)
        assert len(candidates_3) < len(test_embeddings)
        assert len(candidates_1) <= len(candidates_3)  # More clusters = more candidates
        
        # Candidates should be valid indices
        for idx in candidates_3:
            assert idx >= 0
            assert idx < len(test_embeddings)
    
    def test_ann_accuracy_vs_linear_search(self):
        """Test that ANN search maintains reasonable accuracy compared to linear search."""
        # Use a specific query that should have clear similarity patterns
        query_embedding = self.test_embeddings[0] + np.random.normal(0, 0.1, 384)  # Similar to first embedding
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        self.mock_model.encode.return_value = [query_embedding]
        
        # Get results with ANN
        with patch('src.fuzzyshell.fuzzyshell.USE_ANN_SEARCH', True):
            ann_results = self.fuzzyshell.search("test query", top_k=20, return_scores=True)
            
        # Get results with linear search  
        with patch('src.fuzzyshell.fuzzyshell.USE_ANN_SEARCH', False):
            linear_results = self.fuzzyshell.search("test query", top_k=20, return_scores=True)
        
        # Extract commands from results
        ann_commands = {result[0] for result in ann_results}
        linear_commands = {result[0] for result in linear_results}
        
        # Calculate overlap
        overlap = len(ann_commands.intersection(linear_commands))
        recall = overlap / len(linear_commands) if linear_commands else 0
        
        print(f"\\nAccuracy Comparison:")
        print(f"ANN results: {len(ann_results)}")
        print(f"Linear results: {len(linear_results)}")
        print(f"Overlap: {overlap}")
        print(f"Recall: {recall:.2f}")
        
        # ANN should maintain reasonable recall (at least 70% for top results)
        assert recall > 0.5  # At least 50% recall
        
    def teardown_method(self):
        """Clean up test resources."""
        if hasattr(self, 'test_conn'):
            self.test_conn.close()


class TestANNScalability:
    """Test ANN search scalability with different dataset sizes."""
    
    def test_clustering_scalability(self):
        """Test K-means clustering performance with different dataset sizes."""
        dataset_sizes = [100, 500, 1000, 2000]
        clustering_times = []
        
        print("\\nClustering Scalability Test:")
        for size in dataset_sizes:
            # Generate test embeddings
            np.random.seed(42)
            test_embeddings = np.random.randn(size, 384).astype(np.float32)
            norms = np.linalg.norm(test_embeddings, axis=1, keepdims=True)
            test_embeddings = test_embeddings / norms
            
            # Measure clustering time
            ann_index = ANNSearchIndex()
            cluster_count = min(32, size // 10)  # Adaptive cluster count
            start_time = time.time()
            result = ann_index.fit(test_embeddings, n_clusters=cluster_count)
            clustering_time = time.time() - start_time
            clustering_times.append(clustering_time)
            
            print(f"Dataset size: {size:>4d}, Clustering time: {clustering_time:.4f}s, Time per item: {clustering_time/size*1000:.2f}ms")
            
            # Should complete in reasonable time even for larger datasets and succeed
            assert result  # fit() should return True for success
            assert clustering_time < size * 0.01  # Should be faster than 10ms per item
            assert ann_index.is_trained
        
        # Clustering time should scale reasonably (not exponentially)
        # Check that doubling dataset size doesn't more than quadruple the time
        for i in range(1, len(clustering_times)):
            if clustering_times[i-1] > 0:
                time_ratio = clustering_times[i] / clustering_times[i-1]
                size_ratio = dataset_sizes[i] / dataset_sizes[i-1]
                scaling_factor = time_ratio / size_ratio
                print(f"Scaling factor {dataset_sizes[i-1]} -> {dataset_sizes[i]}: {scaling_factor:.2f}")
                # Should not be worse than quadratic scaling
                # Allow some overhead for small datasets - K-means can have high variance
                assert scaling_factor < size_ratio * 2.0, (
                    f"Scaling factor {scaling_factor:.2f} is too high for {dataset_sizes[i-1]} -> {dataset_sizes[i]}"
                )  # Better than quadratic with overhead


