"""
Unit test to prevent search scoring regression after pipeline integration.

This test ensures that semantic similarity scores are varied and not uniform,
which was a regression introduced during the data processing pipeline integration.
"""

import pytest
import tempfile
import numpy as np
from fuzzyshell import FuzzyShell
from tests.test_helpers import create_test_db_connection


class TestSearchScoringRegression:
    """Test to prevent uniform scoring regression in search results."""
    
    def setup_method(self):
        """Set up test with in-memory database."""
        self.test_conn = create_test_db_connection()
        self.fuzzyshell = FuzzyShell(conn=self.test_conn)
        
        # Wait for model to be ready
        model_ready = self.fuzzyshell.wait_for_model(timeout=30.0)
        assert model_ready, "Model should be ready for testing"
    
    def test_semantic_scores_are_varied_not_uniform(self):
        """Test that semantic similarity scores vary for different queries."""
        # Add test commands with clear semantic differences
        test_commands = [
            'docker ps',
            'docker run hello-world', 
            'ls -la',
            'grep pattern file.txt',
            'npm install package',
            'git commit -m "message"'
        ]
        
        # Ingest commands using the pipeline
        result = self.fuzzyshell.data_pipeline.process_ingestion(
            test_commands, 
            use_clustering=False
        )
        assert result.success, "Pipeline ingestion should succeed"
        assert result.commands_processed == len(test_commands)
        
        # Test search for docker-related query
        docker_results = self.fuzzyshell.search('docker', top_k=6, return_scores=True)
        assert len(docker_results) > 0, "Should return docker search results"
        
        # Extract semantic scores
        semantic_scores = []
        for result in docker_results:
            if len(result) >= 4:  # (command, combined, semantic, bm25)
                semantic_scores.append(result[2])  # semantic score
        
        assert len(semantic_scores) > 0, "Should have semantic scores"
        
        # Critical regression test: semantic scores should NOT all be the same
        unique_scores = set(semantic_scores)
        assert len(unique_scores) > 1, (
            f"Semantic scores should vary, but all were: {semantic_scores[0]:.6f}. "
            f"This indicates the numerical overflow regression in search similarity calculation."
        )
        
        # Docker commands should have higher semantic scores than non-docker commands
        docker_commands = [cmd for cmd, _, _, _ in docker_results if 'docker' in cmd.lower()]
        non_docker_commands = [cmd for cmd, _, _, _ in docker_results if 'docker' not in cmd.lower()]
        
        if docker_commands and non_docker_commands:
            docker_scores = [score for cmd, _, score, _ in docker_results if 'docker' in cmd.lower()]
            non_docker_scores = [score for cmd, _, score, _ in docker_results if 'docker' not in cmd.lower()]
            
            # At least some docker commands should score higher than non-docker commands
            max_docker_score = max(docker_scores) if docker_scores else 0
            max_non_docker_score = max(non_docker_scores) if non_docker_scores else 0
            
            assert max_docker_score > max_non_docker_score, (
                f"Docker commands should have higher semantic scores. "
                f"Docker max: {max_docker_score:.6f}, Non-docker max: {max_non_docker_score:.6f}"
            )
    
    def test_exact_match_gets_high_score(self):
        """Test that exact matches get high semantic similarity scores."""
        # Add a distinctive command
        test_command = "docker run hello-world"
        
        result = self.fuzzyshell.data_pipeline.process_ingestion(
            [test_command],
            use_clustering=False
        )
        assert result.success
        
        # Search for exact match
        results = self.fuzzyshell.search(test_command, top_k=1, return_scores=True)
        assert len(results) > 0, "Should find the exact match"
        
        command, combined_score, semantic_score, bm25_score = results[0]
        
        # Exact match should have high semantic score
        assert semantic_score > 0.7, (
            f"Exact match should have high semantic score, got {semantic_score:.6f}"
        )
        
        # Should not be the regression value of exactly 0.5
        assert abs(semantic_score - 0.5) > 0.001, (
            "Semantic score should not be the regression value of 0.5"
        )


