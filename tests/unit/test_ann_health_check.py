#!/usr/bin/env python3
"""
Unit tests for ANN Index Health Check functionality.

Tests the health check heuristics that warn when the ANN index becomes stale
relative to the current database state.
"""

import pytest
import numpy as np
import tempfile
import os
from unittest.mock import Mock

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from fuzzyshell.ann_index_manager import ANNIndexManager, ANNSearchIndex


@pytest.fixture
def temp_db_path():
    """Create temporary database path."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        temp_path = f.name
    yield temp_path
    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def ann_manager(temp_db_path):
    """Create ANNIndexManager instance."""
    return ANNIndexManager(temp_db_path)


@pytest.fixture
def trained_ann_manager(temp_db_path):
    """Create ANNIndexManager with trained index."""
    manager = ANNIndexManager(temp_db_path)
    
    # Create sample embeddings (100 embeddings of 384 dimensions)
    embeddings = np.random.rand(100, 384).astype(np.float32)
    manager.build_index(embeddings, n_clusters=10)
    
    return manager


class TestANNHealthCheck:
    """Test ANN index health check functionality."""
    
    def test_health_check_not_trained(self, ann_manager):
        """Test health check when index is not trained."""
        health = ann_manager.check_health(current_embedding_count=100)
        
        assert health['status'] == 'unhealthy'
        assert health['reason'] == 'not_trained'
        assert health['message'] == 'ANN index not trained'
        assert health['staleness_pct'] == 0
        assert health['recommend_rebuild'] is True
        
    def test_health_check_empty_index(self, ann_manager):
        """Test health check when index is empty."""
        # Force trained state but with no embeddings
        ann_manager.index.is_trained = True
        ann_manager.index.embeddings = None
        
        health = ann_manager.check_health(current_embedding_count=100)
        
        assert health['status'] == 'unhealthy'
        assert health['reason'] == 'empty_index'
        assert health['message'] == 'ANN index is empty'
        assert health['staleness_pct'] == 100
        assert health['recommend_rebuild'] is True
        
    def test_health_check_healthy(self, trained_ann_manager):
        """Test health check when index is healthy (< 5% change)."""
        # Test with same count (0% change)
        health = trained_ann_manager.check_health(current_embedding_count=100)
        
        assert health['status'] == 'healthy'
        assert health['reason'] == 'current'
        assert health['message'] == 'ANN index is current'
        assert health['staleness_pct'] == 0.0
        assert health['indexed_count'] == 100
        assert health['current_count'] == 100
        assert health['recommend_rebuild'] is False
        
        # Test with minimal change (3% change)
        health = trained_ann_manager.check_health(current_embedding_count=103)
        
        assert health['status'] == 'healthy'
        assert health['reason'] == 'current'
        assert health['staleness_pct'] == 3.0
        assert health['recommend_rebuild'] is False
        
    def test_health_check_minor_staleness(self, trained_ann_manager):
        """Test health check with minor staleness (5-15% change)."""
        # Test with 10% increase
        health = trained_ann_manager.check_health(current_embedding_count=110)
        
        assert health['status'] == 'ok'
        assert health['reason'] == 'minor_staleness'
        assert health['message'] == 'ANN index slightly stale: 10% change'
        assert health['staleness_pct'] == 10.0
        assert health['indexed_count'] == 100
        assert health['current_count'] == 110
        assert health['recommend_rebuild'] is False
        
    def test_health_check_degraded(self, trained_ann_manager):
        """Test health check when degraded (15-25% change)."""
        # Test with 20% increase
        health = trained_ann_manager.check_health(current_embedding_count=120)
        
        assert health['status'] == 'degraded'
        assert health['reason'] == 'stale'
        assert 'ANN index stale: 20% change (100 → 120)' in health['message']
        assert health['staleness_pct'] == 20.0
        assert health['recommend_rebuild'] is True
        
    def test_health_check_very_stale(self, trained_ann_manager):
        """Test health check when very stale (≥25% change)."""
        # Test with 50% increase
        health = trained_ann_manager.check_health(current_embedding_count=150)
        
        assert health['status'] == 'unhealthy'
        assert health['reason'] == 'very_stale'
        assert 'ANN index very stale: 50% change (100 → 150)' in health['message']
        assert health['staleness_pct'] == 50.0
        assert health['recommend_rebuild'] is True
        
    def test_health_check_database_shrunk(self, trained_ann_manager):
        """Test health check when database has fewer embeddings than indexed."""
        # Test when database shrunk by 30%
        health = trained_ann_manager.check_health(current_embedding_count=70)
        
        assert health['status'] == 'unhealthy'
        assert health['reason'] == 'very_stale'
        assert 'ANN index very stale: 30% change (100 → 70)' in health['message']
        assert health['staleness_pct'] == 30.0
        assert health['recommend_rebuild'] is True
        
    def test_get_health_warning_healthy(self, trained_ann_manager):
        """Test health warning when index is healthy."""
        # No warning for healthy index
        warning = trained_ann_manager.get_health_warning(current_embedding_count=100)
        assert warning is None
        
        # No warning for minor staleness
        warning = trained_ann_manager.get_health_warning(current_embedding_count=110)
        assert warning is None
        
    def test_get_health_warning_degraded(self, trained_ann_manager):
        """Test health warning when index is degraded."""
        warning = trained_ann_manager.get_health_warning(current_embedding_count=120)
        
        assert warning == "⚠ Stale Clusters (20%)"
        
    def test_get_health_warning_unhealthy(self, trained_ann_manager):
        """Test health warning when index is unhealthy."""
        # Very stale
        warning = trained_ann_manager.get_health_warning(current_embedding_count=150)
        assert warning == "! Outdated Clusters (50% stale)"
        
        # Not trained
        ann_manager = ANNIndexManager(':memory:')
        warning = ann_manager.get_health_warning(current_embedding_count=100)
        assert warning == "! No ANN Index"
        
    def test_health_check_edge_cases(self, trained_ann_manager):
        """Test health check edge cases."""
        # Zero current embeddings
        health = trained_ann_manager.check_health(current_embedding_count=0)
        assert health['status'] == 'unhealthy'
        assert health['staleness_pct'] == 100.0
        
        # Very large increase
        health = trained_ann_manager.check_health(current_embedding_count=1000)
        assert health['status'] == 'unhealthy'
        assert health['staleness_pct'] == 900.0
        assert health['recommend_rebuild'] is True
        
    def test_health_check_threshold_boundaries(self, trained_ann_manager):
        """Test health check at threshold boundaries."""
        # Exactly 5% (should be healthy)  
        health = trained_ann_manager.check_health(current_embedding_count=105)
        assert health['status'] == 'ok'  # 5% is minor staleness boundary
        assert health['staleness_pct'] == 5.0
        
        # Exactly 15% (should be degraded)
        health = trained_ann_manager.check_health(current_embedding_count=115)
        assert health['status'] == 'degraded'
        assert health['staleness_pct'] == 15.0
        
        # Exactly 25% (should be unhealthy)
        health = trained_ann_manager.check_health(current_embedding_count=125)
        assert health['status'] == 'unhealthy'
        assert health['staleness_pct'] == 25.0
        
    def test_health_check_precision(self, trained_ann_manager):
        """Test health check precision with fractional percentages."""
        # Test with fractional staleness
        health = trained_ann_manager.check_health(current_embedding_count=107)
        assert health['staleness_pct'] == 7.0  # Should be rounded to 1 decimal
        
        # Test with very small difference
        health = trained_ann_manager.check_health(current_embedding_count=101)
        assert health['staleness_pct'] == 1.0
        assert health['status'] == 'healthy'
        
    def test_health_check_integration_with_stats(self, trained_ann_manager):
        """Test that health check integrates well with existing stats."""
        stats = trained_ann_manager.get_stats()
        health = trained_ann_manager.check_health(current_embedding_count=150)
        
        # Should have consistent data
        assert stats['n_embeddings'] == health['indexed_count']
        assert stats['status'] == 'trained'
        assert health['status'] == 'unhealthy'  # Due to staleness
        
    def test_health_warning_for_ui_display(self, trained_ann_manager):
        """Test health warning messages are appropriate for UI display."""
        # Test that warning messages are concise and informative
        test_cases = [
            (100, None),  # Healthy - no warning
            (110, None),  # Minor staleness - no warning  
            (120, "⚠ Stale Clusters (20%)"),  # Degraded
            (150, "! Outdated Clusters (50% stale)"),  # Unhealthy
        ]
        
        for count, expected_warning in test_cases:
            warning = trained_ann_manager.get_health_warning(current_embedding_count=count)
            assert warning == expected_warning
            
            # Ensure warnings are short and suitable for status displays
            if warning:
                assert len(warning) < 50  # Should be concise
                assert any(char in warning for char in ['!', '⚠'])  # Should have indicator


class TestANNHealthCheckResponsibilitySeparation:
    """Test that health check responsibility is well-separated for future enhancements."""
    
    def test_health_check_extensibility(self, trained_ann_manager):
        """Test that health check can be easily extended."""
        # The current implementation uses simple percentage-based metrics
        # This test ensures the interface can be extended for more sophisticated metrics
        
        health = trained_ann_manager.check_health(current_embedding_count=120)
        
        # Verify structure allows for future enhancements
        required_fields = ['status', 'reason', 'message', 'staleness_pct', 'recommend_rebuild']
        for field in required_fields:
            assert field in health
            
        # Verify status types are well-defined
        valid_statuses = ['healthy', 'ok', 'degraded', 'unhealthy']
        assert health['status'] in valid_statuses
        
    def test_health_check_single_responsibility(self, trained_ann_manager):
        """Test that health check has single responsibility - health assessment only."""
        # Health check should not modify state
        original_embeddings = trained_ann_manager.index.embeddings.copy()
        original_trained = trained_ann_manager.index.is_trained
        
        trained_ann_manager.check_health(current_embedding_count=150)
        
        # State should be unchanged
        np.testing.assert_array_equal(trained_ann_manager.index.embeddings, original_embeddings)
        assert trained_ann_manager.index.is_trained == original_trained
        
    def test_health_metrics_are_simple_and_interpretable(self, trained_ann_manager):
        """Test that health metrics are simple percentage-based for easy interpretation."""
        health = trained_ann_manager.check_health(current_embedding_count=130)
        
        # Should use simple percentage calculation
        expected_pct = ((130 - 100) / 100) * 100  # 30%
        assert health['staleness_pct'] == expected_pct
        
        # Should have clear thresholds
        assert isinstance(health['staleness_pct'], (int, float))
        assert health['staleness_pct'] >= 0
        
        # Should provide clear boolean recommendation
        assert isinstance(health['recommend_rebuild'], bool)