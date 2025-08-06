#!/usr/bin/env python3
"""
Unit tests for ExpertScreenController class.

Tests comprehensive system information gathering for expert/debug displays
including models, database stats, search configuration, and error handling.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from fuzzyshell.expert_screen_controller import ExpertScreenController


@pytest.fixture
def mock_fuzzyshell():
    """Create mock FuzzyShell instance with all required attributes."""
    mock_fs = Mock()
    mock_fs.db_path = "/test/path/fuzzyshell.db"
    mock_fs.total_commands = 1500
    mock_fs.avg_length = 8.5
    mock_fs.k1 = 1.5
    mock_fs.b = 0.75
    mock_fs._model = Mock()
    mock_fs.ann_manager = Mock()
    
    # Mock DAL objects
    mock_fs.command_dal = Mock()
    mock_fs.command_dal.get_command_count.return_value = 1500
    mock_fs.command_dal.get_embedding_count.return_value = 1400
    
    mock_fs.cache_dal = Mock()
    mock_fs.cache_dal.get_cache_count.return_value = 25
    
    return mock_fs


@pytest.fixture
def expert_controller(mock_fuzzyshell):
    """Create ExpertScreenController with mock FuzzyShell."""
    return ExpertScreenController(mock_fuzzyshell)


def test_initialization(mock_fuzzyshell):
    """Test ExpertScreenController initializes correctly."""
    controller = ExpertScreenController(mock_fuzzyshell)
    assert controller.fuzzyshell == mock_fuzzyshell


def test_get_system_info_basic_structure(expert_controller):
    """Test that get_system_info returns correct basic structure."""
    with patch('fuzzyshell.expert_screen_controller.__version__', '0.2.0'):
        with patch('fuzzyshell.expert_screen_controller.USE_ANN_SEARCH', True):
            with patch('fuzzyshell.expert_screen_controller.ANN_NUM_CLUSTERS', 32):
                info = expert_controller.get_system_info()
    
    # Check main structure
    assert isinstance(info, dict)
    assert 'version' in info
    assert 'database' in info
    assert 'search_configuration' in info
    assert 'bm25_parameters' in info
    assert 'embedding_model' in info
    assert 'description_model' in info
    assert 'ann_index' in info


def test_database_info_structure(expert_controller):
    """Test database information structure and content."""
    info = expert_controller.get_system_info()
    
    db_info = info['database']
    assert db_info['path'] == "/test/path/fuzzyshell.db"
    assert db_info['total_commands'] == 1500
    assert db_info['avg_command_length'] == 8.5
    assert db_info['actual_command_count'] == 1500
    assert db_info['embedding_count'] == 1400
    assert db_info['embedding_coverage'] == "93.3%"
    assert db_info['cached_queries'] == 25


def test_search_configuration_info(expert_controller):
    """Test search configuration information."""
    with patch('fuzzyshell.expert_screen_controller.USE_ANN_SEARCH', True):
        with patch('fuzzyshell.expert_screen_controller.ANN_NUM_CLUSTERS', 64):
            with patch('fuzzyshell.expert_screen_controller.ANN_CLUSTER_CANDIDATES', 8):
                with patch('fuzzyshell.expert_screen_controller.EMBEDDING_DTYPE') as mock_dtype:
                    mock_dtype.__name__ = 'float32'
                    with patch('fuzzyshell.expert_screen_controller.MODEL_OUTPUT_DIM', 512):
                        info = expert_controller.get_system_info()
    
    search_config = info['search_configuration']
    assert search_config['use_ann_search'] is True
    assert search_config['ann_num_clusters'] == 64
    assert search_config['ann_cluster_candidates'] == 8
    assert search_config['embedding_dtype'] == 'float32'
    assert search_config['embedding_dimensions'] == 512


def test_search_configuration_ann_disabled(expert_controller):
    """Test search configuration when ANN is disabled."""
    with patch('fuzzyshell.expert_screen_controller.USE_ANN_SEARCH', False):
        info = expert_controller.get_system_info()
    
    search_config = info['search_configuration']
    assert search_config['use_ann_search'] is False
    assert search_config['ann_num_clusters'] == 'N/A'
    assert search_config['ann_cluster_candidates'] == 'N/A'


def test_bm25_parameters(expert_controller):
    """Test BM25 parameters extraction."""
    info = expert_controller.get_system_info()
    
    bm25_params = info['bm25_parameters']
    assert bm25_params['k1'] == 1.5
    assert bm25_params['b'] == 0.75


def test_embedding_model_info_success(expert_controller):
    """Test embedding model info when model is available."""
    mock_model_info = {
        'status': 'Ready',
        'model_key': 'test-minilm',
        'model_name': 'sentence-transformers/test-model',
        'dimensions': 384
    }
    expert_controller.fuzzyshell._model.get_embedding_model_info.return_value = mock_model_info
    
    info = expert_controller.get_system_info()
    
    assert info['embedding_model'] == mock_model_info


def test_embedding_model_info_fallback(expert_controller):
    """Test embedding model info fallback when model not initialized."""
    expert_controller.fuzzyshell._model = None
    
    with patch('fuzzyshell.expert_screen_controller.get_active_model_key') as mock_get_key:
        with patch('fuzzyshell.expert_screen_controller.get_model_config') as mock_get_config:
            mock_get_key.return_value = 'fallback-model'
            mock_get_config.return_value = {
                'repo': 'test/fallback-model',
                'description': 'Fallback model for testing'
            }
            
            info = expert_controller.get_system_info()
    
    embedding_model = info['embedding_model']
    assert embedding_model['status'] == 'Not initialized'
    assert embedding_model['model_key'] == 'fallback-model'
    assert embedding_model['model_name'] == 'test/fallback-model'
    assert embedding_model['description'] == 'Fallback model for testing'


def test_embedding_model_info_error(expert_controller):
    """Test embedding model info error handling."""
    expert_controller.fuzzyshell._model.get_embedding_model_info.side_effect = Exception("Model error")
    
    info = expert_controller.get_system_info()
    
    assert 'error' in info['embedding_model']
    assert info['embedding_model']['error'] == "Model error"


def test_description_model_info_success(expert_controller):
    """Test description model info when available."""
    mock_desc_info = {
        'model_name': 'codet5-small',
        'status': 'Ready',
        'tokenizer': 'T5Tokenizer'
    }
    
    with patch('fuzzyshell.expert_screen_controller.DescriptionHandler') as mock_handler_class:
        mock_handler = Mock()
        mock_handler.get_model_info.return_value = mock_desc_info
        mock_handler_class.return_value = mock_handler
        
        info = expert_controller.get_system_info()
    
    assert info['description_model'] == mock_desc_info


def test_description_model_info_error(expert_controller):
    """Test description model info error handling."""
    with patch('fuzzyshell.expert_screen_controller.DescriptionHandler') as mock_handler_class:
        mock_handler_class.side_effect = Exception("Description model error")
        
        info = expert_controller.get_system_info()
    
    assert 'error' in info['description_model']
    assert info['description_model']['error'] == "Description model error"


def test_database_statistics_error(expert_controller):
    """Test database statistics error handling."""
    expert_controller.fuzzyshell.command_dal.get_command_count.side_effect = Exception("DB error")
    
    info = expert_controller.get_system_info()
    
    assert 'query_error' in info['database']
    assert info['database']['query_error'] == "DB error"


def test_zero_commands_coverage(expert_controller):
    """Test embedding coverage calculation with zero commands."""
    expert_controller.fuzzyshell.command_dal.get_command_count.return_value = 0
    expert_controller.fuzzyshell.command_dal.get_embedding_count.return_value = 0
    
    info = expert_controller.get_system_info()
    
    assert info['database']['embedding_coverage'] == "0%"


def test_ann_index_enabled(expert_controller):
    """Test ANN index status when enabled."""
    mock_stats = {
        'status': 'Ready',
        'clusters': 32,
        'embeddings': 1400,
        'last_built': '2023-12-01'
    }
    expert_controller.fuzzyshell.ann_manager.get_stats.return_value = mock_stats
    
    with patch('fuzzyshell.expert_screen_controller.USE_ANN_SEARCH', True):
        info = expert_controller.get_system_info()
    
    assert info['ann_index'] == mock_stats


def test_ann_index_disabled(expert_controller):
    """Test ANN index status when disabled."""
    with patch('fuzzyshell.expert_screen_controller.USE_ANN_SEARCH', False):
        info = expert_controller.get_system_info()
    
    assert info['ann_index']['status'] == 'Disabled'


def test_ann_index_no_manager(expert_controller):
    """Test ANN index status when manager is None."""
    expert_controller.fuzzyshell.ann_manager = None
    
    with patch('fuzzyshell.expert_screen_controller.USE_ANN_SEARCH', True):
        info = expert_controller.get_system_info()
    
    assert info['ann_index']['status'] == 'Disabled'


def test_version_import_success(expert_controller):
    """Test version import success."""
    with patch('fuzzyshell.expert_screen_controller.__version__', '1.2.3'):
        info = expert_controller.get_system_info()
    
    assert info['version'] == '1.2.3'


def test_comprehensive_system_info_integration(expert_controller):
    """Test comprehensive system info with all components."""
    # Set up complex mock scenario
    expert_controller.fuzzyshell.total_commands = 2500
    expert_controller.fuzzyshell.avg_length = 12.8
    expert_controller.fuzzyshell.command_dal.get_command_count.return_value = 2500
    expert_controller.fuzzyshell.command_dal.get_embedding_count.return_value = 2300
    expert_controller.fuzzyshell.cache_dal.get_cache_count.return_value = 150
    
    mock_embedding_info = {'status': 'Ready', 'model_key': 'advanced-model'}
    expert_controller.fuzzyshell._model.get_embedding_model_info.return_value = mock_embedding_info
    
    mock_ann_stats = {'status': 'Ready', 'clusters': 64}
    expert_controller.fuzzyshell.ann_manager.get_stats.return_value = mock_ann_stats
    
    with patch('fuzzyshell.expert_screen_controller.DescriptionHandler') as mock_desc_class:
        mock_desc = Mock()
        mock_desc.get_model_info.return_value = {'model_name': 'codet5-small'}
        mock_desc_class.return_value = mock_desc
        
        with patch('fuzzyshell.expert_screen_controller.__version__', '2.0.0'):
            with patch('fuzzyshell.expert_screen_controller.USE_ANN_SEARCH', True):
                info = expert_controller.get_system_info()
    
    # Verify comprehensive data
    assert info['version'] == '2.0.0'
    assert info['database']['total_commands'] == 2500
    assert info['database']['embedding_coverage'] == "92.0%"  # 2300/2500
    assert info['embedding_model'] == mock_embedding_info
    assert info['description_model']['model_name'] == 'codet5-small'
    assert info['ann_index'] == mock_ann_stats


@pytest.mark.parametrize("command_count,embedding_count,expected_coverage", [
    (0, 0, "0%"),
    (100, 95, "95.0%"),
    (1000, 850, "85.0%"),
    (50, 50, "100.0%"),
    (1, 0, "0.0%")
])
def test_embedding_coverage_calculation(expert_controller, command_count, embedding_count, expected_coverage):
    """Test embedding coverage calculation with various scenarios."""
    expert_controller.fuzzyshell.command_dal.get_command_count.return_value = command_count
    expert_controller.fuzzyshell.command_dal.get_embedding_count.return_value = embedding_count
    
    info = expert_controller.get_system_info()
    
    assert info['database']['embedding_coverage'] == expected_coverage


def test_error_isolation(expert_controller):
    """Test that errors in one component don't break others."""
    # Make embedding model fail
    expert_controller.fuzzyshell._model.get_embedding_model_info.side_effect = Exception("Embedding error")
    
    # Make description model fail
    with patch('fuzzyshell.expert_screen_controller.DescriptionHandler') as mock_desc_class:
        mock_desc_class.side_effect = Exception("Description error")
        
        # Make database stats fail
        expert_controller.fuzzyshell.command_dal.get_command_count.side_effect = Exception("DB error")
        
        info = expert_controller.get_system_info()
    
    # Basic structure should still be there
    assert 'version' in info
    assert 'database' in info
    assert 'search_configuration' in info
    assert 'bm25_parameters' in info
    
    # Error states should be recorded
    assert 'error' in info['embedding_model']
    assert 'error' in info['description_model']
    assert 'query_error' in info['database']
    
    # But other parts should still work
    assert info['bm25_parameters']['k1'] == 1.5
    assert info['database']['path'] == "/test/path/fuzzyshell.db"


def test_get_database_info_success(expert_controller):
    """Test get_database_info with successful data retrieval."""
    # Mock MetadataManager database info
    mock_db_info = {
        'uuid': 'test-uuid-123',
        'created': '2023-12-01 10:00:00',
        'item_count': 1000,
        'embedding_model': 'test-model'
    }
    expert_controller.fuzzyshell.metadata_manager.get_database_info.return_value = mock_db_info
    
    # Mock file system and metadata
    expert_controller.fuzzyshell.db_path = "/test/fuzzyshell.db"
    expert_controller.fuzzyshell.get_metadata.side_effect = lambda key, default: {
        'embedding_dtype': 'float32',
        'schema_version': '2.0'
    }.get(key, default)
    expert_controller.fuzzyshell.get_indexed_count.return_value = 1000
    
    with patch('os.path.exists', return_value=True):
        with patch('os.path.getsize', return_value=1048576):  # 1MB
            info = expert_controller.get_database_info()
    
    # Verify structure
    assert info['uuid'] == 'test-uuid-123'
    assert info['created'] == '2023-12-01 10:00:00'
    assert info['item_count'] == 1000
    assert info['embedding_model'] == 'test-model'
    assert info['embedding_dtype'] == 'float32'
    assert info['schema_version'] == '2.0'
    assert info['db_size_bytes'] == 1048576
    assert info['db_size_human'] == '1.0MB'


def test_get_database_info_no_file(expert_controller):
    """Test get_database_info when database file doesn't exist."""
    mock_db_info = {'item_count': 500}
    expert_controller.fuzzyshell.metadata_manager.get_database_info.return_value = mock_db_info
    expert_controller.fuzzyshell.db_path = "/nonexistent/fuzzyshell.db"
    expert_controller.fuzzyshell.get_indexed_count.return_value = 500
    expert_controller.fuzzyshell.get_metadata.side_effect = lambda key, default: default
    
    with patch('os.path.exists', return_value=False):
        info = expert_controller.get_database_info()
    
    # Should estimate size based on command count
    assert info['db_size_bytes'] == 500 * 1024  # 500 commands * 1KB estimate
    assert info['db_size_human'] == '500.0KB'


@pytest.mark.parametrize("size_bytes,expected", [
    (512, "512B"),
    (1536, "1.5KB"),  # 1.5 * 1024
    (2097152, "2.0MB"),  # 2 * 1024^2
    (3221225472, "3.0GB"),  # 3 * 1024^3
])
def test_format_bytes(expert_controller, size_bytes, expected):
    """Test byte formatting with various sizes."""
    result = expert_controller._format_bytes(size_bytes)
    assert result == expected


def test_get_database_size_file_exists(expert_controller):
    """Test database size calculation when file exists."""
    expert_controller.fuzzyshell.db_path = "/test/fuzzyshell.db"
    
    with patch('os.path.exists', return_value=True):
        with patch('os.path.getsize', return_value=2048):
            size = expert_controller._get_database_size()
    
    assert size == 2048


def test_get_database_size_file_missing(expert_controller):
    """Test database size estimation when file doesn't exist."""
    expert_controller.fuzzyshell.db_path = "/missing/fuzzyshell.db"
    expert_controller.fuzzyshell.get_indexed_count.return_value = 750
    
    with patch('os.path.exists', return_value=False):
        size = expert_controller._get_database_size()
    
    assert size == 750 * 1024  # 750 commands * 1KB estimate