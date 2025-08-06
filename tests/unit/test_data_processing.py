"""
Unit tests for data processing pipeline components.

Tests the Command + Facade pattern implementation for data processing operations.
"""

import pytest
import tempfile
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path

from fuzzyshell.data_processing import (
    DataProcessingPipeline, 
    ProcessingContext, 
    ProcessingResult,
    ExtractEmbeddingsCommand,
    CalculateIDFCommand,
    BuildANNIndexCommand,
    ClearCacheCommand,
    DatabaseWriteCommand
)
from fuzzyshell.data.datastore import ProductionDatabaseProvider, CommandDAL


class TestProcessingContext:
    """Test ProcessingContext dataclass."""
    
    def test_processing_context_initialization(self):
        """Test ProcessingContext creates with defaults."""
        context = ProcessingContext(commands=["test command"])
        
        assert context.commands == ["test command"]
        assert context.embeddings is None
        assert context.command_ids is None
        assert context.batch_size == 1000
        assert context.use_clustering is True
        assert context.progress_callback is None
        assert context.processing_times == {}
    
    def test_processing_context_with_data(self):
        """Test ProcessingContext with full data."""
        embeddings = [np.array([1, 2, 3]), np.array([4, 5, 6])]
        
        context = ProcessingContext(
            commands=["cmd1", "cmd2"],
            embeddings=embeddings,
            command_ids=[1, 2],
            batch_size=500,
            use_clustering=False
        )
        
        assert len(context.commands) == 2
        assert len(context.embeddings) == 2
        assert context.command_ids == [1, 2]
        assert context.batch_size == 500
        assert context.use_clustering is False


class TestProcessingResult:
    """Test ProcessingResult dataclass."""
    
    def test_processing_result_success(self):
        """Test successful ProcessingResult."""
        result = ProcessingResult(
            success=True,
            commands_processed=100,
            embeddings_processed=100,
            ann_index_built=True,
            processing_time_seconds=5.0
        )
        
        assert result.success is True
        assert result.commands_processed == 100
        assert result.ann_index_built is True
        assert result.processing_time_seconds == 5.0
        
        summary = result.get_summary()
        assert "Successfully processed 100 commands" in summary
        assert "ANN index: built" in summary
    
    def test_processing_result_failure(self):
        """Test failed ProcessingResult."""
        result = ProcessingResult(
            success=False,
            commands_processed=0,
            embeddings_processed=0,
            ann_index_built=False,
            processing_time_seconds=1.0,
            error_message="Test error"
        )
        
        assert result.success is False
        summary = result.get_summary()
        assert "Processing failed: Test error" in summary
    
    def test_performance_breakdown(self):
        """Test performance breakdown."""
        result = ProcessingResult(
            success=True,
            commands_processed=10,
            embeddings_processed=10,
            ann_index_built=False,
            processing_time_seconds=2.0,
            embedding_extraction_time=0.5,
            idf_calculation_time=0.3,
            ann_building_time=0.7,
            database_write_time=0.5
        )
        
        breakdown = result.get_performance_breakdown()
        assert breakdown['embedding_extraction'] == 0.5
        assert breakdown['idf_calculation'] == 0.3
        assert breakdown['ann_building'] == 0.7
        assert breakdown['database_write'] == 0.5
        assert breakdown['total'] == 2.0


class TestExtractEmbeddingsCommand:
    """Test ExtractEmbeddingsCommand."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_model = Mock()
        self.mock_model.encode.return_value = [
            np.array([0.1, 0.2, 0.3] * 128),  # 384 dims
            np.array([0.4, 0.5, 0.6] * 128)
        ]
        
    def test_extract_embeddings_command_execution(self):
        """Test ExtractEmbeddingsCommand executes successfully."""
        command = ExtractEmbeddingsCommand(self.mock_model)
        context = ProcessingContext(commands=["test cmd 1", "test cmd 2"])
        
        command.execute(context)
        
        # Check embeddings were generated
        assert context.processed_embeddings is not None
        assert len(context.processed_embeddings) == 2
        
        # Check model was called
        self.mock_model.encode.assert_called()
        
        # Check timing was recorded
        assert 'ExtractEmbeddings' in context.processing_times
        assert context.processing_times['ExtractEmbeddings'] > 0
    
    def test_extract_embeddings_with_progress_callback(self):
        """Test ExtractEmbeddingsCommand with progress callback."""
        progress_calls = []
        def progress_callback(percent, message):
            progress_calls.append((percent, message))
        
        command = ExtractEmbeddingsCommand(self.mock_model)
        context = ProcessingContext(
            commands=["cmd"] * 100,  # Large batch to trigger progress
            progress_callback=progress_callback
        )
        
        command.execute(context)
        
        # Should have progress updates
        assert len(progress_calls) > 0
        
        # Progress should be in 20-80% range
        for percent, message in progress_calls:
            assert percent >= 20
            assert percent <= 80
            assert "Processing" in message
    
    def test_extract_embeddings_quantization(self):
        """Test embedding quantization fallback."""
        command = ExtractEmbeddingsCommand(self.mock_model, embedding_dtype=np.int8)
        
        # Test int8 quantization using fallback method
        test_embedding = np.array([0.5, -0.3, 1.0])
        quantized = command._quantize_embedding_fallback(test_embedding)
        
        assert quantized.dtype == np.int8
        # Values should be scaled to int8 range
        expected = (test_embedding * 127.0).astype(np.int8)
        np.testing.assert_array_equal(quantized, expected)
    
    def test_can_execute_validation(self):
        """Test can_execute validation."""
        command = ExtractEmbeddingsCommand(self.mock_model)
        
        # Should execute with commands
        context_with_commands = ProcessingContext(commands=["test"])
        assert command.can_execute(context_with_commands) is True
        
        # Should not execute without commands
        context_empty = ProcessingContext(commands=[])
        assert command.can_execute(context_empty) is False
        
        context_none = ProcessingContext(commands=None)
        assert command.can_execute(context_none) is False


class TestDatabaseWriteCommand:
    """Test DatabaseWriteCommand."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_dal_provider = Mock()
        self.mock_command_dal = Mock()
        self.mock_dal_provider.command_dal = self.mock_command_dal
        
        # Mock the DAL methods
        self.mock_command_dal.add_commands_batch.return_value = [1, 2]
        self.mock_command_dal.add_embeddings_batch.return_value = None
        self.mock_command_dal.add_term_frequencies_batch.return_value = None
    
    def test_database_write_command_execution(self):
        """Test DatabaseWriteCommand executes successfully."""
        command = DatabaseWriteCommand(self.mock_dal_provider)
        
        # Create test context with embeddings
        embeddings = [
            np.array([1, 2, 3], dtype=np.int8),
            np.array([4, 5, 6], dtype=np.int8)
        ]
        
        context = ProcessingContext(
            commands=["test cmd 1", "test cmd 2"],
            processed_embeddings=embeddings
        )
        
        command.execute(context)
        
        # Check DAL methods were called
        self.mock_command_dal.add_commands_batch.assert_called_once()
        self.mock_command_dal.add_embeddings_batch.assert_called_once()
        self.mock_command_dal.add_term_frequencies_batch.assert_called_once()
        
        # Check command IDs were stored in context
        assert context.command_ids == [1, 2]
        
        # Check timing was recorded
        assert 'DatabaseWrite' in context.processing_times
    
    def test_database_write_with_progress_callback(self):
        """Test DatabaseWriteCommand with progress callback."""
        progress_calls = []
        def progress_callback(percent, message):
            progress_calls.append((percent, message))
        
        command = DatabaseWriteCommand(self.mock_dal_provider)
        
        # Mock should return 1 ID for 1 command
        self.mock_command_dal.add_commands_batch.return_value = [1]
        
        embeddings = [np.array([1, 2, 3], dtype=np.int8)]
        context = ProcessingContext(
            commands=["test cmd"],
            processed_embeddings=embeddings,
            progress_callback=progress_callback
        )
        
        command.execute(context)
        
        # Should have progress updates
        assert len(progress_calls) > 0
        
        # Check progress messages
        messages = [msg for _, msg in progress_calls]
        assert any("database" in msg.lower() for msg in messages)
    
    def test_can_execute_validation(self):
        """Test can_execute validation."""
        command = DatabaseWriteCommand(self.mock_dal_provider)
        
        # Should execute with matching commands and embeddings
        context_valid = ProcessingContext(
            commands=["cmd1", "cmd2"],
            processed_embeddings=[np.array([1, 2]), np.array([3, 4])]
        )
        assert command.can_execute(context_valid) is True
        
        # Should not execute with mismatched lengths
        context_mismatch = ProcessingContext(
            commands=["cmd1"],
            processed_embeddings=[np.array([1, 2]), np.array([3, 4])]
        )
        assert command.can_execute(context_mismatch) is False
        
        # Should not execute without commands
        context_no_commands = ProcessingContext(
            commands=None,
            processed_embeddings=[np.array([1, 2])]
        )
        assert command.can_execute(context_no_commands) is False


class TestClearCacheCommand:
    """Test ClearCacheCommand."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = str(Path(self.temp_dir) / "test.db")
        
        # Create mock cache files
        self.cluster_cache_path = self.db_path.replace('.db', '_clusters.pkl')
        self.ann_cache_path = self.db_path.replace('.db', '_ann_index.pkl')
        
        Path(self.cluster_cache_path).touch()
        Path(self.ann_cache_path).touch()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_clear_cache_command_execution(self):
        """Test ClearCacheCommand removes cache files."""
        command = ClearCacheCommand(self.db_path)
        context = ProcessingContext(commands=[])
        
        # Verify cache files exist before
        assert Path(self.cluster_cache_path).exists() is True
        assert Path(self.ann_cache_path).exists() is True
        
        command.execute(context)
        
        # Verify cache files were removed
        assert Path(self.cluster_cache_path).exists() is False
        assert Path(self.ann_cache_path).exists() is False
        
        # Check timing was recorded
        assert 'ClearCache' in context.processing_times
    
    def test_clear_cache_no_files(self):
        """Test ClearCacheCommand when no cache files exist."""
        # Remove the files we created
        Path(self.cluster_cache_path).unlink()
        Path(self.ann_cache_path).unlink()
        
        command = ClearCacheCommand(self.db_path)
        context = ProcessingContext(commands=[])
        
        # Should not raise error
        command.execute(context)
        
        # Check timing was still recorded
        assert 'ClearCache' in context.processing_times


class TestDataProcessingPipeline:
    """Test DataProcessingPipeline facade."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_dal_provider = Mock()
        self.mock_model_handler = Mock()
        self.mock_ann_index_handler = Mock()
        
        self.pipeline = DataProcessingPipeline(
            dal_provider=self.mock_dal_provider,
            db_path="test.db",
            model_handler=self.mock_model_handler,
            ann_index_handler=self.mock_ann_index_handler
        )
    
    @patch('fuzzyshell.data_processing.pipeline.ExtractEmbeddingsCommand')
    @patch('fuzzyshell.data_processing.pipeline.DatabaseWriteCommand')
    @patch('fuzzyshell.data_processing.pipeline.CalculateIDFCommand')
    @patch('fuzzyshell.data_processing.pipeline.BuildANNIndexCommand')
    def test_process_ingestion_success(self, mock_ann_cmd, mock_idf_cmd, mock_db_cmd, mock_emb_cmd):
        """Test successful ingestion processing."""
        # Mock command instances
        mock_emb_instance = Mock()
        mock_db_instance = Mock()
        mock_idf_instance = Mock()
        mock_ann_instance = Mock()
        
        mock_emb_cmd.return_value = mock_emb_instance
        mock_db_cmd.return_value = mock_db_instance
        mock_idf_cmd.return_value = mock_idf_instance
        mock_ann_cmd.return_value = mock_ann_instance
        
        # Mock can_execute to return True
        mock_emb_instance.can_execute.return_value = True
        mock_db_instance.can_execute.return_value = True
        mock_idf_instance.can_execute.return_value = True
        mock_ann_instance.can_execute.return_value = True
        
        # Mock command names
        mock_emb_instance.name = "ExtractEmbeddings"
        mock_db_instance.name = "DatabaseWrite"
        mock_idf_instance.name = "CalculateIDF"
        mock_ann_instance.name = "BuildANNIndex"
        
        commands = ["test command 1", "test command 2"]
        
        result = self.pipeline.process_ingestion(
            commands=commands,
            use_clustering=True
        )
        
        # Check result
        assert isinstance(result, ProcessingResult)
        assert result.success is True
        assert result.commands_processed == 2
        
        # Check commands were executed
        mock_emb_instance.execute.assert_called_once()
        mock_db_instance.execute.assert_called_once()
        mock_idf_instance.execute.assert_called_once()
        mock_ann_instance.execute.assert_called_once()
    
    def test_clear_all_caches(self):
        """Test clear_all_caches facade method."""
        with patch('fuzzyshell.data_processing.commands.ClearCacheCommand') as mock_clear_cmd:
            mock_clear_instance = Mock()
            mock_clear_cmd.return_value = mock_clear_instance
            
            result = self.pipeline.clear_all_caches()
            
            assert isinstance(result, ProcessingResult)
            assert result.success is True
            mock_clear_instance.execute.assert_called_once()


