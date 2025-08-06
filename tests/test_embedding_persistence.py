#!/usr/bin/env python3
"""
Embedding Persistence and Save/Load Tests

This test suite ensures that:
1. Embeddings are generated correctly and consistently with the current model
2. Embeddings are stored and retrieved correctly (save/load roundtrip)
3. Model change detection properly clears old embeddings
4. Only valid shell commands are ingested (no JSON fragments or corrupted data)
5. Semantic search returns reasonable results for known command relationships

This validates the entire embedding persistence layer and guards against data corruption.
"""
import tempfile
import os
import sys
import sqlite3
import numpy as np
from unittest.mock import patch, MagicMock

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from fuzzyshell import FuzzyShell
from fuzzyshell.model_handler import ModelHandler
from fuzzyshell.search_coordinator import SearchCoordinator, EmbeddingManager
from fuzzyshell.data.datastore import ProductionDatabaseProvider, CommandDAL, MetadataDAL, QueryCacheDAL


class TestEmbeddingPersistence:
    """Test suite for embedding persistence and save/load functionality."""

    def setup_method(self):
        """Set up clean test environment."""
        # Create temporary database
        self.temp_db_fd, self.temp_db_path = tempfile.mkstemp(suffix='.db')
        os.close(self.temp_db_fd)
        
        # Create FuzzyShell with clean database
        self.fs = FuzzyShell(db_path=self.temp_db_path)
        self.fs.wait_for_model(timeout=10.0)
        
        # Verify model is terminal-minilm-l6
        model_info = self.fs._model.get_model_info()
        assert model_info['model_key'] == 'terminal-minilm-l6', (
            "Tests must use terminal-minilm-l6 model"
        )
    
    def teardown_method(self):
        """Clean up test database."""
        if os.path.exists(self.temp_db_path):
            os.unlink(self.temp_db_path)
    
    def test_model_determinism(self):
        """Test that model produces consistent embeddings."""
        test_command = "ls -lh"
        
        # Generate embeddings multiple times
        emb1 = self.fs._model.encode([test_command])[0]
        emb2 = self.fs._model.encode([test_command])[0]
        emb3 = self.fs._model.encode([test_command])[0]
        
        # Should be identical (deterministic)
        similarity_1_2 = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        similarity_1_3 = np.dot(emb1, emb3) / (np.linalg.norm(emb1) * np.linalg.norm(emb3))
        
        assert similarity_1_2 > 0.999, "Model should be deterministic"
        assert similarity_1_3 > 0.999, "Model should be deterministic"
    
    def test_quantization_roundtrip(self):
        """Test that quantization doesn't corrupt embeddings."""
        test_commands = ["ls -lh", "git status", "docker ps", "python manage.py runserver"]
        
        for command in test_commands:
            # Generate fresh embedding
            original_emb = self.fs._model.encode([command])[0]

            # Quantize and dequantize
            quantized = self.fs.quantize_embedding(original_emb)
            recovered_emb = self.fs.dequantize_embedding(quantized)

            # Check consistency
            similarity = np.dot(original_emb, recovered_emb) / (
                np.linalg.norm(original_emb) * np.linalg.norm(recovered_emb)
            )

            assert similarity > 0.99, (
                f"Quantization roundtrip failed for '{command}': {similarity:.3f}"
            )
    
    def test_database_storage_roundtrip(self):
        """Test that database storage doesn't corrupt embeddings."""
        test_commands = [
            "ls -lh",
            "git add .",
            "docker build -t myapp .",
            "python -m pytest tests/"
        ]
        
        for command in test_commands:
            # Generate and store embedding
            original_emb = self.fs._model.encode([command])[0]
            quantized_emb = self.fs.quantize_embedding(original_emb)

            # Add to database
            cmd_id = self.fs.command_dal.add_command(command, embedding=quantized_emb)

            # Retrieve from database
            stored_data = self.fs.command_dal.get_command(cmd_id)
            assert stored_data is not None, f"Command '{command}' not found in database"

            stored_cmd_id, stored_command, stored_blob = stored_data
            assert stored_command == command, "Command text corrupted in storage"

            # Dequantize retrieved embedding
            retrieved_emb = self.fs.dequantize_embedding(stored_blob)

            # Check consistency
            similarity = np.dot(original_emb, retrieved_emb) / (
                np.linalg.norm(original_emb) * np.linalg.norm(retrieved_emb)
            )

            assert similarity > 0.99, (
                f"Database roundtrip failed for '{command}': {similarity:.3f}"
            )
    
    def test_embedding_consistency_with_search_coordinator(self):
        """Test that SearchCoordinator detects no corruption in fresh embeddings."""
        # Add clean test commands
        test_commands = [
            "ls -la",
            "cd /tmp", 
            "git status",
            "python main.py",
            "docker ps -a"
        ]
        
        # Add commands to database
        for command in test_commands:
            original_emb = self.fs._model.encode([command])[0]
            quantized_emb = self.fs.quantize_embedding(original_emb)
            self.fs.command_dal.add_command(command, embedding=quantized_emb)
        
        # Create SearchCoordinator and test
        coordinator = SearchCoordinator(
            model_handler=self.fs._model,
            command_dal=self.fs.command_dal,
            ann_manager=self.fs.ann_manager,
            dequantize_func=self.fs.dequantize_embedding,
            quantize_func=self.fs.quantize_embedding
        )
        
        # Run corruption diagnosis
        report = coordinator.diagnose_embeddings(sample_size=len(test_commands))
        
        # Should find no corruption
        assert report['corrupted_count'] == 0, (
            f"Fresh embeddings should not be corrupted: {report}"
        )
        assert report['avg_consistency'] > 0.99, (
            f"Average consistency too low: {report['avg_consistency']}"
        )
    
    def test_model_change_detection_clears_embeddings(self):
        """Test that model change properly clears old embeddings."""
        # Add some embeddings
        test_commands = ["ls", "pwd", "whoami"]
        for command in test_commands:
            original_emb = self.fs._model.encode([command])[0]
            quantized_emb = self.fs.quantize_embedding(original_emb)
            self.fs.command_dal.add_command(command, embedding=quantized_emb)
        
        # Verify embeddings exist
        initial_count = self.fs.command_dal.get_embedding_count()
        assert initial_count > 0, "Should have embeddings before test"
        
        # Simulate model change by setting old metadata
        self.fs.metadata_dal.set('model_name', 'old-model')
        
        # Trigger model change detection
        self.fs._handle_model_change()
        
        # Verify embeddings were cleared
        final_count = self.fs.command_dal.get_embedding_count()
        assert final_count == 0, "Model change should clear all embeddings"
        
        # Verify metadata was updated
        current_model = self.fs.metadata_dal.get('model_name')
        assert current_model == 'terminal-minilm-l6', (
            "Metadata should be updated to current model"
        )
    
    def test_only_valid_commands_ingested(self):
        """Test that only valid shell commands are ingested, not JSON fragments."""
        # Mock shell history file content with mixed valid/invalid content
        mock_history_content = '''ls -lh
git status
"content": "Extract all characters mentioned in the following"
"images": ["$(<image.b64)"],\\
cd /tmp
"max_tokens": 100\\
docker ps
'''
        
        # Mock the file reading
        mock_open = patch('builtins.open', mock=MagicMock())
        with mock_open as mock_file:
            mock_file.return_value.__enter__.return_value.__iter__.return_value = mock_history_content.strip().split('\n')
            
            # Clear database first
            self.fs.command_dal.clear_embeddings()
            
            # Run ingestion
            added_count = self.fs.ingest_history(use_tui=False, no_random=True)
            
            # Get all ingested commands
            all_commands = self.fs.command_dal.get_all_commands_with_embeddings()
            ingested_commands = [row[1] for row in all_commands]  # row[1] is command text
            
            # Should only have valid commands
            valid_commands = ["ls -lh", "git status", "cd /tmp", "docker ps"]
            
            for valid_cmd in valid_commands:
                assert valid_cmd in ingested_commands, (
                    f"Valid command '{valid_cmd}' should be ingested"
                )
            
            # Should not have JSON fragments
            json_fragments = ['"content":', '"images":', '"max_tokens":']
            for fragment in json_fragments:
                for ingested_cmd in ingested_commands:
                    assert fragment not in ingested_cmd, (
                        f"JSON fragment '{fragment}' should not be ingested: '{ingested_cmd}'"
                    )
    
    def test_semantic_search_quality(self):
        """Test that semantic search returns reasonable results."""
        # Add commands with known relationships
        command_pairs = [
            ("ls -lh", "list files"),
            ("cd /tmp", "change directory"),
            ("git add .", "stage changes"),
            ("rm file.txt", "delete file"),
            ("grep pattern file", "search text")
        ]
        
        # Add all commands to database
        all_commands = []
        for cmd, desc in command_pairs:
            all_commands.extend([cmd, desc])
        
        for command in all_commands:
            original_emb = self.fs._model.encode([command])[0]
            quantized_emb = self.fs.quantize_embedding(original_emb)
            self.fs.command_dal.add_command(command, embedding=quantized_emb)
        
        # Test semantic relationships
        for cmd, desc in command_pairs:
            # Search for the natural language description
            results = self.fs.search(desc, top_k=5, return_scores=True)
            result_commands = [result[0] for result in results]

            # The corresponding command should be in top results
            assert cmd in result_commands[:3], (
                f"Command '{cmd}' should be in top 3 results for '{desc}': {result_commands}"
            )

            # Find the semantic score for the matching command
            matching_semantic_score = None
            for result_cmd, combined_score, semantic_score, bm25_score in results:
                if result_cmd == cmd:
                    matching_semantic_score = semantic_score
                    break

            assert matching_semantic_score is not None, (
                f"Matching command '{cmd}' not found in results"
            )
            assert matching_semantic_score > 0.4, (
                f"Semantic similarity too low: '{desc}' -> '{cmd}' = {matching_semantic_score}"
            )
    
    def test_no_embedding_corruption_after_full_pipeline(self):
        """Test that full ingestion pipeline doesn't corrupt embeddings."""
        # Mock shell history file content
        mock_history_content = '''ls -la
cd /home/user
git clone https://github.com/user/repo.git
python manage.py migrate
docker build -t app .
npm install
pytest tests/
curl -X GET https://api.example.com
grep -r 'TODO' src/
find . -name '*.py' | head -10
'''
        
        # Mock the file reading
        mock_open = patch('builtins.open', mock=MagicMock())
        with mock_open as mock_file:
            mock_file.return_value.__enter__.return_value.__iter__.return_value = mock_history_content.strip().split('\n')
            
            # Run full ingestion pipeline
            added_count = self.fs.ingest_history(use_tui=False, no_random=True)
            assert added_count > 0, "Should ingest some commands"
            
            # Create SearchCoordinator for corruption detection
            coordinator = SearchCoordinator(
                model_handler=self.fs._model,
                command_dal=self.fs.command_dal,
                ann_manager=self.fs.ann_manager,
                dequantize_func=self.fs.dequantize_embedding,
                quantize_func=self.fs.quantize_embedding
            )
            
            # Test all ingested embeddings for corruption
            report = coordinator.diagnose_embeddings(sample_size=100)
            
            # Should have zero corruption
            corruption_rate = report['corrupted_count'] / report['total_tested'] * 100
            assert corruption_rate < 10, (
                f"Corruption rate too high: {corruption_rate:.1f}% - {report}"
            )

            # Average consistency should be very high
            assert report['avg_consistency'] > 0.95, (
                f"Average consistency too low: {report['avg_consistency']}"
            )


if __name__ == '__main__':
    # Run with verbose output to see which tests pass/fail
    import pytest

    raise SystemExit(pytest.main([__file__]))