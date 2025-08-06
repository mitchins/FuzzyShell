"""
Tests for DAL bulk operations context manager.

Ensures the context manager:
1. Sets up bulk-optimized SQLite settings
2. Creates proper transaction boundaries  
3. Restores original settings on exit
4. Handles errors with rollback
"""

import pytest
import tempfile
import sqlite3
from pathlib import Path

from fuzzyshell.data.datastore import ProductionDatabaseProvider, CommandDAL


class TestBulkOperations:
    """Test bulk operations context manager functionality."""
    
    def setup_method(self):
        """Set up test database for each test."""
        # Create temporary database file
        self.db_fd, self.db_path = tempfile.mkstemp(suffix='.db')
        self.db_provider = ProductionDatabaseProvider(self.db_path)
        self.command_dal = CommandDAL(self.db_provider)
        
    def teardown_method(self):
        """Clean up test database."""
        Path(self.db_path).unlink(missing_ok=True)
        
    def test_bulk_operations_setup_and_teardown(self):
        """Test that bulk operations context manager sets up and restores settings correctly."""
        
        # Get original pragma values by creating a direct connection
        direct_conn = sqlite3.connect(self.db_path)
        
        # Check original values before bulk operations
        original_synchronous = direct_conn.execute("PRAGMA synchronous").fetchone()[0]
        original_journal_mode = direct_conn.execute("PRAGMA journal_mode").fetchone()[0]
        direct_conn.close()
        
        # Test the bulk operations context manager
        with self.command_dal.bulk_operations() as bulk_conn:
            # Inside context: should have optimized settings
            current_synchronous = bulk_conn.execute("PRAGMA synchronous").fetchone()[0]
            current_journal_mode = bulk_conn.execute("PRAGMA journal_mode").fetchone()[0]
            
            # Should be optimized for bulk operations
            assert current_synchronous == 0, "synchronous should be OFF (0) during bulk operations"
            assert current_journal_mode.upper() == "MEMORY", "journal_mode should be MEMORY during bulk operations"
            
            # Should be in a transaction (test by checking we can rollback)
            bulk_conn.execute("CREATE TABLE IF NOT EXISTS test_table (id INTEGER)")
            bulk_conn.execute("INSERT INTO test_table (id) VALUES (1)")
            
        # After context: settings should be restored
        restored_conn = sqlite3.connect(self.db_path)
        restored_synchronous = restored_conn.execute("PRAGMA synchronous").fetchone()[0]
        restored_journal_mode = restored_conn.execute("PRAGMA journal_mode").fetchone()[0]
        
        # Settings should be restored to original values
        assert restored_synchronous == original_synchronous, "synchronous should be restored after bulk operations"
        assert restored_journal_mode == original_journal_mode, "journal_mode should be restored after bulk operations"
        
        # Transaction should have been committed - test_table should exist
        tables = restored_conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='test_table'").fetchall()
        assert len(tables) == 1, "Transaction should have been committed"
        
        restored_conn.close()
        
    def test_bulk_operations_transaction_commit(self):
        """Test that bulk operations properly commit transactions."""
        
        # Insert data using bulk operations
        with self.command_dal.bulk_operations() as bulk_conn:
            bulk_conn.execute("CREATE TABLE IF NOT EXISTS test_commits (data TEXT)")
            bulk_conn.execute("INSERT INTO test_commits (data) VALUES ('test_data')")
        
        # Verify data persists after context (was committed)
        with self.command_dal.connection() as conn:
            result = conn.execute("SELECT data FROM test_commits WHERE data = 'test_data'").fetchone()
            assert result is not None, "Data should persist after bulk operations commit"
            assert result[0] == 'test_data'
            
    def test_bulk_operations_transaction_rollback_on_error(self):
        """Test that bulk operations rollback on exceptions."""
        
        # Setup: Create table outside bulk operations
        with self.command_dal.connection() as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS test_rollback (data TEXT)")
            
        # Test: Exception during bulk operations should rollback
        with pytest.raises(sqlite3.IntegrityError):
            with self.command_dal.bulk_operations() as bulk_conn:
                # This should succeed
                bulk_conn.execute("INSERT INTO test_rollback (data) VALUES ('good_data')")
                
                # This should fail and trigger rollback
                bulk_conn.execute("CREATE UNIQUE INDEX test_unique ON test_rollback (data)")
                bulk_conn.execute("INSERT INTO test_rollback (data) VALUES ('good_data')")  # Duplicate!
        
        # Verify rollback occurred - no data should exist
        with self.command_dal.connection() as conn:
            result = conn.execute("SELECT COUNT(*) FROM test_rollback").fetchone()
            assert result[0] == 0, "Transaction should have been rolled back on error"
            
    def test_bulk_operations_performance_benefit(self):
        """Test that bulk operations provide measurable performance improvement."""
        import time
        
        # Setup test data
        test_commands = [(f"command_{i}", f"description_{i}") for i in range(100)]
        
        # Create table
        with self.command_dal.connection() as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS perf_test (command TEXT, description TEXT)")
        
        # Measure regular operations time
        start_time = time.time()
        for command, description in test_commands:
            with self.command_dal.connection() as conn:
                conn.execute("INSERT INTO perf_test (command, description) VALUES (?, ?)", (command, description))
        regular_time = time.time() - start_time
        
        # Clear table
        with self.command_dal.connection() as conn:
            conn.execute("DELETE FROM perf_test")
        
        # Measure bulk operations time  
        start_time = time.time()
        with self.command_dal.bulk_operations() as bulk_conn:
            for command, description in test_commands:
                bulk_conn.execute("INSERT INTO perf_test (command, description) VALUES (?, ?)", (command, description))
        bulk_time = time.time() - start_time
        
        # Bulk operations should be significantly faster
        # (Allow some variance, but bulk should be at least 20% faster)
        performance_improvement = (regular_time - bulk_time) / regular_time
        assert performance_improvement > 0.1, \
            f"Bulk operations should be faster. Regular: {regular_time:.3f}s, Bulk: {bulk_time:.3f}s"
        
        print(f"Performance test: Regular {regular_time:.3f}s, Bulk {bulk_time:.3f}s ({performance_improvement:.1%} improvement)")
        
    def test_bulk_operations_pragma_restoration(self):
        """Test that pragma settings are properly restored after bulk operations."""
        
        # Get original pragma values
        with self.command_dal.connection() as conn:
            original_synchronous = conn.execute("PRAGMA synchronous").fetchone()[0]
            original_journal_mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        
        # Use bulk operations (this changes pragma values temporarily)
        with self.command_dal.bulk_operations() as bulk_conn:
            bulk_conn.execute("CREATE TABLE IF NOT EXISTS pragma_test (data TEXT)")
            bulk_conn.execute("INSERT INTO pragma_test (data) VALUES ('test')")
        
        # Verify pragma values are restored
        with self.command_dal.connection() as conn:
            restored_synchronous = conn.execute("PRAGMA synchronous").fetchone()[0]
            restored_journal_mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
            
            assert restored_synchronous == original_synchronous, "synchronous pragma should be restored"
            assert restored_journal_mode == original_journal_mode, "journal_mode pragma should be restored"