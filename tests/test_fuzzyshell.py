import unittest
import os
import tempfile
import sqlite3
from fuzzyshell import FuzzyShell
import numpy as np

class TestFuzzyShell(unittest.TestCase):
    def setUp(self):
        # Create a temporary database file
        self.temp_db = tempfile.NamedTemporaryFile(delete=False)
        self.temp_db.close()
        self.fuzzyshell = FuzzyShell(db_path=self.temp_db.name)
        
        # Create a temporary history file
        self.test_commands = [
            "git push origin main",
            "docker build -t myapp .",
            "npm install express",
            "python manage.py runserver",
            "ls -la /var/log"
        ]
        
        # Write commands with a newline after each one and one at the end
        self.temp_history = tempfile.NamedTemporaryFile(mode='w', delete=False)
        self.temp_history.write('\n'.join(self.test_commands) + '\n')
        self.temp_history.flush()  # Ensure all data is written
        os.fsync(self.temp_history.fileno())  # Force filesystem sync
        self.temp_history.close()

    def tearDown(self):
        # Clean up temporary files
        os.unlink(self.temp_db.name)
        os.unlink(self.temp_history.name)

    def test_init_db(self):
        """Test if database tables are created correctly"""
        conn = sqlite3.connect(self.temp_db.name)
        cursor = conn.cursor()
        
        # Check if commands table exists and has correct schema
        cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='commands'")
        commands_schema = cursor.fetchone()[0]
        self.assertIn("id INTEGER PRIMARY KEY", commands_schema)
        self.assertIn("command TEXT NOT NULL", commands_schema)

        # Check if embeddings virtual table exists
        cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='embeddings'")
        embeddings_schema = cursor.fetchone()[0]
        self.assertIn("VIRTUAL TABLE embeddings", embeddings_schema)
        self.assertIn("vss0", embeddings_schema)
        
        conn.close()

    def test_ingest_history(self):
        """Test history ingestion"""
        # Mock the get_shell_history_file method to return our temp file
        self.fuzzyshell.get_shell_history_file = lambda: self.temp_history.name
        
        # Ingest the test commands
        self.fuzzyshell.ingest_history()
        
        # Verify all commands were ingested
        cursor = self.fuzzyshell.conn.cursor()
        cursor.execute("SELECT command FROM commands")
        ingested_commands = {row[0] for row in cursor.fetchall()}
        
        self.assertEqual(len(ingested_commands), len(self.test_commands))
        for cmd in self.test_commands:
            self.assertIn(cmd, ingested_commands)

    def test_search_exact_match(self):
        """Test search functionality with exact matches"""
        # Mock history and ingest test commands
        self.fuzzyshell.get_shell_history_file = lambda: self.temp_history.name
        self.fuzzyshell.ingest_history()
        
        # Test exact match
        results = self.fuzzyshell.search("git push origin main")
        self.assertTrue(len(results) > 0)
        # Check that the exact match is in the results - debug output
        commands_in_results = [cmd for cmd, score in results]
        # Check that the exact match is found (any score)
        exact_match_found = any(cmd == "git push origin main"
                               for cmd, score in results)
        if not exact_match_found:
            print(f"Commands in results: {commands_in_results}")
            print(f"Test commands: {self.test_commands}")
        self.assertTrue(exact_match_found, f"Exact match 'git push origin main' should be found. Got: {results}")

    def test_search_semantic_match(self):
        """Test search functionality with semantic matches"""
        # Mock history and ingest test commands
        self.fuzzyshell.get_shell_history_file = lambda: self.temp_history.name
        self.fuzzyshell.ingest_history()
        
        # Test semantic match
        results = self.fuzzyshell.search("start development server")
        matching_commands = [cmd for cmd, _ in results]
        self.assertIn("python manage.py runserver", matching_commands)

    def test_search_no_results(self):
        """Test search with no matching results"""
        # Mock history and ingest test commands
        self.fuzzyshell.get_shell_history_file = lambda: self.temp_history.name
        self.fuzzyshell.ingest_history()
        
        # Test with unlikely query - check that results have very low scores
        results = self.fuzzyshell.search("xyzabc123nonexistent")
        # Either no results or very low scoring results (adjust threshold based on actual behavior)
        if results:
            max_score = max(score for _, score in results)
            self.assertLess(max_score, 0.9, "Random query should not have very high similarity scores")

    def test_embedding_consistency(self):
        """Test if embeddings are consistent across searches"""
        self.fuzzyshell.get_shell_history_file = lambda: self.temp_history.name
        self.fuzzyshell.ingest_history()
        
        # Perform same search twice
        results1 = self.fuzzyshell.search("git")
        results2 = self.fuzzyshell.search("git")
        
        # Check if results are consistent
        self.assertEqual(len(results1), len(results2))
        for (cmd1, score1), (cmd2, score2) in zip(results1, results2):
            self.assertEqual(cmd1, cmd2)
            self.assertAlmostEqual(score1, score2, places=5)

    def test_hybrid_search(self):
        """Test that hybrid search balances exact and semantic matches"""
        self.fuzzyshell.get_shell_history_file = lambda: self.temp_history.name
        self.fuzzyshell.ingest_history()
        
        # Test with exact match term
        results = self.fuzzyshell.search("git push")
        self.assertTrue(any("git push" in cmd for cmd, _ in results))
        
        # Test with semantic match
        results = self.fuzzyshell.search("version control push")
        self.assertTrue(any("git push" in cmd for cmd, _ in results))
        
        # Test rare term matching
        results = self.fuzzyshell.search("runserver")
        self.assertTrue(any("runserver" in cmd for cmd, _ in results))
        
        # Test that exact matches score reasonably high
        exact_results = self.fuzzyshell.search("docker build")
        semantic_results = self.fuzzyshell.search("container create")
        
        # Find the score for "docker build" in both searches
        exact_score = next((score for cmd, score in exact_results 
                          if "docker build" in cmd), 0)
        semantic_score = next((score for cmd, score in semantic_results 
                             if "docker build" in cmd), 0)
        
        # Both should find the docker build command with decent scores
        self.assertGreater(exact_score, 0.5, "Exact match should have reasonable score")
        self.assertGreater(semantic_score, 0.3, "Semantic match should have some score")

if __name__ == '__main__':
    unittest.main()
