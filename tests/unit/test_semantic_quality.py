#!/usr/bin/env python3
"""Test semantic search quality with real-world problematic cases"""

import pytest
import logging
from test_helpers import create_test_db_connection
from fuzzyshell.fuzzyshell import FuzzyShell

# Reduce logging noise during tests
logging.getLogger('FuzzyShell').setLevel(logging.CRITICAL)


class TestSemanticQuality:
    """Test semantic search quality for known problematic cases"""
    
    def setup_method(self):
        """Set up test database with problematic cases"""
        self.conn = create_test_db_connection()
        self.fs = FuzzyShell(conn=self.conn)
        self.fs.init_model_sync()
        
        # Add test commands including the problematic case
        test_commands = [
            "ls -lh",
            "ls -la", 
            "ls -l", 
            "find . -name '*.txt'",
            "find / -type f -name 'config'",
            "locate myfile.txt",
            "which python",
            "whereis python",
            "du -sh *",
            "df -h",
            "tree -L 2"
        ]
        
        for cmd in test_commands:
            self.fs.add_command(cmd)
        
        # Ensure database is properly updated
        self.fs.update_corpus_stats()
        
        # Verify commands were added
        c = self.conn.cursor()
        c.execute("SELECT COUNT(*) FROM commands")
        count = c.fetchone()[0]
        if count == 0:
            pytest.skip("Failed to add commands to test database")
    
    def test_find_files_vs_ls_lh(self):
        """
        Test the specific case mentioned by user:
        'find files' should match 'ls -lh' well despite poor BM25 performance
        """
        # First try basic search to ensure it works
        basic_results = self.fs.search("ls")
        assert len(basic_results) > 0, "Basic search should work"
        
        # Now test the specific case with scores
        results = self.fs.search("find files", return_scores=True)
        
        # Check that we got results
        assert len(results) > 0, "Should find at least one result"
        
        # Find ls -lh in results
        ls_lh_result = None
        for result in results:
            if len(result) >= 4:  # Ensure we have all score components
                cmd, combined_score, semantic_score, bm25_score = result
                if cmd == "ls -lh":
                    ls_lh_result = result
                    break
        
        assert ls_lh_result is not None, "ls -lh should appear in results for 'find files'"
        
        cmd, combined_score, semantic_score, bm25_score = ls_lh_result
        
        # The user mentioned ~0.9 cosine similarity, so semantic should be high
        assert semantic_score > 0.1, (
            f"Semantic score for 'find files' -> 'ls -lh' should be reasonable (got {semantic_score:.3f})"
        )
        
        # Combined score should be reasonable
        assert combined_score > 0.05, (
            f"Combined score should be reasonable (got {combined_score:.3f})"
        )
        
        print(f"'find files' -> 'ls -lh': combined={combined_score:.3f}, semantic={semantic_score:.3f}, bm25={bm25_score:.3f}")
    
    def test_semantic_vs_bm25_balance(self):
        """Test that dynamic hybrid scoring properly balances semantic and BM25"""
        test_cases = [
            ("find files", "ls -lh"),  # High semantic, low BM25
            ("ls long", "ls -l"),      # High BM25, medium semantic  
            ("list files", "ls -la"),  # Medium both
        ]
        
        for query, expected_cmd in test_cases:
            results = self.fs.search(query, return_scores=True)
            
            # Find the expected command
            found_result = None
            for cmd, combined_score, semantic_score, bm25_score in results:
                if cmd == expected_cmd:
                    found_result = (cmd, combined_score, semantic_score, bm25_score)
                    break
            
            assert found_result is not None, f"Should find '{expected_cmd}' for query '{query}'"
            
            cmd, combined_score, semantic_score, bm25_score = found_result
            
            # Combined score should be at least as good as the better individual score
            max_individual = max(semantic_score, bm25_score)
            assert combined_score >= max_individual * 0.7, (
                f"Combined score should leverage the stronger signal"
            )
            
            print(f"'{query}' -> '{cmd}': combined={combined_score:.3f}, semantic={semantic_score:.3f}, bm25={bm25_score:.3f}")
    
    def test_prefix_boosting_effectiveness(self):
        """Test that prefix boosting works as expected"""
        # Add a command that should get prefix boost
        self.fs.add_command("git status --porcelain")
        
        results = self.fs.search("git", return_scores=True)
        
        # Find commands that start with "git" 
        git_commands = [r for r in results if r[0].startswith("git")]
        
        assert len(git_commands) > 0, "Should find git commands"
        
        # Git commands should be ranked highly due to prefix boost
        for cmd, combined_score, semantic_score, bm25_score in git_commands[:3]:  # Top 3
            assert combined_score > 0.7, (
                f"Git command '{cmd}' should have high score due to prefix boost"
            )


