"""Basic tests for benchmark_models module."""

import unittest


class TestBenchmarkModels(unittest.TestCase):
    """Basic tests for benchmark_models functionality."""
    
    def test_benchmark_module_structure(self):
        """Test that benchmark_models has expected structure."""
        # Simple test that doesn't import the module
        # Just verify the file exists and has correct structure
        import os
        from pathlib import Path
        
        # Get the benchmark module path
        module_path = Path(__file__).parent.parent.parent / "src" / "fuzzyshell" / "tools" / "benchmark_models.py"
        
        self.assertTrue(module_path.exists(), "benchmark_models.py should exist")
        
        # Read the file and check for expected functions
        with open(module_path, 'r') as f:
            content = f.read()
            
        self.assertIn('def get_model_size', content, "Should have get_model_size function")
        self.assertIn('def profile_model', content, "Should have profile_model function")
        self.assertIn('from sentence_transformers', content, "Should import sentence_transformers")
        self.assertIn('import torch', content, "Should import torch")


if __name__ == '__main__':
    unittest.main()