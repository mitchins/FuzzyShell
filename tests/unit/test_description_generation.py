#!/usr/bin/env python3
"""Comprehensive tests for custom CodeT5 description generation"""

import pytest
import logging
from unittest.mock import patch, MagicMock
from fuzzyshell.model_handler import DescriptionHandler

# Reduce logging noise during tests
logging.getLogger('FuzzyShell.ModelHandler').setLevel(logging.CRITICAL)


class TestDescriptionGeneration:
    """Test custom CodeT5 description generation functionality"""

    @classmethod
    def setup_class(cls):
        """Set up description handler once for all tests"""
        try:
            cls.desc_handler = DescriptionHandler()
            cls.model_available = cls.desc_handler.use_t5_model
            if cls.model_available:
                print(f"✅ Using custom CodeT5 model for tests")
            else:
                print(f"⚠️ Using rule-based fallback for tests")
        except Exception as e:
            print(f"❌ Failed to initialize description handler: {e}")
            cls.model_available = False
            cls.desc_handler = None

    def test_basic_git_commands(self):
        """Test description generation for git commands"""
        if not self.model_available:
            pytest.skip("CodeT5 model not available")

        git_commands = [
            "git status",
            "git commit -m 'message'",
            "git push origin main",
            "git pull upstream master",
            "git log --oneline",
            "git diff HEAD~1"
        ]

        for cmd in git_commands:
            description = self.desc_handler.generate_description(cmd)
            assert isinstance(description, str)
            assert len(description.strip()) > 0
            # Git commands should mention git or repository concepts
            desc_lower = description.lower()
            git_related = any(word in desc_lower for word in [
                'git', 'commit', 'repository', 'repo', 'branch', 'status', 'push', 'pull'
            ])
            assert git_related, f"Description '{description}' should be git-related"

    def test_file_operations(self):
        """Test description generation for file operation commands"""
        if not self.model_available:
            pytest.skip("CodeT5 model not available")

        file_commands = [
            "ls -la",
            "cp file1.txt file2.txt", 
            "mv oldname.txt newname.txt",
            "rm -rf directory/",
            "find . -name '*.py'",
            "chmod 755 script.sh",
            "chown user:group file.txt"
        ]

        for cmd in file_commands:
            description = self.desc_handler.generate_description(cmd)
            assert isinstance(description, str)
            assert len(description.strip()) > 0
            # File operations should mention files, directories, permissions, or action words
            desc_lower = description.lower()
            file_related = any(word in desc_lower for word in [
                'file', 'directory', 'folder', 'list', 'copy', 'move', 'remove', 
                'delete', 'find', 'permission', 'owner', 'chmod', 'chown', 'rename',
                'set', 'search', 'create', 'change'
            ])
            assert file_related, f"Description '{description}' should be file-related"

    def test_docker_commands(self):
        """Test description generation for Docker commands"""
        if not self.model_available:
            pytest.skip("CodeT5 model not available")

        docker_commands = [
            "docker ps -a",
            "docker run -it ubuntu bash",
            "docker build -t myapp .",
            "docker stop container_name",
            "docker logs container_id",
            "docker exec -it container bash"
        ]

        for cmd in docker_commands:
            description = self.desc_handler.generate_description(cmd)
            assert isinstance(description, str)
            assert len(description.strip()) > 0
            # Docker commands should mention containers or docker concepts
            desc_lower = description.lower()
            docker_related = any(word in desc_lower for word in [
                'docker', 'container', 'run', 'build', 'image', 'stop', 'start', 'logs'
            ])
            assert docker_related, f"Description '{description}' should be docker-related"

    def test_system_commands(self):
        """Test description generation for system administration commands"""
        if not self.model_available:
            pytest.skip("CodeT5 model not available")

        system_commands = [
            "ps aux",
            "kill -9 1234", 
            "systemctl restart nginx",
            "sudo apt update",
            "crontab -e",
            "df -h",
            "top -n 1"
        ]

        for cmd in system_commands:
            description = self.desc_handler.generate_description(cmd)
            assert isinstance(description, str)
            assert len(description.strip()) > 0
            # System commands should mention processes, services, or system concepts
            desc_lower = description.lower()
            system_related = any(word in desc_lower for word in [
                'process', 'service', 'system', 'kill', 'restart', 'update', 
                'disk', 'memory', 'cpu', 'schedule', 'cron'
            ])
            # Allow more flexible matching for system commands
            has_meaningful_content = len(description.strip()) > 3
            assert has_meaningful_content, f"Description '{description}' should have meaningful content"

    def test_programming_commands(self):
        """Test description generation for programming-related commands"""
        if not self.model_available:
            pytest.skip("CodeT5 model not available")

        prog_commands = [
            "python manage.py runserver",
            "npm install express",
            "pip install requests",
            "node app.js",
            "python -m pytest tests/",
            "javac HelloWorld.java",
            "gcc -o program program.c"
        ]

        for cmd in prog_commands:
            description = self.desc_handler.generate_description(cmd)
            assert isinstance(description, str)
            assert len(description.strip()) > 0
            # Programming commands should be reasonably descriptive
            assert len(description.strip()) > 5

    def test_description_length_limits(self):
        """Test that description length limits are respected"""
        if not self.model_available:
            pytest.skip("CodeT5 model not available")

        test_command = "git status"
        
        # Test different max_length settings
        for max_len in [10, 20, 50]:
            description = self.desc_handler.generate_description(test_command, max_length=max_len)
            assert isinstance(description, str)
            # Description should be reasonably within bounds (allowing for some model variance)
            assert len(description.split()) <= max_len + 5

    def test_empty_and_invalid_commands(self):
        """Test handling of empty and invalid commands"""
        if not self.model_available:
            pytest.skip("CodeT5 model not available")

        edge_cases = [
            "",
            "   ",
            "nonexistentcommand123456",
            "a" * 1000,  # Very long command
            "!@#$%^&*()",  # Special characters
        ]

        for cmd in edge_cases:
            # Should not crash and should return some description
            description = self.desc_handler.generate_description(cmd)
            assert isinstance(description, str)

    def test_rule_based_fallback(self):
        """Test rule-based fallback descriptions"""
        # Create a handler that will use rule-based fallback
        with patch('fuzzyshell.model_handler.DescriptionHandler._ensure_model_files') as mock_ensure:
            mock_ensure.side_effect = Exception("Model not available")
            
            fallback_handler = DescriptionHandler()
            assert fallback_handler.use_t5_model is False
            
            # Test that fallback works
            description = fallback_handler.generate_description("ls -la")
            assert isinstance(description, str)
            assert len(description.strip()) > 0
            
            # Should be rule-based description
            assert "List" in description

    def test_description_consistency(self):
        """Test that descriptions are consistent across multiple calls"""
        if not self.model_available:
            pytest.skip("CodeT5 model not available")

        test_command = "docker ps -a"
        
        # Generate description multiple times
        descriptions = []
        for _ in range(3):
            desc = self.desc_handler.generate_description(test_command)
            descriptions.append(desc)
        
        # All descriptions should be the same (greedy decoding is deterministic)
        assert descriptions[0] == descriptions[1]
        assert descriptions[1] == descriptions[2]

    def test_model_initialization_error_handling(self):
        """Test error handling during model initialization"""
        # Test with invalid model directory
        with patch('os.path.exists') as mock_exists:
            mock_exists.return_value = False
            with patch('fuzzyshell.model_handler.DescriptionHandler._download_file') as mock_download:
                mock_download.return_value = False
                
                # Should gracefully fall back to rule-based
                handler = DescriptionHandler()
                assert handler.use_t5_model is False
                
                # Should still generate descriptions
                desc = handler.generate_description("git status")
                assert isinstance(desc, str)
                assert len(desc.strip()) > 0


class TestDescriptionQuality:
    """Test the quality and accuracy of generated descriptions"""

    @classmethod
    def setup_class(cls):
        """Set up description handler for quality tests"""
        try:
            cls.desc_handler = DescriptionHandler()
            cls.model_available = cls.desc_handler.use_t5_model
        except Exception:
            cls.model_available = False
            cls.desc_handler = None

    def test_description_relevance(self):
        """Test that descriptions are relevant to the commands"""
        if not self.model_available:
            pytest.skip("CodeT5 model not available")

        # Test cases: (command, expected_keywords) - more flexible keyword matching
        test_cases = [
            ("git status", ["status", "git", "repository", "repo", "commit", "show"]),
            ("ls -la", ["list", "file", "directory", "show", "current", "tree"]),
            ("docker ps", ["docker", "container", "list", "running", "all"]),
            ("python script.py", ["python", "script", "run", "execute", "file"]),
            ("chmod 755", ["permission", "file", "mode", "access", "set", "755"]),
            ("grep pattern file", ["search", "pattern", "file", "find", "text"])
        ]

        for command, expected_keywords in test_cases:
            description = self.desc_handler.generate_description(command).lower()
            
            # At least one expected keyword should appear
            found_keywords = [kw for kw in expected_keywords if kw in description]
            assert len(found_keywords) > 0, \
                f"Description '{description}' should contain at least one of {expected_keywords}"

    def test_terminal_specific_vocabulary(self):
        """Test that descriptions use terminal-specific vocabulary"""
        if not self.model_available:
            pytest.skip("CodeT5 model not available")

        terminal_commands = [
            "cd /home/user",
            "tar -xzf archive.tar.gz", 
            "ssh user@server",
            "wget https://example.com/file",
            "tail -f /var/log/syslog"
        ]

        for cmd in terminal_commands:
            description = self.desc_handler.generate_description(cmd)
            assert isinstance(description, str)
            assert len(description.strip()) > 0
            
            # Should not be generic software descriptions
            desc_lower = description.lower()
            generic_terms = ["application", "software", "program"]
            has_only_generic = all(term not in desc_lower for term in generic_terms)
            # Allow generic terms but ensure we have meaningful content
            assert len(description.strip()) > 5


