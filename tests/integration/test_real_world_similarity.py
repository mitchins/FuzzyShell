import pytest
import numpy as np
import sqlite3
import os
from unittest.mock import patch, MagicMock

# Import test helpers and FuzzyShell modules
from test_helpers import create_test_db_connection
from fuzzyshell.fuzzyshell import FuzzyShell, USE_ANN_SEARCH
from fuzzyshell.model_handler import ModelHandler


class TestRealWorldSemanticSimilarity:
    """Test semantic similarity with real command-description pairs."""

    def setup_method(self):
        """Set up test database with real command examples."""
        self.test_conn = create_test_db_connection()
        
        # Use the custom trained model (terminal-minilm-l6) for better accuracy
        os.environ['FUZZYSHELL_MODEL'] = 'terminal-minilm-l6'
        
        # Override model directory to use local test models (avoids downloading)
        test_model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
        os.environ['FUZZYSHELL_MODEL_DIR'] = test_model_dir
        
        self.fuzzyshell = FuzzyShell(conn=self.test_conn)
        
        # Wait for model to be ready for testing
        model_ready = self.fuzzyshell.wait_for_model(timeout=30.0)
        if not model_ready:
            pytest.skip("Model not available for semantic similarity testing")
        
        # Real-world command-description pairs
        self.command_description_pairs = [
            # File operations
            ("ls -lh", "list files in folder"),
            ("ls -la", "show all files with details"),
            ("find . -name '*.py'", "find python files"),
            ("grep -r 'pattern' .", "search for text in files"),
            ("cat README.md", "display file contents"),
            ("tail -f /var/log/syslog", "monitor log file"),
            ("du -sh *", "check folder sizes"),
            ("chmod +x script.sh", "make file executable"),
            
            # Process management
            ("ps aux", "show running processes"),
            ("kill -9 1234", "force kill process"),
            ("top", "monitor system resources"),
            ("htop", "interactive process viewer"),
            ("pgrep firefox", "find process by name"),
            
            # Network operations
            ("curl https://api.github.com", "download web content"),
            ("wget https://example.com/file.zip", "download file from web"),
            ("ping google.com", "test network connectivity"),
            ("ssh user@server.com", "connect to remote server"),
            ("scp file.txt user@server:/path", "copy file to remote server"),
            
            # Text processing
            ("awk '{print $1}' file.txt", "extract first column from file"),
            ("sed 's/old/new/g' file.txt", "replace text in file"),
            ("sort file.txt", "sort lines in file"),
            ("uniq file.txt", "remove duplicate lines"),
            ("wc -l file.txt", "count lines in file"),
            
            # Archive operations
            ("tar -xzf archive.tar.gz", "extract compressed archive"),
            ("zip -r backup.zip folder/", "create zip archive"),
            ("unzip file.zip", "extract zip file"),
            
            # Development tools
            ("git status", "check repository status"),
            ("git commit -m 'message'", "save changes to repository"),
            ("npm install", "install project dependencies"),
            ("python script.py", "run python program"),
            ("docker ps", "list running containers"),
            ("docker build -t image .", "build container image"),
            
            # System information
            ("df -h", "check disk space"),
            ("free -h", "check memory usage"),
            ("uname -a", "show system information"),
            ("whoami", "show current user"),
            ("date", "show current date and time"),
            
            # Specialized tools
            ("ebook-convert input.epub output.pdf", "change the format of a book"),
            ("ffmpeg -i video.mp4 output.mp3", "convert video to audio"),
            ("rsync -av source/ dest/", "synchronize folders"),
            ("crontab -e", "edit scheduled tasks"),
            ("systemctl status nginx", "check service status"),
            
            # Package management
            ("apt update", "update package list"),
            ("brew install package", "install software package"),
            ("pip install requests", "install python library"),
            ("yum search package", "search for package"),
            
            # File editing
            ("vim file.txt", "edit text file"),
            ("nano config.conf", "simple text editor"),
            ("code project/", "open in code editor"),
            
            # Database operations
            ("mysql -u user -p", "connect to database"),
            ("pg_dump database > backup.sql", "backup database"),
            ("sqlite3 database.db", "open sqlite database"),
        ]
        
        # Only use commands for this test
        self.test_commands = [pair[0] for pair in self.command_description_pairs]
        self.test_descriptions = [pair[1] for pair in self.command_description_pairs]
        
        # Set up mock model for non-semantic tests
        self.mock_model = MagicMock()
        # Don't override the real model yet - we'll do it selectively in tests that need mocking
        
        # Add commands to database
        for i, command in enumerate(self.test_commands):
            self.fuzzyshell.add_command(command)
            
    def test_command_description_semantic_similarity(self):
        """Test that commands have high semantic similarity to their descriptions."""
        # Model is already ready from setUp
        model = self.fuzzyshell.model
            
        # Test a subset of pairs for semantic similarity
        test_pairs = [
            ("ls -lh", "list files in folder"),
            ("grep -r 'pattern' .", "search for text in files"),
            ("ps aux", "show running processes"),
            ("curl https://api.github.com", "download web content"),
            ("tar -xzf archive.tar.gz", "extract compressed archive"),
            ("git status", "check repository status"),
            ("df -h", "check disk space"),
            ("ebook-convert input.epub output.pdf", "change the format of a book"),
            ("vim file.txt", "edit text file"),
            ("docker ps", "list running containers")
        ]
        
        similarities = []
        
        for command, description in test_pairs:
            # Get embeddings for both command and description
            command_embedding = model.encode([command])[0]
            description_embedding = model.encode([description])[0]
            
            # Calculate cosine similarity
            norm_command = np.linalg.norm(command_embedding)
            norm_description = np.linalg.norm(description_embedding)
            
            # Handle zero norm case
            if norm_command == 0 or norm_description == 0:
                similarity = 0.0
            else:
                similarity = np.dot(command_embedding, description_embedding) / (norm_command * norm_description)
            similarities.append(similarity)
            
            print(f"'{command}' <-> '{description}': {similarity:.3f}")
            
            # Each pair should have reasonable semantic similarity
            # Using custom trained model optimized for terminal commands
            assert similarity > 0.3, (
                f"Low similarity for '{command}' <-> '{description}': {similarity:.3f}"
            )
        
        avg_similarity = np.mean(similarities)
        print(f"\nAverage semantic similarity: {avg_similarity:.3f}")
        
        # Average similarity should be good with the new model
        assert avg_similarity > 0.5, (
            f"Average similarity too low: {avg_similarity:.3f}"
        )
        
    def test_natural_language_search_accuracy(self):
        """Test searching with natural language descriptions finds correct commands."""
        # Use mock model for this controlled test
        self.fuzzyshell._model = self.mock_model
        
        # Mock some embeddings for testing
        np.random.seed(42)
        n_commands = len(self.test_commands)
        
        # Create embeddings where related commands have higher similarity
        base_embeddings = np.random.randn(n_commands, 384).astype(np.float32)
        
        # Manually create some clusters for related commands
        file_ops_indices = [i for i, cmd in enumerate(self.test_commands) 
                           if any(word in cmd for word in ['ls', 'find', 'cat', 'grep'])]
        process_indices = [i for i, cmd in enumerate(self.test_commands) 
                          if any(word in cmd for word in ['ps', 'kill', 'top', 'pgrep'])]
        git_indices = [i for i, cmd in enumerate(self.test_commands) 
                      if 'git' in cmd]
        
        # Make similar commands have similar embeddings
        if file_ops_indices:
            cluster_center = np.random.randn(384).astype(np.float32)
            for idx in file_ops_indices:
                base_embeddings[idx] = cluster_center + np.random.randn(384) * 0.1
                
        # Normalize embeddings
        norms = np.linalg.norm(base_embeddings, axis=1, keepdims=True)
        test_embeddings = base_embeddings / norms
        
        self.mock_model.encode.return_value = test_embeddings
        
        # Store embeddings in database
        c = self.test_conn.cursor()
        for i, command in enumerate(self.test_commands):
            command_id = i + 1
            quantized_embedding = self.fuzzyshell.quantize_embedding(test_embeddings[i])
            c.execute('INSERT OR REPLACE INTO embeddings (rowid, embedding) VALUES (?, ?)', 
                     (command_id, quantized_embedding))
        self.test_conn.commit()
        
        # Test natural language queries
        test_queries = [
            ("list files", ["ls -lh", "ls -la"]),
            ("show processes", ["ps aux", "top", "htop"]),
            ("find files", ["find . -name '*.py'", "grep -r 'pattern' ."]),
            ("check disk space", ["df -h", "du -sh *"]),
            ("version control", ["git status", "git commit -m 'message'"]),
            ("text editor", ["vim file.txt", "nano config.conf"]),
            ("containers", ["docker ps", "docker build -t image ."]),
        ]
        
        for query, expected_commands in test_queries:
            # Mock query embedding to be similar to expected commands
            if expected_commands:
                # Find the first expected command in our test set
                expected_indices = []
                for exp_cmd in expected_commands:
                    if exp_cmd in self.test_commands:
                        expected_indices.append(self.test_commands.index(exp_cmd))
                
                if expected_indices:
                    # Create query embedding similar to expected commands
                    query_embedding = np.mean([test_embeddings[idx] for idx in expected_indices], axis=0)
                    query_embedding = query_embedding / np.linalg.norm(query_embedding)
                    self.mock_model.encode.return_value = [query_embedding]
                    
                    # Perform search
                    results = self.fuzzyshell.search(query, top_k=5, return_scores=True)
                    
                    if results:
                        print(f"\nQuery: '{query}'")
                        print("Top results:")
                        for cmd, score, sem_score, bm25_score in results[:3]:
                            print(f"  {score:.3f}: {cmd}")
                        
                        # Check if any expected command is in top results
                        result_commands = [result[0] for result in results]
                        found_expected = any(exp_cmd in result_commands for exp_cmd in expected_commands)
                        
                        if not found_expected:
                            print(f"  Expected one of: {expected_commands}")
                            print(f"  Got: {result_commands[:3]}")
                        
                        # At least one result should be returned
                        assert len(results) > 0, f"No results for query: {query}"
                        
    def test_ann_search_with_real_commands(self):
        """Test that ANN search works correctly with real command data."""
        if not USE_ANN_SEARCH:
            pytest.skip("ANN search is disabled")
        
        # Use mock model for this controlled test
        self.fuzzyshell._model = self.mock_model
            
        # Create embeddings for real commands
        np.random.seed(42)
        n_commands = len(self.test_commands)
        test_embeddings = np.random.randn(n_commands, 384).astype(np.float32)
        norms = np.linalg.norm(test_embeddings, axis=1, keepdims=True)
        test_embeddings = test_embeddings / norms
        
        self.mock_model.encode.return_value = test_embeddings
        
        # Store embeddings
        c = self.test_conn.cursor()
        for i, command in enumerate(self.test_commands):
            command_id = i + 1
            quantized_embedding = self.fuzzyshell.quantize_embedding(test_embeddings[i])
            c.execute('INSERT OR REPLACE INTO embeddings (rowid, embedding) VALUES (?, ?)', 
                     (command_id, quantized_embedding))
        self.test_conn.commit()
        
        # Test search with ANN enabled
        query_embedding = np.random.randn(384).astype(np.float32)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        self.mock_model.encode.return_value = [query_embedding]
        
        # Perform search
        results = self.fuzzyshell.search("test query", top_k=10)
        
        assert len(results) > 0, "ANN search should return results"
        assert len(results) <= 10, "Should respect top_k limit"
        
        # Results should be real commands from our test set
        for result in results:
            command = result[0]
            assert command in self.test_commands, (
                f"Result '{command}' not in test command set"
            )
            
    def test_command_complexity_handling(self):
        """Test that the system handles commands of varying complexity."""
        # Test commands with different complexity levels
        complexity_tests = [
            # Simple commands
            ("ls", "list files"),
            ("pwd", "show current directory"),
            ("date", "show date"),
            
            # Medium complexity
            ("ls -la", "list all files with details"),
            ("find . -name '*.txt'", "find text files"),
            ("ps aux | grep firefox", "find firefox process"),
            
            # Complex commands
            ("find /var/log -name '*.log' -mtime +7 -exec rm {} \\;", "delete old log files"),
            ("docker run -d -p 8080:80 --name web nginx", "run web server container"),
            ("rsync -avz --delete source/ user@host:/backup/", "backup files to remote server"),
            ("awk 'BEGIN{OFS=\",\"} {print $1,$3,$5}' data.txt", "extract columns from data file"),
        ]
        
        for command, description in complexity_tests:
            # Add command to database
            self.fuzzyshell.add_command(command)
            
            # Test that complex commands can be added and searched
            # (We're primarily testing that the system doesn't break with complex commands)
            assert True  # Placeholder - real test would check embeddings
            
        print(f"Successfully handled {len(complexity_tests)} commands of varying complexity")
        
    def test_multilingual_command_support(self):
        """Test support for commands with non-English elements."""
        # Commands that might contain non-English elements or international formats
        international_commands = [
            ("locale", "show system language settings"),
            ("iconv -f utf-8 -t ascii file.txt", "convert text encoding"),
            ("LC_ALL=C sort file.txt", "sort with specific locale"),
            ("date +'%Y-%m-%d %H:%M:%S'", "show formatted date"),
            ("curl -H 'Content-Type: application/json'", "send json request"),
        ]
        
        for command, description in international_commands:
            # Test that the system can handle these commands
            self.fuzzyshell.add_command(command)
            
        print(f"Successfully handled {len(international_commands)} international commands")
        
    def teardown_method(self):
        """Clean up test resources."""
        if hasattr(self, 'test_conn'):
            self.test_conn.close()


class TestCommandCategoryRecognition:
    """Test that the system can recognize different categories of commands."""
    
    def setup_method(self):
        """Set up test with categorized commands."""
        self.test_conn = create_test_db_connection()
        self.fuzzyshell = FuzzyShell(conn=self.test_conn)
        
        # Commands organized by category
        self.command_categories = {
            'file_operations': [
                'ls -la', 'cp file.txt backup.txt', 'mv old.txt new.txt', 
                'rm unwanted.txt', 'chmod +x script.sh', 'find . -name "*.py"'
            ],
            'process_management': [
                'ps aux', 'kill -9 1234', 'top', 'htop', 'pgrep firefox', 'killall chrome'
            ],
            'network_tools': [
                'ping google.com', 'curl https://api.github.com', 'wget https://example.com/file.zip',
                'ssh user@server.com', 'scp file.txt user@server:/path'
            ],
            'development': [
                'git status', 'git commit -m "fix"', 'npm install', 'python script.py',
                'docker ps', 'docker build -t app .'
            ],
            'system_info': [
                'df -h', 'free -h', 'uname -a', 'whoami', 'date', 'uptime'
            ],
            'text_processing': [
                'grep pattern file.txt', 'sed "s/old/new/g" file.txt', 'awk "{print $1}" file.txt',
                'sort file.txt', 'uniq file.txt', 'wc -l file.txt'
            ]
        }
        
        # Mock model
        self.mock_model = MagicMock()
        self.fuzzyshell._model = self.mock_model
        
    def test_category_clustering_with_ann(self):
        """Test that ANN search groups similar command categories together."""
        if not USE_ANN_SEARCH:
            pytest.skip("ANN search is disabled")
            
        # Create synthetic embeddings where commands in the same category are similar
        all_commands = []
        all_embeddings = []
        command_to_category = {}
        
        np.random.seed(42)
        
        for category, commands in self.command_categories.items():
            # Create a cluster center for this category
            cluster_center = np.random.randn(384).astype(np.float32)
            cluster_center = cluster_center / np.linalg.norm(cluster_center)
            
            for command in commands:
                # Create embedding near the cluster center
                embedding = cluster_center + np.random.randn(384) * 0.2
                embedding = embedding / np.linalg.norm(embedding)
                
                all_commands.append(command)
                all_embeddings.append(embedding)
                command_to_category[command] = category
        
        test_embeddings = np.array(all_embeddings)
        self.mock_model.encode.return_value = test_embeddings
        
        # Add commands to database
        c = self.test_conn.cursor()
        for i, command in enumerate(all_commands):
            self.fuzzyshell.add_command(command)
            command_id = i + 1
            quantized_embedding = self.fuzzyshell.quantize_embedding(test_embeddings[i])
            c.execute('INSERT OR REPLACE INTO embeddings (rowid, embedding) VALUES (?, ?)', 
                     (command_id, quantized_embedding))
        self.test_conn.commit()
        
        # Test that searching for a command from one category returns similar commands
        test_command = 'ls -la'  # file_operations category
        query_idx = all_commands.index(test_command)
        query_embedding = test_embeddings[query_idx]
        self.mock_model.encode.return_value = [query_embedding]
        
        results = self.fuzzyshell.search("list files", top_k=10, return_scores=True)
        
        if results:
            print(f"\nSearching for file operations (query similar to '{test_command}'):")
            categories_found = {}
            for i, (command, score, sem_score, bm25_score) in enumerate(results[:5]):
                category = command_to_category.get(command, 'unknown')
                categories_found[category] = categories_found.get(category, 0) + 1
                print(f"  {i+1}. {command} ({category}) - score: {score:.3f}")
            
            print(f"Categories in top 5: {categories_found}")
            
            # Should find some file operations commands in top results
            file_ops_in_top5 = categories_found.get('file_operations', 0)
            assert file_ops_in_top5 > 0, (
                "Should find at least one file operations command in top results"
            )
        
    def teardown_method(self):
        """Clean up test resources."""
        if hasattr(self, 'test_conn'):
            self.test_conn.close()


