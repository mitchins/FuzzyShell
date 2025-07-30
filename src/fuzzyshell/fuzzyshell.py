import sqlite3
import os
import sqlite_vss
import numpy as np
from collections import Counter
import math
import re
import asyncio
import threading
import logging
import time
from pathlib import Path
import pickle
import hashlib
from threading import Event
from .model_handler import ModelHandler, MODEL_OUTPUT_DIM
from .fuzzy_tui import FuzzyShellApp

# Configure logging
# File handler for debug logging
file_handler = logging.FileHandler('debug.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
))

# Console handler for important messages only
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.WARNING)
console_handler.setFormatter(logging.Formatter('%(message)s'))

# Configure root logger
logger = logging.getLogger('FuzzyShell')
logger.setLevel(logging.DEBUG)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

class FuzzyShell:
    def __init__(self, db_path='fuzzyshell.db'):
        self.db_path = db_path
        self._conn = None
        self._model = None
        self._initialized = False
        self._model_ready = threading.Event()
        
        # BM25 parameters
        self.k1 = 1.5
        self.b = 0.75
        
        # Cache for corpus stats
        self.total_commands = 0
        self.avg_length = 0
        
        # Initialize database on startup
        _ = self.conn  # This will trigger database initialization
    
    @property
    def conn(self):
        """Lazy initialize database connection"""
        if self._conn is None:
            start_time = time.time()
            if os.path.exists(self.db_path):
                logger.debug("Existing database found, connecting")
            else:
                logger.debug("No existing database, creating new one")
            # Enable memory-mapped I/O and other performance optimizations
            self._conn = sqlite3.connect(
                f"file:{self.db_path}?mode=rwc", 
                uri=True,
                timeout=60.0
            )
            self.optimize_db_connection()
            
            # Try to enable extensions if available
            try:
                self._conn.enable_load_extension(True)
                sqlite_vss.load(self._conn)
            except (AttributeError, Exception) as e:
                logger.warning("Could not load SQLite extensions: %s", str(e))
                logger.warning("Vector search may not be available")
            
            # Initialize DB structure if needed
            self.init_db()
            
            # Load corpus stats
            self.update_corpus_stats()
            logger.debug("Database connection initialized in %.3fs", time.time() - start_time)
        return self._conn
    
    def _init_model(self):
        """Initialize the ONNX model handler"""
        start_time = time.time()
        logger.debug("Starting model initialization")
        try:
            self._model = ModelHandler()
            if not hasattr(self._model, 'encode'):
                raise RuntimeError("Model initialized but encode method not found")
            logger.info("Model initialized successfully in %.3fs", time.time() - start_time)
            return self._model
        except Exception as e:
            logger.error("Failed to initialize model: %s", str(e), exc_info=True)
            self._model = None
            return None
            
    def quantize_embedding(self, embedding, scale_factor=127):
        """Quantize embedding to INT8 (-127 to 127 range)"""
        start_time = time.time()
        # Normalize the embedding to unit length
        embedding = embedding / np.linalg.norm(embedding)
        # Scale to INT8 range and quantize
        result = np.clip(np.round(embedding * scale_factor), -127, 127).astype(np.int8)
        logger.debug("Quantized embedding in %.3fs", time.time() - start_time)
        return result
    
    def dequantize_embedding(self, quantized_embedding, scale_factor=127):
        """Dequantize INT8 embedding back to float32"""
        return (quantized_embedding.astype(np.float32) / scale_factor)

    def _init_model_async(self):
        """Initialize model in background thread"""
        def load_model():
            try:
                logger.debug("Initializing model (loading cached or downloading)")
                self._init_model()
                if self._model is None:
                    raise RuntimeError("Model initialization returned None")
                logger.debug("Model ready and verified")
            except Exception as e:
                logger.error("Error in model initialization: %s", str(e), exc_info=True)
                # Clear the event so other threads know initialization failed
                if hasattr(self, '_model_ready') and self._model_ready is not None:
                    self._model_ready.clear()
                return
            else:
                # Only set the event if initialization succeeded
                if hasattr(self, '_model_ready') and self._model_ready is not None:
                    self._model_ready.set()
                    logger.debug("Model ready event set")

        # Only reset state if not already initializing
        event_ready = (hasattr(self, '_model_ready') and 
                      self._model_ready is not None and 
                      self._model_ready.is_set())
        
        if not event_ready:
            logger.debug("Creating new model initialization state")
            self._model = None
            self._model_ready = threading.Event()
            
            # Start background thread
            thread = threading.Thread(target=load_model, name="ModelLoader", daemon=True)
            thread.start()
        else:
            logger.debug("Model initialization already in progress, skipping")

    @property
    def model(self):
        """Get model, waiting if necessary"""
        if self._model is None:
            start_time = time.time()
            logger.debug("Model access requested, checking cache first")
            
            # Reset state if previous initialization failed
            if hasattr(self, '_model_ready') and not self._model_ready.is_set():
                logger.debug("Previous initialization failed, resetting state")
                self._model_ready = None
            
            # Initialize if needed
            if not hasattr(self, '_model_ready') or self._model_ready is None:
                logger.debug("No model state found, starting async initialization")
                self._init_model_async()
            else:
                logger.debug("Model initialization already in progress")
            
            # Wait for initialization with appropriate timeout
            timeout = 30.0 if not hasattr(self, '_model') or self._model is None else 2.0
            logger.debug("Waiting for model with timeout=%.1fs", timeout)
            
            if not hasattr(self, '_model_ready') or self._model_ready is None or not self._model_ready.wait(timeout=timeout):
                logger.error("Model initialization timed out or failed after %.1fs. Check debug.log for errors.", timeout)
                # Reset state so next attempt can start fresh
                self._model_ready = None
                return None
            
            # Now that we've waited, check if initialization succeeded
            if self._model is None:
                logger.error("Model initialization failed after waiting")
                return None
                
            logger.debug("Model initialization completed in %.3fs", time.time() - start_time)
                
        return self._model
        
    def basic_search(self, query, top_k=5):
        """Fallback search using simple substring matching"""
        if not query or query.isspace():
            return []
            
        try:
            c = self.conn.cursor()
            # Use LIKE for basic substring matching
            c.execute("""
                SELECT command FROM commands 
                WHERE command LIKE ? 
                ORDER BY last_used DESC 
                LIMIT ?
            """, (f"%{query}%", top_k))
            
            results = [(row[0], 0.0) for row in c.fetchall()]
            return results or [("No matches found", 0.0)]
            
        except Exception as e:
            print(f"Basic search failed: {str(e)}")
            return [("Search error occurred", 0.0)]
    
    def is_model_ready(self):
        """Check if model is ready without waiting"""
        return (hasattr(self, '_model_ready') and 
                self._model_ready is not None and 
                self._model_ready.is_set())
    
    def optimize_db_connection(self):
        """Apply SQLite optimizations"""
        c = self.conn.cursor()
        # Memory-mapped I/O
        c.execute('PRAGMA mmap_size = 30000000000')  # 30GB max mmap
        # Other performance optimizations
        c.execute('PRAGMA journal_mode = WAL')  # Write-Ahead Logging
        c.execute('PRAGMA synchronous = NORMAL')
        c.execute('PRAGMA cache_size = -2000000')  # 2GB cache
        c.execute('PRAGMA temp_store = MEMORY')
        c.execute('PRAGMA case_sensitive_like = false')

    def init_db(self):
        start_time = time.time()
        logger.debug("Initializing database structure")
        c = self.conn.cursor()
        # Create table for commands with optimized indexing
        c.execute('''
            CREATE TABLE IF NOT EXISTS commands (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                command TEXT NOT NULL UNIQUE,
                length INTEGER NOT NULL DEFAULT 0,
                last_used TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create indices for better query performance
        c.execute('CREATE INDEX IF NOT EXISTS idx_commands_last_used ON commands(last_used)')
        
        # Create table for embeddings with quantized storage
        # Note: VSS module handles its own indexing, fallback to regular table if not available
        try:
            c.execute('''
                CREATE VIRTUAL TABLE IF NOT EXISTS embeddings USING vss0(
                    embedding(384)
                )
            ''')
        except Exception as e:
            logger.warning("Could not create VSS table, using regular table: %s", str(e))
            c.execute('''
                CREATE TABLE IF NOT EXISTS embeddings (
                    rowid INTEGER PRIMARY KEY,
                    embedding BLOB NOT NULL
                )
            ''')
        
        # Create table for term frequencies with optimized indexing
        c.execute('''
            CREATE TABLE IF NOT EXISTS term_frequencies (
                term TEXT NOT NULL,
                command_id INTEGER NOT NULL,
                freq INTEGER NOT NULL,
                FOREIGN KEY (command_id) REFERENCES commands (id),
                PRIMARY KEY (term, command_id)
            )
        ''')
        
        # Create indices for term frequency lookups
        c.execute('CREATE INDEX IF NOT EXISTS idx_term_freq_term ON term_frequencies(term)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_term_freq_command ON term_frequencies(command_id)')
        
        # Create table for corpus statistics
        c.execute('''
            CREATE TABLE IF NOT EXISTS corpus_stats (
                total_commands INTEGER NOT NULL DEFAULT 0,
                avg_length REAL NOT NULL DEFAULT 0
            )
        ''')
        
        # Insert initial corpus stats if not exists
        c.execute("INSERT OR IGNORE INTO corpus_stats VALUES (0, 0)")
        
        # Create a cache table for frequently used queries
        c.execute('''
            CREATE TABLE IF NOT EXISTS query_cache (
                query_hash TEXT PRIMARY KEY,
                results BLOB,  -- Serialized results
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Index for cache cleanup
        c.execute('CREATE INDEX IF NOT EXISTS idx_cache_timestamp ON query_cache(timestamp)')
        
        self.conn.commit()
        
        # Clean old cache entries
        self.cleanup_cache()
        
        logger.debug("Database initialization completed in %.3fs", time.time() - start_time)

    def add_command(self, raw_command):
        """Add a single command to the history and update its embedding."""
        if not raw_command or len(raw_command.strip()) == 0:
            return
        
        # Clean the command to remove shell history format
        command = self.clean_shell_command(raw_command)
        if not command or len(command.strip()) == 0:
            return
        
        c = self.conn.cursor()
        
        try:
            # Insert or ignore the command
            c.execute('''
                INSERT OR IGNORE INTO commands (command, length)
                VALUES (?, ?)
            ''', (command, len(command.split())))
            
            # Get the command ID
            c.execute('SELECT id FROM commands WHERE command = ?', (command,))
            command_id = c.fetchone()[0]
            
            # Update term frequencies
            terms = command.lower().split()
            for term in terms:
                c.execute('''
                    INSERT OR REPLACE INTO term_frequencies (term, command_id, freq)
                    VALUES (?, ?, ?)
                ''', (term, command_id, terms.count(term)))
            
            # Update corpus stats
            c.execute('''
                UPDATE corpus_stats 
                SET total_commands = total_commands + 1,
                    avg_length = (avg_length * (total_commands) + ?) / (total_commands + 1)
                WHERE rowid = 1
            ''', (len(terms),))
            
            # Generate and store embedding
            if self._model_ready and self._model_ready.is_set():
                embedding = self.model.encode([command])[0]
                embedding = embedding[:384]  # Ensure correct dimensions
                quantized = self.quantize_embedding(embedding)
                c.execute('INSERT INTO embeddings (rowid, embedding) VALUES (?, ?)', 
                         (command_id, quantized))
            
            self.conn.commit()
            
        except sqlite3.Error as e:
            logger.error(f"Error adding command to database: {e}")
            self.conn.rollback()

    def get_shell_history_file(self):
        if os.path.exists(".zsh_history"):
            return ".zsh_history"
        elif os.path.exists(".bash_history"):
            return ".bash_history"
        else:
            shell = os.environ.get("SHELL")
            if shell and "bash" in shell:
                return os.path.expanduser("~/.bash_history")
            elif shell and "zsh" in shell:
                return os.path.expanduser("~/.zsh_history")
            else:
                raise ValueError(f"Unsupported shell: {shell}. Only bash and zsh are currently supported.")

    def clean_shell_command(self, raw_command):
        """Clean shell history format to extract actual command"""
        # Handle zsh/bash history format: ': <timestamp>:<duration>;<command>'
        if raw_command.startswith(': ') and ';' in raw_command:
            # Find the semicolon that separates timestamp from command
            semicolon_pos = raw_command.find(';')
            if semicolon_pos != -1:
                # Extract command part after semicolon (no space expected after ;)
                command = raw_command[semicolon_pos + 1:].strip()
                return command
        
        # If no shell history format detected, return as-is
        return raw_command.strip()

    def tokenize(self, text):
        """Convert text to lowercase and split into terms"""
        start_time = time.time()
        tokens = re.findall(r'\w+|[^\w\s]', text.lower())
        logger.debug("Tokenized text in %.3fs (%d tokens)", time.time() - start_time, len(tokens))
        return tokens

    def update_corpus_stats(self):
        """Update the average document length and total document count"""
        c = self.conn.cursor()
        
        # Get the stats, handling NULL case for avg_length
        c.execute("""
            SELECT COUNT(*), COALESCE(AVG(length), 0) 
            FROM commands
        """)
        total_commands, avg_length = c.fetchone()
        
        # Update the stats
        c.execute("""
            UPDATE corpus_stats 
            SET total_commands = ?, avg_length = ?
        """, (total_commands, avg_length))
        self.conn.commit()

        # Cache the values
        self.total_commands = total_commands
        self.avg_length = avg_length

    def calculate_idf(self, term):
        """Calculate IDF for a term"""
        c = self.conn.cursor()
        c.execute("""
            SELECT COUNT(DISTINCT command_id) 
            FROM term_frequencies 
            WHERE term = ?
        """, (term,))
        doc_freq = c.fetchone()[0] or 0
        return math.log((self.total_commands - doc_freq + 0.5) / (doc_freq + 0.5) + 1)

    def ingest_history(self):
        start_time = time.time()
        history_file = self.get_shell_history_file()
        logger.info("Starting history ingestion from: %s", history_file)
        
        # Initialize model if needed and wait for it
        if self._model is None:
            logger.debug("Model not initialized, starting async initialization")
            self._init_model_async()
        
        logger.debug("Waiting for model to be ready...")
        if not hasattr(self, '_model_ready') or self._model_ready is None or not self._model_ready.wait(timeout=30.0):  # Longer timeout for initial load
            logger.error("Model failed to load in time or initialization failed (timeout after 30s)")
            return
            
        try:
            with open(history_file, 'r', encoding='utf-8', errors='ignore') as f:
                commands = set(line.strip() for line in f)
        except FileNotFoundError:
            print(f"History file not found at: {history_file}")
            return

        logger.info("Starting command processing...")
        total_processed = 0
        total_commands = len(commands)
        c = self.conn.cursor()
        process_start = time.time()
        
        for raw_command in commands:
            if not raw_command or raw_command.isspace():
                continue
            
            # Clean the command to remove shell history format
            command = self.clean_shell_command(raw_command)
            if not command or command.isspace():
                continue
                
            # Check if command already exists
            c.execute("SELECT id FROM commands WHERE command = ?", (command,))
            if c.fetchone() is None:
                # Process terms for BM25
                terms = self.tokenize(command)
                term_freq = Counter(terms)
                
                # Insert command and get its id
                c.execute("INSERT INTO commands (command, length) VALUES (?, ?)", 
                         (command, len(terms)))
                command_id = c.lastrowid
                
                # Insert term frequencies
                for term, freq in term_freq.items():
                    c.execute("""
                        INSERT INTO term_frequencies (term, command_id, freq)
                        VALUES (?, ?, ?)
                    """, (term, command_id, freq))
                
                # Generate and insert embedding
                # Get embedding and ensure it's the right size
                embedding = self.model.encode([command])[0]  # Get first element as we're encoding a single command
                if len(embedding) > MODEL_OUTPUT_DIM:
                    embedding = embedding[:MODEL_OUTPUT_DIM]
                embedding = self.quantize_embedding(embedding)  # Quantize before storage
                c.execute("INSERT INTO embeddings (rowid, embedding) VALUES (?, ?)", 
                         (command_id, embedding.tobytes()))
                
                total_processed += 1
                if total_processed % 100 == 0:
                    print(f"Processed {total_processed} commands...")
        
        self.conn.commit()
        self.update_corpus_stats()
        
        total_time = time.time() - start_time
        process_time = time.time() - process_start
        logger.info("Ingestion complete: %d/%d commands processed", total_processed, total_commands)
        logger.info("Total ingestion time: %.3fs (%.3fs/command)", 
                   total_time, total_time/total_processed if total_processed > 0 else 0)
        logger.debug("Pure processing time: %.3fs (%.3fs/command)", 
                    process_time, process_time/total_processed if total_processed > 0 else 0)

    def calculate_bm25_scores_batch(self, query_terms, command_ids):
        """Calculate BM25 scores for multiple commands at once"""
        if not query_terms:
            return np.zeros(len(command_ids))
            
        c = self.conn.cursor()
        
        # Get all document lengths at once
        placeholders = ','.join('?' * len(command_ids))
        c.execute(f"SELECT id, length FROM commands WHERE id IN ({placeholders})", command_ids)
        doc_lengths = dict(c.fetchall())
        
        # Get all term frequencies at once
        term_freqs = {}
        for term in query_terms:
            c.execute(f"""
                SELECT command_id, freq 
                FROM term_frequencies 
                WHERE term = ? AND command_id IN ({placeholders})
            """, (term, *command_ids))
            term_freqs[term] = dict(c.fetchall())
        
        # Calculate IDF values once
        idf_values = {term: self.calculate_idf(term) for term in query_terms}
        
        # Calculate scores vectorized
        scores = np.zeros(len(command_ids))
        for i, cmd_id in enumerate(command_ids):
            doc_length = doc_lengths.get(cmd_id, 0)
            score = 0.0
            
            for term in query_terms:
                tf = term_freqs.get(term, {}).get(cmd_id, 0)
                if tf == 0:
                    continue
                    
                idf = idf_values[term]
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_length / self.avg_length)
                score += idf * (numerator / denominator)
            
            scores[i] = score
            
        return scores

    def cleanup_cache(self, max_age_hours=24):
        """Clean up old cache entries"""
        c = self.conn.cursor()
        c.execute("""
            DELETE FROM query_cache 
            WHERE timestamp < datetime('now', '-' || ? || ' hours')
        """, (max_age_hours,))
        self.conn.commit()

    def get_cached_results(self, query, return_scores=False):
        """Get cached results for a query with specific return_scores setting"""
        import hashlib
        cache_key = f"{query}|return_scores={return_scores}"
        query_hash = hashlib.md5(cache_key.encode()).hexdigest()
        
        c = self.conn.cursor()
        c.execute("""
            SELECT results FROM query_cache 
            WHERE query_hash = ? 
            AND timestamp > datetime('now', '-1 hour')
        """, (query_hash,))
        
        result = c.fetchone()
        if result:
            import pickle
            return pickle.loads(result[0])
        return None

    def cache_results(self, query, results, return_scores=False):
        """Cache results for a query with specific return_scores setting"""
        import hashlib, pickle
        cache_key = f"{query}|return_scores={return_scores}"
        query_hash = hashlib.md5(cache_key.encode()).hexdigest()
        
        c = self.conn.cursor()
        c.execute("""
            INSERT OR REPLACE INTO query_cache (query_hash, results)
            VALUES (?, ?)
        """, (query_hash, pickle.dumps(results)))
        self.conn.commit()

    def search(self, query, top_k=5, return_scores=False):
        """Search for commands using hybrid BM25 + semantic search"""
        start_time = time.time()
        logger.debug("Starting search for query: %s (return_scores=%s)", query, return_scores)
        
        if not query or query.isspace():
            return []
            
        # Check cache first
        try:
            cache_start = time.time()
            cached_results = self.get_cached_results(query, return_scores)
            if cached_results is not None:
                logger.debug("Cache hit! Retrieved results in %.3fs", time.time() - cache_start)
                return cached_results[:top_k]
            logger.debug("Cache miss, took %.3fs to check", time.time() - cache_start)
        except Exception as e:
            logger.error("Cache lookup failed: %s", str(e))
            
        # Ensure model is ready
        model = self.model  # Use property to handle initialization
        if model is None:
            return [("Model initialization in progress... Please wait.", 0.0)]
            
        try:
            # Generate and quantize query embedding
            embed_start = time.time()
            query_embedding = model.encode([query])[0]  # Get first element as we're encoding a single query
            # Ensure query embedding is the right size before quantizing
            if len(query_embedding) > MODEL_OUTPUT_DIM:
                logger.debug("Truncating query embedding from %d to %d dimensions", 
                           len(query_embedding), MODEL_OUTPUT_DIM)
                query_embedding = query_embedding[:MODEL_OUTPUT_DIM]
            query_embedding = self.quantize_embedding(query_embedding)
            logger.debug("Generated query embedding in %.3fs (shape: %s)", 
                        time.time() - embed_start, query_embedding.shape)
        except Exception as e:
            logger.error("Error generating query embedding: %s", str(e))
            # Fallback to basic search with score 0.0
            logger.debug("Falling back to basic search")
            return [(cmd, 0.0) for cmd in self.basic_search(query, top_k)]
        
        c = self.conn.cursor()
        # Get commands with embeddings, prioritizing exact matches
        # First try to find commands that contain the search term
        exact_match_query = """
            SELECT c.id, c.command, v.embedding
            FROM commands c
            JOIN embeddings v ON c.id = v.rowid
            WHERE c.command LIKE ?
            ORDER BY c.last_used DESC
            LIMIT 500
        """
        
        # Get broader set for semantic matching
        broader_query = """
            SELECT c.id, c.command, v.embedding
            FROM commands c
            JOIN embeddings v ON c.id = v.rowid
            ORDER BY c.last_used DESC
            LIMIT 1000
        """
        
        # Try exact matches first
        c.execute(exact_match_query, (f"%{query}%",))
        exact_matches = c.fetchall()
        
        # If we have enough exact matches, use those; otherwise get broader set
        if len(exact_matches) >= 20:
            all_commands_data = exact_matches
            logger.debug("Using %d exact matches for query '%s'", len(exact_matches), query)
        else:
            c.execute(broader_query)
            all_commands_data = c.fetchall()
            logger.debug("Using broader search with %d commands", len(all_commands_data))

        if not all_commands_data:
            logger.warning("No commands found in database - have you run --ingest?")
            return []

        # Calculate scores using numpy vectorization for performance
        calc_start = time.time()
        query_terms = self.tokenize(query)
        command_ids, commands, embeddings = zip(*all_commands_data)
        
        logger.debug("Processing %d commands for ranking (query terms: %s)", len(command_ids), query_terms)
        
        # Convert embeddings to numpy array for vectorized operations
        embed_start = time.time()
        embeddings_array = np.vstack([
            np.frombuffer(emb, dtype=np.int8)[:384] for emb in embeddings  # Note: stored as INT8, ensure 384 dimensions
        ])
        logger.debug("Loaded embeddings with shape %s", embeddings_array.shape)
        
        # First ensure query embedding is the right shape
        logger.debug("Query embedding shape before processing: %s", query_embedding.shape)
        query_embedding = np.array(query_embedding).flatten()[:384]  # Ensure 1D array of right size
        logger.debug("Query embedding shape after flattening and truncating: %s", query_embedding.shape)
        
        # Dequantize embeddings for comparison (convert INT8 back to float32)
        embeddings_array = self.dequantize_embedding(embeddings_array)
        query_embedding = self.dequantize_embedding(query_embedding)
        logger.debug("Query embedding shape after dequantizing: %s", query_embedding.shape)
        
        # Reshape for matrix multiplication
        query_embedding = query_embedding.reshape(1, -1)  # Should now be (1, 384)
        logger.debug("Final shapes - embeddings: %s, query: %s", 
                    embeddings_array.shape, query_embedding.shape)
        
        # Vectorized semantic similarity calculation
        semantic_scores = np.dot(embeddings_array, query_embedding.T).reshape(-1)
        norms = np.linalg.norm(embeddings_array, axis=1) * np.linalg.norm(query_embedding)
        logger.debug("Semantic scoring completed in %.3fs (max score: %.3f)", 
                    time.time() - embed_start,
                    np.max(semantic_scores) if len(semantic_scores) > 0 else 0.0)
        
        # Avoid division by zero and handle NaN values
        semantic_scores = np.divide(
            semantic_scores, 
            norms, 
            out=np.zeros_like(semantic_scores), 
            where=norms != 0
        )
        
        # Ensure scores are in [0, 1] range
        semantic_scores = (semantic_scores + 1) / 2  # Convert from [-1, 1] to [0, 1]
        semantic_scores = np.clip(semantic_scores, 0, 1)
        
        # Calculate BM25 scores in batch
        bm25_scores = self.calculate_bm25_scores_batch(query_terms, command_ids)
        
        # Normalize BM25 scores
        if len(bm25_scores) > 0:
            # Add small epsilon to avoid division by zero
            max_score = np.maximum(bm25_scores.max(), 1e-6)
            bm25_scores = bm25_scores / max_score  # Normalize to [0, 1]
            bm25_scores = np.clip(bm25_scores, 0, 1)
        
        # Combine scores with weighted average (strongly favoring exact matches)
        # BM25 gets 0.85 weight (exact matches), semantic gets 0.15 (meaning/context)
        combined_scores = 0.15 * semantic_scores + 0.85 * bm25_scores
        results = []
        for i, (command, combined_score) in enumerate(zip(commands, combined_scores)):
            if return_scores:
                results.append((
                    command, float(combined_score), 
                    float(semantic_scores[i]), float(bm25_scores[i])
                ))
            else:
                results.append((command, float(combined_score)))

        # Sort results with priority for exact prefix matches
        def sort_key(result):
            command = result[0]
            score = result[1]
            
            # Boost score significantly if command starts with the query
            if command.lower().startswith(query.lower()):
                return score + 10.0  # Large boost for prefix matches
            
            # Boost score if any word in command starts with query
            command_words = command.lower().split()
            for word in command_words:
                if word.startswith(query.lower()):
                    return score + 5.0  # Medium boost for word prefix matches
            
            return score
        
        results.sort(key=sort_key, reverse=True)
        results = results[:top_k]
        
        # Log top results with scores for debugging
        if results:
            logger.debug("Top %d results:", len(results))
            for i, result in enumerate(results[:3], 1):  # Log top 3 for debugging
                if return_scores and len(result) == 4:
                    cmd, combined, semantic, bm25 = result
                    logger.debug("  %d. %s (combined: %.3f)", i, cmd, combined)
                else:
                    cmd, score = result
                    logger.debug("  %d. %s (score: %.3f)", i, cmd, score)
        else:
            logger.warning("No results found for query: %s", query)
            
        # Cache the results
        cache_start = time.time()
        self.cache_results(query, results, return_scores)
        logger.debug("Results cached in %.3fs", time.time() - cache_start)
        
        total_time = time.time() - start_time
        logger.info("Total search completed in %.3fs with %d results", total_time, len(results))
        
        return results
    
    def tui(self, show_scoring=False):
        """Launch the interactive TUI for this FuzzyShell instance"""
        def search_callback(query: str) -> list:
            """Callback function for the UI to perform searches"""
            if not query:
                return []
            return self.search(query, return_scores=show_scoring)

        app = FuzzyShellApp(search_callback, fuzzyshell_instance=self)
        selected_command = app.run()
        return selected_command

__version__ = "0.1.0"

def interactive_search(show_scoring=False):
    """Launch the interactive search UI"""
    fuzzyshell = FuzzyShell()
    
    def search_callback(query: str) -> list:
        """Callback function for the UI to perform searches"""
        if not query:
            return []
        return fuzzyshell.search(query, return_scores=show_scoring)

    app = FuzzyShellApp(search_callback, fuzzyshell_instance=fuzzyshell)
    selected_command = app.run()
    return selected_command

def main():
    """Main entry point for the application"""
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="FuzzyShell: A semantic search for your command history.")
    parser.add_argument('--version', action='version', version=f'%(prog)s {__version__}')
    parser.add_argument('--ingest', action='store_true', help='Ingest commands from shell history.')
    parser.add_argument('--scoring', action='store_true', help='Show semantic and BM25 scores for each result.')

    args = parser.parse_args()
    fuzzyshell = FuzzyShell()

    if args.ingest:
        fuzzyshell.ingest_history()
    else:
        result = interactive_search(show_scoring=args.scoring)
        if result:
            # Print the selected command so it can be captured by the shell wrapper
            print(result)
        sys.exit(0)

if __name__ == '__main__':
    main()
