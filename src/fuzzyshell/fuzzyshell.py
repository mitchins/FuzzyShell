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

# Configuration constants for embedding storage
EMBEDDING_DTYPE = np.float16  # Options: np.int8, np.float16, np.float32
EMBEDDING_SCALE_FACTOR = 127  # Only used for INT8 quantization

class FuzzyShell:
    def __init__(self, db_path='fuzzyshell.db', conn=None):
        self.db_path = db_path
        self._conn = conn  # Allow dependency injection of connection
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
        if self._conn is not None:
            # Connection was injected, need to initialize it manually
            self.optimize_db_connection()
            self.init_db()
            self.update_corpus_stats()
        else:
            # Will be lazy-initialized when conn property is accessed
            _ = self.conn
    
    @property
    def conn(self):
        """Get database connection (lazy initialize if not injected)"""
        if self._conn is None:
            start_time = time.time()
            if os.path.exists(self.db_path):
                logger.debug("Existing database found, connecting")
            else:
                logger.debug("No existing database, creating new one")
            # Enable memory-mapped I/O and other performance optimizations
            # Handle both file paths and URI paths (for in-memory databases)
            if self.db_path.startswith('file:'):
                # Already a URI, use as-is
                connect_string = self.db_path
            else:
                # Regular file path, add file: prefix and mode
                connect_string = f"file:{self.db_path}?mode=rwc"
            
            self._conn = sqlite3.connect(
                connect_string,
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
    
    def init_model_sync(self):
        """Initialize the ONNX model handler synchronously"""
        start_time = time.time()
        logger.debug("[SYNC] Starting synchronous model initialization")
        try:
            logger.debug("[SYNC] Creating ModelHandler instance...")
            handler_start = time.time()
            
            self._model = ModelHandler()
            
            handler_time = time.time() - handler_start
            logger.debug(f"[SYNC] ModelHandler created in {handler_time:.3f}s")
            
            logger.debug("[SYNC] Checking for encode method...")
            if not hasattr(self._model, 'encode'):
                raise RuntimeError("Model initialized but encode method not found")
                
            total_time = time.time() - start_time
            logger.info(f"[SYNC] Model initialized successfully in {total_time:.3f}s")
            
            if hasattr(self, '_model_ready'):
                self._model_ready.set()
            return self._model
            
        except Exception as e:
            total_time = time.time() - start_time
            logger.error(f"[SYNC] Failed to initialize model after {total_time:.3f}s: %s", str(e), exc_info=True)
            self._model = None
            if hasattr(self, '_model_ready'):
                self._model_ready.clear()
            return None
            
    def quantize_embedding(self, embedding, scale_factor=EMBEDDING_SCALE_FACTOR):
        """
        Store embedding in the configured format (INT8, FP16, or FP32).
        Returns bytes ready for database storage.
        """
        start_time = time.time()
        
        if EMBEDDING_DTYPE == np.int8:
            # Legacy INT8 quantization with normalization
            embedding = embedding / np.linalg.norm(embedding)
            result = np.clip(np.round(embedding * scale_factor), -127, 127).astype(np.int8)
            logger.debug("Quantized embedding to INT8 in %.3fs", time.time() - start_time)
        elif EMBEDDING_DTYPE == np.float16:
            # FP16 storage - good balance of quality vs size
            result = embedding.astype(np.float16)
            logger.debug("Converted embedding to FP16 in %.3fs", time.time() - start_time)
        elif EMBEDDING_DTYPE == np.float32:
            # FP32 storage - full precision
            result = embedding.astype(np.float32)
            logger.debug("Stored embedding as FP32 in %.3fs", time.time() - start_time)
        else:
            raise ValueError(f"Unsupported EMBEDDING_DTYPE: {EMBEDDING_DTYPE}")
            
        return result
    
    def dequantize_embedding(self, stored_embedding, scale_factor=EMBEDDING_SCALE_FACTOR):
        """
        Convert stored embedding back to float32 for computation.
        Handles INT8, FP16, and FP32 storage formats.
        Also ensures dimensional compatibility with current model.
        """
        # Ensure we have a valid numpy array first
        if not isinstance(stored_embedding, np.ndarray):
            stored_embedding = np.array(stored_embedding)
            
        # Check for invalid values before casting
        if len(stored_embedding) == 0:
            logger.warning("Empty embedding received, returning zeros")
            return np.zeros(MODEL_OUTPUT_DIM, dtype=np.float32)
            
        if EMBEDDING_DTYPE == np.int8:
            # Dequantize INT8 back to float32, with safe casting
            try:
                result = stored_embedding.astype(np.float32) / scale_factor
            except (ValueError, RuntimeWarning) as e:
                logger.warning(f"Error converting INT8 embedding to float32: {e}")
                return np.zeros(MODEL_OUTPUT_DIM, dtype=np.float32)
        elif EMBEDDING_DTYPE == np.float16:
            # Convert FP16 to FP32 for computation, with safe casting
            try:
                result = stored_embedding.astype(np.float32)
                # Check for NaN/inf values after conversion
                if np.any(np.isnan(result)) or np.any(np.isinf(result)):
                    logger.warning("NaN or inf values detected in FP16 embedding after conversion")
                    result = np.nan_to_num(result, nan=0.0, posinf=1.0, neginf=-1.0)
            except (ValueError, RuntimeWarning) as e:
                logger.warning(f"Error converting FP16 embedding to float32: {e}")
                return np.zeros(MODEL_OUTPUT_DIM, dtype=np.float32)
        elif EMBEDDING_DTYPE == np.float32:
            # Already FP32, but check for invalid values
            result = stored_embedding.copy()
            if np.any(np.isnan(result)) or np.any(np.isinf(result)):
                logger.warning("NaN or inf values detected in FP32 embedding")
                result = np.nan_to_num(result, nan=0.0, posinf=1.0, neginf=-1.0)
        else:
            raise ValueError(f"Unsupported EMBEDDING_DTYPE: {EMBEDDING_DTYPE}")
            
        # Ensure embedding matches current model output dimension
        if len(result) > MODEL_OUTPUT_DIM:
            logger.debug(f"Truncating stored embedding from {len(result)} to {MODEL_OUTPUT_DIM} dimensions")
            result = result[:MODEL_OUTPUT_DIM]
        elif len(result) < MODEL_OUTPUT_DIM:
            logger.warning(f"Stored embedding has {len(result)} dimensions, expected {MODEL_OUTPUT_DIM}. Padding with zeros.")
            # Pad with zeros if stored embedding is smaller
            padded = np.zeros(MODEL_OUTPUT_DIM, dtype=np.float32)
            padded[:len(result)] = result
            result = padded
            
        return result
    
    def _dynamic_hybrid_score(self, semantic_score, bm25_score):
        """
        Dynamic hybrid scoring that adjusts weights based on relative strengths:
        - High BM25, low semantic: prioritize keywords (favor BM25)
        - High semantic, low BM25: prioritize embeddings (favor semantic)
        - Both high/low or similar: use balanced 50/50 weighting
        """
        # Threshold for considering a score "high"
        high_threshold = 0.5
        
        # Calculate absolute difference to determine if scores are similar
        score_diff = abs(semantic_score - bm25_score)
        
        if bm25_score > high_threshold and semantic_score < high_threshold and score_diff > 0.3:
            # High BM25, low semantic → prioritize keywords (70% BM25)
            return 0.3 * semantic_score + 0.7 * bm25_score
        elif semantic_score > high_threshold and bm25_score < high_threshold and score_diff > 0.3:
            # High semantic, low BM25 → prioritize embeddings (70% semantic)
            return 0.7 * semantic_score + 0.3 * bm25_score
        else:
            # Balanced scoring for all other cases (50/50)
            return 0.5 * semantic_score + 0.5 * bm25_score
    
    def _kmeans_numpy(self, X, k, max_iters=100, tol=1e-4):
        """
        Pure NumPy K-means clustering implementation for coarse-to-fine pruning.
        
        Args:
            X: Data points (n_samples, n_features)
            k: Number of clusters
            max_iters: Maximum iterations
            tol: Convergence tolerance
            
        Returns:
            centroids: Cluster centers (k, n_features)
            labels: Cluster labels for each point (n_samples,)
        """
        n_samples, n_features = X.shape
        
        # Initialize centroids randomly
        np.random.seed(42)  # For reproducibility
        centroids = X[np.random.choice(n_samples, k, replace=False)]
        
        for _ in range(max_iters):
            # Assign points to nearest centroid
            distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
            labels = np.argmin(distances, axis=0)
            
            # Update centroids
            new_centroids = np.array([
                X[labels == i].mean(axis=0) if np.sum(labels == i) > 0 else centroids[i]
                for i in range(k)
            ])
            
            # Check convergence
            if np.allclose(centroids, new_centroids, atol=tol):
                break
                
            centroids = new_centroids
            
        return centroids, labels
    
    def _build_embedding_clusters(self, force_rebuild=False):
        """
        Build K-means clusters for embeddings for coarse-to-fine search.
        Clusters are cached and rebuilt only when necessary.
        """
        cluster_cache_path = os.path.join(self.db_path.replace('.db', '_clusters.pkl'))
        
        # Check if we need to rebuild clusters
        if not force_rebuild and os.path.exists(cluster_cache_path):
            # Check if cache is newer than database
            cache_mtime = os.path.getmtime(cluster_cache_path)
            db_mtime = os.path.getmtime(self.db_path)
            
            if cache_mtime > db_mtime:
                try:
                    with open(cluster_cache_path, 'rb') as f:
                        cluster_data = pickle.load(f)
                    logger.debug("Loaded cached embedding clusters")
                    return cluster_data
                except Exception as e:
                    logger.warning("Failed to load cached clusters: %s", e)
        
        # Build new clusters
        logger.info("Building embedding clusters for faster search...")
        
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute("SELECT c.id, c.command, e.embedding FROM commands c JOIN embeddings e ON c.id = e.rowid")
            all_data = c.fetchall()
        
        if len(all_data) < 50:  # Not enough data for clustering
            return None
            
        # Extract embeddings based on storage format
        embeddings_list = []
        for _, _, emb in all_data:
            if EMBEDDING_DTYPE == np.int8:
                stored_emb = np.frombuffer(emb, dtype=np.int8)[:384]
            elif EMBEDDING_DTYPE == np.float16:
                stored_emb = np.frombuffer(emb, dtype=np.float16)[:384]
            elif EMBEDDING_DTYPE == np.float32:
                stored_emb = np.frombuffer(emb, dtype=np.float32)[:384]
            else:
                raise ValueError(f"Unsupported EMBEDDING_DTYPE: {EMBEDDING_DTYPE}")
            
            embeddings_list.append(self.dequantize_embedding(stored_emb))
        
        embeddings = np.vstack(embeddings_list)
        
        # Determine number of clusters (roughly sqrt(n) clusters)
        n_clusters = max(10, min(100, int(np.sqrt(len(all_data)))))
        logger.debug("Creating %d clusters for %d commands", n_clusters, len(all_data))
        
        # Build clusters
        centroids, labels = self._kmeans_numpy(embeddings, n_clusters)
        
        # Build cluster mapping
        cluster_data = {
            'centroids': centroids,
            'cluster_map': {},  # cluster_id -> [command_ids]
            'command_clusters': {}  # command_id -> cluster_id
        }
        
        for i, (cmd_id, command, _) in enumerate(all_data):
            cluster_id = labels[i]
            if cluster_id not in cluster_data['cluster_map']:
                cluster_data['cluster_map'][cluster_id] = []
            cluster_data['cluster_map'][cluster_id].append((cmd_id, command))
            cluster_data['command_clusters'][cmd_id] = cluster_id
        
        # Cache the clusters
        try:
            with open(cluster_cache_path, 'wb') as f:
                pickle.dump(cluster_data, f)
            logger.debug("Cached embedding clusters to %s", cluster_cache_path)
        except Exception as e:
            logger.warning("Failed to cache clusters: %s", e)
            
        return cluster_data

    def _init_model_async(self):
        """Initialize model in background thread"""
        def load_model():
            try:
                logger.debug("Initializing model (loading cached or downloading)")
                self.init_model_sync()
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
        
    def basic_search(self, query, top_k=50):
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
    
    def _keyword_only_search(self, query, top_k=100, return_scores=False):
        """Perform BM25-only keyword search when embedding model is unavailable"""
        logger.debug("Performing keyword-only search for: %s", query)
        
        c = self.conn.cursor()
        # Get all commands for BM25 scoring
        c.execute("""
            SELECT id, command FROM commands 
            ORDER BY last_used DESC
            LIMIT 2000
        """)
        
        commands_data = c.fetchall()
        if not commands_data:
            logger.debug("No commands found in database")
            return []
        
        # Extract commands and compute BM25 scores
        commands = [row[1] for row in commands_data]
        logger.debug("Computing BM25 scores for %d commands", len(commands))
        
        try:
            bm25_scores = self.compute_bm25_scores(query, commands)
            
            # Create results with scores
            scored_results = list(zip(commands, bm25_scores))
            # Sort by BM25 score (descending)
            scored_results.sort(key=lambda x: x[1], reverse=True)
            
            # Filter out zero scores and limit results
            filtered_results = [(cmd, score) for cmd, score in scored_results if score > 0.0][:top_k]
            
            if return_scores:
                # Return format: (command, combined_score, semantic_score, bm25_score)
                # Since we only have BM25, semantic_score = 0
                return [(cmd, score, 0.0, score) for cmd, score in filtered_results]
            else:
                return filtered_results
                
        except Exception as e:
            logger.error("Error in keyword-only search: %s", str(e))
            # Final fallback to basic search
            basic_results = self.basic_search(query, top_k)
            if return_scores:
                return [(cmd, 0.5, 0.0, 0.5) for cmd, score in basic_results]  # Give basic results moderate score
            else:
                return basic_results
    
    def is_model_ready(self):
        """Check if model is ready without waiting"""
        return (hasattr(self, '_model_ready') and 
                self._model_ready is not None and 
                self._model_ready.is_set())
    
    def get_indexed_count(self):
        """Get the total number of indexed commands from metadata (more accurate)."""
        try:
            return int(self.get_metadata('item_count', '0'))
        except Exception as e:
            logger.error("Error getting indexed count from metadata: %s", str(e))
            # Fallback to direct count
            try:
                c = self.conn.cursor()
                c.execute('SELECT COUNT(*) FROM commands')
                return c.fetchone()[0]
            except Exception:
                return 0
    
    def get_metadata(self, key, default=None):
        """Get a metadata value by key."""
        try:
            c = self.conn.cursor()
            c.execute('SELECT value FROM metadata WHERE key = ?', (key,))
            result = c.fetchone()
            return result[0] if result else default
        except Exception as e:
            logger.error("Error getting metadata key '%s': %s", key, str(e))
            return default
    
    def set_metadata(self, key, value):
        """Set a metadata key-value pair."""
        try:
            c = self.conn.cursor()
            c.execute('''
                INSERT OR REPLACE INTO metadata (key, value, updated_at) 
                VALUES (?, ?, CURRENT_TIMESTAMP)
            ''', (key, str(value)))
            self.conn.commit()
        except Exception as e:
            logger.error("Error setting metadata key '%s': %s", key, str(e))
    
    def _init_metadata(self):
        """Initialize metadata with default values."""
        # Set schema version for future migrations
        if not self.get_metadata('schema_version'):
            self.set_metadata('schema_version', '1.0')
        
        # Set embedding model version
        if not self.get_metadata('embedding_model'):
            self.set_metadata('embedding_model', 'minilm-l6-v2-terminal-describer')
        
        # Store embedding quantization level for compatibility checking
        dtype_str = str(EMBEDDING_DTYPE).replace("<class 'numpy.", "").replace("'>", "")
        if not self.get_metadata('embedding_dtype'):
            self.set_metadata('embedding_dtype', dtype_str)
        else:
            # Warn if dtype has changed (requires re-ingestion)
            stored_dtype = self.get_metadata('embedding_dtype')
            if stored_dtype != dtype_str:
                logger.warning("Embedding dtype changed from %s to %s - you may need to regenerate the database for best quality", 
                              stored_dtype, dtype_str)
        
        # Set initial item count if not exists
        if not self.get_metadata('item_count'):
            # Count existing items for initial setup
            try:
                c = self.conn.cursor()
                c.execute('SELECT COUNT(*) FROM commands')
                count = c.fetchone()[0]
                self.set_metadata('item_count', count)
            except Exception:
                self.set_metadata('item_count', 0)
    
    def _update_item_count(self):
        """Update the item count in metadata to ensure accuracy."""
        try:
            c = self.conn.cursor()
            c.execute('SELECT COUNT(*) FROM commands')
            count = c.fetchone()[0]
            self.set_metadata('item_count', count)
            logger.debug("Updated item count to %d", count)
        except Exception as e:
            logger.error("Error updating item count: %s", str(e))
    
    def get_database_info(self):
        """Get comprehensive database information for status display."""
        # Calculate database size
        db_size_bytes = self._get_database_size()
        db_size_human = self._format_bytes(db_size_bytes)
        
        return {
            'item_count': self.get_indexed_count(),
            'embedding_model': self.get_metadata('embedding_model', 'unknown'),
            'embedding_dtype': self.get_metadata('embedding_dtype', 'int8'),
            'schema_version': self.get_metadata('schema_version', '1.0'),
            'last_updated': self.get_metadata('last_updated', 'never'),
            'db_size_bytes': db_size_bytes,
            'db_size_human': db_size_human
        }
    
    def _get_database_size(self):
        """Get database size in bytes."""
        if hasattr(self, '_conn') and self._conn:
            # For in-memory databases, estimate size based on page count
            c = self._conn.cursor()
            try:
                c.execute('PRAGMA page_count')
                page_count = c.fetchone()[0]
                c.execute('PRAGMA page_size')
                page_size = c.fetchone()[0]
                return page_count * page_size
            except Exception:
                # Fallback: estimate based on record count
                return self.get_indexed_count() * 1024  # Rough estimate
        elif os.path.exists(self.db_path):
            # For file-based databases
            return os.path.getsize(self.db_path)
        else:
            return 0
    
    def _format_bytes(self, size_bytes):
        """Format bytes into human readable format."""
        if size_bytes < 1024:
            return f"{size_bytes}B"
        elif size_bytes < 1024**2:
            return f"{size_bytes/1024:.1f}KB"
        elif size_bytes < 1024**3:
            return f"{size_bytes/(1024**2):.1f}MB"
        else:
            return f"{size_bytes/(1024**3):.1f}GB"
    
    def optimize_db_connection(self):
        """Apply SQLite optimizations (defensive for in-memory databases)"""
        c = self.conn.cursor()
        try:
            # These optimizations work for both file and in-memory databases
            c.execute('PRAGMA synchronous = NORMAL')
            c.execute('PRAGMA cache_size = -2000000')  # 2GB cache
            c.execute('PRAGMA temp_store = MEMORY')
            c.execute('PRAGMA case_sensitive_like = false')
            
            # These only work for file-based databases
            if not self.db_path.startswith('file:') or 'memory' not in self.db_path:
                c.execute('PRAGMA mmap_size = 30000000000')  # 30GB max mmap
                c.execute('PRAGMA journal_mode = WAL')  # Write-Ahead Logging
        except Exception as e:
            logger.debug("Some database optimizations failed: %s", str(e))

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
        
        # Create metadata table for tracking database state and versions
        c.execute('''
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
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
        
        # Initialize metadata with default values
        self._init_metadata()
        
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
            # Update item count metadata after successful addition
            self._update_item_count()
            
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
        """
        Optimized tokenization for BM25 search:
        - Focus on mid-to-low frequency terms
        - Filter out ultra-common stopwords and single characters
        - Remove one-off typos by requiring minimum length
        """
        start_time = time.time()
        
        # Common shell/command stopwords to exclude from BM25 indexing
        SHELL_STOPWORDS = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'in', 'is', 'it', 'of', 'on', 'or', 'to', 'the', 'that', 'this',
            'with', 'will', 'was', 'were', 'been', 'have', 'has', 'had',
            # Shell-specific common terms that add little search value
            'run', 'cmd', 'sh', 'bash', 'exe', 'bin', 'usr', 'var', 'tmp',
            'home', 'root', 'etc', 'opt', 'dev', 'proc', 'sys'
        }
        
        # Extract tokens including punctuation for command structure
        tokens = re.findall(r'\w+|[^\w\s]', text.lower())
        
        # Filter tokens for better BM25 precision
        filtered_tokens = []
        for token in tokens:
            # Keep punctuation as it's important for command structure
            if not token.isalnum():
                filtered_tokens.append(token)
                continue
                
            # Skip ultra-short tokens (likely typos or noise)
            if len(token) < 2:
                continue
            # Skip common stopwords
            if token in SHELL_STOPWORDS:
                continue
            # Skip purely numeric tokens that are too long (likely timestamps)
            if token.isdigit() and len(token) > 5:
                continue
            
            filtered_tokens.append(token)
        
        logger.debug("Tokenized text in %.3fs (%d->%d tokens)", 
                    time.time() - start_time, len(tokens), len(filtered_tokens))
        return filtered_tokens

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
        
        # Update accurate item count in metadata
        self._update_item_count()
        
        # Update last ingestion timestamp
        self.set_metadata('last_updated', time.strftime('%Y-%m-%d %H:%M:%S'))
        
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

    def wait_for_model(self, timeout=10.0):
        """Wait for the model to be ready - using sync approach since model loads quickly"""
        logger.debug(f"[WAIT] wait_for_model called")
        
        # Check if model is already ready
        if hasattr(self, '_model') and self._model is not None:
            logger.debug("[WAIT] Model already loaded")
            return True
            
        # Model loads in ~0.3s, so just do it synchronously
        logger.debug("[WAIT] Loading model synchronously")
        try:
            start_time = time.time()
            result = self.init_model_sync()
            load_time = time.time() - start_time
            
            if result is not None:
                logger.info(f"[WAIT] Model loaded successfully in {load_time:.3f}s")
                return True
            else:
                logger.error(f"[WAIT] Model initialization failed after {load_time:.3f}s")
                return False
                
        except Exception as e:
            logger.error(f"[WAIT] Model initialization exception: {e}")
            return False

    def search(self, query, top_k=100, return_scores=False, progress_callback=None):
        """
        Search for commands using hybrid BM25 + semantic search
        
        Args:
            query: Search query
            top_k: Maximum number of results to return
            return_scores: Whether to return detailed scores
            progress_callback: Optional callback function(current, total, stage, partial_results)
        """
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
            
        # Ensure model is ready - NO FALLBACK, fail fast to expose issues
        model_ready = self.wait_for_model(timeout=5.0)
        if not model_ready:
            error_msg = "❌ SEMANTIC MODEL FAILED TO INITIALIZE - This is the core issue that needs fixing!"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        model = self.model
            
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
            error_msg = f"❌ QUERY EMBEDDING GENERATION FAILED: {str(e)} - This needs to be fixed, not hidden!"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
        
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
        
        total_records = len(command_ids)
        logger.debug("Processing %d commands for ranking (query terms: %s)", total_records, query_terms)
        
        # Progress callback: Starting processing
        if progress_callback:
            progress_callback(0, total_records, "Loading embeddings...", [])
        
        # Convert embeddings to numpy array for vectorized operations
        embed_start = time.time()
        
        # Load embeddings based on configured storage format - with progress tracking
        batch_size = 1000  # Process in batches to show progress for large datasets
        num_batches = (len(embeddings) + batch_size - 1) // batch_size
        embeddings_list = []
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(embeddings))
            batch_embeddings = embeddings[start_idx:end_idx]
            
            # Progress callback for batch processing
            if progress_callback and len(embeddings) > 500:  # Only show progress for larger datasets
                progress_callback(end_idx, total_records, f"Loading embeddings batch {batch_idx + 1}/{num_batches}...", [])
            
            if EMBEDDING_DTYPE == np.int8:
                # Batch process INT8 embeddings
                raw_batch = np.array([np.frombuffer(emb, dtype=np.int8)[:384] for emb in batch_embeddings])
                batch_array = raw_batch.astype(np.float32) / EMBEDDING_SCALE_FACTOR
            elif EMBEDDING_DTYPE == np.float16:
                # Batch process FP16 embeddings
                raw_batch = np.array([np.frombuffer(emb, dtype=np.float16)[:384] for emb in batch_embeddings])
                batch_array = raw_batch.astype(np.float32)
                # Clean any NaN/inf values
                batch_array = np.nan_to_num(batch_array, nan=0.0, posinf=1.0, neginf=-1.0)
            elif EMBEDDING_DTYPE == np.float32:
                # Batch process FP32 embeddings
                raw_batch = np.array([np.frombuffer(emb, dtype=np.float32)[:384] for emb in batch_embeddings])
                batch_array = raw_batch.astype(np.float32)
                # Clean any NaN/inf values
                batch_array = np.nan_to_num(batch_array, nan=0.0, posinf=1.0, neginf=-1.0)
            else:
                raise ValueError(f"Unsupported EMBEDDING_DTYPE: {EMBEDDING_DTYPE}")
            
            embeddings_list.append(batch_array)
        
        # Combine all batches
        embeddings_array = np.vstack(embeddings_list) if embeddings_list else np.array([]).reshape(0, MODEL_OUTPUT_DIM)
        
        # Ensure all embeddings are exactly 384 dimensions
        if embeddings_array.shape[1] > MODEL_OUTPUT_DIM:
            embeddings_array = embeddings_array[:, :MODEL_OUTPUT_DIM]
        elif embeddings_array.shape[1] < MODEL_OUTPUT_DIM:
            # Pad with zeros if needed
            padding = np.zeros((embeddings_array.shape[0], MODEL_OUTPUT_DIM - embeddings_array.shape[1]), dtype=np.float32)
            embeddings_array = np.hstack([embeddings_array, padding])
        logger.debug("Loaded %s embeddings with shape %s", 
                    str(EMBEDDING_DTYPE).split('.')[-1], embeddings_array.shape)
        
        # Progress callback: Embeddings loaded
        if progress_callback:
            progress_callback(total_records, total_records, "Processing query embedding...", [])
        
        # First ensure query embedding is the right shape
        logger.debug("Query embedding shape before processing: %s", query_embedding.shape)
        query_embedding = np.array(query_embedding).flatten()[:384]  # Ensure 1D array of right size
        logger.debug("Query embedding shape after flattening and truncating: %s", query_embedding.shape)
        
        # Query embedding needs dequantization - optimize for consistency with batch processing
        if EMBEDDING_DTYPE == np.int8:
            query_embedding = query_embedding.astype(np.float32) / EMBEDDING_SCALE_FACTOR
        elif EMBEDDING_DTYPE == np.float16:
            query_embedding = query_embedding.astype(np.float32)
            query_embedding = np.nan_to_num(query_embedding, nan=0.0, posinf=1.0, neginf=-1.0)
        elif EMBEDDING_DTYPE == np.float32:
            query_embedding = query_embedding.astype(np.float32)
            query_embedding = np.nan_to_num(query_embedding, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Ensure exactly 384 dimensions
        if len(query_embedding) < MODEL_OUTPUT_DIM:
            padded = np.zeros(MODEL_OUTPUT_DIM, dtype=np.float32)
            padded[:len(query_embedding)] = query_embedding
            query_embedding = padded
        
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
        
        # Progress callback: Semantic scoring complete
        if progress_callback:
            progress_callback(total_records, total_records, "Computing BM25 scores...", [])
        
        # Avoid division by zero and handle NaN values
        semantic_scores = np.divide(
            semantic_scores, 
            norms, 
            out=np.zeros_like(semantic_scores), 
            where=norms != 0
        )
        
        # Additional NaN/inf cleanup after division
        semantic_scores = np.nan_to_num(semantic_scores, nan=0.0, posinf=1.0, neginf=0.0)
        
        # Ensure scores are in [0, 1] range
        semantic_scores = (semantic_scores + 1) / 2  # Convert from [-1, 1] to [0, 1]
        semantic_scores = np.clip(semantic_scores, 0, 1)
        
        # Final check for any remaining invalid values
        semantic_scores = np.nan_to_num(semantic_scores, nan=0.0, posinf=1.0, neginf=0.0)
        
        # Calculate BM25 scores in batch
        bm25_scores = self.calculate_bm25_scores_batch(query_terms, command_ids)
        
        # Normalize BM25 scores
        if len(bm25_scores) > 0:
            # Clean any NaN/inf values before normalization
            bm25_scores = np.nan_to_num(bm25_scores, nan=0.0, posinf=1.0, neginf=0.0)
            # Add small epsilon to avoid division by zero
            max_score = np.maximum(bm25_scores.max(), 1e-6)
            bm25_scores = bm25_scores / max_score  # Normalize to [0, 1]
            bm25_scores = np.clip(bm25_scores, 0, 1)
            # Final cleanup after normalization
            bm25_scores = np.nan_to_num(bm25_scores, nan=0.0, posinf=1.0, neginf=0.0)
        
        # Progress callback: Combining scores
        if progress_callback:
            progress_callback(total_records, total_records, "Combining scores...", [])
        
        # Dynamic hybrid scoring: adjust weights based on score strengths
        combined_scores = np.array([
            self._dynamic_hybrid_score(sem, bm25) 
            for sem, bm25 in zip(semantic_scores, bm25_scores)
        ])
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
        
        # Progress callback: Sorting results
        if progress_callback:
            progress_callback(total_records, total_records, "Sorting results...", [])
        
        results.sort(key=sort_key, reverse=True)
        final_results = results[:top_k]
        
        # Progress callback: Complete with final results
        if progress_callback:
            progress_callback(total_records, total_records, "Complete", final_results)
        
        # Log top results with scores for debugging
        if final_results:
            logger.debug("Top %d results:", len(final_results))
            for i, result in enumerate(final_results[:3], 1):  # Log top 3 for debugging
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
        self.cache_results(query, final_results, return_scores)
        logger.debug("Results cached in %.3fs", time.time() - cache_start)
        
        total_time = time.time() - start_time
        logger.info("Total search completed in %.3fs with %d results", total_time, len(final_results))
        
        return final_results
    
    def tui(self, show_scoring=False):
        """Launch the interactive TUI for this FuzzyShell instance"""
        def search_callback(query: str) -> list:
            """Callback function for the UI to perform searches"""
            if not query:
                return []
            # Always return detailed scores for the TUI
            return self.search(query, return_scores=True)

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
        # Always return detailed scores for the TUI
        return fuzzyshell.search(query, return_scores=True)

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
