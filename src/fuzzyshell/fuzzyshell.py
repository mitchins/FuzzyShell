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
from .model_configs import get_active_model_key, get_model_config
from .fuzzy_tui import FuzzyShellApp, IngestionProgressTUI

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
console_handler.setLevel(logging.CRITICAL)
console_handler.setFormatter(logging.Formatter('%(message)s'))

# Configure root logger
logger = logging.getLogger('FuzzyShell')
logger.setLevel(logging.DEBUG)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Configuration constants for embedding storage
EMBEDDING_DTYPE = np.float32  # Options: np.int8, np.float16, np.float32 (using FP32 for better precision)
EMBEDDING_SCALE_FACTOR = 127  # Only used for INT8 quantization

# ANN Search Configuration
USE_ANN_SEARCH = True  # Enable K-means clustering for approximate nearest neighbor search
ANN_NUM_CLUSTERS = 32  # Number of clusters for K-means (tune based on dataset size)
ANN_CLUSTER_CANDIDATES = 6  # Number of closest clusters to search in (optimized for speed)

class ANNSearchIndex:
    """
    K-means based Approximate Nearest Neighbor search index for fast embedding similarity search.
    Uses numpy-only implementation to avoid heavy dependencies.
    """
    
    def __init__(self, n_clusters=ANN_NUM_CLUSTERS, max_iterations=100):
        self.n_clusters = n_clusters
        self.max_iterations = max_iterations
        self.cluster_centers = None
        self.cluster_assignments = None
        self.cluster_indices = None  # Maps cluster_id -> list of embedding indices
        self.embeddings = None
        self.is_trained = False
        
    def fit(self, embeddings):
        """Train K-means clustering on embeddings."""
        if len(embeddings) < self.n_clusters:
            # Too few embeddings for clustering, use linear search
            self.is_trained = False
            return
            
        logger.debug("Training K-means clustering with %d clusters on %d embeddings", 
                    self.n_clusters, len(embeddings))
        start_time = time.time()
        
        self.embeddings = embeddings.copy()
        n_samples, n_features = embeddings.shape
        
        # Initialize centroids using k-means++ for better initial placement
        self.cluster_centers = self._init_centroids_plus_plus(embeddings)
        
        # Run K-means iterations
        for iteration in range(self.max_iterations):
            # Assign points to closest centroids
            distances = np.linalg.norm(embeddings[:, np.newaxis, :] - self.cluster_centers, axis=2)
            new_assignments = np.argmin(distances, axis=1)
            
            # Check for convergence
            if iteration > 0 and np.array_equal(new_assignments, self.cluster_assignments):
                logger.debug("K-means converged after %d iterations", iteration + 1)
                break
                
            self.cluster_assignments = new_assignments
            
            # Update centroids
            for k in range(self.n_clusters):
                cluster_mask = self.cluster_assignments == k
                if np.any(cluster_mask):
                    self.cluster_centers[k] = np.mean(embeddings[cluster_mask], axis=0)
        
        # Build cluster index mapping
        self.cluster_indices = {}
        for k in range(self.n_clusters):
            self.cluster_indices[k] = np.where(self.cluster_assignments == k)[0].tolist()
            
        self.is_trained = True
        train_time = time.time() - start_time
        logger.debug("K-means training completed in %.3fs", train_time)
        
    def _init_centroids_plus_plus(self, embeddings):
        """Initialize centroids using k-means++ algorithm for better clustering."""
        n_samples, n_features = embeddings.shape
        centroids = np.zeros((self.n_clusters, n_features))
        
        # Choose first centroid randomly
        centroids[0] = embeddings[np.random.randint(n_samples)]
        
        # Choose remaining centroids
        for k in range(1, self.n_clusters):
            # Calculate distances to closest existing centroid
            distances = np.array([
                min([np.linalg.norm(x - c)**2 for c in centroids[:k]]) 
                for x in embeddings
            ])
            
            # Choose next centroid with probability proportional to squared distance
            distance_sum = distances.sum()
            if distance_sum > 0:
                probabilities = distances / distance_sum
                cumulative_probabilities = probabilities.cumsum()
                r = np.random.rand()
            else:
                # All points are identical - choose randomly
                r = 0.5
                cumulative_probabilities = np.linspace(0, 1, len(distances))
            
            for i, p in enumerate(cumulative_probabilities):
                if r < p:
                    centroids[k] = embeddings[i]
                    break
                    
        return centroids
        
    def search_candidates(self, query_embedding, n_candidates=ANN_CLUSTER_CANDIDATES):
        """
        Find candidate embedding indices using ANN search.
        Returns indices of embeddings that are likely to be similar to query.
        """
        if not self.is_trained or self.cluster_centers is None:
            # Fall back to all indices if not trained
            logger.warning("ANN index not properly trained, falling back to linear search")
            return list(range(len(self.embeddings))) if self.embeddings is not None else []
            
        # Find closest clusters to query
        cluster_distances = np.linalg.norm(self.cluster_centers - query_embedding, axis=1)
        closest_clusters = np.argsort(cluster_distances)[:n_candidates]
        
        # Collect all indices from closest clusters
        candidate_indices = []
        for cluster_id in closest_clusters:
            candidate_indices.extend(self.cluster_indices.get(cluster_id, []))
            
        logger.info("ANN search: query mapped to %d clusters out of %d total, returning %d candidates out of %d total embeddings", 
                   len(closest_clusters), len(self.cluster_centers), len(candidate_indices), len(self.embeddings))
        return candidate_indices

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
        
        # ANN search index
        self.ann_index = ANNSearchIndex() if USE_ANN_SEARCH else None
        if USE_ANN_SEARCH and self.ann_index:
            # Try to load pre-trained index, rebuild if missing/outdated
            if not self._load_ann_index():
                print("🔧 ANN index missing or outdated - rebuilding automatically...")
                print("⏳ This may take a few seconds for large command histories...")
                logger.info("ANN index missing or outdated - rebuilding automatically...")
                
                # Check if database has commands before attempting rebuild
                if self._conn is not None:
                    # Ensure database tables exist first
                    try:
                        c = self._conn.cursor()
                        c.execute("SELECT COUNT(*) FROM commands")
                        command_count = c.fetchone()[0]
                        if command_count == 0:
                            logger.info("Database is empty, skipping ANN index build for now")
                            print("⚠️  Database is empty, ANN index will be built after commands are added")
                        else:
                            self._rebuild_ann_index()
                            print("✅ ANN index rebuild complete")
                            if not self.ann_index.is_trained:
                                logger.error("Failed to build ANN index during startup")
                                raise RuntimeError("ANN index build failed despite having commands.")
                    except sqlite3.OperationalError:
                        # Tables don't exist yet, database not initialized
                        logger.info("Database tables not initialized yet, skipping ANN index build for now")
                        print("⚠️  Database not initialized, ANN index will be built after setup")
                else:
                    # No connection yet, defer ANN building
                    logger.info("No database connection yet, deferring ANN index build")
        
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
    
    def create_thread_connection(self):
        """Create a new database connection for use in background threads."""
        if self.db_path == ":memory:":
            # For in-memory databases, we need to share the connection
            return self._conn
        
        # Create new connection for this thread
        if self.db_path.startswith(":"):
            connect_string = self.db_path
        else:
            connect_string = f"file:{self.db_path}?mode=rwc"
        
        conn = sqlite3.connect(
            connect_string,
            uri=True,
            timeout=60.0
        )
        
        # Apply same optimizations
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=10000")
        conn.execute("PRAGMA temp_store=MEMORY")
        
        # Try to enable extensions if available
        try:
            conn.enable_load_extension(True)
            sqlite_vss.load(conn)
        except (AttributeError, Exception) as e:
            logger.warning("Could not load SQLite extensions on thread connection: %s", str(e))
        
        return conn
    
    def init_model_sync(self):
        """Initialize the ONNX model handler synchronously"""
        start_time = time.time()
        logger.debug("[SYNC] Starting synchronous model initialization")
        
        # Create thread-local database connection for model initialization
        thread_conn = None
        try:
            logger.debug("[SYNC] Creating thread-local database connection...")
            thread_conn = self.create_thread_connection()
            
            logger.debug("[SYNC] Creating ModelHandler instance...")
            handler_start = time.time()
            
            self._model = ModelHandler()
            
            handler_time = time.time() - handler_start
            logger.debug(f"[SYNC] ModelHandler created in {handler_time:.3f}s")
            
            # Check for model changes and handle them using thread connection
            logger.debug("[SYNC] Checking for model changes...")
            self._handle_model_change_with_conn(thread_conn)
            
            # Store current model metadata if not already stored
            metadata_model = self._get_metadata_with_conn(thread_conn, 'model_name')
            if not metadata_model:
                self._store_model_metadata_with_conn(thread_conn)
            
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
            raise
        finally:
            # Close thread connection if it's not the shared one
            if thread_conn and thread_conn != self._conn:
                thread_conn.close()
            
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
    
    def _dynamic_hybrid_score(self, semantic_score, bm25_score, command_text="", query_text=""):
        """
        Rarity-based hybrid scoring:
        1. Semantic similarity is primary ranking signal (captures intent)
        2. BM25 only boosts when query contains genuinely rare terms
        3. Common terms like 'list', 'file', 'get' don't get BM25 boost
        4. Rare terms like 'lfs', 'kubectl', specific flags do get boost
        """
        # Apply phrase proximity penalty to BM25 for multi-word queries
        if len(query_text.split()) > 1 and len(command_text.split()) > 1:
            original_bm25 = bm25_score
            bm25_score = self._apply_phrase_penalty(bm25_score, query_text, command_text)
            # Debug: Always log for git lfs ls-files to see what's happening
            if "git lfs ls-files" in command_text and "list" in query_text:
                logger.info("DEBUG: BM25 penalty for problematic case: '%s' vs '%s': %.3f -> %.3f", 
                           query_text, command_text, original_bm25, bm25_score)
            elif abs(bm25_score - original_bm25) > 0.01:  # Log other significant changes
                logger.info("BM25 phrase penalty: '%s' vs '%s': %.3f -> %.3f", 
                           query_text, command_text, original_bm25, bm25_score)
        # Check if query contains rare terms that deserve BM25 boost
        has_rare_terms = self._query_has_rare_terms(query_text)
        
        # Debug rarity detection for problematic cases (can be removed in production)
        # if "ollama list" in command_text and "list" in query_text:
        #     logger.info("🔍 RARITY DEBUG: query='%s', has_rare_terms=%s, bm25=%.3f", query_text, has_rare_terms, bm25_score)
        
        # If no rare terms, almost entirely favor semantic similarity
        if not has_rare_terms:
            result = 0.99 * semantic_score + 0.01 * bm25_score
            if ("ollama list" in command_text or "ls -lh" in command_text) and "list" in query_text:
                logger.info("🎯 COMMON TERMS: '%s' vs '%s' → Sem:%.3f BM25:%.3f → Final:%.3f", 
                           query_text, command_text, semantic_score, bm25_score, result)
            return result
        
        # With rare terms, use adaptive weighting based on semantic confidence
        high_semantic_threshold = 0.4
        low_semantic_threshold = 0.2 
        high_bm25_threshold = 0.6
        
        # Calculate score difference
        score_diff = abs(semantic_score - bm25_score)
        
        # Prioritize semantic similarity more with the improved model
        if semantic_score > high_semantic_threshold:
            if bm25_score > high_bm25_threshold and score_diff > 0.4:
                # Both high but very different - balanced but favor semantic
                return 0.6 * semantic_score + 0.4 * bm25_score
            else:
                # High semantic confidence - strongly favor semantic (90%)
                return 0.90 * semantic_score + 0.10 * bm25_score
                
        elif semantic_score < low_semantic_threshold:
            if bm25_score > high_bm25_threshold:
                # Low semantic, high BM25 - favor BM25 but not as strongly (60%)
                return 0.4 * semantic_score + 0.6 * bm25_score
            else:
                # Both low - slight semantic preference for rare term matching
                return 0.55 * semantic_score + 0.45 * bm25_score
        else:
            # Medium semantic score - favor semantic more strongly  
            return 0.8 * semantic_score + 0.2 * bm25_score
    
    def _apply_phrase_penalty(self, bm25_score, query_text, command_text):
        """
        Apply penalty to BM25 score for poor phrase matching in multi-word queries.
        Penalizes commands where query words are out of order or far apart.
        """
        query_words = query_text.lower().split()
        command_words = command_text.lower().split()
        
        if len(query_words) < 2:
            return bm25_score
            
        # Find positions of query words in command
        word_positions = {}
        for i, cmd_word in enumerate(command_words):
            for query_word in query_words:
                # Allow partial matches (ls matches list)
                if query_word in cmd_word or cmd_word in query_word:
                    if query_word not in word_positions:
                        word_positions[query_word] = []
                    word_positions[query_word].append(i)
        
        # Calculate phrase penalties
        penalty_factor = 1.0
        
        # Penalty 1: Missing words penalty
        found_words = len(word_positions)
        if found_words < len(query_words):
            missing_ratio = (len(query_words) - found_words) / len(query_words)
            penalty_factor *= (1.0 - missing_ratio * 0.6)  # Up to 60% penalty for missing words
        
        # Penalty 2: Word order penalty  
        if found_words >= 2:
            query_pairs = [(query_words[i], query_words[i+1]) for i in range(len(query_words)-1)]
            order_violations = 0
            
            for word1, word2 in query_pairs:
                if word1 in word_positions and word2 in word_positions:
                    # Check if any occurrence of word1 comes before word2
                    found_correct_order = False
                    for pos1 in word_positions[word1]:
                        for pos2 in word_positions[word2]:
                            if pos1 < pos2:
                                found_correct_order = True
                                break
                        if found_correct_order:
                            break
                    
                    if not found_correct_order:
                        order_violations += 1
            
            if order_violations > 0:
                order_penalty = order_violations / len(query_pairs)
                penalty_factor *= (1.0 - order_penalty * 0.4)  # Up to 40% penalty for wrong order
        
        # Debug logging for phrase penalties
        if penalty_factor < 0.9:  # Only log when significant penalty applied
            logger.debug("Phrase penalty: '%s' -> '%s': factor=%.3f (was %.3f, now %.3f)", 
                        query_text, command_text, penalty_factor, bm25_score, bm25_score * penalty_factor)
        
        return bm25_score * penalty_factor
    
    def _query_has_rare_terms(self, query_text):
        """
        Determine if query contains rare/specific terms that deserve BM25 boost.
        Common shell terms like 'list', 'file', 'get' are not rare.
        Technical terms like 'lfs', 'kubectl', specific flags are rare.
        """
        # Common shell/command terms that shouldn't get BM25 boost
        common_terms = {
            'list', 'lists', 'file', 'files', 'get', 'show', 'display', 'view',
            'open', 'create', 'make', 'new', 'delete', 'remove', 'copy', 'move',
            'find', 'search', 'run', 'execute', 'start', 'stop', 'install',
            'update', 'upgrade', 'download', 'upload', 'save', 'load', 'edit',
            'change', 'modify', 'set', 'config', 'configuration', 'settings',
            'help', 'info', 'status', 'version', 'test', 'check', 'verify'
        }
        
        query_words = set(word.lower().strip() for word in query_text.split())
        
        # Remove common terms
        rare_words = query_words - common_terms
        
        # Also remove single characters and very short words (often noise)
        rare_words = {word for word in rare_words if len(word) > 2}
        
        # If we have any remaining words, they're potentially rare
        has_rare = bool(rare_words)
        
        if has_rare:
            logger.debug("Rare terms detected in '%s': %s", query_text, rare_words)
        
        return has_rare
    
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
    
    def _get_metadata_with_conn(self, conn, key, default=None):
        """Get a metadata value by key using provided connection."""
        try:
            c = conn.cursor()
            c.execute('SELECT value FROM metadata WHERE key = ?', (key,))
            result = c.fetchone()
            return result[0] if result else default
        except Exception as e:
            logger.error("Error getting metadata key '%s': %s", key, str(e))
            return default
    
    def _set_metadata_with_conn(self, conn, key, value):
        """Set a metadata key-value pair using provided connection."""
        try:
            c = conn.cursor()
            c.execute('''
                INSERT OR REPLACE INTO metadata (key, value, updated_at) 
                VALUES (?, ?, CURRENT_TIMESTAMP)
            ''', (key, str(value)))
            conn.commit()
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
    
    def store_model_metadata(self, model_key: str = None):
        """Store current model information in database metadata."""
        if model_key is None:
            model_key = get_active_model_key()
        
        config = get_model_config(model_key)
        self.set_metadata('model_name', model_key)
        self.set_metadata('model_repo', config['repo'])
        self.set_metadata('model_dimensions', str(config['files']['dimensions']))
        self.set_metadata('model_tokenizer_type', config['files']['tokenizer_type'])
        logger.debug(f"Stored model metadata for: {model_key}")
    
    def _store_model_metadata_with_conn(self, conn, model_key: str = None):
        """Store current model information in database metadata using provided connection."""
        if model_key is None:
            model_key = get_active_model_key()
        
        config = get_model_config(model_key)
        self._set_metadata_with_conn(conn, 'model_name', model_key)
        self._set_metadata_with_conn(conn, 'model_repo', config['repo'])
        self._set_metadata_with_conn(conn, 'model_dimensions', str(config['files']['dimensions']))
        self._set_metadata_with_conn(conn, 'model_tokenizer_type', config['files']['tokenizer_type'])
        logger.debug(f"Stored model metadata for: {model_key}")
    
    def _handle_model_change_with_conn(self, conn):
        """Handle model change using provided connection."""
        current_model = get_active_model_key()
        stored_model = self._get_metadata_with_conn(conn, 'model_name', 'multilingual-l12')
        changed = current_model != stored_model
        
        if changed:
            logger.info(f"Model change detected: {stored_model} -> {current_model}")
            print(f"🔄 Model changed from {stored_model} to {current_model}")
            print("   Clearing old embeddings - they will be rebuilt with the new model...")
            
            # Clear embeddings and related caches using thread connection
            self._clear_embeddings_with_conn(conn)
            self._clear_all_caches_with_conn(conn)
            
            # Update model metadata
            self._store_model_metadata_with_conn(conn, current_model)
            
            # Reset ANN index count since embeddings changed
            self._set_metadata_with_conn(conn, 'ann_command_count', '0')
            self._set_metadata_with_conn(conn, 'poorly_clustered_commands', '0')
            
            return True
        return False
    
    def detect_model_change(self) -> tuple[bool, str, str]:
        """
        Check if the active model has changed since last run.
        
        Returns:
            tuple: (changed, current_model, stored_model)
        """
        current_model = get_active_model_key()
        stored_model = self.get_metadata('model_name', 'multilingual-l12')  # Default to old model
        changed = current_model != stored_model
        
        if changed:
            logger.info(f"Model change detected: {stored_model} -> {current_model}")
        
        return changed, current_model, stored_model
    
    def handle_model_change(self):
        """Handle model change by clearing embeddings and updating metadata."""
        changed, current_model, stored_model = self.detect_model_change()
        
        if changed:
            print(f"🔄 Model changed from {stored_model} to {current_model}")
            print("   Clearing old embeddings - they will be rebuilt with the new model...")
            
            # Clear embeddings and related caches
            self.clear_embeddings()
            self.clear_all_caches()
            
            # Update model metadata
            self.store_model_metadata(current_model)
            
            # Reset ANN index count since embeddings changed
            self.set_metadata('ann_command_count', '0')
            self.set_metadata('poorly_clustered_commands', '0')
            
            return True
        return False
    
    def clear_embeddings(self):
        """Clear all embeddings from the database."""
        try:
            c = self.conn.cursor()
            c.execute("DELETE FROM embeddings")
            self.conn.commit()
            logger.info("Cleared all embeddings due to model change")
        except Exception as e:
            logger.error("Error clearing embeddings: %s", str(e))
    
    def _clear_embeddings_with_conn(self, conn):
        """Clear all embeddings using provided connection."""
        try:
            c = conn.cursor()
            c.execute("DELETE FROM embeddings")
            conn.commit()
            logger.info("Cleared all embeddings due to model change")
        except Exception as e:
            logger.error("Error clearing embeddings: %s", str(e))
    
    def _clear_all_caches_with_conn(self, conn):
        """Clear all caches using provided connection."""
        try:
            # Clear query cache
            c = conn.cursor()
            c.execute("DELETE FROM query_cache")
            conn.commit()
            logger.info("Cleared query cache")
            
            # Clear ANN cluster cache (file-based cache)
            if not self.db_path.startswith(':'):  # Skip for in-memory DBs
                cluster_cache_path = os.path.join(self.db_path.replace('.db', '_clusters.pkl'))
                if os.path.exists(cluster_cache_path):
                    os.remove(cluster_cache_path)
                    logger.info("Cleared ANN cluster cache")
                    
                # Clear ANN index cache (file-based cache)
                index_cache_path = os.path.join(self.db_path.replace('.db', '_index.ann'))
                if os.path.exists(index_cache_path):
                    os.remove(index_cache_path)
                    logger.info("Cleared ANN index cache")
        except Exception as e:
            logger.error("Error clearing caches: %s", str(e))
    
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

    def ingest_history(self, use_tui=True):
        """
        Ingest shell history with optional TUI progress display.
        
        Args:
            use_tui: If True, use TUI progress display. If False or TUI fails, fall back to console.
        """
        start_time = time.time()
        history_file = self.get_shell_history_file()
        logger.info("Starting history ingestion from: %s", history_file)
        
        # Clear all caches when ingesting to ensure fresh results
        logger.info("Clearing caches due to data ingestion...")
        self.clear_all_caches()
        
        # Initialize model if needed and wait for it
        if self._model is None:
            logger.debug("Model not initialized, starting async initialization")
            self._init_model_async()
        
        logger.debug("Waiting for model to be ready...")
        if not hasattr(self, '_model_ready') or self._model_ready is None or not self._model_ready.wait(timeout=30.0):
            logger.error("Model failed to load in time or initialization failed (timeout after 30s)")
            return
            
        try:
            with open(history_file, 'r', encoding='utf-8', errors='ignore') as f:
                commands = set(line.strip() for line in f)
        except FileNotFoundError:
            print(f"History file not found at: {history_file}")
            return

        raw_command_count = len(commands)
        if raw_command_count == 0:
            print("No commands to process.")
            return
        
        logger.info("Starting command processing...")
        c = self.conn.cursor()
        process_start = time.time()
        
        # Filter and clean commands first, maintaining set for deduplication
        print(f"Cleaning {raw_command_count} raw commands...")
        clean_commands = set()
        for raw_command in commands:
            if not raw_command or raw_command.isspace():
                continue
            command = self.clean_shell_command(raw_command)
            if command and not command.isspace():
                clean_commands.add(command)  # Use set to maintain deduplication
        
        if not clean_commands:
            print("No valid commands found after cleaning.")
            return
        
        # Convert to list for batch processing (already deduplicated by set)
        new_commands = list(clean_commands)
        final_command_count = len(new_commands)
        
        print(f"After cleaning and deduplication: {final_command_count} unique commands to process")
        
        # Now initialize TUI with the correct final count
        tui_progress = None
        console_fallback = False
        
        if use_tui:
            try:
                tui_progress = IngestionProgressTUI()
                tui_progress.set_total_commands(final_command_count)
                
                # Start TUI in a separate thread to avoid blocking
                def run_tui():
                    try:
                        loop = tui_progress.start()
                        loop.run()
                    except Exception as e:
                        logger.debug(f"TUI loop ended: {e}")
                
                tui_thread = threading.Thread(target=run_tui, daemon=True)
                tui_thread.start()
                
                # Give TUI a moment to initialize
                time.sleep(0.1)
                
                # Update initial status
                tui_progress.update_progress(0, "Starting batch processing...", "Initialization")
                
            except Exception as e:
                logger.warning(f"Failed to initialize TUI progress: {e}. Falling back to console.")
                console_fallback = True
                tui_progress = None
        else:
            console_fallback = True
        
        # If using console fallback, show simple message
        if console_fallback:
            print(f"Processing {final_command_count} commands...")
        
        # Enable SQLite fast mode for bulk ingestion
        c = self.conn.cursor()
        
        # Store original settings
        original_synchronous = c.execute("PRAGMA synchronous").fetchone()[0]
        original_journal_mode = c.execute("PRAGMA journal_mode").fetchone()[0]
        
        try:
            # Fast mode optimizations for bulk insert
            c.execute("PRAGMA synchronous = OFF")        # Don't wait for disk writes
            c.execute("PRAGMA journal_mode = MEMORY")    # Keep journal in memory
            c.execute("PRAGMA cache_size = -64000")      # 64MB cache
            c.execute("PRAGMA temp_store = MEMORY")      # Temp tables in memory
            logger.debug("Enabled SQLite fast mode for bulk ingestion")
        except Exception as e:
            logger.warning(f"Could not enable fast mode: {e}")
        
        total_processed = 0
        
        if tui_progress:
            try:
                tui_progress.update_progress(10, f"Processing {final_command_count} unique commands", "Command Processing")
            except:
                pass
        
        # DEFERRED APPROACH: Generate all embeddings first, then do database operations
        if console_fallback:
            print("Generating embeddings for all commands...")
        
        # Generate embeddings in batches but store them in memory
        BATCH_SIZE = 32 # Optimal batch size for best performance (256+ commands/sec)
        all_embeddings = []
        all_command_data = []
        
        for batch_start in range(0, final_command_count, BATCH_SIZE):
            batch_commands = new_commands[batch_start:batch_start + BATCH_SIZE]
            
            if tui_progress:
                try:
                    commands_processed = min(batch_start + BATCH_SIZE, final_command_count)
                    total_batches = (final_command_count + BATCH_SIZE - 1) // BATCH_SIZE
                    current_batch = batch_start // BATCH_SIZE + 1
                    tui_progress.update_progress(
                        20 + int((batch_start / final_command_count) * 60),
                        f"Processing {commands_processed}/{final_command_count} commands (batch {current_batch}/{total_batches})",
                        "Embedding Generation"
                    )
                except:
                    pass
            
            # Generate embeddings (the fast part)
            batch_embeddings = self.model.encode(batch_commands)
            all_embeddings.extend(batch_embeddings)
            
            # Prepare command data (no database operations yet)
            for command in batch_commands:
                all_command_data.append((command, len(command.split())))
        
        # Now do ALL database operations in one go
        if console_fallback:
            print("Saving all data to database...")
        if tui_progress:
            try:
                tui_progress.update_progress(80, "Saving to database...", "Database Operations")
            except:
                pass
        
        # Single bulk insert for all commands
        c.executemany("INSERT OR IGNORE INTO commands (command, length) VALUES (?, ?)", all_command_data)
        
        # Get the IDs for all processed commands to ensure correctness
        placeholders = ','.join('?' for _ in new_commands)
        c.execute(f"SELECT id, command FROM commands WHERE command IN ({placeholders})", new_commands)
        command_to_id = {cmd: cmd_id for cmd_id, cmd in c.fetchall()}
        
        # Prepare embedding and term frequency data in one pass
        embedding_data = []
        term_freq_data = []
        
        # Note: all_embeddings is in the same order as new_commands
        for i, command in enumerate(new_commands):
            command_id = command_to_id.get(command)
            if not command_id:
                continue  # Skip if command wasn't inserted for some reason

            # 1. Prepare embedding data
            embedding = all_embeddings[i]
            if len(embedding) > MODEL_OUTPUT_DIM:
                embedding = embedding[:MODEL_OUTPUT_DIM]
            quantized_embedding = self.quantize_embedding(embedding)
            embedding_data.append((command_id, quantized_embedding.tobytes()))

            # 2. Prepare term frequency data for BM25
            terms = self.tokenize(command)
            term_counts = Counter(terms)
            for term, freq in term_counts.items():
                term_freq_data.append((term, command_id, freq))

        if tui_progress:
            try:
                tui_progress.update_progress(85, "Saving embeddings and keyword index...", "Database Write")
            except:
                pass

        # Bulk insert embeddings
        c.executemany("INSERT OR REPLACE INTO embeddings (rowid, embedding) VALUES (?, ?)", embedding_data)
        
        # Bulk insert term frequencies
        c.executemany("INSERT OR REPLACE INTO term_frequencies (term, command_id, freq) VALUES (?, ?, ?)", term_freq_data)
        
        total_processed = len(command_to_id)
        
        if tui_progress:
            tui_progress.processed_commands = total_processed
        
        self.conn.commit()
        self.update_corpus_stats()
        
        # Update metadata
        self._update_item_count()
        self.set_metadata('last_updated', time.strftime('%Y-%m-%d %H:%M:%S'))
        
        # Rebuild ANN index (67-100%)
        if total_processed > 0:
            if tui_progress:
                try:
                    tui_progress.update_progress(67, "Building search index...", "ANN Indexing")
                except:
                    pass
            elif console_fallback:
                print("Building search index...")
                
            logger.info("Rebuilding ANN index after ingestion...")
            self._rebuild_ann_index_with_tui_progress(tui_progress, console_fallback)
        else:
            if tui_progress:
                try:
                    tui_progress.update_progress(100, "No new commands to index", "Complete")
                except:
                    pass
        
        # Finish progress display
        if tui_progress:
            try:
                tui_progress.finish("Ingestion complete!")
            except:
                pass
        elif console_fallback:
            print(f"✅ Ingestion complete! {total_processed} commands processed.")
        
        # Restore original SQLite settings
        try:
            c.execute(f"PRAGMA synchronous = {original_synchronous}")
            c.execute(f"PRAGMA journal_mode = {original_journal_mode}")
            c.execute("PRAGMA cache_size = -2000")  # Reset to default
            self.conn.commit()  # Final commit with restored settings
            logger.debug("Restored SQLite settings after bulk ingestion")
        except Exception as e:
            logger.warning(f"Could not restore SQLite settings: {e}")
        
        total_time = time.time() - start_time
        process_time = time.time() - process_start
        logger.info("Ingestion complete: %d/%d commands processed", total_processed, final_command_count)
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
                # Prevent division by zero if avg_length is 0
                avg_len = max(self.avg_length, 1.0)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_length / avg_len)
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
    
    def clear_all_caches(self):
        """Clear all caches to force fresh results"""
        # Clear query cache
        c = self.conn.cursor()
        c.execute("DELETE FROM query_cache")
        self.conn.commit()
        logger.info("Cleared query cache")
        
        # Clear ANN cluster cache
        cluster_cache_path = os.path.join(self.db_path.replace('.db', '_clusters.pkl'))
        if os.path.exists(cluster_cache_path):
            os.remove(cluster_cache_path)
            logger.info("Cleared ANN cluster cache")
            
        # Clear ANN index cache
        ann_cache_path = os.path.join(self.db_path.replace('.db', '_ann_index.pkl'))
        if os.path.exists(ann_cache_path):
            os.remove(ann_cache_path)
            logger.info("Cleared ANN index cache")
            
        # Reset ANN index to force rebuild
        if hasattr(self, 'ann_index') and self.ann_index:
            self.ann_index = ANNSearchIndex()
            logger.info("Reset ANN index")
    
    def _rebuild_ann_index(self):
        """Rebuild ANN index using all embeddings from database"""
        if not USE_ANN_SEARCH or not hasattr(self, 'ann_index'):
            return
            
        try:
            ann_start = time.time()
            
            # Load all embeddings from database
            c = self.conn.cursor()
            c.execute("SELECT c.id, c.command, e.embedding FROM commands c JOIN embeddings e ON c.id = e.rowid")
            all_data = c.fetchall()
            
            if len(all_data) < ANN_NUM_CLUSTERS:
                logger.info("Too few commands (%d) for ANN clustering, skipping", len(all_data))
                return
                
            # Extract embeddings
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
            
            # Train the index
            logger.info("Training ANN index with %d embeddings (%d clusters, %d candidates)", 
                       len(embeddings), ANN_NUM_CLUSTERS, ANN_CLUSTER_CANDIDATES)
            self.ann_index.fit(embeddings)
            
            ann_time = time.time() - ann_start
            logger.info("ANN index rebuilt in %.3fs", ann_time)
            
        except Exception as e:
            logger.error("Error rebuilding ANN index: %s", e)
            
    def _rebuild_ann_index_with_progress(self, progress):
        """Rebuild ANN index with progress updates (67-100%)"""
        if not USE_ANN_SEARCH or not hasattr(self, 'ann_index'):
            progress.update(100, "ANN indexing skipped")
            return
            
        try:
            ann_start = time.time()
            
            # Load all embeddings from database (67-75%)
            progress.update(67, "Loading embeddings...")
            c = self.conn.cursor()
            c.execute("SELECT c.id, c.command, e.embedding FROM commands c JOIN embeddings e ON c.id = e.rowid")
            all_data = c.fetchall()
            
            if len(all_data) < ANN_NUM_CLUSTERS:
                logger.info("Too few commands (%d) for ANN clustering, skipping", len(all_data))
                progress.update(100, f"Too few commands ({len(all_data)}) for clustering")
                return
                
            progress.update(75, "Preparing embeddings...")
            
            # Extract embeddings (75-85%)
            embeddings_list = []
            for i, (_, _, emb) in enumerate(all_data):
                if EMBEDDING_DTYPE == np.int8:
                    stored_emb = np.frombuffer(emb, dtype=np.int8)[:384]
                elif EMBEDDING_DTYPE == np.float16:
                    stored_emb = np.frombuffer(emb, dtype=np.float16)[:384]
                elif EMBEDDING_DTYPE == np.float32:
                    stored_emb = np.frombuffer(emb, dtype=np.float32)[:384]
                else:
                    raise ValueError(f"Unsupported EMBEDDING_DTYPE: {EMBEDDING_DTYPE}")
                
                embeddings_list.append(self.dequantize_embedding(stored_emb))
                
                # Update progress occasionally
                if i % max(1, len(all_data) // 10) == 0:
                    current_progress = 75 + int((i / len(all_data)) * 10)
                    progress.update(current_progress, f"Processing embeddings... ({i+1}/{len(all_data)})")
            
            embeddings = np.vstack(embeddings_list)
            
            # Train the index (85-100%)
            progress.update(85, f"Training ANN index ({ANN_NUM_CLUSTERS} clusters)...")
            logger.info("Training ANN index with %d embeddings (%d clusters, %d candidates)", 
                       len(embeddings), ANN_NUM_CLUSTERS, ANN_CLUSTER_CANDIDATES)
            self.ann_index.fit(embeddings)
            
            ann_time = time.time() - ann_start
            progress.update(100, f"ANN index complete! ({ann_time:.1f}s)")
            logger.info("ANN index rebuilt in %.3fs", ann_time)
            
        except Exception as e:
            logger.error("Error rebuilding ANN index: %s", e)
            progress.update(100, f"ANN indexing failed: {e}")
            
    def _rebuild_ann_index_with_tui_progress(self, tui_progress, console_fallback):
        """Rebuild ANN index with TUI progress updates (67-100%)"""
        if not USE_ANN_SEARCH or not hasattr(self, 'ann_index'):
            if tui_progress:
                try:
                    tui_progress.update_progress(100, "ANN indexing skipped")
                except:
                    pass
            elif console_fallback:
                print("  Search indexing skipped")
            return
            
        try:
            ann_start = time.time()
            
            # Load all embeddings from database (67-75%)
            if tui_progress:
                try:
                    tui_progress.update_progress(67, "Loading embeddings...", "ANN Indexing")
                except:
                    pass
            elif console_fallback:
                print("  Loading embeddings...")
                
            c = self.conn.cursor()
            c.execute("SELECT c.id, c.command, e.embedding FROM commands c JOIN embeddings e ON c.id = e.rowid")
            all_data = c.fetchall()
            
            if len(all_data) < ANN_NUM_CLUSTERS:
                message = f"Too few commands ({len(all_data)}) for clustering"
                logger.info(message)
                if tui_progress:
                    try:
                        tui_progress.update_progress(100, message, "Complete")
                    except:
                        pass
                elif console_fallback:
                    print(f"  {message}")
                return
                
            if tui_progress:
                try:
                    tui_progress.update_progress(75, "Preparing embeddings...", "ANN Indexing")
                except:
                    pass
            elif console_fallback:
                print("  Preparing embeddings...")
            
            # Extract embeddings (75-85%)
            embeddings_list = []
            for i, (_, _, emb) in enumerate(all_data):
                if EMBEDDING_DTYPE == np.int8:
                    stored_emb = np.frombuffer(emb, dtype=np.int8)[:384]
                elif EMBEDDING_DTYPE == np.float16:
                    stored_emb = np.frombuffer(emb, dtype=np.float16)[:384]
                elif EMBEDDING_DTYPE == np.float32:
                    stored_emb = np.frombuffer(emb, dtype=np.float32)[:384]
                else:
                    raise ValueError(f"Unsupported EMBEDDING_DTYPE: {EMBEDDING_DTYPE}")
                
                embeddings_list.append(self.dequantize_embedding(stored_emb))
                
                # Update progress occasionally
                if tui_progress and i % max(1, len(all_data) // 10) == 0:
                    try:
                        current_progress = 75 + int((i / len(all_data)) * 10)
                        tui_progress.update_progress(
                            current_progress, 
                            f"Processing embeddings... ({i+1}/{len(all_data)})",
                            "ANN Indexing"
                        )
                    except:
                        pass
            
            embeddings = np.vstack(embeddings_list)
            
            # Train the index (85-100%)
            if tui_progress:
                try:
                    tui_progress.update_progress(85, f"Training search index ({ANN_NUM_CLUSTERS} clusters)...", "ANN Indexing")
                except:
                    pass
            elif console_fallback:
                print(f"  Training search index ({ANN_NUM_CLUSTERS} clusters)...")
                
            logger.info("Training ANN index with %d embeddings (%d clusters, %d candidates)", 
                       len(embeddings), ANN_NUM_CLUSTERS, ANN_CLUSTER_CANDIDATES)
            self.ann_index.fit(embeddings)
            
            ann_time = time.time() - ann_start
            final_message = f"Search index complete! ({ann_time:.1f}s)"
            
            if tui_progress:
                try:
                    tui_progress.update_progress(100, final_message, "Complete")
                except:
                    pass
            elif console_fallback:
                print(f"  {final_message}")
                
            logger.info("ANN index rebuilt in %.3fs", ann_time)
            
        except Exception as e:
            error_message = f"Search indexing failed: {e}"
            logger.error("Error rebuilding ANN index: %s", e)
            if tui_progress:
                try:
                    tui_progress.update_progress(100, error_message, "Error")
                except:
                    pass
            elif console_fallback:
                print(f"  {error_message}")
            
            self.ann_index = ANNSearchIndex()
    
    def _save_ann_index(self):
        """Save ANN index to disk for reuse"""
        if not hasattr(self, 'ann_index') or not self.ann_index.is_trained:
            return
            
        try:
            import pickle
            ann_cache_path = os.path.join(self.db_path.replace('.db', '_ann_index.pkl'))
            with open(ann_cache_path, 'wb') as f:
                pickle.dump({
                    'cluster_centers': self.ann_index.cluster_centers,
                    'cluster_indices': self.ann_index.cluster_indices,
                    'embeddings': self.ann_index.embeddings,
                    'n_clusters': self.ann_index.n_clusters,
                    'is_trained': self.ann_index.is_trained
                }, f)
            logger.info("ANN index saved to %s", ann_cache_path)
        except Exception as e:
            logger.warning("Failed to save ANN index: %s", str(e))
    
    def _load_ann_index(self):
        """Load ANN index from disk"""
        try:
            import pickle
            ann_cache_path = os.path.join(self.db_path.replace('.db', '_ann_index.pkl'))
            
            if not os.path.exists(ann_cache_path):
                return False
                
            # Check if cache is newer than database
            cache_mtime = os.path.getmtime(ann_cache_path)
            db_mtime = os.path.getmtime(self.db_path)
            
            if cache_mtime < db_mtime:
                logger.info("ANN index cache is outdated (database modified after index). Run --ingest to rebuild.")
                return False
                
            with open(ann_cache_path, 'rb') as f:
                data = pickle.load(f)
                
            # Restore the index
            self.ann_index.cluster_centers = data['cluster_centers']
            self.ann_index.cluster_indices = data['cluster_indices'] 
            self.ann_index.embeddings = data['embeddings']
            self.ann_index.n_clusters = data['n_clusters']
            self.ann_index.is_trained = data['is_trained']
            
            logger.info("ANN index loaded from cache")
            return True
            
        except Exception as e:
            logger.warning("Failed to load ANN index: %s", str(e))
            return False

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
        logger.info("🔍 SEARCH STARTING: query='%s' (return_scores=%s)", query, return_scores)
        
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
        """
        
        # Get broader set for semantic matching - no limits, search entire dataset
        broader_query = """
            SELECT c.id, c.command, v.embedding
            FROM commands c
            JOIN embeddings v ON c.id = v.rowid
            ORDER BY c.last_used DESC
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
            logger.debug("Using broader search with %d commands (full dataset)", len(all_commands_data))

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
        
        # Use ANN index (should always be available due to auto-rebuild at startup)
        use_ann_for_this_search = USE_ANN_SEARCH
        if USE_ANN_SEARCH and (not self.ann_index or not self.ann_index.is_trained):
            logger.warning("ANN index unexpectedly unavailable, falling back to linear search")
            use_ann_for_this_search = False
        
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
        
        # Use ANN search to get candidate indices, or all indices for linear search
        if use_ann_for_this_search and self.ann_index and self.ann_index.is_trained:
            candidate_indices = self.ann_index.search_candidates(query_embedding.flatten())
            # Filter embeddings and commands to only candidates
            if candidate_indices:
                # Safety check for index bounds
                valid_indices = [i for i in candidate_indices if i < len(embeddings_array)]
                if len(valid_indices) != len(candidate_indices):
                    logger.warning("ANN returned %d invalid indices, using %d valid ones", 
                                 len(candidate_indices) - len(valid_indices), len(valid_indices))
                
                candidate_embeddings = embeddings_array[valid_indices]
                candidate_command_ids = [command_ids[i] for i in valid_indices]  
                candidate_commands = [commands[i] for i in valid_indices]
                logger.debug("ANN search: using %d/%d candidates for similarity calculation", 
                           len(candidate_indices), len(embeddings_array))
            else:
                # Fallback to full search if no candidates
                candidate_embeddings = embeddings_array
                candidate_command_ids = list(command_ids)
                candidate_commands = list(commands)
                candidate_indices = list(range(len(embeddings_array)))
        else:
            # Linear search - use all embeddings
            candidate_embeddings = embeddings_array
            candidate_command_ids = list(command_ids)
            candidate_commands = list(commands)
            candidate_indices = list(range(len(embeddings_array)))
            logger.debug("Using linear search for %d embeddings", len(embeddings_array))
        
        # Vectorized semantic similarity calculation on candidates
        semantic_scores = np.dot(candidate_embeddings, query_embedding.T).reshape(-1)
        norms = np.linalg.norm(candidate_embeddings, axis=1) * np.linalg.norm(query_embedding)
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
        
        # Calculate BM25 scores in batch for candidates
        bm25_scores = self.calculate_bm25_scores_batch(query_terms, candidate_command_ids)
        
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
        combined_scores = []
        for i, (sem, bm25) in enumerate(zip(semantic_scores, bm25_scores)):
            hybrid = self._dynamic_hybrid_score(sem, bm25, candidate_commands[i], query)
            combined_scores.append(hybrid)
            
            # Log top scores for debugging (can be removed in production)
            if i < 5:  # Only log top 5 for performance
                logger.debug("Score: '%s' → Sem:%.3f BM25:%.3f Hybrid:%.3f", 
                           candidate_commands[i][:50], sem, bm25, hybrid)
        
        combined_scores = np.array(combined_scores)
        results = []
        for i, (command, combined_score) in enumerate(zip(candidate_commands, combined_scores)):
            if return_scores:
                results.append((
                    command, float(combined_score), 
                    float(semantic_scores[i]), float(bm25_scores[i])
                ))
            else:
                results.append((command, float(combined_score)))

        # Sort results - hybrid scoring already handles relevance, minimal additional boosts
        def sort_key(result):
            command = result[0]
            score = result[1]
            
            # Only boost for exact command matches (not partial word matches)
            if command.lower() == query.lower():
                return score + 2.0  # Small boost for exact matches
            
            # Very small boost for exact prefix matches only
            if command.lower().startswith(query.lower()):
                return score + 0.1  # Minimal boost for prefix matches
            
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
    
    def get_system_info(self) -> dict:
        """Get comprehensive system information including models, database, and configuration."""
        info = {
            'version': __version__,
            'database': {
                'path': self.db_path if self._conn is None else 'In-memory (test mode)',
                'total_commands': self.total_commands,
                'avg_command_length': self.avg_length,
            },
            'search_configuration': {
                'use_ann_search': USE_ANN_SEARCH,
                'ann_num_clusters': ANN_NUM_CLUSTERS if USE_ANN_SEARCH else 'N/A',
                'ann_cluster_candidates': ANN_CLUSTER_CANDIDATES if USE_ANN_SEARCH else 'N/A',
                'embedding_dtype': EMBEDDING_DTYPE.__name__,
                'embedding_dimensions': MODEL_OUTPUT_DIM,
            },
            'bm25_parameters': {
                'k1': self.k1,
                'b': self.b,
            }
        }
        
        # Get embedding model information
        try:
            if self._model and hasattr(self._model, 'get_embedding_model_info'):
                info['embedding_model'] = self._model.get_embedding_model_info()
            else:
                # Fallback when model not initialized
                current_model = get_active_model_key()
                config = get_model_config(current_model)
                info['embedding_model'] = {
                    'status': 'Not initialized',
                    'model_key': current_model,
                    'model_name': config['repo'],
                    'description': config['description']
                }
        except Exception as e:
            info['embedding_model'] = {'error': str(e)}
        
        # Get description model information  
        try:
            from .model_handler import DescriptionHandler
            desc_handler = DescriptionHandler()
            info['description_model'] = desc_handler.get_model_info()
        except Exception as e:
            info['description_model'] = {'error': str(e)}
            
        # Get database statistics
        try:
            c = self.conn.cursor()
            
            # Command count
            c.execute("SELECT COUNT(*) FROM commands")
            command_count = c.fetchone()[0]
            info['database']['actual_command_count'] = command_count
            
            # Embedding count
            c.execute("SELECT COUNT(*) FROM embeddings")
            embedding_count = c.fetchone()[0]
            info['database']['embedding_count'] = embedding_count
            info['database']['embedding_coverage'] = f"{embedding_count/command_count*100:.1f}%" if command_count > 0 else "0%"
            
            # Cache statistics
            c.execute("SELECT COUNT(*) FROM query_cache")
            cache_count = c.fetchone()[0]
            info['database']['cached_queries'] = cache_count
            
        except Exception as e:
            info['database']['query_error'] = str(e)
            
        # ANN index status
        if USE_ANN_SEARCH and self.ann_index:
            info['ann_index'] = {
                'trained': self.ann_index.is_trained,
                'n_clusters': self.ann_index.n_clusters,
                'embeddings_indexed': len(self.ann_index.embeddings) if self.ann_index.embeddings is not None else 0,
            }
        else:
            info['ann_index'] = {'status': 'Disabled'}
            
        return info
    
    def print_system_info(self, detailed: bool = False):
        """Print formatted system information."""
        info = self.get_system_info()
        
        print(f"FuzzyShell v{info['version']} - System Status")
        print("=" * 45)
        
        # Database info
        print("DATABASE:")
        print(f"  Path: {info['database']['path']}")
        print(f"  Commands: {info['database'].get('actual_command_count', 'Unknown')}")
        print(f"  Embeddings: {info['database'].get('embedding_count', 'Unknown')} ({info['database'].get('embedding_coverage', 'Unknown')} coverage)")
        if detailed:
            print(f"  Cached queries: {info['database'].get('cached_queries', 'Unknown')}")
        
        # Embedding model info  
        print("\nEMBEDDING MODEL:")
        emb_model = info.get('embedding_model', {})
        if 'error' in emb_model:
            print(f"  Error: {emb_model['error']}")
        else:
            # Use description if available, fallback to model name
            if 'description' in emb_model:
                print(f"  Model: {emb_model['description']}")
            else:
                model_name = emb_model.get('model_name', 'Unknown')
                print(f"  Model: {model_name}")
            
            print(f"  Status: {emb_model.get('status', 'Unknown')}")
            print(f"  Dimensions: {emb_model.get('dimensions', 'Unknown')}")
            
            if detailed:
                if 'model_key' in emb_model:
                    print(f"  Model Key: {emb_model['model_key']}")
                if 'tokenizer_type' in emb_model:
                    print(f"  Tokenizer: {emb_model['tokenizer_type']}")
                if 'actual_size_mb' in emb_model:
                    print(f"  Size: {emb_model['actual_size_mb']} MB")
                elif 'model_size_mb' in emb_model:
                    print(f"  Expected Size: {emb_model['model_size_mb']} MB")
        
        # Description model info
        print("\nDESCRIPTION MODEL:")
        desc_model = info.get('description_model', {})
        if 'error' in desc_model:
            print(f"  Error: {desc_model['error']}")
        else:
            print(f"  Model: {desc_model.get('model_name', 'Unknown')}")
            print(f"  Status: {desc_model.get('status', 'Unknown')}")
            if detailed and 'model_files' in desc_model:
                print(f"  Files: {len(desc_model['model_files'])} components")
        
        # Search configuration
        print("\nSEARCH CONFIG:")
        search_config = info['search_configuration']
        ann_enabled = search_config['use_ann_search']
        print(f"  ANN Search: {'Enabled' if ann_enabled else 'Disabled'}")
        if ann_enabled:
            print(f"  Clusters: {search_config['ann_num_clusters']}")
            ann_info = info.get('ann_index', {})
            index_status = "Trained" if ann_info.get('trained') else "Not trained"
            print(f"  Index: {index_status}")
        
        print(f"  Storage: {search_config['embedding_dtype']} ({search_config['embedding_dimensions']}D)")
        
        if detailed:
            bm25_params = info['bm25_parameters']
            print(f"  BM25: k1={bm25_params['k1']}, b={bm25_params['b']}")
        
        print("=" * 45)
    
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

def print_technical_details():
    """Print technical details and internal configuration."""
    print(f"FuzzyShell v{__version__} - Technical Internals")
    print("=" * 50)
    
    print("\nARCHITECTURE:")
    print("Hybrid semantic + keyword search using neural embeddings")
    print("and BM25 ranking with K-means approximate nearest neighbors.")
    
    print("\nEMBEDDING MODEL:")
    print("  Mitchins/multilingual-minilm-l12-h384-terminal-describer-embeddings-ONNX")
    print("  - Architecture: MiniLM-L12-H384 transformer")
    print("  - Training: Custom terminal command + description pairs")
    print("  - Performance: 2x cosine similarity improvement")
    print("  - Quantization: INT8 for storage, float32 for computation")
    print("  - Dimensions: 384 (optimized for speed/accuracy balance)")
    print("  - Languages: 50+ multilingual support")
    
    print("\nDESCRIPTION MODEL:")
    print("  Mitchins/codet5-small-terminal-describer-ONNX")
    print("  - Architecture: CodeT5-Small encoder-decoder transformer")
    print("  - Components: Separate encoder, decoder, decoder-with-past")
    print("  - Tokenizer: RoBERTa with vocab.json + merges.txt")  
    print("  - Fallback: Rule-based pattern matching for reliability")
    
    print("\nSEARCH ALGORITHM:")
    print(f"  Approximate Nearest Neighbors: {'Enabled' if USE_ANN_SEARCH else 'Disabled'}")
    if USE_ANN_SEARCH:
        print(f"  - Method: K-means clustering (pure numpy)")
        print(f"  - Clusters: {ANN_NUM_CLUSTERS} (configurable)")
        print(f"  - Candidates: {ANN_CLUSTER_CANDIDATES} closest clusters searched")
        print("  - Complexity: O(k) where k << n for large datasets")
        print("  - Training: K-means++ initialization, convergence detection")
    print(f"  Storage format: {EMBEDDING_DTYPE.__name__} ({MODEL_OUTPUT_DIM} dimensions)")
    
    print("\nHYBRID SCORING ALGORITHM:")
    print("  Dynamic weighting based on confidence levels:")
    print("  - High semantic confidence (>0.4): 75% semantic, 25% BM25")
    print("  - Medium confidence (0.2-0.4): 60% semantic, 40% BM25")
    print("  - Low confidence (<0.2): Adaptive BM25 preference")
    print("  - Score normalization: Cosine similarity [-1,1] -> [0,1]")
    print(f"  BM25 parameters: k1={1.5} (term frequency), b={0.75} (length norm)")
    
    print("\nPERFORMANCE OPTIMIZATIONS:")
    print("  - Vectorized operations: NumPy batch processing")
    print("  - Memory efficiency: float16 embeddings, int8 quantization")
    print("  - Caching: Query results (1-hour TTL)")
    print("  - Database: SQLite with optimized indices")
    print("  - Batch processing: Progress callbacks for large datasets")
    print("  - Expected speedup: 11.5x for 1000+ commands with ANN")
    
    print("\nSTORAGE LAYOUT:")
    print("  Database: fuzzyshell.db (SQLite)")
    print("  - commands: id, command, last_used, length")
    print("  - embeddings: rowid -> command_id, quantized vectors")
    print("  - term_frequencies: BM25 preprocessing")
    print("  - query_cache: LRU cache with timestamps")
    print("  Models: ~/.fuzzyshell/model/ ~/.fuzzyshell/description_model/")
    print("  Logs: debug.log (development mode)")
    
    print("=" * 50)

def interactive_search(show_scoring=False, show_profiling=False):
    """Launch the interactive search UI"""
    fuzzyshell = FuzzyShell()
    
    def search_callback(query: str) -> list:
        """Callback function for the UI to perform searches"""
        if not query:
            return []
        # Always return detailed scores for the TUI
        return fuzzyshell.search(query, return_scores=True)

    if show_profiling:
        print("🔍 Profiling mode enabled - check logs for detailed timing")
        
    app = FuzzyShellApp(search_callback, fuzzyshell_instance=fuzzyshell)
    selected_command = app.run()
    return selected_command

def main():
    """Main entry point for the application"""
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="FuzzyShell: A semantic search for your command history.")
    parser.add_argument('--version', action='version', version=f'%(prog)s {__version__}')
    parser.add_argument('--ingest', action='store_true', help='Full re-ingestion from shell history (use for repairs/model changes).')
    parser.add_argument('--rebuild-ann', action='store_true', help='Rebuild the ANN index for optimal search performance.')
    parser.add_argument('--scoring', action='store_true', help='Show semantic and BM25 scores for each result.')
    parser.add_argument('--status', action='store_true', help='Show system status and model information.')
    parser.add_argument('--info', action='store_true', help='Show detailed system information including model details.')
    parser.add_argument('--open-hood', action='store_true', help='Show technical details and internal configuration.')
    parser.add_argument('--clear-cache', action='store_true', help='Clear all caches and force fresh results.')
    parser.add_argument('--no-ann', action='store_true', help='Disable ANN search for debugging (uses full linear search).')
    parser.add_argument('--profile', action='store_true', help='Show detailed timing and performance information.')

    args = parser.parse_args()
    
    # Handle technical details
    if args.open_hood:
        print_technical_details()
        sys.exit(0)
    
    # Handle cache clearing (do this before creating FuzzyShell instance)
    if args.clear_cache:
        fuzzyshell_temp = FuzzyShell()
        fuzzyshell_temp.clear_all_caches()
        print("✅ All caches cleared. Search results will be recalculated from scratch.")
        sys.exit(0)
    
    fuzzyshell = FuzzyShell()
    
    # Perform ingestion on startup for interactive mode
    if not any([args.status, args.info, args.ingest, args.rebuild_ann, args.clear_cache, args.open_hood]):
        try:
            indexed_count = fuzzyshell.get_indexed_count()
            
            if indexed_count == 0:
                # Empty database - perform full initial ingestion
                print("🎉 First time setup - ingesting command history...")
                added_count = fuzzyshell.ingest_history()
                if added_count > 0:
                    print(f"✅ Ingested {added_count} commands from your shell history")
                else:
                    print("⚠️  No commands found in shell history")
            else:
                # Existing database - perform micro-ingest for new commands
                from .micro_ingest import micro_ingest, suggest_ann_rebuild
                new_commands, should_rebuild = micro_ingest(fuzzyshell)
                if new_commands > 0:
                    logger.info("Micro-ingest added %d new commands", new_commands)
                
                # Check if ANN rebuild is suggested
                if should_rebuild:
                    suggestion = suggest_ann_rebuild(fuzzyshell)
                    if suggestion:
                        print(suggestion)
        except Exception as e:
            logger.warning("Ingestion failed: %s", str(e))

    # Handle status and info commands
    if args.status:
        fuzzyshell.print_system_info(detailed=False)
        sys.exit(0)
    elif args.info:
        fuzzyshell.print_system_info(detailed=True)  
        sys.exit(0)
    elif args.ingest:
        print("🔄 Starting full re-ingestion from shell history...")
        print("⚠️  This will clear all caches and rebuild the database.")
        fuzzyshell.ingest_history()
        print("✅ Ingestion complete!")
        sys.exit(0)
    elif args.rebuild_ann:
        print("🔧 Rebuilding ANN index...")
        fuzzyshell._rebuild_ann_index()
        fuzzyshell.set_metadata('ann_command_count', str(fuzzyshell.get_indexed_count()))
        fuzzyshell.set_metadata('poorly_clustered_commands', '0')
        print("✅ ANN index rebuilt successfully!")
        sys.exit(0)
    else:
        # Handle ANN disable flag
        if args.no_ann:
            global USE_ANN_SEARCH
            USE_ANN_SEARCH = False
            print("🔍 ANN search disabled - using full linear search for debugging")
            
        # Handle profiling flag
        if args.profile:
            logger.setLevel(logging.INFO)  # Enable more detailed logging
            print("🔍 Profiling mode enabled - detailed timing will be shown")
        
        result = interactive_search(show_scoring=args.scoring, show_profiling=args.profile)
        if result:
            # Print the selected command so it can be captured by the shell wrapper
            print(result)
        sys.exit(0)

if __name__ == '__main__':
    main()
