import sqlite3
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, List, Optional, Tuple
from fuzzyshell.db_wrapper import ThreadSafeDatabase


class DatabaseProvider(ABC):
    """Abstract interface for database connection providers."""
    
    @abstractmethod
    def _get_connection(self):
        """Get a database connection."""
        pass


class ProductionDatabaseProvider(DatabaseProvider):
    """Production database provider using ThreadSafeDatabase with singleton pattern."""
    
    _instances = {}  # Class variable to store singleton instances by db_path
    
    def __new__(cls, db_path: str):
        # Ensure only one instance per database path
        if db_path not in cls._instances:
            instance = super().__new__(cls)
            cls._instances[db_path] = instance
        return cls._instances[db_path]
    
    def __init__(self, db_path: str):
        # Only initialize once per db_path
        if not hasattr(self, '_initialized'):
            self._thread_safe_db = ThreadSafeDatabase(db_path)
            self._initialized = True
    
    def _get_connection(self):
        return self._thread_safe_db._get_connection()


class MockDatabaseProvider(DatabaseProvider):
    """Mock database provider using injected connection for testing."""
    
    def __init__(self, connection):
        self._connection = connection
    
    def _get_connection(self):
        return self._connection


class BaseDAL:
    """Base class providing a thread-safe DB connection context."""
    def __init__(self, db_provider: DatabaseProvider, load_vss: bool = True):
        self._db_provider = db_provider
        
        # Initialize schema and extensions for production providers only
        if isinstance(db_provider, ProductionDatabaseProvider):
            with self.connection() as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                if load_vss:
                    self._load_vss_extension(conn)
    
    def _load_vss_extension(self, conn):
        """Load VSS extension - can be overridden in tests"""
        try:
            import sqlite_vss
            conn.enable_load_extension(True)
            sqlite_vss.load(conn)
        except (ImportError, AttributeError, Exception) as e:
            # Fallback to loading SQLite extension by module name
            try:
                conn.enable_load_extension(True)
                conn.load_extension("vss0")
                conn.enable_load_extension(False)
            except Exception:
                # VSS not available - continue without it
                pass

    @contextmanager
    def connection(self):
        """Yield the thread-local SQLite connection."""
        conn = self._db_provider._get_connection()
        try:
            yield conn
            # Commit the transaction on successful completion
            conn.commit()
        except Exception:
            # Rollback on any exception
            conn.rollback()
            raise
    
    @contextmanager
    def bulk_operations(self):
        """
        Context manager for bulk database operations with optimized settings.
        
        Sets up SQLite for maximum bulk performance:
        - Sets PRAGMA synchronous = OFF for speed (before transaction)
        - Sets PRAGMA journal_mode = MEMORY (before transaction)
        - Uses explicit transaction boundaries
        - Commits everything at the end
        
        Usage:
            with dal.bulk_operations() as bulk_conn:
                # All operations in this block are optimized for bulk
                dal.insert_many_commands(commands, connection=bulk_conn)
                dal.insert_many_embeddings(embeddings, connection=bulk_conn)
                # Automatic commit/rollback on exit
        """
        conn = self._db_provider._get_connection()
        
        # Store original pragma values (must be done outside transaction)
        original_synchronous = conn.execute("PRAGMA synchronous").fetchone()[0]
        original_journal_mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        
        try:
            # Optimize for bulk operations (must be done BEFORE transaction)
            conn.execute("PRAGMA synchronous = OFF")  # Faster, less safe
            conn.execute("PRAGMA journal_mode = MEMORY")  # Keep journal in memory
            
            # Start explicit transaction after pragma settings
            conn.execute("BEGIN TRANSACTION")
            
            yield conn
            
            # Commit the bulk transaction
            conn.execute("COMMIT")
            
        except Exception:
            # Rollback on any exception
            try:
                conn.execute("ROLLBACK")
            except:
                pass
            raise
            
        finally:
            # Restore original pragma values (must be done outside transaction)
            try:
                conn.execute(f"PRAGMA synchronous = {original_synchronous}")
                conn.execute(f"PRAGMA journal_mode = {original_journal_mode}")
            except:
                # Don't fail on pragma restoration
                pass


# Helper type for rows
Row = sqlite3.Row


class CommandDAL(BaseDAL):
    """Data access for shell commands and their embeddings."""

    def __init__(self, db_provider: DatabaseProvider, load_vss: bool = True):
        super().__init__(db_provider, load_vss=load_vss)
        self._ensure_schema()

    def _ensure_schema(self):
        with self.connection() as conn:
            # Match the exact schema from fuzzyshell.py
            conn.execute("""
            CREATE TABLE IF NOT EXISTS commands (
                id INTEGER PRIMARY KEY,
                command TEXT UNIQUE NOT NULL,
                length INTEGER,
                frequency INTEGER DEFAULT 1,
                last_used TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                context TEXT
            )""")
            conn.execute('CREATE INDEX IF NOT EXISTS idx_commands_last_used ON commands(last_used)')
            
            # Create VSS virtual table for semantic search
            # Get model dimension from config
            from fuzzyshell.model_configs import get_model_config
            try:
                model_config = get_model_config()
                model_dim = model_config.get('dimensions', 384)
                
                conn.execute(f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS embeddings USING vss0(
                    embedding({model_dim})
                )""")
            except (sqlite3.OperationalError, ImportError, Exception):
                # Fallback to regular table if VSS not available
                conn.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    rowid INTEGER PRIMARY KEY,
                    embedding BLOB
                )""")
            
            # Term frequencies for BM25
            conn.execute("""
            CREATE TABLE IF NOT EXISTS term_frequencies (
                term TEXT,
                command_id INTEGER,
                freq REAL,
                PRIMARY KEY (term, command_id),
                FOREIGN KEY (command_id) REFERENCES commands(id) ON DELETE CASCADE
            )""")
            conn.execute('CREATE INDEX IF NOT EXISTS idx_term_freq_term ON term_frequencies(term)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_term_freq_command ON term_frequencies(command_id)')
            
            # Corpus statistics
            conn.execute("""
            CREATE TABLE IF NOT EXISTS corpus_stats (
                total_docs INTEGER DEFAULT 0,
                avg_doc_length REAL DEFAULT 0
            )""")
            conn.execute("INSERT OR IGNORE INTO corpus_stats VALUES (0, 0)")
            
            # Query cache
            conn.execute("""
            CREATE TABLE IF NOT EXISTS query_cache (
                query_hash TEXT PRIMARY KEY,
                results TEXT,
                timestamp INTEGER,
                query_type TEXT
            )""")
            conn.execute('CREATE INDEX IF NOT EXISTS idx_cache_timestamp ON query_cache(timestamp)')

    def add_command(self, command: str, length: int = None, embedding: bytes = None) -> int:
        """Insert or update a command. Returns command_id."""
        with self.connection() as conn:
            if length is None:
                length = len(command)
                
            cursor = conn.execute(
                "INSERT OR IGNORE INTO commands (command, length) VALUES (?, ?)", 
                (command, length)
            )
            
            if cursor.lastrowid:
                cmd_id = cursor.lastrowid
            else:
                # Command already exists - get its ID and update frequency
                row = conn.execute("SELECT id FROM commands WHERE command = ?", (command,)).fetchone()
                cmd_id = row[0]
                conn.execute("""
                    UPDATE commands 
                    SET frequency = frequency + 1, 
                        last_used = CURRENT_TIMESTAMP 
                    WHERE id = ?
                """, (cmd_id,))
            
            # Add embedding if provided
            if embedding is not None:
                conn.execute(
                    "INSERT OR REPLACE INTO embeddings (rowid, embedding) VALUES (?, ?)",
                    (cmd_id, embedding)
                )
                
            return cmd_id
    
    def add_commands_batch(self, commands_data: List[Tuple[str, int]]) -> List[int]:
        """Batch insert commands. Returns list of command IDs IN THE SAME ORDER as input."""
        with self.connection() as conn:
            conn.executemany("INSERT OR IGNORE INTO commands (command, length) VALUES (?, ?)", commands_data)
            
            # Get all IDs for the commands
            placeholders = ','.join(['?'] * len(commands_data))
            command_texts = [cmd[0] for cmd in commands_data]
            rows = conn.execute(
                f"SELECT id, command FROM commands WHERE command IN ({placeholders})", 
                command_texts
            ).fetchall()
            
            # Create a mapping of command text to ID
            command_to_id = {row[1]: row[0] for row in rows}
            
            # Return IDs in the same order as input commands
            ordered_ids = []
            for cmd_text, _ in commands_data:
                if cmd_text in command_to_id:
                    ordered_ids.append(command_to_id[cmd_text])
                else:
                    # Command was not inserted (possibly duplicate), need to get its ID
                    row = conn.execute("SELECT id FROM commands WHERE command = ?", (cmd_text,)).fetchone()
                    if row:
                        ordered_ids.append(row[0])
                    else:
                        raise ValueError(f"Failed to get ID for command: {cmd_text}")
            
            return ordered_ids
    
    def add_embeddings_batch(self, embeddings_data: List[Tuple[int, bytes]]):
        """Batch insert embeddings."""
        with self.connection() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO embeddings (rowid, embedding) VALUES (?, ?)", 
                embeddings_data
            )
    
    def add_term_frequencies_batch(self, term_freq_data: List[Tuple[str, int, float]]):
        """Batch insert term frequencies for BM25."""
        with self.connection() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO term_frequencies (term, command_id, freq) VALUES (?, ?, ?)",
                term_freq_data
            )

    def get_command(self, command_id: int) -> Optional[Row]:
        """Get command with embedding by ID."""
        with self.connection() as conn:
            return conn.execute(
                "SELECT c.id, c.command, e.embedding FROM commands c "
                "JOIN embeddings e ON c.id = e.rowid "
                "WHERE c.id = ?", (command_id,)
            ).fetchone()
    
    def get_all_commands_with_embeddings(self) -> List[Row]:
        """Get all commands with their embeddings."""
        with self.connection() as conn:
            return conn.execute(
                "SELECT c.id, c.command, e.embedding FROM commands c "
                "JOIN embeddings e ON c.id = e.rowid"
            ).fetchall()
    
    def get_command_count(self) -> int:
        """Get total number of commands."""
        with self.connection() as conn:
            return conn.execute("SELECT COUNT(*) FROM commands").fetchone()[0]
    
    def get_embedding_count(self) -> int:
        """Get total number of embeddings."""
        with self.connection() as conn:
            return conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
    
    def get_average_command_length(self) -> float:
        """Get average command length."""
        with self.connection() as conn:
            row = conn.execute("SELECT COALESCE(AVG(length), 0) FROM commands").fetchone()
            return float(row[0]) if row else 0.0

    def search_similar_vss(self, embedding: bytes, top_k: int = 100) -> List[Tuple[int, float]]:
        """
        VSS search - returns list of (rowid, score) for the closest embeddings.
        Falls back to cosine similarity if VSS virtual table not available.
        """
        with self.connection() as conn:
            try:
                # Try VSS virtual table first
                rows = conn.execute(
                    "SELECT rowid, distance FROM embeddings WHERE vss_search(embedding, vss_search_params(?, ?))",
                    (embedding, top_k)
                ).fetchall()
                return [(row[0], row[1]) for row in rows]
            except sqlite3.OperationalError:
                # VSS not available - fallback to manual cosine similarity
                # This is much slower but functional
                import numpy as np
                
                # Get all embeddings from regular table
                all_rows = conn.execute("SELECT rowid, embedding FROM embeddings").fetchall()
                if not all_rows:
                    return []
                
                # Convert query embedding from bytes to numpy
                query_emb = np.frombuffer(embedding, dtype=np.float32)
                
                similarities = []
                for rowid, stored_blob in all_rows:
                    try:
                        stored_emb = np.frombuffer(stored_blob, dtype=np.float32)
                        # Compute cosine similarity
                        similarity = np.dot(query_emb, stored_emb) / (
                            np.linalg.norm(query_emb) * np.linalg.norm(stored_emb)
                        )
                        # Convert to distance (1 - similarity) for compatibility
                        distance = 1.0 - similarity
                        similarities.append((rowid, distance))
                    except Exception:
                        continue
                
                # Sort by distance (ascending) and return top_k
                similarities.sort(key=lambda x: x[1])
                return similarities[:top_k]
    
    def update_corpus_stats(self, total_docs: int, avg_doc_length: float):
        """Update corpus statistics for BM25."""
        with self.connection() as conn:
            conn.execute(
                "UPDATE corpus_stats SET total_docs = ?, avg_doc_length = ?",
                (total_docs, avg_doc_length)
            )
    
    def clear_embeddings(self):
        """Clear all embeddings."""
        with self.connection() as conn:
            conn.execute("DELETE FROM embeddings")
    
    def clear_query_cache(self):
        """Clear query cache."""
        with self.connection() as conn:
            conn.execute("DELETE FROM query_cache")
    
    def optimize_for_bulk_operations(self):
        """Set pragmas for bulk operations."""
        with self.connection() as conn:
            conn.execute("PRAGMA synchronous = OFF")
            conn.execute("PRAGMA journal_mode = MEMORY")
            conn.execute("PRAGMA cache_size = -64000")
            conn.execute("PRAGMA temp_store = MEMORY")
    
    def restore_normal_pragmas(self):
        """Restore normal pragmas after bulk operations."""  
        with self.connection() as conn:
            conn.execute("PRAGMA synchronous = NORMAL")
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA cache_size = -2000")
    
    def basic_search(self, query: str, top_k: int = 50) -> List[Tuple[str, float]]:
        """Basic substring search returning (command, score) tuples."""
        with self.connection() as conn:
            rows = conn.execute("""
                SELECT command FROM commands 
                WHERE command LIKE ? 
                ORDER BY last_used DESC 
                LIMIT ?
            """, (f"%{query}%", top_k)).fetchall()
            return [(row[0], 0.0) for row in rows]
    
    def get_commands_for_bm25(self, limit: int = 2000) -> List[Tuple[int, str]]:
        """Get commands with IDs for BM25 scoring."""
        with self.connection() as conn:
            rows = conn.execute("""
                SELECT id, command FROM commands 
                ORDER BY last_used DESC
                LIMIT ?
            """, (limit,)).fetchall()
            return [(row[0], row[1]) for row in rows]
    
    def get_all_commands_with_embeddings_for_clustering(self) -> List[Tuple[int, str, bytes]]:
        """Get all commands with embeddings for clustering analysis."""
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT c.id, c.command, e.embedding FROM commands c JOIN embeddings e ON c.id = e.rowid"
            ).fetchall()
            return [(row[0], row[1], row[2]) for row in rows]
    
    def add_single_command_with_terms(self, command: str, terms: List[str], embedding: bytes = None) -> int:
        """Add a single command with its term frequencies and embedding."""
        with self.connection() as conn:
            length = len(command.split())
            
            # Insert or ignore the command
            cursor = conn.execute(
                "INSERT OR IGNORE INTO commands (command, length) VALUES (?, ?)", 
                (command, length)
            )
            
            if cursor.lastrowid:
                cmd_id = cursor.lastrowid
            else:
                # Command already exists - get its ID and update frequency
                row = conn.execute("SELECT id FROM commands WHERE command = ?", (command,)).fetchone()
                cmd_id = row[0]
                conn.execute("""
                    UPDATE commands 
                    SET frequency = frequency + 1, 
                        last_used = CURRENT_TIMESTAMP 
                    WHERE id = ?
                """, (cmd_id,))
            
            # Add term frequencies
            for term in terms:
                conn.execute("""
                    INSERT OR REPLACE INTO term_frequencies (term, command_id, freq)
                    VALUES (?, ?, ?)
                """, (term, cmd_id, terms.count(term)))
            
            # Add embedding if provided
            if embedding is not None:
                conn.execute(
                    "INSERT OR REPLACE INTO embeddings (rowid, embedding) VALUES (?, ?)",
                    (cmd_id, embedding)
                )
            
            return cmd_id
    
    def get_term_document_frequency(self, term: str) -> int:
        """Get the number of documents containing a term."""
        with self.connection() as conn:
            row = conn.execute(
                "SELECT COUNT(DISTINCT command_id) FROM term_frequencies WHERE term = ?",
                (term,)
            ).fetchone()
            return row[0] if row else 0
    
    def get_command_ids_for_commands(self, commands: List[str]) -> dict:
        """Get command IDs for a list of commands. Returns dict mapping command -> id."""
        with self.connection() as conn:
            placeholders = ','.join('?' for _ in commands)
            rows = conn.execute(
                f"SELECT id, command FROM commands WHERE command IN ({placeholders})",
                commands
            ).fetchall()
            return {cmd: cmd_id for cmd_id, cmd in rows}
    
    def get_commands_with_embeddings_matching(self, query_pattern: str) -> List[Tuple[int, str, bytes]]:
        """Get commands with embeddings that match a pattern."""
        with self.connection() as conn:
            rows = conn.execute("""
                SELECT c.id, c.command, e.embedding
                FROM commands c
                JOIN embeddings e ON c.id = e.rowid
                WHERE c.command LIKE ?
                ORDER BY c.last_used DESC
            """, (query_pattern,)).fetchall()
            return [(row[0], row[1], row[2]) for row in rows]
    
    def get_all_commands_with_embeddings_ordered(self) -> List[Tuple[int, str, bytes]]:
        """Get all commands with embeddings ordered by usage."""
        with self.connection() as conn:
            rows = conn.execute("""
                SELECT c.id, c.command, e.embedding
                FROM commands c
                JOIN embeddings e ON c.id = e.rowid
                ORDER BY c.last_used DESC
            """).fetchall()
            return [(row[0], row[1], row[2]) for row in rows]


class MetadataDAL(BaseDAL):
    """Example of another table for storing arbitrary metadata."""

    def __init__(self, db_provider: DatabaseProvider):
        super().__init__(db_provider, load_vss=False)  # Metadata doesn't need VSS
        self._ensure_schema()
        self._ensure_database_uuid()

    def _ensure_schema(self):
        with self.connection() as conn:
            conn.execute("""
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )""")

    def set(self, key: str, value: Any):
        with self.connection() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
                (key, str(value))
            )

    def get(self, key: str) -> Optional[str]:
        with self.connection() as conn:
            row = conn.execute(
                "SELECT value FROM metadata WHERE key = ?", (key,)
            ).fetchone()
            return row["value"] if row else None
    
    def _ensure_database_uuid(self):
        """Ensure database has a unique UUID for tracking."""
        import uuid
        import logging
        
        logger = logging.getLogger('FuzzyShell.MetadataDAL')
        
        db_uuid = self.get('database_uuid')
        if not db_uuid:
            # Generate new UUID for this database
            db_uuid = str(uuid.uuid4())
            self.set('database_uuid', db_uuid)
            created_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
            self.set('database_created', created_time)
            logger.info(f"🆔 NEW DATABASE CREATED: {db_uuid} at {created_time}")
        else:
            created_time = self.get('database_created') or 'unknown'
            logger.info(f"🆔 DATABASE LOADED: {db_uuid} (created: {created_time})")
    
    def get_database_info(self) -> dict:
        """Get database identification information."""
        return {
            'uuid': self.get('database_uuid'),
            'created': self.get('database_created'),
            'model_name': self.get('model_name'),
            'model_key': self.get('model_key')
        }
    
    def get_scoring_preference(self) -> str:
        """Get user's scoring preference."""
        return self.get('scoring_preference') or 'balanced'
    
    def set_scoring_preference(self, preference: str):
        """Set user's scoring preference.
        
        Options:
        - 'less_semantic': 30% semantic, 70% BM25 (keyword-focused)
        - 'balanced': 50% semantic, 50% BM25 (default)  
        - 'more_semantic': 70% semantic, 30% BM25 (meaning-focused)
        - 'semantic_only': 100% semantic, 0% BM25 (experimental)
        """
        valid_options = ['less_semantic', 'balanced', 'more_semantic', 'semantic_only']
        if preference not in valid_options:
            raise ValueError(f"Invalid scoring preference. Must be one of: {valid_options}")
        self.set('scoring_preference', preference)
    
    def get_scoring_weights(self) -> tuple:
        """Get semantic and BM25 weights based on user preference."""
        preference = self.get_scoring_preference()
        weights = {
            'less_semantic': (0.3, 0.7),    # 30% semantic, 70% BM25
            'balanced': (0.5, 0.5),         # 50% semantic, 50% BM25  
            'more_semantic': (0.7, 0.3),    # 70% semantic, 30% BM25
            'semantic_only': (1.0, 0.0)     # 100% semantic, 0% BM25
        }
        return weights.get(preference, (0.5, 0.5))  # Default to balanced


class QueryCacheDAL(BaseDAL):
    """Data access for query cache."""
    
    def __init__(self, db_provider: DatabaseProvider):
        super().__init__(db_provider, load_vss=False)  # Cache doesn't need VSS
        # Schema is created by CommandDAL
    
    def add_to_cache(self, query_hash: str, results: str, query_type: str):
        """Add query results to cache."""
        with self.connection() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO query_cache (query_hash, results, timestamp, query_type) VALUES (?, ?, ?, ?)",
                (query_hash, results, int(time.time()), query_type)
            )
    
    def get_from_cache(self, query_hash: str) -> Optional[Row]:
        """Get cached results for a query."""
        with self.connection() as conn:
            return conn.execute(
                "SELECT results, timestamp FROM query_cache WHERE query_hash = ?",
                (query_hash,)
            ).fetchone()
    
    def get_cached_results(self, query_hash: str, max_age_hours: int = 1) -> Optional[bytes]:
        """Get cached results for a query hash with time validation."""
        with self.connection() as conn:
            row = conn.execute("""
                SELECT results FROM query_cache 
                WHERE query_hash = ? 
                AND timestamp > datetime('now', '-' || ? || ' hours')
            """, (query_hash, max_age_hours)).fetchone()
            return row[0] if row else None
    
    def cache_query_results(self, query_hash: str, results_data: bytes):
        """Cache query results with current timestamp."""
        with self.connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO query_cache (query_hash, results)
                VALUES (?, ?)
            """, (query_hash, results_data))
    
    def clear_cache(self):
        """Clear all cached queries."""
        with self.connection() as conn:
            conn.execute("DELETE FROM query_cache")
    
    def get_cache_count(self) -> int:
        """Get number of cached queries."""
        with self.connection() as conn:
            return conn.execute("SELECT COUNT(*) FROM query_cache").fetchone()[0]
    
    def cleanup_cache(self, max_age_hours: int = 24):
        """Clean up old cache entries."""
        with self.connection() as conn:
            conn.execute("""
                DELETE FROM query_cache 
                WHERE timestamp < datetime('now', '-' || ? || ' hours')
            """, (max_age_hours,))


class CorpusStatsDAL(BaseDAL):
    """Data access for corpus statistics."""
    
    def __init__(self, db_provider: DatabaseProvider):
        super().__init__(db_provider, load_vss=False)  # Stats doesn't need VSS
        # Schema is created by CommandDAL
    
    def get_stats(self) -> Optional[Row]:
        """Get corpus statistics."""
        with self.connection() as conn:
            return conn.execute("SELECT total_docs, avg_doc_length FROM corpus_stats").fetchone()
    
    def update_stats(self, total_docs: int, avg_doc_length: float):
        """Update corpus statistics."""
        with self.connection() as conn:
            conn.execute(
                "UPDATE corpus_stats SET total_docs = ?, avg_doc_length = ?",
                (total_docs, avg_doc_length)
            )
    
    def increment_stats(self, avg_length: float):
        """Increment corpus statistics when adding a single command."""
        with self.connection() as conn:
            conn.execute("""
                UPDATE corpus_stats 
                SET total_docs = total_docs + 1,
                    avg_doc_length = (avg_doc_length * (total_docs) + ?) / (total_docs + 1)
                WHERE rowid = 1
            """, (avg_length,))