"""
Thread-safe database wrapper for FuzzyShell.
Guards all database entry points and handles threading automatically.
"""

import sqlite3
import threading
import logging
from contextlib import contextmanager
from typing import Optional, Any, List, Tuple

logger = logging.getLogger('FuzzyShell.DatabaseWrapper')


class ThreadSafeDatabase:
    """
    Thread-safe database wrapper that creates connections per thread.
    Guards all entry points and prevents SQLite threading issues.
    """
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._local = threading.local()
        self._initialized = False
        
    def _get_connection(self) -> sqlite3.Connection:
        """Get or create a connection for the current thread."""
        if not hasattr(self._local, 'connection') or self._local.connection is None:
            logger.debug(f"Creating new DB connection for thread {threading.get_ident()}")
            
            # Handle both file paths and URI paths
            if self.db_path.startswith('file:'):
                connect_string = self.db_path
            else:
                connect_string = f"file:{self.db_path}?mode=rwc"
            
            conn = sqlite3.connect(
                connect_string,
                timeout=30.0,
                uri=True,
                check_same_thread=False
            )
            
            # Configure the connection
            conn.row_factory = sqlite3.Row
            
            # Try to load SQLite extensions
            try:
                conn.enable_load_extension(True)
                try:
                    conn.load_extension("vss0")
                    logger.debug("Loaded sqlite-vss extension")
                except sqlite3.OperationalError:
                    pass
                conn.enable_load_extension(False)
            except (AttributeError, Exception) as e:
                logger.debug(f"Could not load SQLite extensions: {e}")
            
            self._local.connection = conn
            
        return self._local.connection
    
    @contextmanager
    def get_cursor(self):
        """Get a cursor for the current thread with automatic cleanup."""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            yield cursor
        finally:
            cursor.close()
    
    def execute_one(self, query: str, params: Tuple = ()) -> Optional[sqlite3.Row]:
        """Execute a query and return one result."""
        with self.get_cursor() as cursor:
            cursor.execute(query, params)
            return cursor.fetchone()
    
    def execute_all(self, query: str, params: Tuple = ()) -> List[sqlite3.Row]:
        """Execute a query and return all results."""
        with self.get_cursor() as cursor:
            cursor.execute(query, params)
            return cursor.fetchall()
    
    def execute_many(self, query: str, params_list: List[Tuple]) -> None:
        """Execute a query multiple times with different parameters."""
        with self.get_cursor() as cursor:
            cursor.executemany(query, params_list)
    
    def execute_script(self, script: str) -> None:
        """Execute a SQL script."""
        with self.get_cursor() as cursor:
            cursor.executescript(script)
    
    def execute_modify(self, query: str, params: Tuple = ()) -> int:
        """Execute a modifying query and return number of affected rows."""
        with self.get_cursor() as cursor:
            cursor.execute(query, params)
            return cursor.rowcount
    
    def commit(self) -> None:
        """Commit the current transaction."""
        conn = self._get_connection()
        conn.commit()
    
    def rollback(self) -> None:
        """Rollback the current transaction."""
        conn = self._get_connection()
        conn.rollback()
    
    @contextmanager
    def transaction(self):
        """Context manager for transactions with automatic commit/rollback."""
        try:
            yield self
            self.commit()
        except Exception:
            self.rollback()
            raise
    
    def close_current_thread(self) -> None:
        """Close the connection for the current thread."""
        if hasattr(self._local, 'connection') and self._local.connection:
            logger.debug(f"Closing DB connection for thread {threading.get_ident()}")
            self._local.connection.close()
            self._local.connection = None
    
    def get_raw_connection(self) -> sqlite3.Connection:
        """Get raw connection for advanced operations (use with caution)."""
        return self._get_connection()


class DatabaseProxy:
    """
    Proxy object that provides the same interface as a direct SQLite connection
    but delegates to the thread-safe wrapper.
    """
    
    def __init__(self, db_wrapper: ThreadSafeDatabase):
        self._wrapper = db_wrapper
    
    def cursor(self):
        """Get a cursor - returns a proxy that delegates to thread-safe wrapper."""
        return CursorProxy(self._wrapper)
    
    def commit(self):
        """Commit transaction."""
        self._wrapper.commit()
    
    def rollback(self):
        """Rollback transaction."""
        self._wrapper.rollback()
    
    def close(self):
        """Close current thread's connection."""
        self._wrapper.close_current_thread()
    
    def execute(self, query: str, params: Tuple = ()):
        """Execute a query directly on the connection."""
        return self._wrapper.get_raw_connection().execute(query, params)


class CursorProxy:
    """
    Proxy object that provides cursor interface but uses thread-safe wrapper.
    """
    
    def __init__(self, db_wrapper: ThreadSafeDatabase):
        self._wrapper = db_wrapper
    
    def execute(self, query: str, params: Tuple = ()):
        """Execute a query."""
        with self._wrapper.get_cursor() as cursor:
            return cursor.execute(query, params)
    
    def executemany(self, query: str, params_list: List[Tuple]):
        """Execute query multiple times."""
        self._wrapper.execute_many(query, params_list)
    
    def fetchone(self):
        """This method should not be used with proxy - use execute_one instead."""
        raise NotImplementedError("Use db.execute_one() instead of cursor.fetchone()")
    
    def fetchall(self):
        """This method should not be used with proxy - use execute_all instead."""
        raise NotImplementedError("Use db.execute_all() instead of cursor.fetchall()")
    
    def close(self):
        """Cursors are automatically closed in the wrapper."""
        pass