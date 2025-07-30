#!/usr/bin/env python3
"""Test helper utilities for FuzzyShell tests"""

import sqlite3


def create_test_db_connection():
    """Create an in-memory SQLite database connection for testing.
    
    This is much cleaner than managing database paths and URIs.
    Each test gets its own isolated in-memory database.
    
    Returns:
        sqlite3.Connection: In-memory database connection
    """
    conn = sqlite3.connect(':memory:')
    # Enable foreign keys for proper referential integrity
    conn.execute('PRAGMA foreign_keys = ON')
    return conn