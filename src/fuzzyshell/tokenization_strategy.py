"""
Tokenization Strategy for FuzzyShell.

Provides optimized tokenization strategies for BM25 search:
- Command-aware stopword filtering
- Punctuation preservation for command structure
- Noise reduction through length filtering
"""

import re
import time
import logging

logger = logging.getLogger('FuzzyShell.TokenizationStrategy')


class CommandTokenizationStrategy:
    """Tokenization strategy optimized for shell commands and BM25 search."""
    
    def __init__(self):
        """Initialize with shell-specific stopwords."""
        # Common shell/command stopwords to exclude from BM25 indexing
        self.SHELL_STOPWORDS = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'in', 'is', 'it', 'of', 'on', 'or', 'to', 'the', 'that', 'this',
            'with', 'will', 'was', 'were', 'been', 'have', 'has', 'had',
            # Shell-specific common terms that add little search value
            'run', 'cmd', 'sh', 'bash', 'exe', 'bin', 'usr', 'var', 'tmp',
            'home', 'root', 'etc', 'opt', 'dev', 'proc', 'sys'
        }
    
    def tokenize(self, text):
        """
        Optimized tokenization for BM25 search:
        - Focus on mid-to-low frequency terms
        - Filter out ultra-common stopwords and single characters
        - Remove one-off typos by requiring minimum length
        """
        start_time = time.time()
        
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
            if token in self.SHELL_STOPWORDS:
                continue
            # Skip purely numeric tokens that are too long (likely timestamps)
            if token.isdigit() and len(token) > 5:
                continue
            
            filtered_tokens.append(token)
        
        logger.debug("Tokenized text in %.3fs (%d->%d tokens)", 
                    time.time() - start_time, len(tokens), len(filtered_tokens))
        return filtered_tokens