"""
BM25 Scoring System for FuzzyShell.

Implements BM25 (Best Matching 25) algorithm for keyword-based search scoring.
Extracted from the main FuzzyShell monolith for clean separation of concerns.
"""

import logging
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict, Counter
import math

logger = logging.getLogger(__name__)


class BM25Scorer:
    """
    BM25 (Best Matching 25) scoring implementation.
    
    BM25 is a probabilistic ranking function that scores documents based on
    query term frequency, document length, and corpus statistics.
    
    Parameters:
        k1: Controls term frequency impact (typical: 1.2-2.0)
        b: Controls length normalization (typical: 0.75)
    """
    
    def __init__(self, command_dal, tokenizer_func=None, k1: float = 1.5, b: float = 0.75):
        """
        Initialize BM25 scorer.
        
        Args:
            command_dal: Data access layer for commands and term frequencies
            tokenizer_func: Function to tokenize text (defaults to simple word split)
            k1: Term frequency saturation parameter
            b: Length normalization parameter
        """
        self.command_dal = command_dal
        self.tokenizer_func = tokenizer_func or self._default_tokenize
        self.k1 = k1
        self.b = b
        
        # Corpus statistics
        self.total_docs = 0
        self.avg_doc_length = 0.0
        self.idf_cache = {}
        self._corpus_stats_cached = False
        
    def calculate_scores_batch(self, query_terms: List[str], 
                              command_ids: List[int]) -> np.ndarray:
        """
        Calculate BM25 scores for a batch of commands.
        
        Args:
            query_terms: List of query terms to score against
            command_ids: List of command IDs to score
            
        Returns:
            numpy array of BM25 scores for each command
        """
        if not query_terms or not command_ids:
            return np.zeros(len(command_ids))
            
        # Ensure corpus stats are up to date
        if not self._corpus_stats_cached:
            self._update_corpus_stats()
            
        # Pre-calculate IDF values for all query terms
        idf_values = {}
        for term in query_terms:
            idf_values[term] = self._get_idf(term)
            
        # Calculate BM25 scores
        scores = []
        for cmd_id in command_ids:
            score = self._calculate_single_score(query_terms, cmd_id, idf_values)
            scores.append(score)
            
        return np.array(scores)
    
    def _calculate_single_score(self, query_terms: List[str], command_id: int, 
                               idf_values: Dict[str, float]) -> float:
        """Calculate BM25 score for a single command."""
        try:
            # Get term frequencies for this command
            term_freqs = self._get_command_term_frequencies(command_id)
            if not term_freqs:
                return 0.0
                
            # Get command length for normalization
            cmd_length = self._get_command_length(command_id)
            if cmd_length == 0:
                return 0.0
                
            # Calculate BM25 score
            score = 0.0
            query_term_counts = Counter(query_terms)
            
            for term, query_freq in query_term_counts.items():
                if term not in term_freqs:
                    continue
                    
                tf = term_freqs[term]  # Term frequency in document
                idf = idf_values.get(term, 0.0)
                
                # BM25 formula
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * (cmd_length / self.avg_doc_length))
                
                term_score = idf * (numerator / denominator) * query_freq
                score += term_score
                
            return max(0.0, score)  # Ensure non-negative scores
            
        except Exception as e:
            logger.warning("Error calculating BM25 score for command %d: %s", command_id, e)
            return 0.0
    
    def _get_idf(self, term: str) -> float:
        """
        Calculate Inverse Document Frequency (IDF) for a term.
        
        IDF = log((N - df + 0.5) / (df + 0.5))
        where N is total documents and df is document frequency
        """
        if term in self.idf_cache:
            return self.idf_cache[term]
            
        try:
            # Get document frequency for this term
            doc_freq = self.command_dal.get_term_document_frequency(term)
            
            if doc_freq == 0:
                self.idf_cache[term] = 0.0
                return 0.0
                
            # BM25 IDF formula with smoothing
            idf = math.log((self.total_docs - doc_freq + 0.5) / (doc_freq + 0.5))
            
            # Ensure IDF is non-negative
            idf = max(0.0, idf)
            
            self.idf_cache[term] = idf
            return idf
            
        except Exception as e:
            logger.warning("Error calculating IDF for term '%s': %s", term, e)
            self.idf_cache[term] = 0.0
            return 0.0
    
    def _get_command_term_frequencies(self, command_id: int) -> Dict[str, float]:
        """Get term frequencies for a specific command."""
        try:
            with self.command_dal.connection() as conn:
                rows = conn.execute("""
                    SELECT term, freq FROM term_frequencies 
                    WHERE command_id = ?
                """, (command_id,)).fetchall()
                
                return {row[0]: float(row[1]) for row in rows}
                
        except Exception as e:
            logger.warning("Error getting term frequencies for command %d: %s", command_id, e)
            return {}
    
    def _get_command_length(self, command_id: int) -> int:
        """Get length of a specific command."""
        try:
            with self.command_dal.connection() as conn:
                row = conn.execute("""
                    SELECT length FROM commands WHERE id = ?
                """, (command_id,)).fetchone()
                
                return int(row[0]) if row and row[0] else 0
                
        except Exception as e:
            logger.warning("Error getting command length for command %d: %s", command_id, e)
            return 0
    
    def _update_corpus_stats(self):
        """Update corpus statistics from database."""
        try:
            # Get total document count
            self.total_docs = self.command_dal.get_command_count()
            
            # Get average document length
            self.avg_doc_length = self.command_dal.get_average_command_length()
            
            if self.avg_doc_length == 0.0:
                self.avg_doc_length = 1.0  # Avoid division by zero
                
            self._corpus_stats_cached = True
            
            logger.debug("Updated BM25 corpus stats: %d docs, avg length %.2f", 
                        self.total_docs, self.avg_doc_length)
                        
        except Exception as e:
            logger.error("Error updating corpus stats: %s", e)
            # Set sensible defaults
            self.total_docs = 1
            self.avg_doc_length = 1.0
            self._corpus_stats_cached = True
    
    def invalidate_cache(self):
        """Invalidate IDF cache and corpus stats (call when corpus changes)."""
        self.idf_cache.clear()
        self._corpus_stats_cached = False
        logger.debug("BM25 caches invalidated")
    
    def get_stats(self) -> Dict[str, any]:
        """Get current BM25 scorer statistics."""
        return {
            'total_docs': self.total_docs,
            'avg_doc_length': self.avg_doc_length,
            'cached_terms': len(self.idf_cache),
            'k1': self.k1,
            'b': self.b
        }
    
    def _default_tokenize(self, text: str) -> List[str]:
        """Default tokenization - simple word splitting."""
        import re
        return re.findall(r'\b\w+\b', text.lower())


class BM25ScoreCalculator:
    """
    Utility class for BM25 calculations without database dependencies.
    
    Useful for testing or when working with in-memory data structures.
    """
    
    @staticmethod
    def calculate_idf(term_doc_freq: int, total_docs: int) -> float:
        """Calculate IDF for a single term."""
        if term_doc_freq == 0 or total_docs == 0:
            return 0.0
            
        return math.log((total_docs - term_doc_freq + 0.5) / (term_doc_freq + 0.5))
    
    @staticmethod
    def calculate_bm25_score(query_terms: List[str], document_terms: List[str],
                           doc_length: int, avg_doc_length: float,
                           term_idf_values: Dict[str, float],
                           k1: float = 1.5, b: float = 0.75) -> float:
        """
        Calculate BM25 score for a document given all parameters.
        
        Args:
            query_terms: Terms in the query
            document_terms: Terms in the document  
            doc_length: Length of the document
            avg_doc_length: Average document length in corpus
            term_idf_values: Pre-calculated IDF values for terms
            k1: BM25 k1 parameter
            b: BM25 b parameter
            
        Returns:
            BM25 score for the document
        """
        if not query_terms or not document_terms:
            return 0.0
            
        # Count term frequencies in document
        doc_term_freqs = Counter(document_terms)
        query_term_counts = Counter(query_terms)
        
        score = 0.0
        for term, query_freq in query_term_counts.items():
            if term not in doc_term_freqs:
                continue
                
            tf = doc_term_freqs[term]
            idf = term_idf_values.get(term, 0.0)
            
            # BM25 formula
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * (doc_length / avg_doc_length))
            
            term_score = idf * (numerator / denominator) * query_freq
            score += term_score
            
        return max(0.0, score)