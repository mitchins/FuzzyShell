"""
Embedding Processor for FuzzyShell.

Handles embedding quantization and dequantization in different formats:
- INT8: Quantized with scale factor for space efficiency
- FP16: Half precision for balanced quality/size
- FP32: Full precision for maximum accuracy
"""

import numpy as np
import time
import logging

logger = logging.getLogger('FuzzyShell.EmbeddingProcessor')

# Import constants - these should match the main fuzzyshell module
try:
    from .model_handler import MODEL_OUTPUT_DIM
except ImportError:
    MODEL_OUTPUT_DIM = 384  # Default fallback

# Embedding storage configuration
EMBEDDING_DTYPE = np.float32  # Default to FP32 for maximum compatibility  
EMBEDDING_SCALE_FACTOR = 127.0  # For INT8 quantization


class EmbeddingProcessor:
    """Handles embedding quantization and dequantization strategies."""
    
    def __init__(self, dtype=EMBEDDING_DTYPE, scale_factor=EMBEDDING_SCALE_FACTOR, output_dim=MODEL_OUTPUT_DIM):
        """Initialize with embedding configuration."""
        self.dtype = dtype
        self.scale_factor = scale_factor
        self.output_dim = output_dim
        self.logger = logger
    
    def quantize_embedding(self, embedding, scale_factor=None):
        """
        Store embedding in the configured format (INT8, FP16, or FP32).
        Returns numpy array ready for database storage.
        """
        if scale_factor is None:
            scale_factor = self.scale_factor
            
        start_time = time.time()
        
        if self.dtype == np.int8:
            # Legacy INT8 quantization with normalization
            embedding = embedding / np.linalg.norm(embedding)
            result = np.clip(np.round(embedding * scale_factor), -127, 127).astype(np.int8)
            logger.debug("Quantized embedding to INT8 in %.3fs", time.time() - start_time)
        elif self.dtype == np.float16:
            # FP16 storage - good balance of quality vs size
            result = embedding.astype(np.float16)
            logger.debug("Converted embedding to FP16 in %.3fs", time.time() - start_time)
        elif self.dtype == np.float32:
            # FP32 storage - full precision
            result = embedding.astype(np.float32)
            logger.debug("Stored embedding as FP32 in %.3fs", time.time() - start_time)
        else:
            raise ValueError(f"Unsupported embedding dtype: {self.dtype}")
            
        return result
    
    def dequantize_embedding(self, stored_embedding, scale_factor=None):
        """
        Convert stored embedding back to float32 for computation.
        Handles INT8, FP16, and FP32 storage formats.
        Also ensures dimensional compatibility with current model.
        """
        if scale_factor is None:
            scale_factor = self.scale_factor
            
        # Ensure we have a valid numpy array first
        if not isinstance(stored_embedding, np.ndarray):
            if isinstance(stored_embedding, bytes):
                # Convert bytes to numpy array based on configured dtype
                stored_embedding = self._bytes_to_array(stored_embedding)
            else:
                stored_embedding = np.array(stored_embedding)
            
        # Check for invalid values before casting
        if len(stored_embedding) == 0:
            logger.warning("Empty embedding received, returning zeros")
            return np.zeros(self.output_dim, dtype=np.float32)
            
        # Convert to float32 based on storage format
        result = self._convert_to_float32(stored_embedding, scale_factor)
        
        # Handle dimension compatibility
        result = self._ensure_correct_dimensions(result)
            
        return result
    
    def _bytes_to_array(self, embedding_bytes):
        """Convert bytes to numpy array based on configured dtype."""
        if self.dtype == np.int8:
            return np.frombuffer(embedding_bytes, dtype=np.int8)
        elif self.dtype == np.float16:
            return np.frombuffer(embedding_bytes, dtype=np.float16)
        elif self.dtype == np.float32:
            return np.frombuffer(embedding_bytes, dtype=np.float32)
        else:
            raise ValueError(f"Unsupported embedding dtype: {self.dtype}")
    
    def _convert_to_float32(self, stored_embedding, scale_factor):
        """Convert stored embedding to float32 based on storage format."""
        if self.dtype == np.int8:
            # Dequantize INT8 back to float32, with safe casting
            try:
                result = stored_embedding.astype(np.float32) / scale_factor
            except (ValueError, RuntimeWarning) as e:
                logger.warning(f"Error converting INT8 embedding to float32: {e}")
                return np.zeros(self.output_dim, dtype=np.float32)
        elif self.dtype == np.float16:
            # Convert FP16 to FP32 for computation, with safe casting
            try:
                result = stored_embedding.astype(np.float32)
                # Check for NaN/inf values after conversion
                if np.any(np.isnan(result)) or np.any(np.isinf(result)):
                    logger.warning("NaN or inf values detected in FP16 embedding after conversion")
                    result = np.nan_to_num(result, nan=0.0, posinf=1.0, neginf=-1.0)
            except (ValueError, RuntimeWarning) as e:
                logger.warning(f"Error converting FP16 embedding to float32: {e}")
                return np.zeros(self.output_dim, dtype=np.float32)
        elif self.dtype == np.float32:
            # Already FP32, but check for invalid values
            result = stored_embedding.copy()
            if np.any(np.isnan(result)) or np.any(np.isinf(result)):
                logger.warning("NaN or inf values detected in FP32 embedding")
                result = np.nan_to_num(result, nan=0.0, posinf=1.0, neginf=-1.0)
        else:
            raise ValueError(f"Unsupported embedding dtype: {self.dtype}")
        
        return result
    
    def _ensure_correct_dimensions(self, result):
        """Ensure embedding has correct dimensions for current model."""
        # With FP32 storage, dimensions must match exactly - no silent padding/truncation
        if self.dtype == np.float32:
            # For FP32, we expect bit-perfect roundtrips - no dimension changes allowed
            if len(result) != self.output_dim:
                raise ValueError(f"FP32 embedding dimension mismatch: got {len(result)}, expected {self.output_dim}. "
                                "This indicates corrupted storage or model mismatch.")
        else:
            # Legacy handling for quantized formats (INT8/FP16) - may need dimension adjustment
            if len(result) > self.output_dim:
                logger.debug(f"Truncating stored embedding from {len(result)} to {self.output_dim} dimensions")
                result = result[:self.output_dim]
            elif len(result) < self.output_dim:
                logger.warning(f"Stored embedding has {len(result)} dimensions, expected {self.output_dim}. Padding with zeros.")
                # Pad with zeros if stored embedding is smaller
                padded = np.zeros(self.output_dim, dtype=np.float32)
                padded[:len(result)] = result
                result = padded
        
        return result