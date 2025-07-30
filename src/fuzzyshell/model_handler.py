import os
import requests
import numpy as np
from tqdm import tqdm
import onnxruntime as ort
from tokenizers import Tokenizer
from pathlib import Path
import logging
# Modern Python 3.10+ - no need for typing imports for basic types

logger = logging.getLogger('FuzzyShell.ModelHandler')

# Model dimensions
MODEL_OUTPUT_DIM = 384  # Base model output dimension

class ModelHandler:
    def __init__(self, model_dir: str | None = None):
        logger.debug("Initializing ModelHandler")
        self.model_dir = model_dir or str(Path.home() / ".fuzzyshell" / "model")
        self.model_path = os.path.join(self.model_dir, "model_quantized.onnx")
        self.tokenizer_path = os.path.join(self.model_dir, "tokenizer.json")
        
        # Ensure model directory exists
        logger.debug("Using model directory: %s", self.model_dir)
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Configure ONNX Runtime session options for optimal performance
        logger.debug("Configuring ONNX Runtime options")
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.intra_op_num_threads = 1  # Single thread per session for better parallelism
        options.inter_op_num_threads = 1
        options.enable_cpu_mem_arena = False  # Disable memory arena for smaller memory footprint
        options.enable_mem_pattern = False   # Disable memory pattern optimization for faster startup
        
        # Download model and tokenizer if needed
        logger.debug("Checking model files")
        self._ensure_model_files()
        
        # Initialize ONNX session and tokenizer
        logger.debug("Loading ONNX model")
        self.session = ort.InferenceSession(self.model_path, options)
        logger.debug("Loading tokenizer")
        self.tokenizer = Tokenizer.from_file(self.tokenizer_path)
        logger.debug("ModelHandler initialization complete")

    def _download_file(self, url: str, path: str):
        """Download a file with progress bar."""
        logger.debug("Starting download from %s", url)
        try:
            response = requests.get(url, stream=True, timeout=10)
            response.raise_for_status()  # Raise an error for bad status codes
            total_size = int(response.headers.get('content-length', 0))
            
            logger.debug("File size: %.2f MB", total_size / 1024 / 1024)
            downloaded = 0
            
            with open(path, 'wb') as file:
                for data in response.iter_content(chunk_size=8192):  # Larger chunks for faster download
                    size = file.write(data)
                    downloaded += size
                    if downloaded % (1024 * 1024) == 0:  # Log every MB
                        logger.debug("Downloaded %.2f/%.2f MB", 
                                   downloaded / 1024 / 1024,
                                   total_size / 1024 / 1024)
                        
            logger.debug("Download completed successfully")
            return True
        except Exception as e:
            logger.error("Download failed: %s", str(e))
            if os.path.exists(path):
                os.remove(path)  # Remove partial download
            return False

    def _ensure_model_files(self):
        """Download model and tokenizer if they don't exist."""
        files_to_download = {
            'model': (
                self.model_path,
                "https://huggingface.co/Xenova/all-MiniLM-L6-v2/resolve/main/onnx/model_int8.onnx"
            ),
            'tokenizer': (
                self.tokenizer_path,
                "https://huggingface.co/Xenova/all-MiniLM-L6-v2/resolve/main/tokenizer.json"
            )
        }
        
        for name, (path, url) in files_to_download.items():
            if os.path.exists(path):
                size_mb = os.path.getsize(path) / 1024 / 1024
                logger.debug("%s already exists (%.2f MB)", name, size_mb)
            else:
                logger.info("Downloading %s...", name)
                if not self._download_file(url, path):
                    raise RuntimeError(f"Failed to download {name}")
                logger.info("%s download complete", name)

    def encode(self, texts: list[str], truncate_to: int | None = MODEL_OUTPUT_DIM) -> np.ndarray:
        """
        Encode a list of texts to embeddings, returning mean-pooled embeddings with shape (batch_size, MODEL_OUTPUT_DIM).
        
        Args:
            texts: List of texts to encode
            truncate_to: Optional dimension to truncate the embeddings to
        
        Returns:
            np.ndarray: Array of embeddings with shape (len(texts), output_dim)
        """
        # Tokenize the texts
        encoded = self.tokenizer.encode_batch(texts)
        
        # Prepare input tensors
        input_ids = np.array([e.ids for e in encoded], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)
        token_type_ids = np.zeros_like(input_ids)
        
        # Run inference
        outputs = self.session.run(
            None,
            {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'token_type_ids': token_type_ids
            }
        )
        
        # Get token embeddings and convert to numpy array
        token_embeddings = np.asarray(outputs[0])  # Shape: (batch_size, num_tokens, embedding_dim)
        
        # Mean pool embeddings across tokens using attention mask
        # attention_mask has 1s for real tokens and 0s for padding
        attention_mask = attention_mask[:, :, np.newaxis]  # Add embedding dimension
        masked_embeddings = token_embeddings * attention_mask  # Zero out padding tokens
        summed = np.sum(masked_embeddings, axis=1)  # Sum across tokens
        counts = np.sum(attention_mask, axis=1)  # Count real tokens
        mean_pooled = summed / np.maximum(counts, 1)  # Divide by token count, avoid div by 0
        
        # Truncate to desired output dimension if specified
        if truncate_to and truncate_to < mean_pooled.shape[1]:
            mean_pooled = mean_pooled[:, :truncate_to]
            
        return mean_pooled
