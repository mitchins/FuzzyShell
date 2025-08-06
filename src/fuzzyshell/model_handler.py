import os
# Disable Transformers advisory warnings about missing frameworks
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
import logging
from transformers import logging as hf_logging

# Silence the “None of PyTorch, TensorFlow >= 2.0, or Flax have been found…” message
hf_logging.set_verbosity_error()

import time
import requests
import numpy as np
from tqdm import tqdm
import onnxruntime as ort
from tokenizers import Tokenizer
from pathlib import Path
import logging
from typing import Optional, List

from .model_configs import (
    get_active_model_key, get_model_config, get_required_files, 
    TOKENIZER_FILES, validate_model_config
)
from .tokenizer_strategies import get_tokenizer_strategy

logger = logging.getLogger('FuzzyShell.ModelHandler')

# Model dimensions (will be determined by active model config)
MODEL_OUTPUT_DIM = 384  # Default, overridden by model config

class ModelHandler:
    def __init__(self, model_dir: Optional[str] = None):
        init_start = time.time()
        logger.debug("[MODEL] Initializing ModelHandler")
        
        # Get active model configuration
        self.model_key = get_active_model_key()
        self.model_config = get_model_config(self.model_key)
        logger.debug(f"[MODEL] Using model: {self.model_key} ({self.model_config['description']})")
        
        # Set up paths based on configuration
        logger.debug("[MODEL] Setting up paths...")
        
        # Allow model directory override via environment variable (useful for testing and advanced users)
        env_model_dir = os.getenv('FUZZYSHELL_MODEL_DIR')
        self.model_dir = model_dir or env_model_dir or str(Path.home() / ".fuzzyshell" / "model")
        
        # Model file path from config
        model_config_path = self.model_config["files"]["model_path"]
        self.model_path = os.path.join(self.model_dir, model_config_path)
        self.tokenizer_path = os.path.join(self.model_dir, "tokenizer.json")
        
        # Store model dimensions for runtime use
        global MODEL_OUTPUT_DIM
        MODEL_OUTPUT_DIM = self.model_config["files"]["dimensions"]
        
        # Ensure model directory exists
        logger.debug(f"[MODEL] Using model directory: {self.model_dir}")
        os.makedirs(self.model_dir, exist_ok=True)
        logger.debug(f"[MODEL] Directory setup completed in {time.time() - init_start:.3f}s")
        
        # Configure ONNX Runtime session options for optimal performance
        logger.debug("[MODEL] Configuring ONNX Runtime options")
        options_start = time.time()
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.intra_op_num_threads = 0  # Use all available cores for batch processing
        options.inter_op_num_threads = 1
        options.enable_cpu_mem_arena = False  # Disable memory arena for smaller memory footprint
        options.enable_mem_pattern = False   # Disable memory pattern optimization for faster startup
        logger.debug(f"[MODEL] ONNX options configured in {time.time() - options_start:.3f}s")
        
        # Download model and tokenizer if needed
        logger.debug("[MODEL] Checking model files")
        files_start = time.time()
        self._ensure_model_files()
        logger.debug(f"[MODEL] Model files checked in {time.time() - files_start:.3f}s")
        
        # Initialize ONNX session and tokenizer
        logger.debug("[MODEL] Loading ONNX model (THIS IS LIKELY WHERE IT HANGS)")
        session_start = time.time()
        self.session = ort.InferenceSession(self.model_path, options)
        session_time = time.time() - session_start
        logger.debug(f"[MODEL] ONNX session created in {session_time:.3f}s")
        
        logger.debug("[MODEL] Loading tokenizer")
        tokenizer_start = time.time()
        
        # Suppress transformers warnings about missing PyTorch/TensorFlow
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='.*PyTorch.*TensorFlow.*Flax.*')
            from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        
        # Initialize tokenizer strategy based on model config
        tokenizer_type = self.model_config["files"]["tokenizer_type"]
        self.tokenizer_strategy = get_tokenizer_strategy(tokenizer_type)
        logger.debug(f"[MODEL] Using tokenizer strategy: {tokenizer_type}")
        
        tokenizer_time = time.time() - tokenizer_start
        logger.debug(f"[MODEL] Tokenizer loaded in {tokenizer_time:.3f}s")
        
        total_time = time.time() - init_start
        logger.debug(f"[MODEL] ModelHandler initialization complete in {total_time:.3f}s")

    def _download_file(self, url: str, path: str):
        """Download a file with progress bar."""
        logger.debug("Starting download from %s", url)
        try:
            # Ensure parent directories exist
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
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
        """Download model and tokenizer files based on active model configuration."""
        repo = self.model_config["repo"]
        base_url = f"https://huggingface.co/{repo}/resolve/main"
        model_path_in_repo = self.model_config["files"]["model_path"]
        tokenizer_type = self.model_config["files"]["tokenizer_type"]
        
        logger.debug(f"[MODEL] Downloading from repository: {repo}")
        logger.debug(f"[MODEL] Model path in repo: {model_path_in_repo}")
        logger.debug(f"[MODEL] Tokenizer type: {tokenizer_type}")
        
        # Build download list based on configuration
        files_to_download = {
            'model': (
                self.model_path,
                f"{base_url}/{model_path_in_repo}"
            )
        }
        
        # Add tokenizer files based on type
        required_tokenizer_files = TOKENIZER_FILES[tokenizer_type]
        for filename in required_tokenizer_files:
            local_path = os.path.join(self.model_dir, filename)
            download_url = f"{base_url}/{filename}"
            files_to_download[filename] = (local_path, download_url)
        
        for name, (path, url) in files_to_download.items():
            if os.path.exists(path):
                size_mb = os.path.getsize(path) / 1024 / 1024
                logger.debug("%s already exists (%.2f MB)", name, size_mb)
            else:
                logger.info("Downloading custom terminal-trained %s...", name)
                if not self._download_file(url, path):
                    raise RuntimeError(f"Failed to download {name}")
                logger.info("Custom terminal-trained %s download complete", name)

    def encode(self, texts: List[str], truncate_to: Optional[int] = None) -> np.ndarray:
        """
        Encode a list of texts to embeddings, returning mean-pooled embeddings with shape (batch_size, MODEL_OUTPUT_DIM).
        
        Args:
            texts: List of texts to encode
            truncate_to: Optional dimension to truncate the embeddings to
        
        Returns:
            np.ndarray: Array of embeddings with shape (len(texts), output_dim)
        """
        # Use tokenizer strategy to prepare inputs based on model type
        model_inputs = self.tokenizer_strategy.prepare_inputs(self.tokenizer, texts)
        
        # Run inference with dynamically prepared inputs
        outputs = self.session.run(None, model_inputs)
        
        # Get embeddings and convert to numpy array
        embeddings = np.asarray(outputs[0])
        
        # Check if we have token embeddings (3D) or sentence embeddings (2D)
        if embeddings.ndim == 3:
            # Token embeddings: (batch_size, num_tokens, embedding_dim)
            # Mean pool embeddings across tokens using attention mask
            attention_mask = model_inputs['attention_mask'][:, :, np.newaxis]  # Add embedding dimension
            masked_embeddings = embeddings * attention_mask  # Zero out padding tokens
            summed = np.sum(masked_embeddings, axis=1)  # Sum across tokens
            counts = np.sum(attention_mask, axis=1)  # Count real tokens
            mean_pooled = summed / np.maximum(counts, 1)  # Divide by token count, avoid div by 0
        elif embeddings.ndim == 2:
            # Sentence embeddings: (batch_size, embedding_dim) - already pooled
            mean_pooled = embeddings
        else:
            raise ValueError(f"Unexpected embedding shape: {embeddings.shape}. Expected 2D or 3D array.")
        
        # Use model's native dimensions by default, allow truncation if specified
        if truncate_to is not None and truncate_to < mean_pooled.shape[1]:
            mean_pooled = mean_pooled[:, :truncate_to]
            
        return mean_pooled
    
    def get_model_info(self) -> dict:
        """Get model information for status display."""
        # Get model configuration based on active model
        from .model_configs import get_active_model_key, get_model_config
        
        active_model_key = get_active_model_key()
        config = get_model_config(active_model_key)
        
        # Build model info based on configuration
        model_info = {
            'model_key': active_model_key,
            'model_name': config['repo'],
            'model_type': config['description'],
            'dimensions': config['files']['dimensions'],
            'model_dir': self.model_dir,
            'model_path': self.model_path,
            'tokenizer_path': self.tokenizer_path,
        }
        
        # Add model-specific details
        if active_model_key == 'terminal-minilm-l6':
            model_info.update({
                'performance_improvement': '2x better cosine similarity vs base model',
                'languages': 'Multilingual support (50+ languages)',
                'training': 'Custom-trained on terminal commands and descriptions',
                'quantization': 'INT8 quantized for optimal performance',
            })
        elif active_model_key == 'stock-minilm-l6':
            model_info.update({
                'performance_improvement': 'Baseline performance',
                'languages': 'English primary',
                'training': 'General-purpose sentence embeddings',
                'quantization': 'FP32 (full precision)',
            })
        
        # Add file existence and size info
        try:
            if os.path.exists(self.model_path):
                model_size_mb = os.path.getsize(self.model_path) / (1024 * 1024)
                model_info['model_size_mb'] = f"{model_size_mb:.1f} MB"
                model_info['model_status'] = 'Available'
            else:
                model_info['model_status'] = 'Not downloaded'
                
            if os.path.exists(self.tokenizer_path):
                tokenizer_size_kb = os.path.getsize(self.tokenizer_path) / 1024
                model_info['tokenizer_size_kb'] = f"{tokenizer_size_kb:.1f} KB"
                model_info['tokenizer_status'] = 'Available'
            else:
                model_info['tokenizer_status'] = 'Not downloaded'
        except Exception as e:
            model_info['file_check_error'] = str(e)
            
        return model_info


class DescriptionHandler:
    def __init__(self, model_dir: Optional[str] = None):
        logger.debug("Initializing DescriptionHandler for custom CodeT5-small")
        self.model_dir = model_dir or str(Path.home() / ".fuzzyshell" / "description_model")
        self.use_t5_model = False
        self.encoder_session = None
        self.decoder_session = None
        self.tokenizer = None
        
        # CodeT5 has separate encoder and decoder models (using quantized int8 versions)
        self.encoder_path = os.path.join(self.model_dir, "encoder_model.onnx")
        self.decoder_path = os.path.join(self.model_dir, "decoder_model.onnx")
        
        # CodeT5 uses RoBERTa tokenizer files
        self.vocab_path = os.path.join(self.model_dir, "vocab.json")
        self.merges_path = os.path.join(self.model_dir, "merges.txt")
        self.tokenizer_config_path = os.path.join(self.model_dir, "tokenizer_config.json")
        self.special_tokens_path = os.path.join(self.model_dir, "special_tokens_map.json")
        
        # Ensure model directory exists
        logger.debug("Using description model directory: %s", self.model_dir)
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Try to initialize T5 model, fall back to rule-based if it fails
        try:
            # Configure ONNX Runtime session options for optimal performance
            logger.debug("Configuring ONNX Runtime options for T5-small")
            options = ort.SessionOptions()
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            options.intra_op_num_threads = 1
            options.inter_op_num_threads = 1
            options.enable_cpu_mem_arena = False
            options.enable_mem_pattern = False
            
            # Download model and tokenizer if needed
            logger.debug("Checking custom CodeT5-small model files")
            self._ensure_model_files()
            
            # Initialize ONNX sessions
            logger.debug("Loading custom CodeT5-small encoder ONNX model")
            self.encoder_session = ort.InferenceSession(self.encoder_path, options)
            logger.debug("Loading custom CodeT5-small decoder ONNX model") 
            self.decoder_session = ort.InferenceSession(self.decoder_path, options)
            
            # Initialize RoBERTa tokenizer for CodeT5 (uses RoBERTa tokenizer)
            logger.debug("Loading custom CodeT5-small tokenizer")
            
            # Suppress transformers warnings about missing PyTorch/TensorFlow
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='.*PyTorch.*TensorFlow.*Flax.*')
                from transformers import RobertaTokenizer
            # Load from the model directory which should have the tokenizer files
            self.tokenizer = RobertaTokenizer(
                vocab_file=self.vocab_path,
                merges_file=self.merges_path,
                unk_token="<unk>",
                bos_token="<s>",
                eos_token="</s>",
                sep_token="</s>",
                cls_token="<s>",
                pad_token="<pad>",
                mask_token="<mask>"
            )
            
            self.use_t5_model = True
            logger.debug("DescriptionHandler custom CodeT5 model initialization complete")
            
        except Exception as e:
            logger.warning("Failed to initialize custom CodeT5 model, falling back to rule-based descriptions: %s", str(e))
            self.use_t5_model = False
            logger.debug("DescriptionHandler fallback initialization complete")

    def _download_file(self, url: str, path: str):
        """Download a file with progress bar."""
        logger.debug("Starting download from %s", url)
        try:
            # Ensure parent directories exist
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            response = requests.get(url, stream=True, timeout=10)
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))
            
            logger.debug("File size: %.2f MB", total_size / 1024 / 1024)
            downloaded = 0
            
            with open(path, 'wb') as file:
                for data in response.iter_content(chunk_size=8192):
                    size = file.write(data)
                    downloaded += size
                    if downloaded % (1024 * 1024) == 0:
                        logger.debug("Downloaded %.2f/%.2f MB", 
                                   downloaded / 1024 / 1024,
                                   total_size / 1024 / 1024)
                        
            logger.debug("Download completed successfully")
            return True
        except Exception as e:
            logger.error("Download failed: %s", str(e))
            if os.path.exists(path):
                os.remove(path)
            return False

    def _ensure_model_files(self):
        """Download custom CodeT5-small encoder, decoder models and tokenizer files if they don't exist."""
        # Use custom terminal-trained CodeT5 model - quantized versions (int8) for encoder/decoder
        base_url = "https://huggingface.co/Mitchins/codet5-small-terminal-describer-ONNX/resolve/main"
        
        files_to_download = {
            'encoder': (
                self.encoder_path,
                f"{base_url}/int8/encoder_model.onnx"
            ),
            'decoder': (
                self.decoder_path,
                f"{base_url}/int8/decoder_model.onnx"
            ),
            'vocab': (
                self.vocab_path,
                f"{base_url}/vocab.json"
            ),
            'merges': (
                self.merges_path,
                f"{base_url}/merges.txt"
            ),
            'tokenizer_config': (
                self.tokenizer_config_path,
                f"{base_url}/tokenizer_config.json"
            ),
            'special_tokens': (
                self.special_tokens_path,
                f"{base_url}/special_tokens_map.json"
            )
        }
        
        for name, (path, url) in files_to_download.items():
            if os.path.exists(path):
                size_mb = os.path.getsize(path) / 1024 / 1024
                logger.debug("Custom CodeT5-small %s already exists (%.2f MB)", name, size_mb)
            else:
                logger.info("Downloading custom CodeT5-small %s...", name)
                if not self._download_file(url, path):
                    raise RuntimeError(f"Failed to download custom CodeT5-small {name}")
                logger.info("Custom CodeT5-small %s download complete", name)

    def generate_description(self, command: str, max_length: int = 50) -> str:
        """
        Generate a natural language description for a command using custom CodeT5-small or fallback.
        
        Args:
            command: The command to describe
            max_length: Maximum length of the generated description
        
        Returns:
            str: Natural language description of the command
        """
        if self.use_t5_model:
            return self._generate_with_t5(command, max_length)
        else:
            return self._generate_with_rules(command)
    
    def _generate_with_t5(self, command: str, max_length: int) -> str:
        """Generate description using simplified T5 model (encoder + decoder only)."""
        try:
            # Add the required prefix for CodeT5
            input_text = f'describe: {command}'
            input_ids = self.tokenizer(input_text, return_tensors='np').input_ids
            attention_mask = np.ones(input_ids.shape, dtype=np.int64)

            # 1. Encode input
            encoder_outputs = self.encoder_session.run(None, {
                "input_ids": input_ids,
                "attention_mask": attention_mask
            })
            encoder_hidden_states = encoder_outputs[0]

            # 2. Initialize decoder input with pad_token_id
            decoder_input_ids = np.array([[self.tokenizer.pad_token_id]], dtype=np.int64)
            generated_tokens = []

            # Simple generation without past key-value caching
            for _ in range(max_length):
                # Use decoder_session for each token (no caching optimization)
                decoder_outputs = self.decoder_session.run(None, {
                    "input_ids": decoder_input_ids,
                    "encoder_hidden_states": encoder_hidden_states,
                    "encoder_attention_mask": attention_mask
                })
                logits = decoder_outputs[0]

                # Get next token
                next_token_logits = logits[:, -1, :]
                next_token = np.argmax(next_token_logits, axis=-1)

                # Check for end of sequence
                if next_token.item() == self.tokenizer.eos_token_id:
                    break

                # Add token to sequence
                generated_tokens.append(next_token.item())
                decoder_input_ids = np.concatenate([decoder_input_ids, next_token.reshape(1, 1)], axis=-1)

            description = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            description = description.strip()
            
            # Check if T5 description is good enough, otherwise use rule-based fallback
            if (not description or len(description.strip()) < 3 or 
                description.strip().lower() == command.lower() or
                len(description.split()) == 1 or
                self._is_repetitive_description(description)):
                return self._generate_with_rules(command)
            
            return description
            
        except Exception as e:
            logger.error("Error generating T5 description for command '%s': %s", command, str(e))
            return self._generate_with_rules(command)
    
    def _is_repetitive_description(self, description: str) -> bool:
        """Check if description contains repetitive patterns that indicate poor generation."""
        words = description.split()
        if len(words) < 3:
            return False
        
        # Check for identical consecutive words (more than 2)
        consecutive_count = 1
        for i in range(1, len(words)):
            if words[i] == words[i-1]:
                consecutive_count += 1
                if consecutive_count > 2:
                    return True
            else:
                consecutive_count = 1
        
        # Check for high repetition ratio
        unique_words = set(words)
        if len(unique_words) < len(words) / 3:  # More than 2/3 repetition
            return True
            
        return False
    
    def _generate_with_rules(self, command: str) -> str:
        """Generate description using rule-based patterns."""
        import re
        
        # Clean the command for analysis
        clean_cmd = command.strip()
        
        # Basic command patterns
        patterns = {
            r'^ls\b': "List directory contents",
            r'^ls -la\b': "List all files in long format with details",
            r'^ls -l\b': "List files in long format with details", 
            r'^cd\b': "Change directory",
            r'^pwd\b': "Print current working directory",
            r'^mkdir\b': "Create directory",
            r'^rmdir\b': "Remove empty directory",
            r'^rm\b': "Remove files or directories",
            r'^cp\b': "Copy files or directories",
            r'^mv\b': "Move or rename files",
            r'^find\b': "Search for files and directories",
            r'^grep\b': "Search text patterns in files",
            r'^cat\b': "Display file contents",
            r'^less\b': "View file contents page by page",
            r'^head\b': "Display first lines of file",
            r'^tail\b': "Display last lines of file",
            r'^echo\b': "Display text or variables",
            r'^touch\b': "Create empty file or update timestamp",
            r'^chmod\b': "Change file permissions",
            r'^chown\b': "Change file ownership",
            r'^ps\b': "Show running processes",
            r'^kill\b': "Terminate processes",
            r'^top\b': "Display running processes dynamically",
            r'^df\b': "Show disk space usage",
            r'^du\b': "Show directory space usage",
            r'^tar\b': "Archive files",
            r'^zip\b': "Create compressed archive",
            r'^unzip\b': "Extract compressed archive",
            r'^wget\b': "Download files from web",
            r'^curl\b': "Transfer data from servers",
            r'^ssh\b': "Secure shell remote connection",
            r'^scp\b': "Secure copy files over network",
            r'^git\b': "Git version control command",
            r'^python\b': "Execute Python script or interpreter",
            r'^pip\b': "Python package manager",
            r'^node\b': "Execute Node.js script or REPL",
            r'^npm\b': "Node.js package manager",
            r'^docker\b': "Docker container management",
            r'^vim\b|^vi\b': "Edit files with Vim editor",
            r'^nano\b': "Edit files with Nano editor",
            r'^history\b': "Show command history",
            r'^which\b': "Locate command",
            r'^man\b': "Show manual page",
            r'^sudo\b': "Execute command with elevated privileges"
        }
        
        # Try to match common patterns
        for pattern, description in patterns.items():
            if re.match(pattern, clean_cmd, re.IGNORECASE):
                # Add specific details if available
                if pattern == r'^cd\b' and len(clean_cmd.split()) > 1:
                    target = clean_cmd.split()[1]
                    return f"Change to directory '{target}'"
                elif pattern == r'^mkdir\b' and len(clean_cmd.split()) > 1:
                    target = ' '.join(clean_cmd.split()[1:])
                    return f"Create directory '{target}'"
                elif pattern == r'^rm\b' and len(clean_cmd.split()) > 1:
                    args = clean_cmd.split()[1:]
                    if '-r' in args or '-rf' in args:
                        return "Remove files/directories recursively"
                    return "Remove files"
                elif pattern == r'^echo\b' and len(clean_cmd.split()) > 1:
                    return "Print text to output"
                
                return description
        
        # Fallback for complex commands
        if '|' in clean_cmd:
            return "Execute command pipeline"
        elif '&&' in clean_cmd:
            return "Execute commands sequentially"
        elif clean_cmd.startswith('./'):
            return "Execute local script or program"
        elif clean_cmd.endswith('.py'):
            return "Execute Python script"
        elif clean_cmd.endswith('.sh'):
            return "Execute shell script"
        
        # Final fallback
        return f"Execute: {clean_cmd.split()[0] if clean_cmd.split() else clean_cmd}"
    
    def get_model_info(self) -> dict:
        """Get description model information."""
        desc_info = {
            'model_name': 'Mitchins/codet5-small-terminal-describer-ONNX',
            'model_type': 'CodeT5-Small (Terminal-Trained)',
            'architecture': 'Encoder-Decoder Transformer',
            'model_dir': self.model_dir,
            'status': 'Available' if self.use_t5_model else 'Fallback to rule-based',
            'fallback': 'Rule-based pattern matching when T5 unavailable',
        }
        
        if self.use_t5_model:
            desc_info.update({
                'encoder_path': self.encoder_path,
                'decoder_path': self.decoder_path,
                'tokenizer_type': 'RoBERTa tokenizer',
            })
            
            # Add file size information
            try:
                files_info = []
                for name, path in [
                    ('Encoder', self.encoder_path),
                    ('Decoder', self.decoder_path), 
                    ('Vocab', self.vocab_path),
                    ('Merges', self.merges_path),
                ]:
                    if os.path.exists(path):
                        size_mb = os.path.getsize(path) / (1024 * 1024)
                        files_info.append(f"{name}: {size_mb:.1f} MB")
                        
                desc_info['model_files'] = files_info
            except Exception as e:
                desc_info['file_check_error'] = str(e)
        
        return desc_info
    
    def get_embedding_model_info(self) -> dict:
        """Get embedding model information."""
        model_info = {
            'model_key': self.model_key,
            'model_name': self.model_config['repo'],
            'description': self.model_config['description'],
            'dimensions': self.model_config['files']['dimensions'],
            'tokenizer_type': self.model_config['files']['tokenizer_type'],
            'model_size_mb': self.model_config['files'].get('model_size_mb', 'Unknown'),
            'model_path': self.model_path,
        }
        
        # Add file existence checks
        try:
            if os.path.exists(self.model_path):
                actual_size = os.path.getsize(self.model_path) / (1024 * 1024)
                model_info['actual_size_mb'] = f"{actual_size:.1f}"
                model_info['status'] = 'Available'
            else:
                model_info['status'] = 'Not Downloaded'
                
            # Check tokenizer files
            tokenizer_files = TOKENIZER_FILES[self.model_config['files']['tokenizer_type']]
            missing_files = []
            for filename in tokenizer_files:
                file_path = os.path.join(self.model_dir, filename)
                if not os.path.exists(file_path):
                    missing_files.append(filename)
            
            if missing_files:
                model_info['missing_files'] = missing_files
                if model_info['status'] == 'Available':
                    model_info['status'] = 'Incomplete'
                    
        except Exception as e:
            model_info['file_check_error'] = str(e)
            
        return model_info
