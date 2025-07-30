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


class DescriptionHandler:
    def __init__(self, model_dir: str | None = None):
        logger.debug("Initializing DescriptionHandler for T5-small")
        self.model_dir = model_dir or str(Path.home() / ".fuzzyshell" / "description_model")
        self.use_t5_model = False
        self.encoder_session = None
        self.decoder_session = None
        self.tokenizer = None
        
        # T5 has separate encoder and decoder models  
        self.encoder_path = os.path.join(self.model_dir, "encoder_model_quantized.onnx")
        self.decoder_path = os.path.join(self.model_dir, "decoder_model_quantized.onnx")
        
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
            logger.debug("Checking T5-small model files")
            self._ensure_model_files()
            
            # Initialize ONNX sessions
            logger.debug("Loading T5-small encoder ONNX model")
            self.encoder_session = ort.InferenceSession(self.encoder_path, options)
            logger.debug("Loading T5-small decoder ONNX model") 
            self.decoder_session = ort.InferenceSession(self.decoder_path, options)
            
            # Initialize tokenizer from Transformers tokenizer files
            logger.debug("Loading T5-small tokenizer")
            from transformers import T5TokenizerFast
            self.tokenizer = T5TokenizerFast.from_pretrained("t5-small")
            
            self.use_t5_model = True
            logger.debug("DescriptionHandler T5 model initialization complete")
            
        except Exception as e:
            logger.warning("Failed to initialize T5 model, falling back to rule-based descriptions: %s", str(e))
            self.use_t5_model = False
            logger.debug("DescriptionHandler fallback initialization complete")

    def _download_file(self, url: str, path: str):
        """Download a file with progress bar."""
        logger.debug("Starting download from %s", url)
        try:
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
        """Download T5-small encoder and decoder models if they don't exist."""
        files_to_download = {
            'encoder': (
                self.encoder_path,
                "https://huggingface.co/google-t5/t5-small/resolve/main/onnx/encoder_model_quantized.onnx"
            ),
            'decoder': (
                self.decoder_path,
                "https://huggingface.co/google-t5/t5-small/resolve/main/onnx/decoder_model_quantized.onnx"
            )
        }
        
        for name, (path, url) in files_to_download.items():
            if os.path.exists(path):
                size_mb = os.path.getsize(path) / 1024 / 1024
                logger.debug("T5-small %s already exists (%.2f MB)", name, size_mb)
            else:
                logger.info("Downloading T5-small %s...", name)
                if not self._download_file(url, path):
                    raise RuntimeError(f"Failed to download T5-small {name}")
                logger.info("T5-small %s download complete", name)

    def generate_description(self, command: str, max_length: int = 50) -> str:
        """
        Generate a natural language description for a command using T5-small or fallback.
        
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
        """Generate description using T5 model."""
        try:
            # Tokenize the input command with T5 prefix
            input_text = f"summarize: {command}"
            inputs = self.tokenizer(input_text, return_tensors="np", padding=True, truncation=True)
            
            # Run encoder
            encoder_outputs = self.encoder_session.run(
                None,
                {
                    'input_ids': inputs['input_ids'].astype(np.int64),
                    'attention_mask': inputs['attention_mask'].astype(np.int64)
                }
            )
            
            # Get encoder hidden states
            encoder_hidden_states = encoder_outputs[0]
            
            # Initialize decoder with start token (T5 uses pad_token_id as decoder start token)
            start_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
            decoder_input_ids = np.array([[start_token_id]], dtype=np.int64)
            
            # Simple greedy decoding (you might want to implement beam search for better results)
            generated_tokens = []
            for _ in range(max_length):
                decoder_outputs = self.decoder_session.run(
                    None,
                    {
                        'input_ids': decoder_input_ids,
                        'encoder_hidden_states': encoder_hidden_states
                    }
                )
                
                # Get next token logits and select most likely token
                logits = decoder_outputs[0]
                next_token_id = np.argmax(logits[0, -1, :])
                
                # Stop if we hit the end token
                if next_token_id == self.tokenizer.eos_token_id:
                    break
                    
                generated_tokens.append(next_token_id)
                
                # Update decoder input for next iteration
                decoder_input_ids = np.concatenate([
                    decoder_input_ids,
                    np.array([[next_token_id]], dtype=np.int64)
                ], axis=1)
            
            # Decode the generated tokens
            if generated_tokens:
                description = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                description = description.strip()
            else:
                description = ""
            
            # Check if T5 description is good enough, otherwise use rule-based fallback
            if not description or len(description.strip()) < 5 or description.strip().lower() == command.lower():
                return self._generate_with_rules(command)
            
            return description
            
        except Exception as e:
            logger.error("Error generating T5 description for command '%s': %s", command, str(e))
            return self._generate_with_rules(command)
    
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
