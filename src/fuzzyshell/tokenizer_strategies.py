"""
Tokenizer strategy pattern for supporting different model architectures.
Clean abstraction that selects appropriate tokenization based on model config.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Any


class TokenizerStrategy(ABC):
    """Abstract base class for tokenizer strategies."""
    
    @abstractmethod
    def prepare_inputs(self, tokenizer, texts: List[str]) -> Dict[str, np.ndarray]:
        """Prepare model inputs from tokenized texts."""
        pass


class BERTTokenizerStrategy(TokenizerStrategy):
    """Strategy for BERT-based models (requires token_type_ids)."""
    
    def prepare_inputs(self, tokenizer, texts: List[str]) -> Dict[str, np.ndarray]:
        encoded = tokenizer(texts, padding=True, truncation=True, return_tensors="np")
        
        input_ids = encoded['input_ids'].astype(np.int64)
        attention_mask = encoded['attention_mask'].astype(np.int64)
        token_type_ids = np.zeros_like(input_ids)  # BERT requires token_type_ids
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids
        }




class StockBertTokenizerStrategy(TokenizerStrategy):
    """Strategy for stock BERT models (needs token_type_ids like custom BERT)."""
    
    def prepare_inputs(self, tokenizer, texts: List[str]) -> Dict[str, np.ndarray]:
        encoded = tokenizer(texts, padding=True, truncation=True, return_tensors="np")
        
        input_ids = encoded['input_ids'].astype(np.int64)
        attention_mask = encoded['attention_mask'].astype(np.int64)
        token_type_ids = np.zeros_like(input_ids)  # Stock BERT also needs token_type_ids
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids
        }


def get_tokenizer_strategy(tokenizer_type: str) -> TokenizerStrategy:
    """Factory function to get appropriate tokenizer strategy."""
    
    strategies = {
        'bert': BERTTokenizerStrategy(),
        'stock-bert': StockBertTokenizerStrategy(),
    }
    
    strategy = strategies.get(tokenizer_type.lower())
    if not strategy:
        raise ValueError(f"Unsupported tokenizer type: {tokenizer_type}. "
                        f"Supported types: {list(strategies.keys())}")
    
    return strategy