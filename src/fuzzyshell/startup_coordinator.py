"""
Startup Coordinator for FuzzyShell.

Handles model initialization, model change detection, and startup-related
coordination tasks. This centralizes startup logic that was previously
scattered throughout the main FuzzyShell class.
"""

import logging
from typing import Tuple

logger = logging.getLogger('FuzzyShell.StartupCoordinator')

try:
    from .model_configs import get_active_model_key, get_model_config
except ImportError:
    def get_active_model_key():
        return "mock-model"
    def get_model_config(key=None):
        return {"repo": "mock/model", "files": {"dimensions": 384, "tokenizer_type": "mock"}}

try:
    from .fuzzyshell import tui_safe_print
except ImportError:
    def tui_safe_print(*args, **kwargs):
        print(*args, **kwargs)


class StartupCoordinator:
    """
    Coordinator for startup tasks and model change management.
    
    This class centralizes the logic for detecting model changes,
    updating metadata, and coordinating the startup sequence for FuzzyShell.
    It handles the complex interactions between model initialization,
    database cleanup, and cache management.
    """
    
    def __init__(self, fuzzyshell_instance):
        """
        Initialize the startup coordinator.
        
        Args:
            fuzzyshell_instance: FuzzyShell instance to coordinate startup for
        """
        self.fuzzyshell = fuzzyshell_instance
    
    def store_model_metadata(self, model_key: str = None):
        """Store current model information in database metadata."""
        if model_key is None:
            model_key = get_active_model_key()
        
        config = get_model_config(model_key)
        self.fuzzyshell.metadata_dal.set('model_name', model_key)
        self.fuzzyshell.metadata_dal.set('model_repo', config['repo'])
        self.fuzzyshell.metadata_dal.set('model_dimensions', str(config['files']['dimensions']))
        self.fuzzyshell.metadata_dal.set('model_tokenizer_type', config['files']['tokenizer_type'])
        logger.debug(f"Stored model metadata for: {model_key}")
    
    def handle_model_change_during_init(self):
        """Handle model change during initialization using DAL."""
        current_model = get_active_model_key()
        stored_model = self.fuzzyshell.metadata_dal.get('model_name')
        
        # If no stored model, this is first run - no need to clear anything
        if stored_model is None:
            logger.debug("No stored model metadata - first run, storing current model")
            self.store_model_metadata(current_model)
            return
            
        changed = current_model != stored_model
        
        if changed:
            logger.info(f"Model change detected: {stored_model} -> {current_model}")
            tui_safe_print(f"🔄 Model changed from {stored_model} to {current_model}")
            tui_safe_print("   Clearing old embeddings - they will be rebuilt with the new model...")
            
            # Clear embeddings and related caches using DAL
            self.fuzzyshell.command_dal.clear_embeddings()
            self.fuzzyshell.cache_dal.clear_cache()
            
            # Clear ANN index cache since embeddings changed
            if hasattr(self.fuzzyshell, 'ann_manager') and self.fuzzyshell.ann_manager:
                self.fuzzyshell.ann_manager.clear_cache()
            
            # Update model metadata
            self.store_model_metadata(current_model)
    
    def detect_model_change(self) -> Tuple[bool, str, str]:
        """
        Check if the active model has changed since last run.
        
        Returns:
            tuple: (changed, current_model, stored_model)
        """
        current_model = get_active_model_key()
        stored_model = self.fuzzyshell.get_metadata('model_name', 'terminal-minilm-l6')  # Default to default model
        changed = current_model != stored_model
        
        if changed:
            logger.info(f"Model change detected: {stored_model} -> {current_model}")
        
        return changed, current_model, stored_model
    
    def handle_model_change(self):
        """Handle model change by clearing embeddings and updating metadata."""
        changed, current_model, stored_model = self.detect_model_change()
        
        if changed:
            tui_safe_print(f"🔄 Model changed from {stored_model} to {current_model}")
            tui_safe_print("   Clearing old embeddings - they will be rebuilt with the new model...")
            
            # Clear embeddings and related caches
            self.fuzzyshell.clear_embeddings()
            self.fuzzyshell.clear_all_caches()
            
            # Update model metadata
            self.store_model_metadata(current_model)
            
            # Reset ANN index count since embeddings changed
            self.fuzzyshell.set_metadata('ann_command_count', '0')
            self.fuzzyshell.set_metadata('poorly_clustered_commands', '0')
            
            return True
        return False