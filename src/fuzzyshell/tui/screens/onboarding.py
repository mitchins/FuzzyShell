"""
Onboarding and startup progress screen for FuzzyShell TUI.
Handles first-time model downloads and initialization.
"""

import urwid
import time
from enum import Enum
from typing import Optional, Callable


class OnboardingStage(Enum):
    """Stages of the onboarding process."""
    CHECKING_MODELS = "checking_models"
    DOWNLOADING_EMBEDDING = "downloading_embedding"
    DOWNLOADING_DESCRIPTION = "downloading_description"
    INITIALIZING = "initializing"
    COMPLETE = "complete"


class StartupManager:
    """Manages startup progress and model downloads with clean abstractions."""
    
    def __init__(self):
        self.current_stage = OnboardingStage.CHECKING_MODELS
        self.progress_percent = 0
        self.stage_message = ""
        self.error_message = ""
        self.callbacks = []
        
    def add_progress_callback(self, callback: Callable):
        """Add a callback to be notified of progress updates."""
        self.callbacks.append(callback)
    
    def update_progress(self, stage: OnboardingStage, percent: int, message: str = ""):
        """Update the current progress and notify callbacks."""
        self.current_stage = stage
        self.progress_percent = min(100, max(0, percent))
        self.stage_message = message
        
        # Notify all callbacks
        for callback in self.callbacks:
            try:
                callback(self.current_stage, self.progress_percent, self.stage_message)
            except Exception as e:
                pass  # Don't let callback errors break the flow
    
    def set_error(self, error: str):
        """Set an error message."""
        self.error_message = error
        for callback in self.callbacks:
            try:
                callback(self.current_stage, self.progress_percent, self.stage_message, error)
            except Exception:
                pass
    
    def get_stage_info(self):
        """Get current stage information."""
        stage_names = {
            OnboardingStage.CHECKING_MODELS: "Checking Models",
            OnboardingStage.DOWNLOADING_EMBEDDING: "Downloading Embedding Model (1/2)",
            OnboardingStage.DOWNLOADING_DESCRIPTION: "Downloading Description Model (2/2)",
            OnboardingStage.INITIALIZING: "Initializing FuzzyShell",
            OnboardingStage.COMPLETE: "Ready"
        }
        return {
            'stage': self.current_stage,
            'stage_name': stage_names.get(self.current_stage, "Unknown"),
            'progress': self.progress_percent,
            'message': self.stage_message,
            'error': self.error_message
        }


class OnboardingScreen(urwid.WidgetWrap):
    """TUI screen for showing onboarding progress."""
    
    def __init__(self, startup_manager: StartupManager):
        self.startup_manager = startup_manager
        self.startup_manager.add_progress_callback(self._on_progress_update)
        
        # Create UI elements
        self.title_text = urwid.Text("", align='center')
        self.stage_text = urwid.Text("", align='center')
        self.progress_bar = urwid.ProgressBar('progress_normal', 'progress_complete', current=0, done=100)
        self.message_text = urwid.Text("", align='center')
        self.status_text = urwid.Text("", align='center')
        self.error_text = urwid.Text("", align='center')
        
        # Layout
        pile = urwid.Pile([
            ('pack', urwid.Divider()),
            ('pack', urwid.Divider()),
            ('pack', self.title_text),
            ('pack', urwid.Divider()),
            ('pack', self.stage_text),
            ('pack', urwid.Divider()),
            ('pack', urwid.Text("Progress:", align='center')),
            ('pack', self.progress_bar),
            ('pack', urwid.Divider()),
            ('pack', self.message_text),
            ('pack', urwid.Divider()),
            ('pack', self.status_text),
            ('pack', urwid.Divider()),
            ('pack', self.error_text),
            ('pack', urwid.Divider()),
            ('pack', urwid.Divider()),
            ('pack', urwid.Text("Please wait...", align='center')),
        ])
        
        # Add padding
        padded = urwid.Padding(pile, align='center', width=('relative', 80))
        filled = urwid.Filler(padded, valign='middle')
        
        super().__init__(filled)
        
        # Initial update
        self._update_display()
    
    def _on_progress_update(self, stage, percent, message, error=None):
        """Callback for progress updates."""
        self._update_display()
    
    def _update_display(self):
        """Update the display with current progress."""
        info = self.startup_manager.get_stage_info()
        
        # Calculate step number based on progress
        step_num = min(4, max(1, int(info['progress'] / 25) + 1))
        if info['progress'] == 100:
            step_num = 4
        
        # Update title with step counter
        self.title_text.set_text([
            ('bold', 'Getting Started'), f' [{step_num}/4]'
        ])
        
        # Update progress bar
        self.progress_bar.set_completion(info['progress'])
        
        # Update step name (cleaner version)
        step_names = {
            OnboardingStage.CHECKING_MODELS: "Checking Setup",
            OnboardingStage.DOWNLOADING_EMBEDDING: "Downloading Models",
            OnboardingStage.DOWNLOADING_DESCRIPTION: "Processing History",
            OnboardingStage.INITIALIZING: "Finalizing",
            OnboardingStage.COMPLETE: "Complete"
        }
        clean_step_name = step_names.get(info['stage'], "Unknown")
        self.stage_text.set_text([('bold', clean_step_name)])
        
        # Update message (quip or detail)
        if info['message']:
            self.message_text.set_text(info['message'])
        else:
            self.message_text.set_text("")
        
        # Update status - simpler
        if info['progress'] == 100:
            self.status_text.set_text([('bold', '✓ Ready to go!')])
        else:
            self.status_text.set_text("")
        
        # Update error
        if info['error']:
            self.error_text.set_text([('error', f"Error: {info['error']}")])
        else:
            self.error_text.set_text("")
    
    def keypress(self, size, key):
        """Handle key presses - only allow ESC to cancel."""
        if key == 'esc':
            return 'cancel_onboarding'
        # Ignore all other keys during onboarding
        return None


# Color palette for onboarding screen
ONBOARDING_PALETTE = [
    ('progress_normal', 'white', 'dark blue'),
    ('progress_complete', 'white', 'dark green'),
    ('bold', 'white,bold', 'default'),
    ('error', 'light red', 'default'),
]


def create_model_download_progress_callback(startup_manager: StartupManager):
    """Create a progress callback function for model downloads."""
    
    def progress_callback(stage_name: str, current_model: int, total_models: int, model_name: str = ""):
        """
        Progress callback for model downloads.
        
        Args:
            stage_name: Name of the current stage
            current_model: Current model being downloaded (1-based)
            total_models: Total number of models to download
            model_name: Name of the current model being downloaded
        """
        # Map stage names to our enum
        stage_map = {
            'embedding': OnboardingStage.DOWNLOADING_EMBEDDING,
            'description': OnboardingStage.DOWNLOADING_DESCRIPTION,
            'initializing': OnboardingStage.INITIALIZING
        }
        
        stage = stage_map.get(stage_name, OnboardingStage.CHECKING_MODELS)
        
        # Calculate progress based on model completion
        base_progress = ((current_model - 1) / total_models) * 100
        
        # For simplicity, assume each model download is 50% of total progress
        if current_model <= total_models:
            progress = int(base_progress)
        else:
            progress = 100
        
        message = f"Setting up {model_name}..." if model_name else ""
        
        startup_manager.update_progress(stage, progress, message)
    
    return progress_callback