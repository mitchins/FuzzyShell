"""
Smart startup for FuzzyShell that shows onboarding TUI when models need downloading.
Simple integration that hooks into the existing flow.
"""

import os
import time
import threading
from .tui.screens.onboarding import OnboardingScreen, StartupManager, OnboardingStage, ONBOARDING_PALETTE
import urwid


def needs_model_download():
    """Check if models need to be downloaded."""
    model_dir = os.path.expanduser("~/.fuzzyshell/model/")
    desc_model_dir = os.path.expanduser("~/.fuzzyshell/description_model/")
    
    model_exists = os.path.exists(model_dir) and len(os.listdir(model_dir)) > 0
    desc_exists = os.path.exists(desc_model_dir) and len(os.listdir(desc_model_dir)) > 0
    
    return not (model_exists and desc_exists)


def run_with_onboarding_if_needed(fuzzyshell_instance, main_tui_callback):
    """
    Run FuzzyShell with onboarding TUI if models need downloading.
    
    Args:
        fuzzyshell_instance: FuzzyShell instance (not yet initialized)
        main_tui_callback: Function to call after models are ready
    
    Returns:
        Result from main_tui_callback, or False if cancelled
    """
    
    if not needs_model_download():
        # Models exist, just run normally
        return main_tui_callback()
    
    # Models need downloading - show onboarding TUI
    startup_manager = StartupManager()
    onboarding_screen = OnboardingScreen(startup_manager)
    onboarding_complete = False
    startup_error = None
    
    def handle_input(key):
        nonlocal startup_error
        if key == 'esc':
            startup_error = "Cancelled by user"
            raise urwid.ExitMainLoop()
    
    # Create TUI loop
    loop = urwid.MainLoop(
        onboarding_screen,
        palette=ONBOARDING_PALETTE,
        unhandled_input=handle_input
    )
    
    def download_models_async():
        """Download models in background with simulated progress."""
        nonlocal onboarding_complete, startup_error
        
        try:
            # Stage 1: Check models
            startup_manager.update_progress(OnboardingStage.CHECKING_MODELS, 5, "Checking for models...")
            time.sleep(0.5)
            
            # Stage 2: Download embedding model
            startup_manager.update_progress(OnboardingStage.DOWNLOADING_EMBEDDING, 10, "Downloading embedding model...")
            
            # Initialize the fuzzyshell instance (this triggers actual downloads)
            # We'll monitor this with log watching or just simulate for now
            model_ready = fuzzyshell_instance.wait_for_model(timeout=60.0)
            
            if not model_ready:
                raise Exception("Model initialization failed")
            
            # Stage 3: Complete
            startup_manager.update_progress(OnboardingStage.COMPLETE, 100, "Models ready!")
            time.sleep(1.5)
            
            onboarding_complete = True
            loop.set_alarm_in(0.1, lambda loop, data: loop.stop())
            
        except Exception as e:
            startup_error = f"Download failed: {str(e)}"
            startup_manager.set_error(startup_error)
            time.sleep(2)
            loop.set_alarm_in(0.1, lambda loop, data: loop.stop())
    
    # Start download in background
    download_thread = threading.Thread(target=download_models_async, daemon=True)
    download_thread.start()
    
    # Run onboarding TUI
    try:
        loop.run()
    except Exception:
        pass
    
    # Check results
    if startup_error:
        print(f"Setup failed: {startup_error}")
        return False
    
    if not onboarding_complete:
        print("Setup was cancelled or failed")
        return False
    
    # Models are ready, run main TUI
    return main_tui_callback()


# Simple usage example:
def example_integration():
    """Example of how to integrate this into your main flow."""
    
    def create_fuzzyshell():
        """Your existing FuzzyShell creation logic."""
        from .fuzzyshell import FuzzyShell
        return FuzzyShell()
    
    def run_main_tui(fuzzyshell_instance):
        """Your existing main TUI logic."""
        from .fuzzy_tui import run_ui
        return run_ui(lambda query: fuzzyshell_instance.search(query), fuzzyshell_instance)
    
    # Create but don't initialize FuzzyShell yet
    fuzzyshell = create_fuzzyshell()
    
    # Run with onboarding if needed
    def main_tui_callback():
        return run_main_tui(fuzzyshell)
    
    return run_with_onboarding_if_needed(fuzzyshell, main_tui_callback)