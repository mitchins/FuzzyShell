"""
FuzzyShell Progress Indicators - Making waiting fun!
"""

import sys
import time
import threading
from typing import Optional, Callable


class FuzzyProgress:
    """Interactive progress indicator with personality"""
    
    def __init__(self, total_steps: int = 100, show_spinner: bool = True):
        self.total_steps = total_steps
        self.current_step = 0
        self.show_spinner = show_spinner
        self.running = False
        self.spinner_thread = None
        self.message = "Processing..."
        
        # Fun animals for different moods
        self.animals = [
            "🐄", "🐮", "🐰", "🐱", "🐶", "🐸", "🐙", "🦄", "🌈", "⭐"
        ]
        self.current_animal = 0
        
        # Progress bar components
        self.bar_length = 40
        
    def start(self, message: str = "Processing..."):
        """Start the progress indicator"""
        self.message = message
        self.running = True
        self.current_step = 0
        
        if self.show_spinner:
            self.spinner_thread = threading.Thread(target=self._spinner_loop, daemon=True)
            self.spinner_thread.start()
        
        self._draw_initial()
    
    def update(self, step: int, message: Optional[str] = None):
        """Update progress to specific step"""
        if message:
            self.message = message
        
        self.current_step = min(step, self.total_steps)
        
        if not self.show_spinner:  # If not using spinner, update immediately
            self._draw_progress()
    
    def finish(self, message: str = "Complete! ✨"):
        """Finish the progress indicator"""
        self.running = False
        self.current_step = self.total_steps
        self.message = message
        
        if self.spinner_thread:
            self.spinner_thread.join(timeout=0.1)
        
        self._draw_final()
        print()  # Final newline
    
    def _draw_initial(self):
        """Draw the initial progress state"""
        if self.show_spinner:
            # Just show the message, spinner will handle updates
            pass
        else:
            self._draw_progress()
    
    def _draw_progress(self):
        """Draw the current progress state"""
        progress = self.current_step / self.total_steps
        filled_length = int(self.bar_length * progress)
        
        # Create progress bar
        bar = "█" * filled_length + "░" * (self.bar_length - filled_length)
        
        # Add some flair based on progress
        if progress < 0.33:
            emoji = "🚀"
        elif progress < 0.66:
            emoji = "⚡"
        else:
            emoji = "🌟"
        
        percentage = progress * 100
        
        # Build the line
        line = f"\r{emoji} {self.message} [{bar}] {percentage:.1f}%"
        
        # Write to terminal
        sys.stdout.write(line)
        sys.stdout.flush()
    
    def _draw_final(self):
        """Draw the final completed state"""
        bar = "█" * self.bar_length
        line = f"\r🎉 {self.message} [{bar}] 100.0%"
        sys.stdout.write(line)
        sys.stdout.flush()
    
    def _spinner_loop(self):
        """Animation loop for spinner mode"""
        spinner_chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        spinner_idx = 0
        
        while self.running:
            # Calculate progress
            progress = self.current_step / self.total_steps
            filled_length = int(self.bar_length * progress)
            bar = "█" * filled_length + "░" * (self.bar_length - filled_length)
            percentage = progress * 100
            
            # Get current spinner char and animal
            spinner = spinner_chars[spinner_idx % len(spinner_chars)]
            animal = self.animals[self.current_animal % len(self.animals)]
            
            # Occasionally change animal for fun
            if spinner_idx % 20 == 0:
                self.current_animal += 1
            
            # Build animated line
            line = f"\r{spinner} {self.message} {animal} [{bar}] {percentage:.1f}%"
            
            sys.stdout.write(line)
            sys.stdout.flush()
            
            spinner_idx += 1
            time.sleep(0.1)


class NyanProgress:
    """Nyan Cat inspired progress indicator"""
    
    def __init__(self, total_steps: int = 100):
        self.total_steps = total_steps
        self.current_step = 0
        self.running = False
        self.thread = None
        self.message = ""
        
        # Nyan cat frames
        self.nyan_frames = [
            "=^.^= ~",
            "=^o^= ~", 
            "=^.^= ~",
            "=^-^= ~"
        ]
        self.current_frame = 0
        
        # Rainbow trail
        self.rainbow = ["🌈", "🟥", "🟧", "🟨", "🟩", "🟦", "🟪"]
        self.trail_pos = 0
    
    def start(self, message: str = "Nyaning through data..."):
        """Start the nyan progress"""
        self.message = message
        self.running = True
        self.current_step = 0
        
        self.thread = threading.Thread(target=self._nyan_loop, daemon=True)
        self.thread.start()
    
    def update(self, step: int, message: Optional[str] = None):
        """Update nyan progress"""
        if message:
            self.message = message
        self.current_step = min(step, self.total_steps)
    
    def finish(self, message: str = "Nyan complete! 🌈"):
        """Finish nyan progress"""
        self.running = False
        self.message = message
        
        if self.thread:
            self.thread.join(timeout=0.1)
        
        # Final nyan
        progress = 100.0
        bar_length = 30
        filled = int(bar_length * (progress / 100))
        trail = "🌈" * filled + "░" * (bar_length - filled)
        
        line = f"\r🐱 {self.message} {trail} {progress:.1f}%"
        sys.stdout.write(line)
        sys.stdout.flush()
        print()
    
    def _nyan_loop(self):
        """Nyan animation loop"""
        while self.running:
            progress = (self.current_step / self.total_steps) * 100
            
            # Create rainbow trail
            bar_length = 30
            filled = int(bar_length * (progress / 100))
            
            # Animated rainbow trail
            trail = ""
            for i in range(filled):
                color_idx = (i + self.trail_pos) % len(self.rainbow)
                trail += self.rainbow[color_idx]
            trail += "░" * (bar_length - filled)
            
            # Get current nyan frame
            nyan = self.nyan_frames[self.current_frame % len(self.nyan_frames)]
            
            # Build line
            line = f"\r{nyan} {self.message} {trail} {progress:.1f}%"
            
            sys.stdout.write(line)
            sys.stdout.flush()
            
            self.current_frame += 1
            self.trail_pos += 1
            time.sleep(0.2)


class MooProgress:
    """Cow-themed progress indicator"""
    
    def __init__(self, total_steps: int = 100):
        self.total_steps = total_steps
        self.current_step = 0
        self.running = False
        self.thread = None
        self.message = ""
        
        # Cow states
        self.cow_states = ["🐄", "🐮", "🐄", "🐮"]
        self.current_state = 0
        
        # Grass for the cow to "eat"
        self.grass = ["🌱", "🌿", "🍃"]
        self.grass_pos = 0
    
    def start(self, message: str = "Moo-ing through data..."):
        """Start the moo progress"""
        self.message = message
        self.running = True
        self.current_step = 0
        
        self.thread = threading.Thread(target=self._moo_loop, daemon=True)
        self.thread.start()
    
    def update(self, step: int, message: Optional[str] = None):
        """Update moo progress"""
        if message:
            self.message = message
        self.current_step = min(step, self.total_steps)
    
    def finish(self, message: str = "Moo complete! 🐄"):
        """Finish moo progress"""
        self.running = False
        self.message = message
        
        if self.thread:
            self.thread.join(timeout=0.1)
        
        # Final moo
        progress = 100.0
        bar_length = 25
        filled = int(bar_length * (progress / 100))
        field = "🌱" * filled + "░" * (bar_length - filled)
        
        line = f"\r🐄 {self.message} [{field}] {progress:.1f}%"
        sys.stdout.write(line)
        sys.stdout.flush()
        print()
    
    def _moo_loop(self):
        """Moo animation loop"""
        while self.running:
            progress = (self.current_step / self.total_steps) * 100
            
            # Create grass field
            bar_length = 25
            filled = int(bar_length * (progress / 100))
            
            field = ""
            for i in range(filled):
                grass_idx = (i + self.grass_pos) % len(self.grass)
                field += self.grass[grass_idx]
            field += "░" * (bar_length - filled)
            
            # Get current cow state
            cow = self.cow_states[self.current_state % len(self.cow_states)]
            
            # Build line  
            line = f"\r{cow} {self.message} [{field}] {progress:.1f}%"
            
            sys.stdout.write(line)
            sys.stdout.flush()
            
            self.current_state += 1
            self.grass_pos += 1
            time.sleep(0.3)


def create_progress_bar(style: str = "auto", total_steps: int = 100) -> object:
    """
    Create a progress bar with the specified style.
    
    Args:
        style: "auto", "nyan", "moo", "simple", or "spinner"
        total_steps: Total number of steps for the progress
        
    Returns:
        Progress bar instance
    """
    if style == "auto":
        # Auto-select based on terminal capabilities and fun factor
        import random
        style = random.choice(["nyan", "moo", "spinner"])
    
    if style == "nyan":
        return NyanProgress(total_steps)
    elif style == "moo":
        return MooProgress(total_steps)
    elif style == "simple":
        return FuzzyProgress(total_steps, show_spinner=False)
    else:  # "spinner" or default
        return FuzzyProgress(total_steps, show_spinner=True)


# Context manager for easy use
class ProgressContext:
    """Context manager for progress bars"""
    
    def __init__(self, total_steps: int = 100, style: str = "auto", 
                 start_message: str = "Processing...", 
                 end_message: str = "Complete! ✨"):
        self.total_steps = total_steps
        self.style = style
        self.start_message = start_message
        self.end_message = end_message
        self.progress = None
    
    def __enter__(self):
        self.progress = create_progress_bar(self.style, self.total_steps)
        self.progress.start(self.start_message)
        return self.progress
    
    def __exit__(self, exc_type, _exc_val, _exc_tb):
        if self.progress:
            self.progress.finish(self.end_message)


# Example usage functions for testing
def demo_progress_bars():
    """Demo all progress bar styles"""
    print("🎯 FuzzyShell Progress Demo")
    print()
    
    # Spinner demo
    print("Spinner style:")
    with ProgressContext(100, "spinner", "Spinning through data...", "Spin complete!") as progress:
        for i in range(100):
            progress.update(i + 1)
            time.sleep(0.05)
    
    print("\n")
    
    # Nyan demo
    print("Nyan style:")
    with ProgressContext(50, "nyan", "Nyaning files...", "Nyan complete! 🌈") as progress:
        for i in range(50):
            progress.update(i + 1)
            time.sleep(0.1)
    
    print("\n")
    
    # Moo demo
    print("Moo style:")
    with ProgressContext(30, "moo", "Moo-ing commands...", "Moo complete! 🐄") as progress:
        for i in range(30):
            progress.update(i + 1)
            time.sleep(0.1)
    
    print("\n🎉 Demo complete!")


if __name__ == "__main__":
    demo_progress_bars()